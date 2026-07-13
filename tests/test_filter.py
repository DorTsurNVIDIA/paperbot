import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent.fetch import Paper
from agent.filter import score_and_filter


def paper(identifier: str, published_date: str) -> Paper:
    return Paper(
        id=identifier,
        title=identifier,
        abstract="A paper abstract",
        authors=["Ada Researcher"],
        url="https://example.com/paper",
        source="arxiv",
        published_date=published_date,
    )


class FilterTests(unittest.TestCase):
    def test_provider_access_error_aborts_remaining_requests(self):
        class AccessDenied(Exception):
            status_code = 403

        papers = [
            paper("arxiv:first", "2026-07-13T12:00:00+00:00"),
            paper("arxiv:second", "2026-07-13T11:00:00+00:00"),
        ]
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test"},
            clear=True,
        ), patch(
            "agent.filter._call_llm", side_effect=AccessDenied("not allowed")
        ) as call:
            result = score_and_filter(papers)

        self.assertEqual(result.failed_ids, {"arxiv:first", "arxiv:second"})
        self.assertEqual(call.call_count, 1)

    def test_nemotron_super_disables_thinking_for_structured_classification(self):
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
        )
        client = Mock()
        client.chat.completions.create.return_value = completion
        openai_module = SimpleNamespace(OpenAI=Mock(return_value=client))

        with patch.dict("sys.modules", {"openai": openai_module}):
            from agent.filter import _call_openai

            result = _call_openai(
                "test-key",
                "nvidia/nemotron-3-super-120b-a12b",
                "classify",
                384,
                "https://example.com/v1",
            )

        self.assertEqual(result, '{"ok": true}')
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["temperature"], 1.0)
        self.assertEqual(request["top_p"], 0.95)
        self.assertEqual(
            request["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )

    def test_glm_5_disables_thinking_for_structured_classification(self):
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
        )
        client = Mock()
        client.chat.completions.create.return_value = completion
        openai_module = SimpleNamespace(OpenAI=Mock(return_value=client))

        with patch.dict("sys.modules", {"openai": openai_module}):
            from agent.filter import _call_openai

            result = _call_openai(
                "test-key",
                "nvidia/zai-org/eccn-glm-5.2",
                "classify",
                512,
                "https://example.com/v1",
            )

        self.assertEqual(result, '{"ok": true}')
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(
            request["extra_body"],
            {"thinking": {"type": "disabled"}},
        )

    def test_dual_scores_accept_tag_and_sort_specdec_first(self):
        responses = [
            json.dumps(
                {
                    "specdec_score": 2,
                    "inference_score": 9,
                    "tags": ["serving"],
                    "summary": "A serving result.",
                }
            ),
            json.dumps(
                {
                    "specdec_score": 9,
                    "inference_score": 8,
                    "tags": ["draft-model"],
                    "summary": "A speculative decoding result.",
                }
            ),
        ]
        papers = [
            paper("arxiv:2607.00002", "2026-07-13T12:00:00+00:00"),
            paper("arxiv:2607.00001", "2026-07-13T11:00:00+00:00"),
        ]
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test"},
            clear=True,
        ), patch("agent.filter._call_llm", side_effect=responses):
            result = score_and_filter(papers)

        self.assertEqual(
            [item.paper.id for item in result.accepted],
            ["arxiv:2607.00001", "arxiv:2607.00002"],
        )
        self.assertIn("speculative-decoding", result.accepted[0].tags)

    def test_rejected_and_failed_papers_are_distinct(self):
        responses = [
            json.dumps(
                {
                    "specdec_score": 1,
                    "inference_score": 2,
                    "tags": [],
                    "summary": None,
                }
            ),
            "not-json",
        ]
        papers = [
            paper("arxiv:2607.00002", "2026-07-13T12:00:00+00:00"),
            paper("arxiv:2607.00001", "2026-07-13T11:00:00+00:00"),
        ]
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test"},
            clear=True,
        ), patch("agent.filter._call_llm", side_effect=responses):
            result = score_and_filter(papers)

        self.assertEqual(result.rejected_ids, {"arxiv:2607.00002"})
        self.assertEqual(result.failed_ids, {"arxiv:2607.00001"})

    def test_groq_cap_defers_instead_of_losing_oldest_paper(self):
        papers = [
            paper("arxiv:older", "2026-07-12T12:00:00+00:00"),
            paper("arxiv:newer", "2026-07-13T12:00:00+00:00"),
        ]
        response = json.dumps(
            {
                "specdec_score": 8,
                "inference_score": 8,
                "tags": ["speculative-decoding"],
                "summary": "A result.",
            }
        )
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "groq",
                "GROQ_API_KEY": "test",
                "GROQ_MAX_PAPERS": "1",
                "GROQ_DELAY_SEC": "0",
            },
            clear=True,
        ), patch("agent.filter._call_llm", return_value=response):
            result = score_and_filter(papers)

        self.assertEqual(result.accepted[0].paper.id, "arxiv:newer")
        self.assertEqual(result.deferred_ids, {"arxiv:older"})

    def test_full_abstract_is_sent_by_default(self):
        candidate = paper("arxiv:full", "2026-07-13T12:00:00+00:00")
        candidate.abstract = "A" * 2500 + "UNTRUNCATED_TAIL"
        response = json.dumps(
            {
                "specdec_score": 1,
                "inference_score": 1,
                "tags": [],
                "summary": None,
            }
        )
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "test"},
            clear=True,
        ), patch("agent.filter._call_llm", return_value=response) as call:
            score_and_filter([candidate])

        self.assertIn("UNTRUNCATED_TAIL", call.call_args.args[3])


if __name__ == "__main__":
    unittest.main()
