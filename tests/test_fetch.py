import unittest
import datetime
import types
from unittest.mock import Mock, patch

import httpx

from agent.fetch import (
    ARXIV_CATEGORIES,
    ARXIV_QUERIES,
    Paper,
    SourceUnavailableError,
    _s2_request,
    fetch_all,
    fetch_arxiv,
)


def make_paper(identifier: str, source: str, title: str) -> Paper:
    return Paper(
        id=identifier,
        title=title,
        abstract="abstract",
        authors=[],
        url="https://example.com",
        source=source,
        published_date="2026-07-13",
    )


class FetchTests(unittest.TestCase):
    def test_arxiv_uses_client_results_api(self):
        result = types.SimpleNamespace(
            published=datetime.datetime.now(datetime.timezone.utc),
            entry_id="https://arxiv.org/abs/2608.20530v1",
            title="LiLiCorr",
            summary="Speculative decoding with correlated parallel drafts.",
            authors=[types.SimpleNamespace(name="Researcher")],
        )
        client = Mock()
        client.results.side_effect = [[result]] + [
            [] for _ in range(len(ARXIV_QUERIES) + 1)
        ]
        fake_arxiv = types.SimpleNamespace(
            Client=Mock(return_value=client),
            Search=Mock(side_effect=lambda **kwargs: kwargs),
            SortCriterion=types.SimpleNamespace(SubmittedDate="submitted"),
        )

        with patch.dict("sys.modules", {"arxiv": fake_arxiv}):
            papers = fetch_arxiv()

        self.assertEqual([paper.id for paper in papers], ["arxiv:2608.20530"])
        self.assertEqual(client.results.call_count, len(ARXIV_QUERIES) + 2)
        self.assertIsInstance(client.results.call_args_list[0].args[0], dict)

    def test_arxiv_total_outage_fails_source_health_check(self):
        client = Mock()
        client.results.side_effect = RuntimeError("upstream unavailable")
        fake_arxiv = types.SimpleNamespace(
            Client=Mock(return_value=client),
            Search=Mock(side_effect=lambda **kwargs: kwargs),
            SortCriterion=types.SimpleNamespace(SubmittedDate="submitted"),
        )

        with patch.dict("sys.modules", {"arxiv": fake_arxiv}):
            with self.assertRaisesRegex(SourceUnavailableError, "all .* searches failed"):
                fetch_arxiv()

    def test_fetch_all_deduplicates_canonical_identity_across_sources(self):
        arxiv = make_paper("arxiv:2607.01234", "arxiv", "arXiv copy")
        huggingface = make_paper(
            "arxiv:2607.01234", "huggingface", "Hugging Face copy"
        )
        semantic_scholar = make_paper("s2:other", "semantic_scholar", "Other")
        with patch("agent.fetch.fetch_arxiv", return_value=[arxiv]), patch(
            "agent.fetch.fetch_semantic_scholar", return_value=[semantic_scholar]
        ), patch("agent.fetch.fetch_huggingface", return_value=[huggingface]):
            result = fetch_all()

        self.assertEqual([item.id for item in result], ["arxiv:2607.01234", "s2:other"])
        self.assertEqual(result[0].source, "arxiv")

    def test_fetch_all_propagates_required_source_outage(self):
        with patch(
            "agent.fetch.fetch_arxiv",
            side_effect=SourceUnavailableError("arXiv unavailable"),
        ), patch("agent.fetch.fetch_semantic_scholar") as semantic_scholar:
            with self.assertRaises(SourceUnavailableError):
                fetch_all()

        semantic_scholar.assert_not_called()

    def test_semantic_scholar_retries_transient_rate_limit(self):
        request = httpx.Request("GET", "https://example.com")
        client = Mock()
        client.get.side_effect = [
            httpx.Response(429, request=request, headers={"retry-after": "0"}),
            httpx.Response(200, request=request, json={"data": []}),
        ]
        with patch("agent.fetch.time.sleep"):
            self.assertEqual(_s2_request(client, {}, {}), {"data": []})
        self.assertEqual(client.get.call_count, 2)

    def test_semantic_scholar_does_not_retry_authentication_error(self):
        request = httpx.Request("GET", "https://example.com")
        client = Mock()
        client.get.return_value = httpx.Response(401, request=request)
        with self.assertRaises(httpx.HTTPStatusError):
            _s2_request(client, {}, {})
        self.assertEqual(client.get.call_count, 1)


if __name__ == "__main__":
    unittest.main()
