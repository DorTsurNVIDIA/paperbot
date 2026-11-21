import unittest
from unittest.mock import patch

from agent.fetch import Paper
from agent.filter import ScoredPaper, ScoringResult
from agent.main import main
from agent.identity import title_identity_key
from agent.slack import DeliveryResult


class MainTests(unittest.TestCase):
    def test_failed_delivery_does_not_checkpoint_cross_source_aliases(self):
        title = "LoopFill: Pipelined Multi-Token Speculative Decoding for Looped Transformers"
        paper = Paper("alphaxiv:2611.loopfill", title, "abstract", [], "", "alphaxiv", "2026-11-14")
        paper.identity_aliases.add("semantic_scholar:loopfill")
        scored = ScoredPaper(paper, 9, 9, (), "Summary")
        scoring = ScoringResult([scored], set(), set(), set())
        for delivered in (False, True):
            saved = set()
            delivery = DeliveryResult({paper.id} if delivered else set(), set() if delivered else {paper.id})
            with patch("agent.main.fetch_all", return_value=[paper]), patch(
                "agent.main.load_seen", return_value=set()
            ), patch("agent.main.score_and_filter", return_value=scoring), patch(
                "agent.main.post_to_slack", return_value=delivery
            ), patch("agent.main.save_seen", side_effect=lambda ids: saved.update(ids)), patch("agent.main.record_posted_papers"):
                if delivered:
                    main()
                else:
                    with self.assertRaisesRegex(RuntimeError, "Slack delivery failed"):
                        main()
            if delivered:
                self.assertIn("semantic_scholar:loopfill", saved)
                self.assertIn(title_identity_key(title), saved)
            else:
                self.assertEqual(saved, set())

    def test_only_rejected_and_delivered_ids_become_seen(self):
        accepted_paper = Paper(
            id="arxiv:accepted",
            title="Accepted",
            abstract="",
            authors=[],
            url="https://example.com",
            source="arxiv",
            published_date="2026-07-13",
        )
        scored = ScoredPaper(
            paper=accepted_paper,
            specdec_score=9,
            inference_score=9,
            tags=("speculative-decoding",),
            summary="Summary",
        )
        scoring = ScoringResult(
            accepted=[scored],
            rejected_ids={"arxiv:rejected"},
            failed_ids={"arxiv:failed"},
            deferred_ids={"arxiv:deferred"},
            suppressed_ids={"arxiv:suppressed"},
        )
        saved = set()
        with patch("agent.main.fetch_all", return_value=[accepted_paper]), patch(
            "agent.main.load_seen", return_value={"arxiv:old"}
        ), patch("agent.main.filter_new", return_value=[accepted_paper]), patch(
            "agent.main.score_and_filter", return_value=scoring
        ), patch(
            "agent.main.post_to_slack",
            return_value=DeliveryResult({"arxiv:accepted"}, set()),
        ), patch("agent.main.save_seen", side_effect=lambda ids: saved.update(ids)), patch(
            "agent.main.record_posted_papers"
        ) as record:
            main()

        self.assertEqual(
            saved,
            {
                "arxiv:old",
                "arxiv:rejected",
                "arxiv:suppressed",
                "arxiv:accepted",
            },
        )
        record.assert_called_once_with([scored], {"arxiv:accepted"})

    def test_total_scoring_failure_is_saved_for_retry_and_fails_run(self):
        candidate = Paper(
            id="arxiv:failed",
            title="Failed",
            abstract="",
            authors=[],
            url="https://example.com",
            source="arxiv",
            published_date="2026-07-13",
        )
        scoring = ScoringResult(
            accepted=[],
            rejected_ids=set(),
            failed_ids={candidate.id},
            deferred_ids=set(),
        )
        saved = set()
        with patch("agent.main.fetch_all", return_value=[candidate]), patch(
            "agent.main.load_seen", return_value={"arxiv:old"}
        ), patch("agent.main.filter_new", return_value=[candidate]), patch(
            "agent.main.score_and_filter", return_value=scoring
        ), patch(
            "agent.main.post_to_slack",
            return_value=DeliveryResult(set(), set()),
        ), patch("agent.main.save_seen", side_effect=lambda ids: saved.update(ids)):
            with self.assertRaises(RuntimeError):
                main()

        self.assertEqual(saved, {"arxiv:old"})


if __name__ == "__main__":
    unittest.main()
