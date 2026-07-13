import unittest
from unittest.mock import patch

from agent.fetch import Paper
from agent.filter import ScoredPaper, ScoringResult
from agent.main import main
from agent.slack import DeliveryResult


class MainTests(unittest.TestCase):
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
        )
        saved = set()
        with patch("agent.main.fetch_all", return_value=[accepted_paper]), patch(
            "agent.main.load_seen", return_value={"arxiv:old"}
        ), patch("agent.main.filter_new", return_value=[accepted_paper]), patch(
            "agent.main.score_and_filter", return_value=scoring
        ), patch(
            "agent.main.post_to_slack",
            return_value=DeliveryResult({"arxiv:accepted"}, set()),
        ), patch("agent.main.save_seen", side_effect=lambda ids: saved.update(ids)):
            main()

        self.assertEqual(
            saved, {"arxiv:old", "arxiv:rejected", "arxiv:accepted"}
        )

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
