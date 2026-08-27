import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.backfill import run_backfill
from agent.fetch import Paper


def make_paper(identifier: str, title: str) -> Paper:
    return Paper(
        id=identifier,
        title=title,
        abstract="abstract",
        authors=["Author"],
        url=f"https://arxiv.org/abs/{identifier.removeprefix('arxiv:')}",
        source="arxiv",
        published_date="2026-08-20T00:00:00+00:00",
    )


class BackfillTests(unittest.TestCase):
    def test_fetch_only_report_compares_seen_without_writing_state(self):
        known = make_paper("arxiv:known", "Known")
        missed = make_paper("arxiv:missed", "Missed")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with patch("agent.backfill.fetch_all", return_value=[known, missed]) as fetch, patch(
                "agent.backfill.load_seen", return_value={known.id}
            ), patch("agent.backfill.score_and_filter") as score:
                report = run_backfill(days=14, output_path=output, score=False)

            fetch.assert_called_once_with(
                lookback_hours=336,
                arxiv_max_results_per_query=200,
                arxiv_category_max_results=500,
            )
            score.assert_not_called()
            self.assertEqual(report["unseen_count"], 1)
            self.assertEqual(report["unseen_papers"][0]["paper_id"], missed.id)
            self.assertEqual(json.loads(output.read_text())["read_only"], True)

    def test_rejects_clear_seen_override(self):
        with patch.dict("os.environ", {"CLEAR_SEEN_PAPERS": "true"}):
            with self.assertRaises(EnvironmentError):
                run_backfill(score=False)


if __name__ == "__main__":
    unittest.main()
