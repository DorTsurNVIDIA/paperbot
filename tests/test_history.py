import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from agent.fetch import Paper
from agent.filter import ScoredPaper
from agent.history import load_posted_papers, record_posted_papers


def scored(identifier: str, *, specdec: int = 9, inference: int = 8) -> ScoredPaper:
    paper = Paper(
        id=identifier,
        title=f"Title {identifier}",
        abstract="",
        authors=[],
        url=f"https://example.com/{identifier}",
        source="arxiv",
        published_date="2026-07-13",
    )
    return ScoredPaper(
        paper=paper,
        specdec_score=specdec,
        inference_score=inference,
        tags=("speculative-decoding",) if specdec >= 6 else ("serving",),
        summary="Summary",
    )


class HistoryTests(unittest.TestCase):
    def test_records_only_successful_deliveries_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "posted.json"
            candidates = [
                scored("arxiv:delivered"),
                scored("arxiv:failed", specdec=1, inference=9),
            ]
            timestamp = datetime(2026, 7, 13, 12, tzinfo=timezone.utc)

            record_posted_papers(
                candidates,
                {"arxiv:delivered"},
                posted_at=timestamp,
                path=path,
            )
            record_posted_papers(
                candidates,
                {"arxiv:delivered"},
                posted_at=timestamp,
                path=path,
            )

            records = load_posted_papers(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["paper_id"], "arxiv:delivered")
        self.assertEqual(records[0]["lane"], "specdec")
        self.assertEqual(records[0]["posted_at"], timestamp.isoformat())

    def test_corrupt_history_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "posted.json"
            path.write_text(json.dumps({"not": "a list"}))
            with self.assertRaises(RuntimeError):
                load_posted_papers(path)


if __name__ == "__main__":
    unittest.main()
