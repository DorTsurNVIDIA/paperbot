import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.backfill_delivery import deliver_backfill, load_selection


class BackfillDeliveryTests(unittest.TestCase):
    def _report(self, directory: str, count: int = 2) -> Path:
        path = Path(directory) / "report.json"
        path.write_text(
            json.dumps(
                {
                    "read_only": True,
                    "accepted": [
                        {
                            "paper_id": f"arxiv:{index}",
                            "title": f"Paper {index}",
                            "lane": "specdec",
                        }
                        for index in range(count)
                    ],
                }
            )
        )
        return path

    def test_load_selection_requires_exact_count(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._report(directory)
            with self.assertRaisesRegex(ValueError, "expected exactly 3"):
                load_selection(report, 3)

    def test_live_delivery_requires_explicit_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._report(directory)
            with patch.dict(os.environ, {}, clear=True), patch(
                "agent.backfill_delivery.post_backfill_digest"
            ) as post:
                with self.assertRaisesRegex(EnvironmentError, "CONFIRM_BACKFILL_POST"):
                    deliver_backfill(
                        report_path=report, expected_count=2, label="Catch-up"
                    )
            post.assert_not_called()

    def test_confirmed_delivery_posts_exact_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self._report(directory)
            with patch.dict(
                os.environ, {"CONFIRM_BACKFILL_POST": "true"}, clear=True
            ), patch(
                "agent.backfill_delivery.post_backfill_digest", return_value=True
            ) as post:
                deliver_backfill(
                    report_path=report, expected_count=2, label="Catch-up"
                )

            self.assertEqual(len(post.call_args.args[0]), 2)
            self.assertEqual(post.call_args.args[1], "Catch-up")


if __name__ == "__main__":
    unittest.main()
