import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from agent.weekly_digest import _week_bounds, run_weekly_digest


class WeeklyDigestTests(unittest.TestCase):
    def test_default_week_is_the_previous_completed_iso_week(self):
        key, start, end, label = _week_bounds(
            now=datetime(2026, 7, 20, 9, tzinfo=timezone.utc)
        )

        self.assertEqual(key, "2026-W29")
        self.assertEqual(start.isoformat(), "2026-07-13T00:00:00+00:00")
        self.assertEqual(end.isoformat(), "2026-07-20T00:00:00+00:00")
        self.assertEqual(label, "Jul 13–Jul 19, 2026")

    def test_posts_history_once_and_persists_idempotency_state(self):
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "posted.json"
            state_path = Path(directory) / "state.json"
            history_path.write_text(
                json.dumps(
                    [
                        {
                            "paper_id": "arxiv:specdec",
                            "posted_at": "2026-07-14T10:00:00+00:00",
                            "lane": "specdec",
                        },
                        {
                            "paper_id": "arxiv:inference",
                            "posted_at": "2026-07-18T10:00:00+00:00",
                            "lane": "inference",
                        },
                        {
                            "paper_id": "arxiv:older",
                            "posted_at": "2026-07-06T10:00:00+00:00",
                            "lane": "specdec",
                        },
                    ]
                )
            )
            now = datetime(2026, 7, 20, 9, tzinfo=timezone.utc)
            with patch.dict(os.environ, {}, clear=True), patch(
                "agent.weekly_digest.post_weekly_digest", return_value=True
            ) as post:
                run_weekly_digest(
                    history_path=history_path, state_path=state_path, now=now
                )
                run_weekly_digest(
                    history_path=history_path, state_path=state_path, now=now
                )

            state = json.loads(state_path.read_text())

        post.assert_called_once()
        posted_records, label = post.call_args.args
        self.assertEqual(
            {item["paper_id"] for item in posted_records},
            {"arxiv:specdec", "arxiv:inference"},
        )
        self.assertEqual(label, "Jul 13–Jul 19, 2026")
        self.assertEqual(state, {"posted_weeks": ["2026-W29"]})


if __name__ == "__main__":
    unittest.main()
