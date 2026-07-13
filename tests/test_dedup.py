import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import dedup


class DedupTests(unittest.TestCase):
    def test_load_seen_canonicalizes_and_collapses_legacy_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            seen_file = Path(directory) / "seen.json"
            seen_file.write_text(
                json.dumps(["arxiv:2607.01234v1", "hf:2607.01234"])
            )
            with patch.object(dedup, "SEEN_FILE", seen_file), patch.dict(
                os.environ, {}, clear=True
            ):
                self.assertEqual(dedup.load_seen(), {"arxiv:2607.01234"})

    def test_save_seen_is_canonical_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            seen_file = Path(directory) / "seen.json"
            with patch.object(dedup, "SEEN_FILE", seen_file):
                dedup.save_seen({"hf:2607.01234", "arxiv:2607.01234v2"})
            self.assertEqual(json.loads(seen_file.read_text()), ["arxiv:2607.01234"])
            self.assertFalse(Path(f"{seen_file}.tmp").exists())

    def test_corrupt_seen_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            seen_file = Path(directory) / "seen.json"
            seen_file.write_text("not-json")
            with patch.object(dedup, "SEEN_FILE", seen_file):
                with self.assertRaises(RuntimeError):
                    dedup.load_seen()


if __name__ == "__main__":
    unittest.main()
