import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import dedup
from agent.fetch import Paper
from agent.identity import paper_identity_keys


class DedupTests(unittest.TestCase):
    def test_historical_delivery_blocks_a_new_source_with_the_same_title(self):
        title = "LoopFill: Pipelined Multi-Token Speculative Decoding for Looped Transformers"
        paper = Paper("semantic_scholar:loopfill", title.upper(), "", [], "", "semantic_scholar", "2026-11-14")
        with tempfile.TemporaryDirectory() as directory:
            seen_file = Path(directory) / "seen.json"
            seen_file.write_text('["alphaxiv:2611.loopfill"]')
            seen_file.with_name("posted_papers.json").write_text(json.dumps([
                {"paper_id": "alphaxiv:2611.loopfill", "title": title}
            ]))
            with patch.object(dedup, "SEEN_FILE", seen_file), patch.dict(os.environ, {}, clear=True):
                seen = dedup.load_seen()
                self.assertEqual(dedup.filter_new([paper], seen), [])
            with patch.object(dedup, "SEEN_FILE", seen_file), patch.dict(os.environ, {"CLEAR_SEEN_PAPERS": "true"}):
                self.assertEqual(dedup.filter_new([paper], dedup.load_seen()), [paper])

    def test_title_alias_survives_state_roundtrip(self):
        title = "Efficient Speculative Decoding for Large Language Models"
        original = Paper("alphaxiv:2611.exclusive", title, "", [], "", "alphaxiv", "2026-11-14")
        mirror = Paper("semantic_scholar:mirror", title, "", [], "", "semantic_scholar", "2026-11-14")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(dedup, "SEEN_FILE", Path(directory) / "seen.json"), patch.dict(os.environ, {}, clear=True):
                dedup.save_seen(paper_identity_keys(original))
                self.assertEqual(dedup.filter_new([mirror], dedup.load_seen()), [])

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
