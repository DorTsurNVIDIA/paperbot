import os
import unittest
from unittest.mock import patch

from agent.fetch import Paper
from agent.filter import ScoredPaper
from agent.slack import _single_paper_blocks, post_to_slack


class SlackTests(unittest.TestCase):
    def test_specdec_label_scores_tags_and_untrusted_text_are_rendered_safely(self):
        paper = Paper(
            id="arxiv:2607.01234",
            title="A <specdec> paper | @channel",
            abstract="",
            authors=["A & B"],
            url="https://arxiv.org/abs/2607.01234",
            source="arxiv",
            published_date="2026-07-13",
        )
        scored = ScoredPaper(
            paper=paper,
            specdec_score=9,
            inference_score=8,
            tags=("speculative-decoding", "draft-model"),
            summary="Faster <verification> & better throughput.",
        )
        text = _single_paper_blocks(scored)[0]["text"]["text"]

        self.assertIn("SPECDEC", text)
        self.assertIn("9/10", text)
        self.assertIn("`draft-model`", text)
        self.assertIn("&lt;specdec&gt;", text)
        self.assertIn("A &amp; B", text)

    def test_missing_webhook_requires_explicit_dry_run(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(EnvironmentError):
                post_to_slack([])

        with patch.dict(os.environ, {"DRY_RUN": "true"}, clear=True):
            self.assertEqual(post_to_slack([]).failed_ids, set())


if __name__ == "__main__":
    unittest.main()
