"""Post paper summaries to Slack via Incoming Webhook."""

from __future__ import annotations

import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

SLACK_POST_DELAY = 1.2  # seconds between messages to avoid Slack rate limits

SOURCE_LABELS = {
    "arxiv": "arXiv",
    "semantic_scholar": "Semantic Scholar",
    "huggingface": "Hugging Face",
}


def _single_paper_blocks(sp) -> list[dict]:
    """Build blocks for one paper as its own Slack message."""
    p = sp.paper
    source_label = SOURCE_LABELS.get(p.source, p.source)
    authors_str = ", ".join(p.authors[:3])
    if len(p.authors) > 3:
        authors_str += f" +{len(p.authors) - 3} more"
    title = (p.title or "")[:250]
    summary = (sp.summary or "")[:2000]
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*<{p.url}|{title}>*  `{source_label}`  score: {sp.score}/10\n"
                    f":busts_in_silhouette: {authors_str}\n"
                    f":mag: {summary}"
                ),
            },
        },
    ]


def _header_blocks() -> list[dict]:
    return [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "Daily LLM Inference & Speculative Decoding Papers",
                "emoji": True,
            },
        },
        {"type": "divider"},
    ]


def _no_results_blocks() -> list[dict]:
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": ":newspaper: No new speculative decoding / LLM efficiency papers found today.",
            },
        }
    ]


def post_to_slack(scored_papers) -> None:
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.warning(
            "SLACK_WEBHOOK_URL is not set — skipping Slack post. "
            "Add the secret in repo Settings → Secrets and variables → Actions to receive summaries in Slack."
        )
        return

    with httpx.Client(timeout=30) as client:
        if not scored_papers:
            resp = client.post(webhook_url, json={"blocks": _no_results_blocks()})
            if resp.status_code != 200:
                raise RuntimeError(f"Slack webhook returned {resp.status_code}: {resp.text}")
            logger.info("Posted 0 paper(s) to Slack")
            return

        # Header message
        resp = client.post(webhook_url, json={"blocks": _header_blocks()})
        if resp.status_code != 200:
            raise RuntimeError(f"Slack webhook returned {resp.status_code}: {resp.text}")

        # One message per paper
        for i, sp in enumerate(scored_papers):
            if i > 0:
                time.sleep(SLACK_POST_DELAY)
            blocks = _single_paper_blocks(sp)
            resp = client.post(webhook_url, json={"blocks": blocks})
            if resp.status_code != 200:
                raise RuntimeError(f"Slack webhook returned {resp.status_code}: {resp.text}")

        logger.info("Posted %d paper(s) to Slack (%d messages)", len(scored_papers), len(scored_papers) + 1)
