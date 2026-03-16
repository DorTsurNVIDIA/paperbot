"""Post paper summaries to Slack via Incoming Webhook."""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

SOURCE_LABELS = {
    "arxiv": "arXiv",
    "semantic_scholar": "Semantic Scholar",
    "huggingface": "Hugging Face",
}


def _format_blocks(scored_papers) -> list[dict]:
    blocks: list[dict] = [
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

    for sp in scored_papers:
        p = sp.paper
        source_label = SOURCE_LABELS.get(p.source, p.source)
        authors_str = ", ".join(p.authors[:3])
        if len(p.authors) > 3:
            authors_str += f" +{len(p.authors) - 3} more"

        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*<{p.url}|{p.title}>*  `{source_label}`\n"
                        f":busts_in_silhouette: {authors_str}\n"
                        f":mag: {sp.summary}"
                    ),
                },
            }
        )
        blocks.append({"type": "divider"})

    return blocks


def _no_results_blocks() -> list[dict]:
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    ":newspaper: No new speculative decoding / LLM efficiency papers found today."
                ),
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

    if scored_papers:
        blocks = _format_blocks(scored_papers)
    else:
        blocks = _no_results_blocks()

    payload = {"blocks": blocks}

    with httpx.Client(timeout=30) as client:
        resp = client.post(webhook_url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Slack webhook returned {resp.status_code}: {resp.text}"
            )
        logger.info("Posted %d paper(s) to Slack", len(scored_papers))
