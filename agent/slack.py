"""Post paper summaries to Slack via Incoming Webhook."""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

# Slack allows max 50 blocks per message; we use 2 (header + divider) + 2 per paper
PAPERS_PER_SLACK_MESSAGE = 24

SOURCE_LABELS = {
    "arxiv": "arXiv",
    "semantic_scholar": "Semantic Scholar",
    "huggingface": "Hugging Face",
}


def _paper_blocks(scored_papers) -> list[dict]:
    """Build blocks for a list of papers (no header)."""
    blocks: list[dict] = []
    for sp in scored_papers:
        p = sp.paper
        source_label = SOURCE_LABELS.get(p.source, p.source)
        authors_str = ", ".join(p.authors[:3])
        if len(p.authors) > 3:
            authors_str += f" +{len(p.authors) - 3} more"
        # Truncate long summary/title so we don't hit Slack text limits
        title = (p.title or "")[:250]
        summary = (sp.summary or "")[:2000]  # longer LLM summaries (Slack section ~3k limit total)
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*<{p.url}|{title}>*  `{source_label}`\n"
                        f":busts_in_silhouette: {authors_str}\n"
                        f":mag: {summary}"
                    ),
                },
            }
        )
        blocks.append({"type": "divider"})
    return blocks


def _format_blocks(scored_papers) -> list[list[dict]]:
    """Return one list of blocks per message (each ≤50 blocks)."""
    out: list[list[dict]] = []
    for i in range(0, len(scored_papers), PAPERS_PER_SLACK_MESSAGE):
        chunk = scored_papers[i : i + PAPERS_PER_SLACK_MESSAGE]
        header = (
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Daily LLM Inference & Speculative Decoding Papers",
                    "emoji": True,
                },
            },
            {"type": "divider"},
        ) if i == 0 else (
            {"type": "section", "text": {"type": "mrkdwn", "text": f"_— Part {i // PAPERS_PER_SLACK_MESSAGE + 1} —_"}},
            {"type": "divider"},
        )
        blocks = list(header) + _paper_blocks(chunk)
        out.append(blocks)
    return out


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

    with httpx.Client(timeout=30) as client:
        if scored_papers:
            messages_blocks = _format_blocks(scored_papers)
            for blocks in messages_blocks:
                resp = client.post(webhook_url, json={"blocks": blocks})
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"Slack webhook returned {resp.status_code}: {resp.text}"
                    )
            logger.info("Posted %d paper(s) to Slack (%d message(s))", len(scored_papers), len(messages_blocks))
        else:
            blocks = _no_results_blocks()
            resp = client.post(webhook_url, json={"blocks": blocks})
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Slack webhook returned {resp.status_code}: {resp.text}"
                )
            logger.info("Posted 0 paper(s) to Slack")
