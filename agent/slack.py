"""Post paper summaries to Slack via Incoming Webhook."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

SLACK_POST_DELAY = 1.2  # seconds between messages to avoid Slack rate limits
SLACK_MAX_ATTEMPTS = 3

SOURCE_LABELS = {
    "arxiv": "arXiv",
    "semantic_scholar": "Semantic Scholar",
    "huggingface": "Hugging Face",
}


@dataclass
class DeliveryResult:
    delivered_ids: set[str]
    failed_ids: set[str]
    simulated: bool = False


def _escape_mrkdwn(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "¦")
    )


def _safe_url(value: str) -> str:
    return value.replace("|", "%7C").replace(">", "%3E")


def _dry_run_enabled() -> bool:
    return os.environ.get("DRY_RUN", "").strip().lower() in {"1", "true", "yes"}


def _single_paper_blocks(sp) -> list[dict]:
    """Build blocks for one paper as its own Slack message."""
    p = sp.paper
    source_label = SOURCE_LABELS.get(p.source, p.source)
    authors_str = ", ".join(_escape_mrkdwn(author) for author in p.authors[:3])
    if len(p.authors) > 3:
        authors_str += f" +{len(p.authors) - 3} more"
    title = _escape_mrkdwn((p.title or "")[:250])
    summary = _escape_mrkdwn((sp.summary or "")[:1800])
    tags = " ".join(f"`{tag}`" for tag in sp.tags[:6])
    if sp.is_specdec:
        relevance = (
            f":dart: *SPECDEC* {sp.specdec_score}/10"
            f"  ·  inference {sp.inference_score}/10"
        )
    else:
        relevance = (
            f":gear: *INFERENCE* {sp.inference_score}/10"
            f"  ·  specdec {sp.specdec_score}/10"
        )
    metadata = f"*<{_safe_url(p.url)}|{title}>*  `{source_label}`\n{relevance}\n"
    if tags:
        metadata += f"{tags}\n"
    if authors_str:
        metadata += f":busts_in_silhouette: {authors_str}\n"
    metadata += f":mag: {summary}"
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": metadata,
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
                "text": (
                    ":newspaper: No new speculative decoding / LLM efficiency "
                    "papers found today."
                ),
            },
        }
    ]


def _chunk_mrkdwn_lines(lines: list[str], limit: int = 2800) -> list[str]:
    chunks: list[str] = []
    current = ""
    for line in lines:
        candidate = f"{current}\n{line}" if current else line
        if current and len(candidate) > limit:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _weekly_digest_blocks(records: list[dict], label: str) -> list[dict]:
    specdec = [item for item in records if item.get("lane") == "specdec"]
    inference = [item for item in records if item.get("lane") != "specdec"]
    specdec.sort(
        key=lambda item: (
            int(item.get("specdec_score") or 0),
            int(item.get("inference_score") or 0),
        ),
        reverse=True,
    )
    inference.sort(
        key=lambda item: (
            int(item.get("inference_score") or 0),
            int(item.get("specdec_score") or 0),
        ),
        reverse=True,
    )

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"Paperbot Weekly — {label}"[:150],
                "emoji": True,
            },
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"*{len(specdec)}* speculative-decoding · "
                        f"*{len(inference)}* broader-inference paper(s)"
                    ),
                }
            ],
        },
    ]

    def add_lane(title: str, items: list[dict]) -> None:
        if not items:
            return
        blocks.append({"type": "divider"})
        lines = [f"*{title}*"]
        for item in items:
            paper_title = _escape_mrkdwn(str(item.get("title") or "Untitled")[:220])
            url = _safe_url(str(item.get("url") or ""))
            linked_title = f"<{url}|{paper_title}>" if url else paper_title
            specdec_score = int(item.get("specdec_score") or 0)
            inference_score = int(item.get("inference_score") or 0)
            tags = " ".join(
                f"`{_escape_mrkdwn(str(tag))}`"
                for tag in (item.get("tags") or [])[:3]
            )
            suffix = f" — S{specdec_score} · I{inference_score}"
            if tags:
                suffix += f" · {tags}"
            lines.append(f"• {linked_title}{suffix}")
        for chunk in _chunk_mrkdwn_lines(lines):
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": chunk},
                }
            )

    add_lane(":dart: Speculative decoding", specdec)
    add_lane(":gear: Broader inference highlights", inference)
    return blocks


def _post_blocks(client: httpx.Client, webhook_url: str, blocks: list[dict]) -> bool:
    for attempt in range(SLACK_MAX_ATTEMPTS):
        try:
            response = client.post(webhook_url, json={"blocks": blocks})
        except httpx.HTTPError as exc:
            logger.warning("Slack request failed: %s", exc)
            response = None

        if response is not None and response.status_code == 200:
            return True
        if response is not None and response.status_code not in {429, 500, 502, 503, 504}:
            logger.error(
                "Slack webhook returned %d: %s", response.status_code, response.text
            )
            return False
        if attempt + 1 < SLACK_MAX_ATTEMPTS:
            retry_after = response.headers.get("retry-after", "") if response else ""
            try:
                delay = min(float(retry_after), 30.0) if retry_after else 2**attempt
            except ValueError:
                delay = 2**attempt
            time.sleep(delay)
    return False


def post_weekly_digest(records: list[dict], label: str) -> bool:
    """Post one weekly recap message through the existing incoming webhook."""
    if _dry_run_enabled():
        logger.warning(
            "DRY_RUN is enabled — skipping weekly Slack delivery for %s", label
        )
        return True
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        raise EnvironmentError("SLACK_WEBHOOK_URL is required for weekly delivery")
    with httpx.Client(timeout=30) as client:
        delivered = _post_blocks(
            client, webhook_url, _weekly_digest_blocks(records, label)
        )
    if delivered:
        logger.info("Posted weekly digest with %d paper(s)", len(records))
    return delivered


def post_to_slack(scored_papers, *, announce_empty: bool = True) -> DeliveryResult:
    """Post in ranked order and report exactly which papers were delivered."""
    scored_papers = sorted(
        scored_papers,
        key=lambda item: (
            item.is_specdec,
            item.specdec_score if item.is_specdec else item.inference_score,
            item.inference_score if item.is_specdec else item.specdec_score,
        ),
        reverse=True,
    )
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        if _dry_run_enabled():
            logger.warning("DRY_RUN is enabled — skipping Slack delivery")
            return DeliveryResult(
                delivered_ids={sp.paper.id for sp in scored_papers},
                failed_ids=set(),
                simulated=True,
            )
        raise EnvironmentError(
            "SLACK_WEBHOOK_URL is not set; set DRY_RUN=true only when intentionally "
            "running without delivery"
        )

    with httpx.Client(timeout=30) as client:
        if not scored_papers:
            if announce_empty and not _post_blocks(
                client, webhook_url, _no_results_blocks()
            ):
                raise RuntimeError("Slack no-results post failed after retries")
            if announce_empty:
                logger.info("Posted 0 paper(s) to Slack")
            return DeliveryResult(set(), set())

        # Header message
        if not _post_blocks(client, webhook_url, _header_blocks()):
            return DeliveryResult(
                delivered_ids=set(), failed_ids={sp.paper.id for sp in scored_papers}
            )

        # One message per paper
        delivered_ids: set[str] = set()
        failed_ids: set[str] = set()
        for i, sp in enumerate(scored_papers):
            if i > 0:
                time.sleep(SLACK_POST_DELAY)
            blocks = _single_paper_blocks(sp)
            if _post_blocks(client, webhook_url, blocks):
                delivered_ids.add(sp.paper.id)
            else:
                failed_ids.add(sp.paper.id)

        logger.info(
            "Posted %d paper(s) to Slack; %d failed",
            len(delivered_ids),
            len(failed_ids),
        )
        return DeliveryResult(delivered_ids, failed_ids)
