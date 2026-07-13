"""Persist metadata for papers that were successfully posted to Slack."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

POSTED_FILE = Path(__file__).parent.parent / "posted_papers.json"


def load_posted_papers(path: Path | None = None) -> list[dict]:
    history_path = path or POSTED_FILE
    if not history_path.exists():
        return []
    try:
        data = json.loads(history_path.read_text())
        if not isinstance(data, list):
            raise ValueError("posted_papers.json must contain a JSON array")
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("every posted_papers.json entry must be an object")
        return data
    except Exception as exc:
        raise RuntimeError(f"Could not load {history_path}: {exc}") from exc


def save_posted_papers(records: list[dict], path: Path | None = None) -> None:
    history_path = path or POSTED_FILE
    temporary_file = history_path.with_suffix(f"{history_path.suffix}.tmp")
    temporary_file.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    )
    temporary_file.replace(history_path)


def record_posted_papers(
    scored_papers,
    delivered_ids: set[str],
    *,
    posted_at: datetime | None = None,
    path: Path | None = None,
) -> None:
    """Record successful deliveries once so weekly digests never invent history."""
    history_path = path or POSTED_FILE
    records = load_posted_papers(history_path)
    recorded_ids = {str(item.get("paper_id") or "") for item in records}
    timestamp = (posted_at or datetime.now(timezone.utc)).astimezone(timezone.utc)

    for scored in scored_papers:
        paper = scored.paper
        if paper.id not in delivered_ids or paper.id in recorded_ids:
            continue
        records.append(
            {
                "paper_id": paper.id,
                "title": paper.title,
                "url": paper.url,
                "source": paper.source,
                "published_date": paper.published_date,
                "posted_at": timestamp.isoformat(),
                "lane": "specdec" if scored.is_specdec else "inference",
                "specdec_score": scored.specdec_score,
                "inference_score": scored.inference_score,
                "tags": list(scored.tags),
                "summary": scored.summary,
            }
        )
        recorded_ids.add(paper.id)

    records.sort(key=lambda item: (str(item.get("posted_at") or ""), str(item.get("paper_id") or "")))
    save_posted_papers(records, history_path)
