"""Load and save seen paper IDs to avoid duplicate Slack posts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SEEN_FILE = Path(__file__).parent.parent / "seen_papers.json"


def load_seen() -> set[str]:
    if not SEEN_FILE.exists():
        return set()
    try:
        data = json.loads(SEEN_FILE.read_text())
        return set(data)
    except Exception as exc:
        logger.warning("Could not load seen_papers.json: %s", exc)
        return set()


def save_seen(ids: set[str]) -> None:
    SEEN_FILE.write_text(json.dumps(sorted(ids), indent=2))


def filter_new(papers, seen: set[str]):
    """Return only papers whose IDs are not in *seen*."""
    return [p for p in papers if p.id not in seen]
