"""Load and save seen paper IDs to avoid duplicate Slack posts."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from agent.identity import canonicalize_stored_id

logger = logging.getLogger(__name__)

SEEN_FILE = Path(__file__).parent.parent / "seen_papers.json"


def load_seen() -> set[str]:
    """Load seen paper IDs. If CLEAR_SEEN_PAPERS env is set (e.g. 'true'), return empty set."""
    if os.environ.get("CLEAR_SEEN_PAPERS", "").strip().lower() in ("1", "true", "yes"):
        logger.info("CLEAR_SEEN_PAPERS is set — treating all papers as new")
        return set()
    if not SEEN_FILE.exists():
        return set()
    try:
        data = json.loads(SEEN_FILE.read_text())
        if not isinstance(data, list):
            raise ValueError("seen_papers.json must contain a JSON array")
        return {
            canonicalize_stored_id(str(paper_id))
            for paper_id in data
            if str(paper_id).strip()
        }
    except Exception as exc:
        # An empty fallback would make every historical paper look new and could
        # flood Slack. Fail closed; CLEAR_SEEN_PAPERS is the explicit reset path.
        raise RuntimeError(f"Could not load {SEEN_FILE}: {exc}") from exc


def save_seen(ids: set[str]) -> None:
    """Atomically persist canonical IDs so an interrupted write cannot corrupt state."""
    canonical_ids = {
        canonicalize_stored_id(paper_id) for paper_id in ids if paper_id
    }
    temporary_file = SEEN_FILE.with_suffix(f"{SEEN_FILE.suffix}.tmp")
    temporary_file.write_text(json.dumps(sorted(canonical_ids), indent=2) + "\n")
    temporary_file.replace(SEEN_FILE)


def filter_new(papers, seen: set[str]):
    """Return only papers whose IDs are not in *seen*."""
    return [p for p in papers if p.id not in seen]
