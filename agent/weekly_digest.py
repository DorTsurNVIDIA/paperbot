"""Post a recap of Paperbot's own successful Slack deliveries each week."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from agent.history import POSTED_FILE, load_posted_papers
from agent.slack import post_weekly_digest

logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).parent.parent / "weekly_digest_state.json"
WEEK_PATTERN = re.compile(r"^(\d{4})-W(\d{2})$")


def _week_bounds(
    configured_week: str = "", *, now: datetime | None = None
) -> tuple[str, datetime, datetime, str]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if configured_week:
        match = WEEK_PATTERN.fullmatch(configured_week.strip())
        if not match:
            raise ValueError("DIGEST_WEEK must use ISO format YYYY-Www")
        year, week = (int(part) for part in match.groups())
        start_date = date.fromisocalendar(year, week, 1)
    else:
        target = (current - timedelta(days=7)).date()
        year, week, _ = target.isocalendar()
        start_date = date.fromisocalendar(year, week, 1)

    start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=7)
    key = f"{start_date.isocalendar().year}-W{start_date.isocalendar().week:02d}"
    end_date = (end - timedelta(days=1)).date()
    if start_date.year == end_date.year:
        label = f"{start_date.strftime('%b')} {start_date.day}–{end_date.strftime('%b')} {end_date.day}, {start_date.year}"
    else:
        label = f"{start_date.strftime('%b')} {start_date.day}, {start_date.year}–{end_date.strftime('%b')} {end_date.day}, {end_date.year}"
    return key, start, end, label


def _load_state(path: Path | None = None) -> dict:
    state_path = path or STATE_FILE
    if not state_path.exists():
        return {"posted_weeks": []}
    try:
        data = json.loads(state_path.read_text())
        if not isinstance(data, dict) or not isinstance(data.get("posted_weeks"), list):
            raise ValueError("weekly_digest_state.json has an invalid schema")
        return data
    except Exception as exc:
        raise RuntimeError(f"Could not load {state_path}: {exc}") from exc


def _save_state(state: dict, path: Path | None = None) -> None:
    state_path = path or STATE_FILE
    temporary_file = state_path.with_suffix(f"{state_path.suffix}.tmp")
    temporary_file.write_text(json.dumps(state, indent=2) + "\n")
    temporary_file.replace(state_path)


def _parse_posted_at(record: dict) -> datetime:
    value = str(record.get("posted_at") or "")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"Invalid posted_at timestamp in history: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def run_weekly_digest(
    *,
    history_path: Path | None = None,
    state_path: Path | None = None,
    now: datetime | None = None,
) -> None:
    key, start, end, label = _week_bounds(
        os.environ.get("DIGEST_WEEK", ""), now=now
    )
    force = os.environ.get("FORCE_WEEKLY_DIGEST", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    state = _load_state(state_path)
    if key in state["posted_weeks"] and not force:
        logger.info("Weekly digest %s was already posted", key)
        return

    records = [
        record
        for record in load_posted_papers(history_path or POSTED_FILE)
        if start <= _parse_posted_at(record) < end
    ]
    if not records:
        logger.info("No posted papers found for %s; skipping Slack", key)
        return

    if not post_weekly_digest(records, label):
        raise RuntimeError(f"Weekly Slack digest delivery failed for {key}")

    if os.environ.get("DRY_RUN", "").strip().lower() not in {"1", "true", "yes"}:
        posted_weeks = set(str(item) for item in state["posted_weeks"])
        posted_weeks.add(key)
        state["posted_weeks"] = sorted(posted_weeks)
        _save_state(state, state_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_weekly_digest()


if __name__ == "__main__":
    main()
