"""Deliver an explicitly approved backfill report as one Slack message."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from agent.slack import post_backfill_digest

logger = logging.getLogger(__name__)


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def load_selection(report_path: Path, expected_count: int) -> list[dict]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not read backfill report {report_path}: {exc}") from exc
    if not isinstance(report, dict) or report.get("read_only") is not True:
        raise ValueError("input is not a read-only Paperbot backfill report")
    records = report.get("accepted")
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise ValueError("backfill report has no valid accepted-paper list")
    if len(records) != expected_count:
        raise ValueError(
            f"expected exactly {expected_count} accepted papers, found {len(records)}"
        )
    paper_ids = [str(record.get("paper_id") or "") for record in records]
    if not all(paper_ids) or len(set(paper_ids)) != len(paper_ids):
        raise ValueError("accepted-paper IDs must be present and unique")
    return records


def deliver_backfill(
    *,
    report_path: Path,
    expected_count: int,
    label: str,
) -> None:
    records = load_selection(report_path, expected_count)
    if not _enabled("DRY_RUN") and not _enabled("CONFIRM_BACKFILL_POST"):
        raise EnvironmentError(
            "CONFIRM_BACKFILL_POST=true is required for live catch-up delivery"
        )
    if not post_backfill_digest(records, label):
        raise RuntimeError("Slack rejected the backfill catch-up message")
    logger.info("Backfill delivery completed for %d paper(s)", len(records))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=16)
    parser.add_argument(
        "--label", default="14-day recovery · Aug 13–27, 2026"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    deliver_backfill(
        report_path=args.report,
        expected_count=args.expected_count,
        label=args.label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
