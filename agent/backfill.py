"""Read-only historical crawl and scoring report.

This module deliberately does not import Slack delivery or state-writing helpers.
It can therefore be used to inspect a gap without marking papers seen or posting
anything to the channel.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from agent.dedup import filter_new, load_seen
from agent.fetch import Paper, fetch_all
from agent.filter import ScoredPaper, score_and_filter

logger = logging.getLogger(__name__)

BACKFILL_ARXIV_MAX_RESULTS = 200
BACKFILL_ARXIV_CATEGORY_MAX_RESULTS = 500


def _paper_record(paper: Paper) -> dict[str, object]:
    return {
        "paper_id": paper.id,
        "title": paper.title,
        "url": paper.url,
        "source": paper.source,
        "published_date": paper.published_date,
        "authors": paper.authors,
    }


def _scored_record(scored: ScoredPaper) -> dict[str, object]:
    return {
        **_paper_record(scored.paper),
        "lane": "specdec" if scored.is_specdec else "inference",
        "specdec_score": scored.specdec_score,
        "inference_score": scored.inference_score,
        "tags": list(scored.tags),
        "summary": scored.summary,
    }


def run_backfill(
    *,
    days: int = 14,
    output_path: Path = Path("backfill_report.json"),
    score: bool = True,
) -> dict[str, object]:
    """Crawl a historical window and write a report without mutating bot state."""
    if not 1 <= days <= 30:
        raise ValueError("days must be between 1 and 30")
    if os.environ.get("CLEAR_SEEN_PAPERS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        raise EnvironmentError(
            "CLEAR_SEEN_PAPERS must not be set for a backfill audit"
        )

    papers = fetch_all(
        lookback_hours=days * 24,
        arxiv_max_results_per_query=BACKFILL_ARXIV_MAX_RESULTS,
        arxiv_category_max_results=BACKFILL_ARXIV_CATEGORY_MAX_RESULTS,
    )
    seen = load_seen()
    unseen = filter_new(papers, seen)
    by_id = {paper.id: paper for paper in unseen}

    report: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": days,
        "read_only": True,
        "fetched_count": len(papers),
        "known_count": len(papers) - len(unseen),
        "unseen_count": len(unseen),
        "source_counts": dict(sorted(Counter(p.source for p in papers).items())),
        "unseen_papers": [_paper_record(paper) for paper in unseen],
    }

    if score:
        scoring = score_and_filter(unseen)
        report.update(
            {
                "accepted_count": len(scoring.accepted),
                "accepted": [_scored_record(item) for item in scoring.accepted],
                "rejected_count": len(scoring.rejected_ids),
                "failed_ids": sorted(scoring.failed_ids),
                "deferred_ids": sorted(scoring.deferred_ids),
                "suppressed_inference": [
                    _paper_record(by_id[paper_id])
                    for paper_id in sorted(scoring.suppressed_ids)
                    if paper_id in by_id
                ],
            }
        )

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Backfill report written to %s (%d fetched, %d unseen)",
        output_path,
        len(papers),
        len(unseen),
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--output", type=Path, default=Path("backfill_report.json"))
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="crawl and compare state without invoking the LLM scorer",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    report = run_backfill(
        days=args.days,
        output_path=args.output,
        score=not args.fetch_only,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "lookback_days",
                    "fetched_count",
                    "known_count",
                    "unseen_count",
                    "accepted_count",
                    "accepted",
                    "failed_ids",
                    "deferred_ids",
                )
                if key in report
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
