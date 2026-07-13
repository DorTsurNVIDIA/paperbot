"""Entry point: orchestrate fetch → dedup → filter → post → save."""

from __future__ import annotations

import logging
import sys

from agent.dedup import filter_new, load_seen, save_seen
from agent.fetch import fetch_all
from agent.filter import score_and_filter
from agent.slack import post_to_slack

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== papers-agent starting ===")

    # 1. Fetch from all sources
    all_papers = fetch_all()
    logger.info("Total fetched: %d papers", len(all_papers))

    # 2. Remove already-seen papers
    seen = load_seen()
    new_papers = filter_new(all_papers, seen)
    logger.info("New (unseen) papers: %d", len(new_papers))

    # 3. Score & filter with LLM
    scoring = score_and_filter(new_papers)
    logger.info("Relevant papers after LLM filter: %d", len(scoring.accepted))
    if scoring.failed_ids:
        logger.warning(
            "%d paper(s) failed scoring and will be retried", len(scoring.failed_ids)
        )
    if scoring.deferred_ids:
        logger.info(
            "%d paper(s) were deferred by the provider cap and will be retried",
            len(scoring.deferred_ids),
        )

    # 4. Post to Slack
    has_pending_scoring = bool(scoring.failed_ids or scoring.deferred_ids)
    delivery = post_to_slack(
        scoring.accepted, announce_empty=not has_pending_scoring
    )

    # 5. Persist only terminal outcomes. Failed/deferred scoring and failed
    # deliveries remain unseen so a later scheduled run can retry them.
    seen.update(scoring.rejected_ids)
    seen.update(delivery.delivered_ids)
    save_seen(seen)
    logger.info("Saved %d total seen paper IDs", len(seen))

    if delivery.failed_ids:
        raise RuntimeError(
            f"Slack delivery failed for {len(delivery.failed_ids)} paper(s); "
            "their IDs were left unseen for retry"
        )
    if scoring.failed_ids and not (scoring.accepted or scoring.rejected_ids):
        raise RuntimeError(
            f"LLM scoring failed for all {len(scoring.failed_ids)} attempted "
            "paper(s); their IDs were left unseen for retry"
        )

    logger.info("=== papers-agent done ===")


if __name__ == "__main__":
    main()
