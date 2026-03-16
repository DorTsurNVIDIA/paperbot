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

    # 3. Score & filter with Claude
    scored = score_and_filter(new_papers)
    logger.info("Relevant papers after Claude filter: %d", len(scored))

    # 4. Post to Slack
    post_to_slack(scored)

    # 5. Persist seen IDs (all fetched, not just relevant ones — avoids re-scoring)
    seen.update(p.id for p in new_papers)
    save_seen(seen)
    logger.info("Saved %d total seen paper IDs", len(seen))

    logger.info("=== papers-agent done ===")


if __name__ == "__main__":
    main()
