"""alphaXiv discovery with public, credential-free metadata retrieval.

alphaXiv's feed is the interface used by its website, not a versioned public API
contract. Keep its response parsing covered by fixtures and surface schema errors.
"""

from __future__ import annotations

import datetime
import json
import logging
import time

import httpx

from agent.fetch import ARXIV_CATEGORIES, RELEVANCE_KEYWORDS, Paper, _cutoff
from agent.identity import canonical_paper_id, is_arxiv_id

logger = logging.getLogger(__name__)

ALPHAXIV_FEED_URL = "https://api.alphaxiv.org/papers/v3/feed"
SOURCE_HEADERS = {"User-Agent": "paperbot/1.0 (+https://github.com/DorTsurNVIDIA/paperbot)"}
PAGE_SIZE = 100
ALPHAXIV_MAX_PAGES = 100


def _request_json(client: httpx.Client, url: str, params: dict) -> dict:
    """Retry transport errors, rate limits and 5xx responses, with bounded delay."""
    for attempt in range(3):
        try:
            response = client.get(url, params=params)
        except httpx.TransportError:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
            continue
        if (response.status_code == 429 or response.status_code >= 500) and attempt < 2:
            try:
                delay = float(response.headers.get("Retry-After", 2**attempt))
            except ValueError:
                delay = 2**attempt
            time.sleep(max(0, min(delay, 30)))
            continue
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected response object from {url}")
        return data
    raise RuntimeError(f"Request did not complete: {url}")


def _rows(data: dict, key: str) -> list[dict]:
    rows = data.get(key)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Source response must contain a '{key}' array of objects")
    return rows


def _date(value) -> datetime.datetime | None:
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return datetime.datetime.fromtimestamp(value / 1000, datetime.timezone.utc)
        if not isinstance(value, str) or not value.strip():
            return None
        result = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            result = result.replace(tzinfo=datetime.timezone.utc)
        return result.astimezone(datetime.timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None


def fetch_alphaxiv(*, lookback_hours: int | None = None) -> list[Paper]:
    """Page through recent papers in the bot's fields, including exclusive slugs.

    Do not use Hot/For You or the ranked MCP search: popularity-ranked results
    cannot cover every new submission. Recent is the website's date-sorted feed.
    """
    cutoff = _cutoff(lookback_hours)
    papers: list[Paper] = []
    seen: set[str] = set()
    with httpx.Client(timeout=30, headers=SOURCE_HEADERS) as client:
        for page in range(ALPHAXIV_MAX_PAGES):
            data = _request_json(client, ALPHAXIV_FEED_URL, {
                "sort": "Recent",
                "interval": "All time",
                "pageNum": page,
                "pageSize": PAGE_SIZE,
                "topics": json.dumps([*ARXIV_CATEGORIES, "stat.ML"]),
            })
            items = _rows(data, "papers")
            if not items:
                break
            publication_dates = []
            new_ids = 0
            for item in items:
                publication = _date(item.get("publication_date"))
                publication_dates.append(publication)
                # Revisions must not make an old paper look newly published.
                published = _date(item.get("first_publication_date")) or publication
                source_id = item.get("universal_paper_id")
                title = item.get("title")
                if not isinstance(source_id, str) or not source_id or not isinstance(title, str) or not title:
                    logger.warning("Skipping alphaXiv entry without an ID or title")
                    continue
                if source_id in seen:
                    continue
                seen.add(source_id)
                new_ids += 1
                if published is None:
                    logger.warning("Skipping alphaXiv %s without a valid publication date", source_id)
                    continue
                if published < cutoff or item.get("external_blog") or item.get("external_link"):
                    continue
                abstract = item.get("abstract") or ""
                text = (title + " " + abstract).lower()
                if not any(keyword in text for keyword in RELEVANCE_KEYWORDS):
                    continue
                external_ids = {"ArXiv": source_id} if is_arxiv_id(source_id) else {}
                authors = item.get("authors") or []
                papers.append(Paper(
                    id=canonical_paper_id("alphaxiv", source_id, external_ids),
                    title=title,
                    abstract=abstract,
                    authors=[author for author in authors if isinstance(author, str)],
                    url=f"https://www.alphaxiv.org/abs/{source_id}",
                    source="alphaxiv",
                    published_date=published.isoformat(),
                    source_id=source_id,
                    external_ids=external_ids,
                ))
            if all(date is not None and date < cutoff for date in publication_dates):
                break
            if len(items) < PAGE_SIZE:
                break
            if not new_ids:
                raise RuntimeError("alphaXiv feed repeated a page; coverage is incomplete")
        else:
            logger.warning("alphaXiv reached the %d-page cap; lookback coverage may be incomplete", ALPHAXIV_MAX_PAGES)
    return papers
