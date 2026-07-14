"""Fetch papers from arXiv, Semantic Scholar, and Hugging Face."""

from __future__ import annotations

import datetime
import logging
import os
import time
from dataclasses import dataclass, field

import httpx

from agent.identity import canonical_paper_id

logger = logging.getLogger(__name__)

LOOKBACK_HOURS = 168  # 7 days — cast a wider net; LLM filters relevance


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    authors: list[str]
    url: str
    source: str
    published_date: str
    source_id: str = ""
    external_ids: dict[str, str] = field(default_factory=dict)


def _cutoff() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=LOOKBACK_HOURS)


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------

ARXIV_QUERIES = [
    "speculative decoding",
    "inference efficiency",
    "LLM efficiency",
    "token generation",
    "draft model",
    "large language model inference",
    "transformer inference",
    "KV cache",
    "LLM serving",
    "continuous batching",
    "attention optimization",
    "early exit",
    "mixture of experts inference",
]
ARXIV_CATEGORIES = ["cs.CL", "cs.LG", "cs.AI", "cs.DC", "cs.AR"]
ARXIV_MAX_RESULTS_PER_QUERY = 50


def fetch_arxiv() -> list[Paper]:
    try:
        import arxiv  # type: ignore
    except ImportError:
        logger.error("arxiv package not installed")
        return []

    papers: list[Paper] = []
    seen_ids: set[str] = set()
    cutoff = _cutoff()

    for query in ARXIV_QUERIES:
        cat_filter = " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES)
        full_query = f"({query}) AND ({cat_filter})"
        try:
            search = arxiv.Search(
                query=full_query,
                max_results=ARXIV_MAX_RESULTS_PER_QUERY,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
            for result in search.results():
                published = result.published
                if published.tzinfo is None:
                    published = published.replace(tzinfo=datetime.timezone.utc)
                if published < cutoff:
                    continue
                source_id = result.entry_id.split("/")[-1]
                paper_id = canonical_paper_id("arxiv", source_id)
                if not paper_id:
                    continue
                if paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)
                papers.append(
                    Paper(
                        id=paper_id,
                        title=result.title,
                        abstract=result.summary,
                        authors=[a.name for a in result.authors],
                        url=result.entry_id,
                        source="arxiv",
                        published_date=published.isoformat(),
                        source_id=source_id,
                        external_ids={"ArXiv": source_id},
                    )
                )
        except Exception as exc:
            logger.warning("arXiv query '%s' failed: %s", query, exc)

    # Also fetch latest submissions by category only (no keyword) to catch papers we might miss
    for cat in ["cs.CL", "cs.LG"]:
        try:
            search = arxiv.Search(
                query=f"cat:{cat}",
                max_results=80,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
            for result in search.results():
                published = result.published
                if published.tzinfo is None:
                    published = published.replace(tzinfo=datetime.timezone.utc)
                if published < cutoff:
                    continue
                source_id = result.entry_id.split("/")[-1]
                paper_id = canonical_paper_id("arxiv", source_id)
                if not paper_id:
                    continue
                if paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)
                papers.append(
                    Paper(
                        id=paper_id,
                        title=result.title,
                        abstract=result.summary,
                        authors=[a.name for a in result.authors],
                        url=result.entry_id,
                        source="arxiv",
                        published_date=published.isoformat(),
                        source_id=source_id,
                        external_ids={"ArXiv": source_id},
                    )
                )
        except Exception as exc:
            logger.warning("arXiv category '%s' fetch failed: %s", cat, exc)

    return papers


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_FIELDS = "title,authors,year,url,abstract,externalIds,publicationDate"
S2_MAX_ATTEMPTS = 3


def _s2_request(
    client: httpx.Client,
    params: dict[str, object],
    headers: dict[str, str],
) -> dict:
    """Request Semantic Scholar with bounded retry/backoff for transient failures."""
    last_error: Exception | None = None
    for attempt in range(S2_MAX_ATTEMPTS):
        try:
            response = client.get(S2_URL, params=params, headers=headers)
        except httpx.TransportError as exc:
            last_error = exc
            if attempt + 1 < S2_MAX_ATTEMPTS:
                time.sleep(2**attempt)
                continue
            break

        if response.status_code == 429 or response.status_code >= 500:
            last_error = httpx.HTTPStatusError(
                f"Semantic Scholar returned {response.status_code}",
                request=response.request,
                response=response,
            )
            retry_after = response.headers.get("retry-after", "")
            try:
                delay = min(float(retry_after), 30.0) if retry_after else 2**attempt
            except ValueError:
                delay = 2**attempt
            if attempt + 1 < S2_MAX_ATTEMPTS:
                logger.warning(
                    "Semantic Scholar returned %d; retrying in %.1fs",
                    response.status_code,
                    delay,
                )
                time.sleep(delay)
                continue
            break

        # Authentication and query errors are not transient; surface them immediately.
        response.raise_for_status()
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("Semantic Scholar returned invalid JSON") from exc
    raise RuntimeError("Semantic Scholar request failed after retries") from last_error


def fetch_semantic_scholar() -> list[Paper]:
    cutoff = _cutoff()
    papers: list[Paper] = []
    seen_ids: set[str] = set()

    # Fewer queries + delay to avoid Semantic Scholar 429 rate limit
    queries = [
        "speculative decoding LLM efficiency",
        "LLM inference acceleration",
        "large language model inference",
    ]
    S2_LIMIT = 50
    S2_DELAY_SEC = 1
    cutoff_date = cutoff.date().isoformat()
    api_key = (
        os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        or os.environ.get("S2_API_KEY")
        or ""
    )
    headers = {"x-api-key": api_key} if api_key else {}

    with httpx.Client(timeout=30) as client:
        for i, query in enumerate(queries):
            if i > 0:
                time.sleep(S2_DELAY_SEC)
            try:
                data = _s2_request(
                    client,
                    {
                        "query": query,
                        "fields": S2_FIELDS,
                        "limit": S2_LIMIT,
                        "sort": "publicationDate:desc",
                        "publicationDateOrYear": f"{cutoff_date}:",
                    },
                    headers,
                )
            except Exception as exc:
                logger.warning("Semantic Scholar query '%s' failed: %s", query, exc)
                continue

            for item in data.get("data", []):
                pub_date_str = item.get("publicationDate") or ""
                if pub_date_str:
                    try:
                        pub_dt = datetime.datetime.fromisoformat(pub_date_str).replace(
                            tzinfo=datetime.timezone.utc
                        )
                        if pub_dt < cutoff:
                            continue
                    except ValueError:
                        pass

                source_id = item.get("paperId", "")
                external_ids = {
                    str(key): str(value)
                    for key, value in (item.get("externalIds") or {}).items()
                    if value is not None
                }
                paper_id = canonical_paper_id(
                    "semantic_scholar", source_id, external_ids
                )
                if not paper_id:
                    continue
                if paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)

                authors = [a.get("name", "") for a in item.get("authors", [])]
                url = item.get("url") or f"https://www.semanticscholar.org/paper/{source_id}"
                papers.append(
                    Paper(
                        id=paper_id,
                        title=item.get("title", ""),
                        abstract=item.get("abstract") or "",
                        authors=authors,
                        url=url,
                        source="semantic_scholar",
                        published_date=pub_date_str,
                        source_id=source_id,
                        external_ids=external_ids,
                    )
                )

    return papers


# ---------------------------------------------------------------------------
# Hugging Face daily papers
# ---------------------------------------------------------------------------

HF_API_URL = "https://huggingface.co/api/daily_papers"
RELEVANCE_KEYWORDS = [
    "speculative",
    "inference",
    "efficiency",
    "token generation",
    "draft model",
    "decoding",
    "latency",
    "throughput",
    "acceleration",
    "quantization",
    "kv cache",
    "serving",
    "batching",
    "early exit",
    "moe",
    "mixture of experts",
    "vllm",
    "tgi",
    "optimization",
]


def fetch_huggingface() -> list[Paper]:
    papers: list[Paper] = []
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(HF_API_URL)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("HuggingFace fetch failed: %s", exc)
        return []

    for item in data:
        paper_info = item.get("paper", {})
        title = paper_info.get("title", "")
        abstract = paper_info.get("summary") or paper_info.get("abstract") or ""
        text = (title + " " + abstract).lower()
        if not any(kw in text for kw in RELEVANCE_KEYWORDS):
            continue

        source_id = paper_info.get("id", "")
        paper_id = canonical_paper_id("huggingface", source_id)
        if not paper_id:
            continue
        authors_raw = paper_info.get("authors", [])
        authors = [
            a.get("name", a) if isinstance(a, dict) else str(a)
            for a in authors_raw
        ]
        url = f"https://huggingface.co/papers/{source_id}"
        published_date = paper_info.get("publishedAt") or ""

        papers.append(
            Paper(
                id=paper_id,
                title=title,
                abstract=abstract,
                authors=authors,
                url=url,
                source="huggingface",
                published_date=published_date,
                source_id=source_id,
                external_ids={"ArXiv": source_id},
            )
        )

    return papers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_all() -> list[Paper]:
    """Fetch papers from all sources and deduplicate by canonical identity."""
    all_papers: list[Paper] = []
    seen: set[str] = set()

    for fetcher in (fetch_arxiv, fetch_semantic_scholar, fetch_huggingface):
        fetcher_name = getattr(fetcher, "__name__", fetcher.__class__.__name__)
        try:
            batch = fetcher()
            logger.info("%s returned %d papers", fetcher_name, len(batch))
        except Exception as exc:
            logger.error("%s crashed: %s", fetcher_name, exc)
            batch = []

        for p in batch:
            if p.id not in seen:
                seen.add(p.id)
                all_papers.append(p)

    return all_papers
