"""Canonical paper identity across providers and arXiv revisions."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping

_ARXIV_NEW_STYLE = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$", re.IGNORECASE)
_ARXIV_OLD_STYLE = re.compile(
    r"^[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?$", re.IGNORECASE
)
_ARXIV_VERSION = re.compile(r"v\d+$", re.IGNORECASE)


def normalize_arxiv_id(value: str | None) -> str:
    """Return an arXiv identifier without URL, PDF suffix, or version."""
    candidate = (value or "").strip()
    if not candidate:
        return ""

    candidate = re.sub(r"^arxiv:\s*", "", candidate, flags=re.IGNORECASE)
    candidate = candidate.split("?", 1)[0].split("#", 1)[0]
    if "/abs/" in candidate:
        candidate = candidate.rsplit("/abs/", 1)[1]
    elif "/pdf/" in candidate:
        candidate = candidate.rsplit("/pdf/", 1)[1]
    candidate = re.sub(r"\.pdf$", "", candidate, flags=re.IGNORECASE)
    candidate = candidate.strip("/")
    return _ARXIV_VERSION.sub("", candidate).lower()


def is_arxiv_id(value: str | None) -> bool:
    candidate = (value or "").strip()
    if candidate.lower().startswith("arxiv:") or "arxiv.org/" in candidate.lower():
        candidate = normalize_arxiv_id(candidate)
    return bool(
        _ARXIV_NEW_STYLE.fullmatch(candidate)
        or _ARXIV_OLD_STYLE.fullmatch(candidate)
    )


def canonical_paper_id(
    source: str,
    source_id: str | None,
    external_ids: Mapping[str, object] | None = None,
) -> str:
    """Choose a stable ID, preferring arXiv and DOI over provider-local IDs."""
    normalized_external_ids = {
        str(key).lower(): str(value).strip()
        for key, value in (external_ids or {}).items()
        if value is not None and str(value).strip()
    }

    arxiv_id = normalized_external_ids.get("arxiv", "")
    if arxiv_id:
        return f"arxiv:{normalize_arxiv_id(arxiv_id)}"

    normalized_source = (source or "").strip().lower()
    raw_id = (source_id or "").strip()
    if normalized_source == "arxiv" or (
        normalized_source in {"huggingface", "alphaxiv"} and is_arxiv_id(raw_id)
    ):
        normalized = normalize_arxiv_id(raw_id)
        return f"arxiv:{normalized}" if normalized else ""

    doi = normalized_external_ids.get("doi", "")
    if doi:
        doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi, flags=re.IGNORECASE)
        return f"doi:{doi.lower()}"

    if not normalized_source or not raw_id:
        return ""
    return f"{normalized_source}:{raw_id}"


def title_identity_key(title: str) -> str:
    """Match full titles across sources, ignoring case, spacing and punctuation.

    Short/generic titles are deliberately excluded. This is an exact normalized
    match, not fuzzy matching, so similarly named follow-up papers remain distinct.
    """
    normalized = "".join(
        char for char in unicodedata.normalize("NFKC", title).casefold()
        if char.isalnum()
    )
    if len(normalized) < 20:
        return ""
    return "title:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def paper_identity_keys(paper) -> set[str]:
    keys = {paper.id, *paper.identity_aliases}
    title_key = title_identity_key(paper.title)
    if title_key:
        keys.add(title_key)
    for name, value in paper.external_ids.items():
        if name.lower() in {"arxiv", "doi"} and value:
            keys.add(canonical_paper_id("", "", {name: value}))
    return keys - {""}


def canonicalize_stored_id(value: str) -> str:
    """Migrate an ID from the v1 seen file into the v2 canonical namespace."""
    stored = (value or "").strip()
    if ":" not in stored:
        return stored
    source, source_id = stored.split(":", 1)
    normalized_source = source.lower()
    if normalized_source in {"arxiv", "hf", "huggingface"}:
        source_name = "huggingface" if normalized_source in {"hf", "huggingface"} else "arxiv"
        canonical = canonical_paper_id(source_name, source_id)
        return canonical or stored
    if normalized_source == "doi":
        return f"doi:{source_id.lower()}"
    return stored
