"""Use Claude to score relevance and summarize papers."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import anthropic

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 6
MODEL = "claude-haiku-4-5-20251001"

PROMPT_TEMPLATE = """\
Given this paper title and abstract, rate its relevance to LLM inference efficiency \
and/or speculative decoding on a scale of 1-10. If relevant (score >= 6), \
write a 2-sentence summary emphasizing the efficiency/speculative decoding contribution.

Title: {title}
Abstract: {abstract}

Respond as JSON only, with no other text: {{"score": <int>, "summary": "<str or null>"}}"""


@dataclass
class ScoredPaper:
    paper: object  # Paper dataclass
    score: int
    summary: str


def score_and_filter(papers) -> list[ScoredPaper]:
    """Score each paper with Claude; return those above the relevance threshold."""
    if not papers:
        return []

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)
    results: list[ScoredPaper] = []

    for paper in papers:
        prompt = PROMPT_TEMPLATE.format(
            title=paper.title,
            abstract=paper.abstract[:2000],  # guard against very long abstracts
        )
        try:
            message = client.messages.create(
                model=MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            score = int(data.get("score", 0))
            summary = data.get("summary") or ""
        except Exception as exc:
            logger.warning("Claude scoring failed for '%s': %s", paper.title, exc)
            continue

        if score >= RELEVANCE_THRESHOLD and summary:
            results.append(ScoredPaper(paper=paper, score=score, summary=summary))
            logger.info("Accepted (score=%d): %s", score, paper.title)
        else:
            logger.debug("Rejected (score=%d): %s", score, paper.title)

    return results
