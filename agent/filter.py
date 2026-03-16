"""Use an LLM (Claude, ChatGPT, or Gemini) to score relevance and summarize papers."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 6

# Minimize tokens: short prompt, capped abstract, small max output
ABSTRACT_MAX_CHARS = 600   # enough for relevance; tune if needed
LLM_MAX_TOKENS = 128      # score + brief summary

# Model IDs per provider (override with env LLM_MODEL if needed)
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
}

PROMPT_TEMPLATE = """\
Score 1-10 relevance to *LLM inference efficiency* or *speculative decoding*. If >=6, one short summary sentence; else summary null.

Title: {title}
Abstract: {abstract}

JSON only: {{"score": N, "summary": "..." or null}}"""


@dataclass
class ScoredPaper:
    paper: object  # Paper dataclass
    score: int
    summary: str


def _get_provider_and_key() -> tuple[str, str]:
    """Determine which LLM to use from env. Returns (provider, api_key)."""
    provider = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if provider not in ("anthropic", "openai", "gemini"):
        provider = ""
    if provider:
        key_var = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }[provider]
        key = os.environ.get(key_var)
        if key:
            return provider, key
        raise EnvironmentError(f"LLM_PROVIDER={provider} but {key_var} is not set")

    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", os.environ["ANTHROPIC_API_KEY"]
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", os.environ["OPENAI_API_KEY"]
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini", os.environ["GEMINI_API_KEY"]
    raise EnvironmentError(
        "Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY (or LLM_PROVIDER + the matching key)"
    )


def _call_anthropic(api_key: str, model: str, prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=LLM_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return (message.content[0].text or "").strip()


def _call_openai(api_key: str, model: str, prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=LLM_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


def _call_gemini(api_key: str, model: str, prompt: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=LLM_MAX_TOKENS),
    )
    text = getattr(response, "text", None)
    if text is None and hasattr(response, "candidates") and response.candidates:
        part = response.candidates[0].content.parts[0] if response.candidates[0].content.parts else None
        text = part.text if part else ""
    return (text or "").strip()


def _call_llm(provider: str, api_key: str, model: str, prompt: str) -> str:
    if provider == "anthropic":
        return _call_anthropic(api_key, model, prompt)
    if provider == "openai":
        return _call_openai(api_key, model, prompt)
    if provider == "gemini":
        return _call_gemini(api_key, model, prompt)
    raise ValueError(f"Unknown provider: {provider}")


def score_and_filter(papers) -> list[ScoredPaper]:
    """Score each paper with the configured LLM; return those above the relevance threshold."""
    if not papers:
        return []

    provider, api_key = _get_provider_and_key()
    model = os.environ.get("LLM_MODEL") or DEFAULT_MODELS[provider]
    logger.info("Using LLM: %s (%s)", provider, model)

    results: list[ScoredPaper] = []

    for paper in papers:
        prompt = PROMPT_TEMPLATE.format(
            title=paper.title[:500],  # cap title too
            abstract=paper.abstract[:ABSTRACT_MAX_CHARS],
        )
        try:
            raw = _call_llm(provider, api_key, model, prompt)
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            data = json.loads(raw)
            score = int(data.get("score", 0))
            summary = data.get("summary") or ""
        except Exception as exc:
            logger.warning("LLM scoring failed for '%s': %s", paper.title, exc)
            continue

        if score >= RELEVANCE_THRESHOLD and summary:
            results.append(ScoredPaper(paper=paper, score=score, summary=summary))
            logger.info("Accepted (score=%d): %s", score, paper.title)
        else:
            logger.debug("Rejected (score=%d): %s", score, paper.title)

    return results
