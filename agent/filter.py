"""Use an LLM (Claude, ChatGPT, or Gemini) to score relevance and summarize papers."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

RELEVANCE_THRESHOLD = 6

# Minimize tokens: short prompt, capped abstract, small max output
ABSTRACT_MAX_CHARS = 600   # enough for relevance; tune if needed
LLM_MAX_TOKENS = 128      # score + brief summary

# Groq free tier: throttle and cap papers to avoid 429s and keep runs short
GROQ_DELAY_SEC = 3.0   # seconds between requests (env GROQ_DELAY_SEC overrides)
GROQ_MAX_PAPERS = 60   # max papers to score per run (env GROQ_MAX_PAPERS overrides)

# Model IDs per provider (override with env LLM_MODEL if needed)
# Groq: free tier, OpenAI-compatible — https://console.groq.com
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.1-8b-instant",
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
    if provider not in ("anthropic", "openai", "gemini", "groq"):
        provider = ""
    if provider:
        key_var = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "groq": "GROQ_API_KEY",
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
    if os.environ.get("GROQ_API_KEY"):
        return "groq", os.environ["GROQ_API_KEY"]
    raise EnvironmentError(
        "Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY (or LLM_PROVIDER + the matching key)"
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


def _call_openai(api_key: str, model: str, prompt: str, base_url: str | None = None) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=LLM_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


def _call_groq(api_key: str, model: str, prompt: str) -> str:
    return _call_openai(
        api_key=api_key,
        model=model,
        prompt=prompt,
        base_url="https://api.groq.com/openai/v1",
    )


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
    if provider == "groq":
        return _call_groq(api_key, model, prompt)
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

    # Groq: cap number of papers and throttle to avoid 429s
    if provider == "groq":
        max_papers = int(os.environ.get("GROQ_MAX_PAPERS") or GROQ_MAX_PAPERS)
        total = len(papers)
        papers = papers[:max_papers]
        delay_sec = float(os.environ.get("GROQ_DELAY_SEC") or GROQ_DELAY_SEC)
        if len(papers) < total:
            logger.info("Groq: scoring first %d of %d papers (max %d, delay %.1fs)", len(papers), total, max_papers, delay_sec)
        else:
            logger.info("Groq: scoring %d papers (delay %.1fs between calls)", len(papers), delay_sec)

    results: list[ScoredPaper] = []
    groq_delay_sec = float(os.environ.get("GROQ_DELAY_SEC") or GROQ_DELAY_SEC) if provider == "groq" else 0

    for i, paper in enumerate(papers):
        if i > 0 and groq_delay_sec > 0:
            time.sleep(groq_delay_sec)
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
