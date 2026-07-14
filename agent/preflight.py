"""Validate OpenAI-compatible model access before running the full paper pipeline."""

from __future__ import annotations

import os
import sys

import httpx


def _available_model_ids(base_url: str, api_key: str) -> list[str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(timeout=30) as client:
        response = client.get(f"{base_url.rstrip('/')}/models", headers=headers)
        response.raise_for_status()
        payload = response.json()
    return sorted(
        str(item["id"])
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    )


def main() -> int:
    provider = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if provider != "openai_compatible":
        print("Provider preflight skipped: LLM_PROVIDER is not openai_compatible")
        return 0

    base_url = (os.environ.get("LLM_BASE_URL") or "").strip()
    model = (os.environ.get("LLM_MODEL") or "").strip()
    api_key = os.environ.get("LLM_API_KEY") or ""
    if not base_url or not model or not api_key:
        print(
            "Provider preflight failed: LLM_BASE_URL, LLM_MODEL, and "
            "LLM_API_KEY are required",
            file=sys.stderr,
        )
        return 1

    try:
        available = _available_model_ids(base_url, api_key)
    except Exception as exc:
        print(f"Provider model-list request failed: {exc}", file=sys.stderr)
        return 1

    if model in available:
        print(f"Provider preflight passed: configured model is available ({model})")
        return 0

    print(f"Configured model is not available to this key: {model}", file=sys.stderr)
    print(f"Models available to this key ({len(available)}):", file=sys.stderr)
    for model_id in available:
        print(f"  {model_id}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
