"""LLM client wrapper for Phase 4's Research Strategist.

Wraps Google's Gemini API (free tier) -- the problem statement doesn't
mandate any specific LLM (Deliverables explicitly lists "APIs used" as an
open choice: "e.g. OpenAI GPT-4o..."), and a genuinely $0 token cost is a
clean, defensible number for the Feasibility & Practicality judging
criterion, versus reporting an estimated dollar figure from a paid API.

Robustness note: an LLM call is just another external dependency that can
fail (network error, rate limit, malformed JSON, a safety-filter block) --
it gets the same treatment agent/recovery.py gives a training run: retries
with backoff, and callers must be able to fall back safely (skip this
round's LLM-proposed candidate, log the event) instead of letting a flaky
LLM call crash or stall the whole research loop. This module only handles
transport + retry; validating that what comes back is *usable* (a legal
ExperimentConfig) is agent/research_strategist.py's job, deliberately kept
separate -- garbage-in-garbage-out is a different failure mode than "the
API call itself failed," and each needs a different response.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from .paths import REPO_ROOT

DEFAULT_MODEL = "gemini-3.6-flash"  # per Gemini API's own 404 message when the older 2.0-flash was
                                     # retired mid-project; override via GEMINI_MODEL env var if this changes again

_dotenv_loaded = False


def _ensure_dotenv_loaded() -> None:
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass  # python-dotenv not installed -- fall back to whatever's already in os.environ
    _dotenv_loaded = True


class LLMError(Exception):
    """Raised only after every retry is exhausted. Callers must catch this
    and degrade gracefully (skip the LLM-proposed candidate this round,
    log a structured event) -- never let it propagate and crash the run."""


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_s: float
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class GeminiClient:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        _ensure_dotenv_loaded()
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise LLMError(
                "GEMINI_API_KEY not set. Copy .env.example to .env and fill in a real key "
                "(https://aistudio.google.com/apikey), or export it directly in the environment."
            )
        from google import genai  # imported lazily -- agents that never touch Phase 4 never need this installed
        self._client = genai.Client(api_key=key)

    def generate_json(self, prompt: str, max_retries: int = 3,
                       backoff_s: float = 2.0) -> tuple[dict[str, Any], LLMResponse]:
        """Asks the model for a JSON object (schema described in `prompt`
        itself), retrying transient failures with linear backoff. Returns
        (parsed_dict, response_metadata). Raises LLMError only once every
        retry -- including a final malformed-JSON response -- has failed.
        """
        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            t0 = time.time()
            try:
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                latency = time.time() - t0
                usage = getattr(resp, "usage_metadata", None)
                meta = LLMResponse(
                    text=resp.text, model=self.model, latency_s=latency,
                    input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                    output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                )
                parsed = json.loads(resp.text)  # a malformed response raises here, caught below, retried
                return parsed, meta
            except Exception as e:  # noqa: BLE001 -- network, rate limit, and bad-JSON all need the same retry path
                last_err = e
                if attempt < max_retries:
                    time.sleep(backoff_s * attempt)
        raise LLMError(f"Gemini call failed after {max_retries} attempt(s): {last_err}") from last_err
