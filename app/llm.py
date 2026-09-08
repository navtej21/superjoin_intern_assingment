"""Anthropic client wrapper: the one place that talks to the network.

Split deliberately from app.extract, which holds all the logic that can be
tested without a live call. Two responsibilities live here:

1. `extract_json_array` — turning a model's raw text response back into a
   JSON list. Pure string parsing, no network, and tested exhaustively in
   tests/test_llm.py against every shape a real response takes (a code
   fence, a sentence of preamble, trailing commentary). This is the part
   most likely to need tuning as real responses come in, which is exactly
   why it's isolated from the network call it will never need to change.

2. `call_claude` — the actual API call. Requires ANTHROPIC_API_KEY and costs
   real tokens, so it is not exercised by the test suite at all. That is a
   deliberate gap, not an oversight — see scripts/smoke_test_extraction.py
   for how to check this one function works before running it across the
   full corpus.

Caching lives here too, keyed on (chunk content hash, prompt version): the
chunks table already carries a content_hash from Phase A, so a chunk whose
text hasn't changed since the prompt was last run never gets re-sent.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time

from app.config import ANTHROPIC_API_KEY
from app.prompts import PROMPT_VERSION, build_extraction_prompt

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
MAX_TOKENS = 4096
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0

# Strips a leading/trailing ```json or bare ``` fence. Applied before hunting
# for the array itself, so a fenced response and an unfenced one are handled
# by the same bracket-scan below.
_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class ExtractionCallError(RuntimeError):
    """The API call itself failed after retries — a network or auth problem.

    Kept distinct from a parsing failure (ValueError from
    extract_json_array), which means the call succeeded but the model's
    answer wasn't usable. One is an outage; the other is a model-quality
    finding worth logging, not retrying.
    """


def extract_json_array(response_text: str) -> list:
    """Pull a JSON array out of a model's raw text response.

    Handles the three ways a real response reliably deviates from clean JSON
    even when told not to: wrapped in a code fence, prefixed with a sentence
    of prose, or followed by one. Anything past that — truly malformed JSON,
    no array at all — raises rather than guessing further, because patching
    a broken response risks turning a real extraction failure into a
    fabricated fact.

    Tried in two steps, not one bracket scan, because a bracket scan alone
    can't distinguish "there is no array here" from "the whole response
    parses, but it's a JSON object, not an array" — a bare object has no
    square brackets to find at all, so scanning for one first would report
    the wrong failure. Parsing the full (defenced) text first gets that
    distinction right; the bracket scan is only the fallback for when prose
    surrounds an otherwise-clean array.
    """
    text = response_text.strip()
    text = _CODE_FENCE.sub("", text).strip()

    try:
        whole = json.loads(text)
    except json.JSONDecodeError:
        whole = None

    if whole is not None:
        if not isinstance(whole, list):
            raise ValueError(f"expected a JSON array, got {type(whole).__name__}")
        return whole

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON array found in response: {response_text[:200]!r}")
    candidate = text[start : end + 1]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not valid JSON: {exc}") from exc

    if not isinstance(parsed, list):
        raise ValueError(f"expected a JSON array, got {type(parsed).__name__}")
    return parsed


def get_cached_response(conn: sqlite3.Connection, content_hash: str) -> str | None:
    row = conn.execute(
        "SELECT response_text FROM llm_cache WHERE content_hash = ? AND prompt_version = ?",
        (content_hash, PROMPT_VERSION),
    ).fetchone()
    return row["response_text"] if row else None


def store_cached_response(
    conn: sqlite3.Connection, content_hash: str, response_text: str
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO llm_cache (content_hash, prompt_version, response_text)
           VALUES (?, ?, ?)""",
        (content_hash, PROMPT_VERSION, response_text),
    )


def call_claude(chunk_text: str, *, model: str = DEFAULT_MODEL) -> str:
    """One extraction call against the real Anthropic API.

    Requires ANTHROPIC_API_KEY in the environment (see .env.example). This
    function is intentionally the only place in the project that imports the
    anthropic package, so the rest of the test suite never needs the package
    installed with a live key configured to pass.
    """
    import anthropic

    if not ANTHROPIC_API_KEY:
        raise ExtractionCallError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file, or export "
            "it in your shell before running extraction."
        )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = build_extraction_prompt(chunk_text)

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
        except Exception as exc:  # the SDK raises several distinct error
            # types (rate limit, overloaded, connection) — treated uniformly
            # here because the response to all of them is the same: back off
            # and retry, then give up with the real error attached.
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise ExtractionCallError(
        f"extraction call failed after {MAX_RETRIES} attempts: {last_error}"
    )