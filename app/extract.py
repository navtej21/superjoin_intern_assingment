"""Per-chunk extraction: from a model's raw response to grounded, stored facts.

Split deliberately from app.llm. Everything below is pure logic over an
already-produced response string — no network call — which is what makes
almost all of Phase B testable without an API key. The starter corpus is
already sitting in the database from Phase A; a hand-written string standing
in for what the model would say is enough to exercise the entire pipeline:
JSON parsing, shape validation, grounding, and persistence. Only
`run_extraction`'s default caller (app.llm.call_claude) touches the network,
and it is passed in rather than imported directly, so tests can substitute a
fake and prove the orchestration itself is correct.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Callable

from app.db import pages_in_range
from app.facts import Fact, attach_grounding, parse_llm_facts
from app.grounding import verify_quote
from app.llm import extract_json_array, get_cached_response, store_cached_response


@dataclass
class ExtractionOutcome:
    facts: list[Fact] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)  # {"stage", "reason", "raw"}


def extract_facts_for_chunk(
    conn: sqlite3.Connection, chunk: sqlite3.Row, response_text: str
) -> ExtractionOutcome:
    """Turn one chunk's raw model response into grounded facts.

    `chunk` is a row from the chunks table (needs id, doc_id, page_start,
    page_end). `response_text` is the model's raw text, already fetched —
    this function never calls the network itself, which is what makes it
    directly testable with a hand-written response string.

    Three things can go wrong, and each is recorded with which stage it
    failed at rather than merged into one generic error: the response might
    not parse as JSON at all ("parse"), an individual fact object might be
    missing a required field or use an invalid fact_type ("shape"), or a
    fact might parse cleanly but its quote might not actually be on the page
    it claims ("grounding"). That distinction is what lets the assignment's
    required extraction-failure case point at a specific, real cause instead
    of a shrug.
    """
    outcome = ExtractionOutcome()

    try:
        raw_list = extract_json_array(response_text)
    except ValueError as exc:
        outcome.rejections.append(
            {"stage": "parse", "reason": str(exc), "raw": response_text[:2000]}
        )
        return outcome

    candidates, shape_rejections = parse_llm_facts(
        raw_list, doc_id=chunk["doc_id"], chunk_id=chunk["id"]
    )
    for rejection in shape_rejections:
        outcome.rejections.append(
            {"stage": "shape", "reason": rejection["reason"], "raw": rejection["raw"]}
        )

    pages = pages_in_range(conn, chunk["doc_id"], chunk["page_start"], chunk["page_end"])
    page_tuples = [(p["page_number"], p["printed_page"], p["text"]) for p in pages]

    for fact in candidates:
        result = verify_quote(fact.quote, page_tuples)
        if result.ok:
            outcome.facts.append(attach_grounding(fact, result))
        else:
            outcome.rejections.append(
                {
                    "stage": "grounding",
                    "reason": result.reason,
                    "raw": {
                        "entity": fact.entity,
                        "metric": fact.metric,
                        "quote": fact.quote,
                    },
                }
            )

    return outcome


def persist_outcome(
    conn: sqlite3.Connection, doc_id: str, chunk_id: str, outcome: ExtractionOutcome
) -> None:
    """Write grounded facts and rejections. Both are kept: a rejection is the
    raw material for the assignment's required extraction-failure case, not
    something to discard once the good facts are safely stored."""
    for fact in outcome.facts:
        conn.execute(
            """INSERT OR REPLACE INTO facts
               (id, doc_id, chunk_id, entity, metric, fact_type, value, unit,
                period_label, period_start, period_end, basis, status,
                quote, confidence, page_number, printed_page, dimensions_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fact.id, fact.doc_id, fact.chunk_id, fact.entity, fact.metric,
                fact.fact_type, fact.value, fact.unit, fact.period_label,
                fact.period_start, fact.period_end, fact.basis, fact.status,
                fact.quote, fact.confidence, fact.page_number, fact.printed_page,
                json.dumps(fact.dimensions),
            ),
        )
    for rejection in outcome.rejections:
        conn.execute(
            """INSERT INTO extraction_rejections (doc_id, chunk_id, stage, reason, raw_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                doc_id, chunk_id, rejection["stage"], rejection["reason"],
                json.dumps(rejection["raw"], default=str),
            ),
        )


def run_extraction(
    conn: sqlite3.Connection,
    chunks: list[sqlite3.Row],
    call_model: Callable[[str], str],
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Drive extraction over a list of chunks.

    `call_model` is injected rather than imported, so tests can pass a plain
    function instead of app.llm.call_claude and prove this orchestration —
    caching, persistence, counting — is correct without spending a token or
    needing a key. See tests/test_extract.py::TestRunExtraction.

    Caches on the chunk's content_hash, which Phase A already computes: a
    chunk already answered, in this run or a previous one, is never
    re-sent — the same guarantee app.ingest gives at the document level,
    extended to the token-costing step.
    """
    total_facts = 0
    total_rejections = 0
    calls_made = 0

    for i, chunk in enumerate(chunks):
        cached = get_cached_response(conn, chunk["content_hash"])
        if cached is not None:
            response_text = cached
        else:
            response_text = call_model(chunk["text"])
            store_cached_response(conn, chunk["content_hash"], response_text)
            calls_made += 1

        outcome = extract_facts_for_chunk(conn, chunk, response_text)
        persist_outcome(conn, chunk["doc_id"], chunk["id"], outcome)
        total_facts += len(outcome.facts)
        total_rejections += len(outcome.rejections)

        if on_progress:
            on_progress(i + 1, len(chunks))

    conn.commit()
    return {
        "chunks": len(chunks),
        "calls_made": calls_made,
        "facts": total_facts,
        "rejections": total_rejections,
    }