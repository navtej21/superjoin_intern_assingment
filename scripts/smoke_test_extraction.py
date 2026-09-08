"""Live smoke test: run real extraction on two known chunks, print everything.

This is the one thing in Phase B that needs your ANTHROPIC_API_KEY — every
other piece (schema, prompt construction, JSON parsing, grounding wiring) has
already been verified without it. Run this once, look at the output, and
paste it back before running extraction across all 582 chunks. A prompt bug
caught here costs two API calls; caught after a full run, it costs the whole
run.

Deliberately does not write to the database — it only prints — so you can run
it as many times as you like while iterating on app/prompts.py without
polluting the facts table.

Usage:
    python scripts/smoke_test_extraction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import db_session, init_db  # noqa: E402
from app.extract import extract_facts_for_chunk  # noqa: E402
from app.llm import ExtractionCallError, call_claude  # noqa: E402

# Two chunks with known, verified content — see tests/test_corpus.py for the
# quotes these pages are expected to contain. If the model's extraction looks
# thin or wrong on these two, the prompt needs work before it's worth running
# at scale.
TARGETS = [
    ("02-delhivery-annual-report-fy24-excerpt.pdf", 22,
     "revenue split: standalone vs consolidated FY24"),
    ("02-delhivery-annual-report-fy24-excerpt.pdf", 24,
     "director resignations: Sujan and Barasia"),
]


def main() -> int:
    init_db()
    with db_session() as conn:
        for filename, pdf_page, label in TARGETS:
            doc = conn.execute(
                "SELECT id FROM documents WHERE filename = ?", (filename,)
            ).fetchone()
            if doc is None:
                print(f"SKIP: {filename} not ingested yet.")
                print("      Run: python -m app.pipeline ingest data/corpus/delhivery")
                continue

            chunk = conn.execute(
                """SELECT * FROM chunks WHERE doc_id = ? AND page_start <= ? AND page_end >= ?
                   ORDER BY chunk_index LIMIT 1""",
                (doc["id"], pdf_page, pdf_page),
            ).fetchone()
            if chunk is None:
                print(f"SKIP: no chunk covers {filename} pdf page {pdf_page}")
                continue

            print("=" * 78)
            print(f"{label}")
            print(f"({filename}, pdf page {pdf_page}, chunk {chunk['id']}, "
                  f"{len(chunk['text'])} chars)")
            print("=" * 78)

            try:
                response_text = call_claude(chunk["text"])
            except ExtractionCallError as exc:
                print(f"CALL FAILED: {exc}")
                print()
                continue

            print("\n--- RAW MODEL RESPONSE ---")
            print(response_text)

            outcome = extract_facts_for_chunk(conn, chunk, response_text)
            print(f"\n--- PARSED: {len(outcome.facts)} facts, "
                  f"{len(outcome.rejections)} rejections ---")
            for fact in outcome.facts:
                print(
                    f"  [{fact.fact_type:8}] {fact.entity} | {fact.metric} = "
                    f"{fact.value} {fact.unit or ''}  "
                    f"({fact.period_label or '-'}, basis={fact.basis or '-'})  "
                    f"pdf p{fact.page_number} printed={fact.printed_page}  "
                    f"conf={fact.confidence}"
                )
                print(f"             quote: {fact.quote!r}")
            for rej in outcome.rejections:
                print(f"  REJECTED [{rej['stage']}]: {rej['reason']}")
                print(f"             raw: {rej['raw']!r}"[:300])
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())