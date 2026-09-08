"""Budget-conscious extraction: guarantee the required-case chunks first,
then optionally add a small extra sample per document.

Full-corpus extraction (582 chunks) costs real money per chunk — on a tight
budget it's smarter to guarantee the chunks the assignment's four required
cases actually depend on, then spend whatever's left on breadth. This script
does exactly that, reusing the same tested run_extraction/call_claude used
by `python -m app.pipeline extract` — nothing about the extraction logic
itself changes, only which chunks get selected and in what order.

Usage:
    python scripts/extract_priority.py              # required-case chunks only
    python scripts/extract_priority.py --extra 10    # + up to 10 more per document
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import db_session, init_db  # noqa: E402
from app.extract import run_extraction  # noqa: E402
from app.llm import call_claude  # noqa: E402

PROSPECTUS = "01-delhivery-prospectus-2022-excerpt.pdf"
ANNUAL_REPORT = "02-delhivery-annual-report-fy24-excerpt.pdf"
RBI_ANNUAL = "02-rbi-annual-report-2024-25-excerpt.pdf"
IMF_ARTICLE_IV = "03-imf-india-2025-article-iv-excerpt.pdf"

# The exact pages your four required cases depend on (mirrors
# tests/test_corpus.py::TestRequiredCaseEvidence.EVIDENCE + the macro trap).
REQUIRED_PAGES = [
    (ANNUAL_REPORT, 4), (ANNUAL_REPORT, 6), (ANNUAL_REPORT, 22),
    (ANNUAL_REPORT, 24), (ANNUAL_REPORT, 31), (ANNUAL_REPORT, 85),
    (PROSPECTUS, 30),
    (RBI_ANNUAL, 17), (IMF_ARTICLE_IV, 10),
]


def priority_chunk_ids(conn) -> list[str]:
    ids: list[str] = []
    for filename, pdf_page in REQUIRED_PAGES:
        doc = conn.execute(
            "SELECT id FROM documents WHERE filename = ?", (filename,)
        ).fetchone()
        if doc is None:
            print(f"  WARNING: {filename} not ingested, skipping page {pdf_page}")
            continue
        rows = conn.execute(
            """SELECT id FROM chunks WHERE doc_id = ? AND page_start <= ? AND page_end >= ?
               ORDER BY chunk_index""",
            (doc["id"], pdf_page, pdf_page),
        ).fetchall()
        for r in rows:
            if r["id"] not in ids:
                ids.append(r["id"])
    return ids


def extra_sample_chunk_ids(conn, exclude: set[str], per_doc: int) -> list[str]:
    """Evenly spaced extra chunks per document, skipping ones already selected."""
    ids: list[str] = []
    docs = conn.execute("SELECT id, filename FROM documents ORDER BY filename").fetchall()
    for doc in docs:
        rows = conn.execute(
            "SELECT id FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (doc["id"],)
        ).fetchall()
        candidates = [r["id"] for r in rows if r["id"] not in exclude]
        if not candidates:
            continue
        step = max(1, len(candidates) // per_doc)
        picked = candidates[::step][:per_doc]
        ids.extend(picked)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra", type=int, default=0,
                         help="extra chunks per document beyond the required set")
    args = parser.parse_args()

    init_db()
    with db_session() as conn:
        required_ids = priority_chunk_ids(conn)
        print(f"Required-case chunks: {len(required_ids)}")

        selected_ids = list(required_ids)
        if args.extra > 0:
            extras = extra_sample_chunk_ids(conn, set(required_ids), args.extra)
            print(f"Extra sample chunks: {len(extras)}")
            selected_ids.extend(extras)

        if not selected_ids:
            print("No chunks selected — is the corpus ingested?", file=sys.stderr)
            return 1

        placeholders = ",".join("?" for _ in selected_ids)
        chunks = conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})", selected_ids
        ).fetchall()
        print(f"Total chunks to extract: {len(chunks)}\n")

        def progress(done: int, total: int) -> None:
            print(f"  [{done}/{total}] extracted", end="\r", file=sys.stderr)

        result = run_extraction(conn, chunks, call_claude, on_progress=progress)

    print(file=sys.stderr)
    print(
        f"chunks processed: {result['chunks']}   "
        f"api calls: {result['calls_made']}   "
        f"facts: {result['facts']}   "
        f"rejections: {result['rejections']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())