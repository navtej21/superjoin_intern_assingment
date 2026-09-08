"""Command-line entry point for the ingestion and extraction pipeline.

Usage:
    python -m app.pipeline ingest <path>...   # files or directories
    python -m app.pipeline stats              # what is in the database
    python -m app.pipeline extract            # run fact extraction over stored chunks
    python -m app.pipeline relationships      # find corroboration/contradiction/reconciliation
    python -m app.pipeline pages <doc_id>     # page map, incl. printed numbers

Kept as a module rather than folded into the API so the pipeline can be run and
inspected without a server, which is how the tests exercise it and how a
reviewer can reproduce our numbers in one command.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.chunker import chunk_pages
from app.db import db_session, document_exists, init_db
from app.ingest import ingest_document, persist_chunks, persist_document


def _pdf_paths(inputs: list[str]) -> list[Path]:
    """Expand files and directories into a sorted list of PDF paths."""
    found: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            found.extend(sorted(path.rglob("*.pdf")))
        elif path.suffix.lower() == ".pdf":
            found.append(path)
        else:
            print(f"  skipped (not a PDF): {path}", file=sys.stderr)
    return found


def ingest_paths(paths: list[Path], db_path=None) -> dict:
    """Ingest PDFs into the database, skipping any already present.

    Returns a summary dict so callers — tests included — can assert on what
    happened rather than parsing stdout.
    """
    init_db(db_path)
    summary = {"ingested": [], "skipped": [], "pages": 0, "chunks": 0}

    for path in paths:
        content = path.read_bytes()
        with db_session(db_path) as conn:
            from app.ingest import compute_doc_id

            _, content_hash = compute_doc_id(content)
            existing = document_exists(conn, content_hash)
            if existing:
                summary["skipped"].append({"file": path.name, "doc_id": existing})
                continue

            extracted = ingest_document(path.name, content)
            persist_document(conn, path.name, extracted)

            chunks = chunk_pages([(p.page_number, p.text) for p in extracted.pages])
            persist_chunks(conn, extracted.doc_id, chunks)

            summary["ingested"].append(
                {
                    "file": path.name,
                    "doc_id": extracted.doc_id,
                    "pages": extracted.num_pages,
                    "chunks": len(chunks),
                    "low_text_pages": extracted.low_text_pages,
                    "printed_pages_resolved": sum(
                        1 for p in extracted.pages if p.printed_page
                    ),
                }
            )
            summary["pages"] += extracted.num_pages
            summary["chunks"] += len(chunks)

    return summary


def cmd_ingest(args: argparse.Namespace) -> int:
    paths = _pdf_paths(args.paths)
    if not paths:
        print("No PDFs found.", file=sys.stderr)
        return 1

    summary = ingest_paths(paths)
    for item in summary["ingested"]:
        low = len(item["low_text_pages"])
        print(
            f"  {item['file'][:52]:54} {item['pages']:>4} pages "
            f"{item['chunks']:>4} chunks  "
            f"printed {item['printed_pages_resolved']:>3}/{item['pages']:<3} "
            f"low-text {low}"
        )
    for item in summary["skipped"]:
        print(f"  {item['file'][:52]:54} already ingested ({item['doc_id']})")
    print(f"\n{summary['pages']} pages, {summary['chunks']} chunks ingested.")
    return 0


def cmd_stats(_: argparse.Namespace) -> int:
    init_db()
    with db_session() as conn:
        rows = conn.execute(
            """SELECT d.id, d.filename, d.num_pages, d.status,
                      (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.id) AS chunks,
                      (SELECT COUNT(*) FROM pages p WHERE p.doc_id = d.id AND p.low_text = 1) AS low_text,
                      (SELECT COUNT(*) FROM pages p WHERE p.doc_id = d.id AND p.printed_page IS NOT NULL) AS printed
                 FROM documents d ORDER BY d.filename"""
        ).fetchall()

    if not rows:
        print("Database is empty. Run: python -m app.pipeline ingest <path>")
        return 0

    print(f"{'doc_id':18} {'pages':>5} {'chunks':>6} {'printed':>7} {'low':>4}  file")
    for r in rows:
        print(
            f"{r['id']:18} {r['num_pages']:>5} {r['chunks']:>6} "
            f"{r['printed']:>7} {r['low_text']:>4}  {r['filename'][:44]}"
        )
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    from app.extract import run_extraction
    from app.llm import call_claude

    init_db()
    with db_session() as conn:
        sql = "SELECT * FROM chunks"
        params: tuple = ()
        if args.doc_id:
            sql += " WHERE doc_id = ?"
            params = (args.doc_id,)
        sql += " ORDER BY doc_id, chunk_index"
        if args.limit:
            sql += " LIMIT ?"
            params = params + (args.limit,)

        chunks = conn.execute(sql, params).fetchall()
        if not chunks:
            print("No chunks match. Run `ingest` first, or check --doc-id.", file=sys.stderr)
            return 1

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


def cmd_relationships(_: argparse.Namespace) -> int:
    from app.adjudicate import find_relationships

    init_db()
    with db_session() as conn:
        facts = conn.execute("SELECT * FROM facts").fetchall()
        if not facts:
            print("No facts stored yet. Run `extract` first.", file=sys.stderr)
            return 1
        relationships = find_relationships(facts)

    by_type: dict[str, list] = {}
    for rel in relationships:
        by_type.setdefault(rel.relation_type, []).append(rel)

    print(f"{len(facts)} facts analysed, {len(relationships)} relationships found:")
    for rtype in ("corroboration", "contradiction", "context_reconcilable"):
        print(f"  {rtype}: {len(by_type.get(rtype, []))}")
    print()

    for rtype in ("corroboration", "contradiction", "context_reconcilable"):
        rels = by_type.get(rtype, [])
        if not rels:
            continue
        print(f"=== {rtype.upper()} ({len(rels)}) ===")
        for rel in rels:
            print(f"  {rel.explanation}")
            for ev in rel.evidence:
                print(
                    f"    - {ev['entity']} | {ev['metric']} = {ev['value']}  "
                    f"(doc {ev['doc_id']}, pdf p{ev['page_number']}, printed {ev['printed_page']})"
                )
                print(f"      quote: {ev['quote']!r}")
        print()

    return 0


def cmd_pages(args: argparse.Namespace) -> int:
    with db_session() as conn:
        rows = conn.execute(
            """SELECT page_number, printed_page, char_count, low_text
                 FROM pages WHERE doc_id = ? ORDER BY page_number""",
            (args.doc_id,),
        ).fetchall()
    if not rows:
        print(f"No such document: {args.doc_id}", file=sys.stderr)
        return 1
    print(f"{'pdf':>4} {'printed':>9} {'chars':>7}  flag")
    for r in rows:
        flag = "low-text" if r["low_text"] else ""
        print(
            f"{r['page_number']:>4} {str(r['printed_page'] or '-'):>9} "
            f"{r['char_count']:>7}  {flag}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest PDFs from files or directories")
    p_ingest.add_argument("paths", nargs="+")
    p_ingest.set_defaults(func=cmd_ingest)

    sub.add_parser("stats", help="summarise the database").set_defaults(func=cmd_stats)

    p_extract = sub.add_parser("extract", help="run fact extraction over stored chunks")
    p_extract.add_argument("--doc-id", help="restrict to one document")
    p_extract.add_argument("--limit", type=int, help="process at most N chunks")
    p_extract.set_defaults(func=cmd_extract)

    sub.add_parser(
        "relationships", help="find corroboration/contradiction/reconciliation across facts"
    ).set_defaults(func=cmd_relationships)

    p_pages = sub.add_parser("pages", help="page map for one document")
    p_pages.add_argument("doc_id")
    p_pages.set_defaults(func=cmd_pages)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())