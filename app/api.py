"""Phase D: the API surface over everything Phases A-C already built.

Deliberately thin and read-only. The heavy lifting — ingestion, extraction,
grounding, relationship-finding — already exists as tested, CLI-runnable
functions in app.pipeline / app.extract / app.adjudicate; this module's only
job is to expose the *results* of running that pipeline (a database that's
already populated, per the setup story in app.db's module docstring) over
HTTP, so a reviewer can explore facts and relationships without touching a
terminal, and so the demo video has something to click through.

Relationships are computed on request, not stored — the schema has
deliberately no relations table (see app.db). At the corpus size this
project targets (hundreds of facts), find_relationships is fast enough that
persisting its output would only be a cache with no measurable benefit, and
computing it live means the endpoint can never serve a stale answer.

Run with: uvicorn app.api:app --reload
Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.adjudicate import find_relationships
from app.db import db_session, init_db


@asynccontextmanager
async def _lifespan(_: FastAPI):
    # Safe to call on an already-populated database — init_db is additive
    # and idempotent (see app.db._migrate), so this never wipes data.
    init_db()
    yield


app = FastAPI(
    title="Fact Knowledge Layer",
    description=(
        "Extracts grounded facts from PDFs and surfaces corroboration, "
        "contradiction, and context-reconcilable relationships across them."
    ),
    version="1.0.0",
    lifespan=_lifespan,
)

# Wide open by design: this is a local reviewer tool over a read-only API,
# not a multi-tenant service — there is no auth, no user data, and nothing
# a browser-based demo UI should be blocked from calling.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _row_to_dict(row: Any) -> dict:
    return dict(row)


@app.get("/")
def root() -> dict:
    return {
        "name": "Fact Knowledge Layer API",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/documents",
            "/documents/{doc_id}/pages",
            "/facts",
            "/facts/{fact_id}",
            "/relationships",
            "/rejections",
        ],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/documents")
def list_documents() -> list[dict]:
    """Every ingested document with a one-line summary: how many pages,
    chunks, facts, and how many pages needed OCR (low_text) or never
    resolved a printed page number."""
    with db_session() as conn:
        rows = conn.execute(
            """SELECT d.id, d.filename, d.num_pages, d.status, d.created_at,
                      (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.id) AS chunks,
                      (SELECT COUNT(*) FROM facts f WHERE f.doc_id = d.id) AS facts,
                      (SELECT COUNT(*) FROM pages p WHERE p.doc_id = d.id AND p.low_text = 1) AS low_text_pages
                 FROM documents d
                ORDER BY d.filename"""
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


@app.get("/documents/{doc_id}/pages")
def document_pages(doc_id: str) -> list[dict]:
    """Page map for one document — the raw PDF page index next to the page
    number actually printed in the document, so a reviewer can see exactly
    how a fact's page_number/printed_page pair was derived."""
    with db_session() as conn:
        exists = conn.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail=f"No such document: {doc_id}")
        rows = conn.execute(
            """SELECT page_number, printed_page, char_count, low_text
                 FROM pages WHERE doc_id = ? ORDER BY page_number""",
            (doc_id,),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


@app.get("/facts")
def list_facts(
    doc_id: str | None = None,
    entity: str | None = Query(None, description="Case-insensitive substring match"),
    metric: str | None = Query(None, description="Case-insensitive substring match"),
    fact_type: str | None = Query(None, pattern="^(numeric|semantic)$"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict:
    """Every stored fact, each carrying its own grounding evidence (quote,
    page_number, printed_page) — filterable so a reviewer can zoom in on one
    document or one entity instead of scrolling all 400+."""
    where: list[str] = []
    params: list[Any] = []
    if doc_id:
        where.append("doc_id = ?")
        params.append(doc_id)
    if entity:
        where.append("entity LIKE ?")
        params.append(f"%{entity}%")
    if metric:
        where.append("metric LIKE ?")
        params.append(f"%{metric}%")
    if fact_type:
        where.append("fact_type = ?")
        params.append(fact_type)

    clause = f"WHERE {' AND '.join(where)}" if where else ""
    with db_session() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS n FROM facts {clause}", params
        ).fetchone()["n"]
        rows = conn.execute(
            f"""SELECT * FROM facts {clause}
                 ORDER BY doc_id, page_number
                 LIMIT ? OFFSET ?""",
            params + [limit, offset],
        ).fetchall()
    return {
        "total": total,
        "count": len(rows),
        "offset": offset,
        "limit": limit,
        "facts": [_row_to_dict(r) for r in rows],
    }


@app.get("/facts/{fact_id}")
def get_fact(fact_id: str) -> dict:
    with db_session() as conn:
        row = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"No such fact: {fact_id}")
    return _row_to_dict(row)


@app.get("/relationships")
def list_relationships(
    relation_type: str | None = Query(
        None, pattern="^(corroboration|contradiction|context_reconcilable)$"
    ),
    entity: str | None = Query(None, description="Case-insensitive substring match"),
    limit: int = Query(200, ge=1, le=2000),
) -> dict:
    """Corroboration, contradiction, and context-reconcilable relationships
    across every stored fact — computed live by app.adjudicate.
    find_relationships (see that module for the heuristics and their
    documented trade-offs), never persisted, so this can never go stale
    relative to the facts table.
    """
    with db_session() as conn:
        facts = conn.execute("SELECT * FROM facts").fetchall()

    if not facts:
        return {"total": 0, "count": 0, "by_type": {}, "relationships": []}

    relationships = find_relationships(facts)

    if relation_type:
        relationships = [r for r in relationships if r.relation_type == relation_type]
    if entity:
        needle = entity.lower()
        relationships = [
            r
            for r in relationships
            if any(needle in (ev["entity"] or "").lower() for ev in r.evidence)
        ]

    by_type: dict[str, int] = {}
    for r in relationships:
        by_type[r.relation_type] = by_type.get(r.relation_type, 0) + 1

    truncated = relationships[:limit]
    return {
        "total": len(relationships),
        "count": len(truncated),
        "by_type": by_type,
        "relationships": [
            {
                "relation_type": r.relation_type,
                "fact_ids": r.fact_ids,
                "explanation": r.explanation,
                "evidence": r.evidence,
            }
            for r in truncated
        ],
    }


@app.get("/rejections")
def list_rejections(doc_id: str | None = None, limit: int = Query(200, ge=1, le=2000)) -> dict:
    """Extraction candidates that were rejected — at the parse, shape, or
    grounding stage — kept rather than silently discarded. This is the raw
    material for the assignment's extraction-failure case: a system that
    shows what it *couldn't* verify is more trustworthy than one that only
    ever shows successes.
    """
    where = "WHERE doc_id = ?" if doc_id else ""
    params = [doc_id] if doc_id else []
    with db_session() as conn:
        rows = conn.execute(
            f"""SELECT * FROM extraction_rejections {where}
                 ORDER BY created_at DESC LIMIT ?""",
            params + [limit],
        ).fetchall()
    return {"count": len(rows), "rejections": [_row_to_dict(r) for r in rows]}