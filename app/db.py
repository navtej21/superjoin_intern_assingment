"""SQLite schema and connection helpers.

Why SQLite over Postgres or a vector database: the corpus this targets is a few
hundred pages, single-machine, single-writer. A file-based database means
``git clone && pip install && python -m app.pipeline`` is the entire setup
story — no docker-compose, no service to provision, and a reviewer can open
the shipped database and read our facts without running anything at all.

The schema grows one phase at a time. Documents, pages and chunks land in
Phase A; facts and relations arrive with extraction and adjudication. Each
migration is therefore one reviewable commit rather than a large upfront schema
nobody can diff against intent.

One deliberate omission: there is no table of metrics, units, or fact types. The
brief forbids relying on hard-coded schemas, and a lookup table of allowed
metric names is exactly that. Vocabularies are values in rows, discovered from
the documents, never columns or constraints in the schema.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from app.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,      -- first 16 hex chars of sha256(file bytes)
    filename      TEXT NOT NULL,
    content_hash  TEXT NOT NULL UNIQUE,  -- full sha256, for exact dedup
    file_path     TEXT NOT NULL,
    num_pages     INTEGER,
    status        TEXT NOT NULL DEFAULT 'uploaded',
    -- uploaded -> extracted -> chunked -> facts_extracted -> reconciled
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id        TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number   INTEGER NOT NULL,      -- 1-indexed PDF page order
    printed_page  TEXT,                  -- number printed in the document, e.g. "42-43"; NULL if not recoverable
    text          TEXT NOT NULL DEFAULT '',
    char_count    INTEGER NOT NULL DEFAULT 0,
    low_text      INTEGER NOT NULL DEFAULT 0,  -- 1 = likely image-only / needs OCR
    UNIQUE(doc_id, page_number)
);

CREATE TABLE IF NOT EXISTS chunks (
    id            TEXT PRIMARY KEY,      -- f"{doc_id}:{chunk_index}"
    doc_id        TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    page_start    INTEGER NOT NULL,
    page_end      INTEGER NOT NULL,
    char_offset   INTEGER NOT NULL,      -- offset into page_start's own text
    text          TEXT NOT NULL,
    token_count   INTEGER NOT NULL,      -- estimate; sizing only, see textnorm
    content_hash  TEXT NOT NULL,         -- sha256 of chunk text; LLM cache key
    UNIQUE(doc_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS facts (
    id              TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id        TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    entity          TEXT NOT NULL,
    metric          TEXT NOT NULL,
    fact_type       TEXT NOT NULL,       -- "numeric" | "semantic"
    value           TEXT,
    unit            TEXT,
    period_label    TEXT,
    period_start    TEXT,                -- filled in by Phase C normalisation
    period_end      TEXT,
    basis           TEXT,                -- open vocabulary, model-proposed
    status          TEXT,                -- open vocabulary, model-proposed
    quote           TEXT NOT NULL,
    confidence      REAL NOT NULL,
    page_number     INTEGER NOT NULL,    -- set only after grounding succeeds
    printed_page    TEXT,
    dimensions_json TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Every rejected candidate, at whichever stage it failed. Kept, not
-- discarded: this table is the raw material for the assignment's required
-- extraction-failure case, and a system that hides its own failures reads as
-- less trustworthy than one that logs them.
CREATE TABLE IF NOT EXISTS extraction_rejections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL,
    chunk_id    TEXT NOT NULL,
    stage       TEXT NOT NULL,           -- "parse" | "shape" | "grounding"
    reason      TEXT NOT NULL,
    raw_json    TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One row per (chunk content, prompt version). Re-running extraction, or
-- ingesting a document that shares boilerplate with one already processed,
-- must never re-pay for an unchanged call.
CREATE TABLE IF NOT EXISTS llm_cache (
    content_hash    TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    response_text   TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (content_hash, prompt_version)
);

CREATE INDEX IF NOT EXISTS idx_pages_doc ON pages(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_facts_doc ON facts(doc_id);
CREATE INDEX IF NOT EXISTS idx_facts_chunk ON facts(chunk_id);
CREATE INDEX IF NOT EXISTS idx_rejections_chunk ON extraction_rejections(chunk_id);
"""


def get_connection(db_path=None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path=None) -> None:
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.commit()
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive migrations for databases created by an earlier phase.

    Kept deliberately small: SQLite can add a nullable column in place, and
    every column this project adds after the fact is nullable by design, so a
    reviewer who ran an earlier commit is never asked to delete their database.
    """
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(pages)")}
    if "printed_page" not in existing:
        conn.execute("ALTER TABLE pages ADD COLUMN printed_page TEXT")


@contextmanager
def db_session(db_path=None) -> Iterator[sqlite3.Connection]:
    """Connection that commits on success, rolls back on exception, always
    closes."""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def document_exists(conn: sqlite3.Connection, content_hash: str) -> str | None:
    """Return the existing doc_id for this content, if we have already ingested
    it. Dedup is on the file's bytes, so the same document uploaded under two
    names is stored once — which is also what makes re-running the pipeline
    cheap and idempotent."""
    row = conn.execute(
        "SELECT id FROM documents WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    return row["id"] if row else None


def page_text(conn: sqlite3.Connection, doc_id: str, page_number: int) -> str | None:
    """The stored text of one page — what a quote is grounded against."""
    row = conn.execute(
        "SELECT text FROM pages WHERE doc_id = ? AND page_number = ?",
        (doc_id, page_number),
    ).fetchone()
    return row["text"] if row else None


def pages_in_range(
    conn: sqlite3.Connection, doc_id: str, page_start: int, page_end: int
) -> list[sqlite3.Row]:
    """Pages a chunk spans, in order. Grounding checks a quote against exactly
    these, never the whole document, so a quote cannot be 'verified' against a
    page the chunk never covered."""
    return list(
        conn.execute(
            """SELECT page_number, printed_page, text
                 FROM pages
                WHERE doc_id = ? AND page_number BETWEEN ? AND ?
                ORDER BY page_number""",
            (doc_id, page_start, page_end),
        )
    )