"""Schema migration: facts, extraction_rejections, llm_cache.

Phase A's schema (documents/pages/chunks) is already exercised heavily by the
corpus tests. What's new here is narrow on purpose: the migration must add the
three Phase B tables without touching anything Phase A already relies on, and
it must do so idempotently, since init_db() runs on every CLI invocation.
"""
from __future__ import annotations

from app.db import (
    db_session,
    document_exists,
    get_connection,
    init_db,
    page_text,
    pages_in_range,
)


def _table_names(conn):
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {r["name"] for r in rows}


class TestSchema:
    def test_all_expected_tables_exist(self, tmp_path):
        db_path = tmp_path / "test.db"
        init_db(db_path)
        conn = get_connection(db_path)
        try:
            names = _table_names(conn)
            for expected in ("documents", "pages", "chunks", "facts",
                              "extraction_rejections", "llm_cache"):
                assert expected in names, f"{expected} table missing"
        finally:
            conn.close()

    def test_init_db_is_idempotent(self, tmp_path):
        """The CLI calls init_db() on every run — it must never fail or
        duplicate schema objects on a database that already has them."""
        db_path = tmp_path / "test.db"
        init_db(db_path)
        init_db(db_path)  # must not raise
        conn = get_connection(db_path)
        try:
            assert "facts" in _table_names(conn)
        finally:
            conn.close()

    def test_facts_table_has_the_expected_columns(self, tmp_path):
        db_path = tmp_path / "test.db"
        init_db(db_path)
        conn = get_connection(db_path)
        try:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(facts)")}
            for expected in ("id", "doc_id", "chunk_id", "entity", "metric",
                              "fact_type", "value", "unit", "period_label",
                              "basis", "status", "quote", "confidence",
                              "page_number", "printed_page", "dimensions_json"):
                assert expected in columns, f"facts.{expected} missing"
        finally:
            conn.close()

    def test_llm_cache_is_keyed_on_hash_and_prompt_version(self, tmp_path):
        db_path = tmp_path / "test.db"
        init_db(db_path)
        with db_session(db_path) as conn:
            conn.execute(
                """INSERT INTO llm_cache (content_hash, prompt_version, response_text)
                   VALUES ('abc123', 'v1', '[]')"""
            )
        # Re-inserting the same (hash, version) must replace, not duplicate —
        # this is what makes re-running extraction over unchanged chunks free.
        with db_session(db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO llm_cache (content_hash, prompt_version, response_text)
                   VALUES ('abc123', 'v1', '[{"entity": "x"}]')"""
            )
        conn = get_connection(db_path)
        try:
            rows = conn.execute("SELECT * FROM llm_cache").fetchall()
            assert len(rows) == 1
            assert rows[0]["response_text"] == '[{"entity": "x"}]'
        finally:
            conn.close()


class TestPhaseAHelpersStillWork:
    """The migration must not disturb anything Phase A already depends on."""

    def test_document_exists_after_migration(self, tmp_path):
        db_path = tmp_path / "test.db"
        init_db(db_path)
        with db_session(db_path) as conn:
            conn.execute(
                """INSERT INTO documents (id, filename, content_hash, file_path, num_pages)
                   VALUES ('d1', 'f.pdf', 'hash1', '/tmp/f.pdf', 5)"""
            )
        conn = get_connection(db_path)
        try:
            assert document_exists(conn, "hash1") == "d1"
            assert document_exists(conn, "nope") is None
        finally:
            conn.close()

    def test_page_text_and_pages_in_range_after_migration(self, tmp_path):
        db_path = tmp_path / "test.db"
        init_db(db_path)
        with db_session(db_path) as conn:
            conn.execute(
                """INSERT INTO documents (id, filename, content_hash, file_path, num_pages)
                   VALUES ('d1', 'f.pdf', 'hash1', '/tmp/f.pdf', 2)"""
            )
            conn.executemany(
                """INSERT INTO pages (doc_id, page_number, printed_page, text, char_count, low_text)
                   VALUES (?, ?, ?, ?, ?, 0)""",
                [("d1", 1, "1", "first page text", 16),
                 ("d1", 2, "2", "second page text", 17)],
            )
        conn = get_connection(db_path)
        try:
            assert page_text(conn, "d1", 1) == "first page text"
            rows = pages_in_range(conn, "d1", 1, 2)
            assert [r["page_number"] for r in rows] == [1, 2]
        finally:
            conn.close()