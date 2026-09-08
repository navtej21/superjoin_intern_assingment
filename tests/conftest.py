"""Shared fixtures.

The corpus fixture ingests the real starter PDFs into a throwaway database once
per test session. Tests therefore assert against actual extraction output rather
than hand-written samples — which is the point, because every interesting bug in
this phase comes from what real PDFs do, not from what we imagined they do.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "data" / "corpus"

# Filenames used by tests, so a rename shows up in one place.
PROSPECTUS = "01-delhivery-prospectus-2022-excerpt.pdf"
ANNUAL_REPORT = "02-delhivery-annual-report-fy24-excerpt.pdf"
EARNINGS_DECK = "03-delhivery-q4-fy24-earnings-presentation.pdf"
ECONOMIC_SURVEY = "01-india-economic-survey-2024-25-excerpt.pdf"
RBI_ANNUAL = "02-rbi-annual-report-2024-25-excerpt.pdf"
IMF_ARTICLE_IV = "03-imf-india-2025-article-iv-excerpt.pdf"


def corpus_available() -> bool:
    return CORPUS_DIR.exists() and any(CORPUS_DIR.rglob("*.pdf"))


requires_corpus = pytest.mark.skipif(
    not corpus_available(),
    reason="starter corpus not present under data/corpus/",
)


@pytest.fixture(scope="session")
def corpus_db(tmp_path_factory):
    """Ingest the full starter corpus into a temporary database.

    Session-scoped: extraction of 511 pages takes a few seconds and nothing in
    the suite mutates the result.
    """
    from app.pipeline import ingest_paths

    db_path = tmp_path_factory.mktemp("factlayer") / "test.db"
    paths = sorted(CORPUS_DIR.rglob("*.pdf"))
    summary = ingest_paths(paths, db_path=db_path)
    return {"db_path": db_path, "summary": summary}


@pytest.fixture(scope="session")
def conn(corpus_db):
    from app.db import get_connection

    connection = get_connection(corpus_db["db_path"])
    yield connection
    connection.close()


def doc_id_for(connection, filename: str) -> str:
    row = connection.execute(
        "SELECT id FROM documents WHERE filename = ?", (filename,)
    ).fetchone()
    assert row is not None, f"{filename} was not ingested"
    return row["id"]


def page_row(connection, filename: str, page_number: int):
    return connection.execute(
        """SELECT p.* FROM pages p
             JOIN documents d ON d.id = p.doc_id
            WHERE d.filename = ? AND p.page_number = ?""",
        (filename, page_number),
    ).fetchone()