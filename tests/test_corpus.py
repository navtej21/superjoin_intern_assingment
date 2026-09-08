"""End-to-end Phase A behaviour on the real starter corpus.

The last class is the one that matters most. The assignment requires four
demonstrated cases, and each rests on specific sentences in specific documents.
If those sentences are not sitting inside a stored chunk with the right page
range, no amount of prompt or classifier work later can recover them — the fact
was lost at ingest and the failure will look like a model problem.

So these tests assert that the evidence for all four cases survives Phase A:
present on the page it belongs to, and reachable inside a chunk that claims to
cover that page. This is the gate between phases.
"""
from __future__ import annotations

import pytest

from app.grounding import quote_on_page
from app.textnorm import normalize
from tests.conftest import (
    ANNUAL_REPORT,
    EARNINGS_DECK,
    ECONOMIC_SURVEY,
    IMF_ARTICLE_IV,
    PROSPECTUS,
    RBI_ANNUAL,
    doc_id_for,
    page_row,
    requires_corpus,
)

pytestmark = requires_corpus


class TestIngestion:
    EXPECTED_PAGES = {
        PROSPECTUS: 100,
        ANNUAL_REPORT: 100,
        EARNINGS_DECK: 27,
        ECONOMIC_SURVEY: 89,
        RBI_ANNUAL: 100,
        IMF_ARTICLE_IV: 95,
    }

    def test_all_six_documents_ingested(self, conn):
        row = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        assert row["n"] == 6

    @pytest.mark.parametrize("filename,pages", sorted(EXPECTED_PAGES.items()))
    def test_page_counts(self, conn, filename, pages):
        row = conn.execute(
            "SELECT num_pages FROM documents WHERE filename = ?", (filename,)
        ).fetchone()
        assert row["num_pages"] == pages

    def test_total_corpus_size(self, conn):
        row = conn.execute("SELECT COUNT(*) AS n FROM pages").fetchone()
        assert row["n"] == sum(self.EXPECTED_PAGES.values()) == 511

    def test_every_document_produced_chunks(self, conn):
        rows = conn.execute(
            """SELECT d.filename, COUNT(c.id) AS n
                 FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id
                GROUP BY d.id"""
        ).fetchall()
        for row in rows:
            assert row["n"] > 0, f"{row['filename']} produced no chunks"

    def test_documents_reach_chunked_status(self, conn):
        rows = conn.execute("SELECT status FROM documents").fetchall()
        assert all(r["status"] == "chunked" for r in rows)

    # The text layer is clean everywhere except the slide deck, whose figures
    # live inside chart images. That gap is reported, not hidden — it is the
    # raw material for the assignment's extraction-failure case.
    EXPECTED_LOW_TEXT = {
        PROSPECTUS: 0,
        ANNUAL_REPORT: 0,
        EARNINGS_DECK: 3,
        ECONOMIC_SURVEY: 0,
        RBI_ANNUAL: 0,
        IMF_ARTICLE_IV: 1,
    }

    @pytest.mark.parametrize("filename,expected", sorted(EXPECTED_LOW_TEXT.items()))
    def test_low_text_pages_are_flagged(self, conn, filename, expected):
        row = conn.execute(
            """SELECT COUNT(*) AS n FROM pages p JOIN documents d ON d.id = p.doc_id
                WHERE d.filename = ? AND p.low_text = 1""",
            (filename,),
        ).fetchone()
        assert row["n"] == expected

    def test_deck_is_text_sparse_relative_to_its_length(self, conn):
        """A 27-page deck holding under 25k characters is the measurable form of
        'the numbers are in the pictures'."""
        row = conn.execute(
            """SELECT SUM(p.char_count) AS total FROM pages p
                 JOIN documents d ON d.id = p.doc_id WHERE d.filename = ?""",
            (EARNINGS_DECK,),
        ).fetchone()
        assert row["total"] < 25_000


class TestDeduplication:
    def test_reingesting_is_a_noop(self, corpus_db):
        """Re-running the pipeline must not duplicate or re-pay. Identity is the
        file's bytes, so the same document under a different name is one row."""
        from pathlib import Path

        from app.pipeline import ingest_paths
        from tests.conftest import CORPUS_DIR

        paths = sorted(Path(CORPUS_DIR).rglob("*.pdf"))
        summary = ingest_paths(paths, db_path=corpus_db["db_path"])
        assert summary["ingested"] == []
        assert len(summary["skipped"]) == 6


class TestRequiredCaseEvidence:
    """Every sentence the four demonstrated cases depend on, verified present.

    Each entry is (case, filename, pdf page, expected printed page, quote).
    """

    EVIDENCE = [
        # Case 1 — corroboration: one figure, three different labels.
        ("1", ANNUAL_REPORT, 4, "6-7", "Revenue from services"),
        ("1", ANNUAL_REPORT, 22, "42-43",
         "Revenue from Operations 74,540.82 66,586.61 81,415.38 72,253.01"),
        # Case 2 — contradiction: director active in 2022, resigned by FY24.
        ("2", PROSPECTUS, 30, "105",
         "Suvir Suren Sujan(3) Non-Executive Nominee Director"),
        ("2", ANNUAL_REPORT, 24, "46-47",
         "resigned from the Board with effect from August 24, 2023"),
        ("2", ANNUAL_REPORT, 24, "46-47",
         "Sandeep Kumar Barasia (DIN: 01432123) resigned from the office of Executive Director"),
        # Case 3 — reconcilable: a 16.54 gap the corpus explains itself.
        ("3", ANNUAL_REPORT, 6, "10-11", "Revenue from services* (₹ million)"),
        ("3", ANNUAL_REPORT, 22, "42-43",
         "the revenue from operations on standalone basis for FY24 stood at"),
        ("3", ANNUAL_REPORT, 85, "248-249",
         "Revenue from services* 81,415.38 72,236.47"),
        ("3", ANNUAL_REPORT, 85, "248-249", "Revenue from sale of traded goods"),
        # Entity resolution: the same registered office, written two ways.
        ("addr", PROSPECTUS, 30, "105", "Indira Gandhi Intern"),
        ("addr", ANNUAL_REPORT, 31, "60-61", "IGI Airport, New Delhi 110037"),
    ]

    @pytest.mark.parametrize("case,filename,pdf_page,printed,quote", EVIDENCE)
    def test_evidence_is_on_the_page_we_cite(
        self, conn, case, filename, pdf_page, printed, quote
    ):
        row = page_row(conn, filename, pdf_page)
        assert row is not None, f"{filename} pdf page {pdf_page} missing"
        assert quote_on_page(quote, row["text"]), (
            f"case {case}: quote not on {filename} pdf p{pdf_page}"
        )
        assert row["printed_page"] == printed, (
            f"case {case}: citation would read printed page "
            f"{row['printed_page']!r}, expected {printed!r}"
        )

    @pytest.mark.parametrize("case,filename,pdf_page,printed,quote", EVIDENCE)
    def test_evidence_survives_into_a_chunk(
        self, conn, case, filename, pdf_page, printed, quote
    ):
        """The gate between phases: extraction only ever sees chunks, so a
        sentence not inside one is invisible no matter how good the prompt is."""
        doc_id = doc_id_for(conn, filename)
        rows = conn.execute(
            """SELECT text FROM chunks
                WHERE doc_id = ? AND page_start <= ? AND page_end >= ?""",
            (doc_id, pdf_page, pdf_page),
        ).fetchall()
        assert rows, f"no chunk covers {filename} pdf p{pdf_page}"
        needle = normalize(quote).casefold()
        assert any(needle in normalize(r["text"]).casefold() for r in rows), (
            f"case {case}: evidence lost between page and chunk"
        )

    def test_macro_false_corroboration_trap_is_present(self, conn):
        """RBI projects 6.5% for 2025-26; the IMF reports 6.5% actual for
        FY2024/25. Same number, unrelated claims. Both must be extractable, or
        the comparison layer never gets tested against its hardest case."""
        rbi = page_row(conn, RBI_ANNUAL, 17)
        imf = page_row(conn, IMF_ARTICLE_IV, 10)
        assert quote_on_page(
            "real GDP growth for 2025-26 is projected at 6.5 per cent", rbi["text"]
        )
        assert quote_on_page("real GDP grew by 6.5 percent in FY2024/25", imf["text"])