"""Printed-page recovery.

The unit tests use synthetic documents so the intent of each rule is visible.
The corpus tests pin the result against page numbers read by eye out of the real
PDFs — those are the ones that would have caught the two bugs this module
actually had (a tail window too short to reach the annual report's footer, and a
spread label built off the wrong member of the pair).
"""
from __future__ import annotations

import pytest

from app.pagenum import infer_printed_pages, page_number_candidates
from tests.conftest import (
    ANNUAL_REPORT,
    ECONOMIC_SURVEY,
    IMF_ARTICLE_IV,
    PROSPECTUS,
    RBI_ANNUAL,
    page_row,
    requires_corpus,
)


class TestCandidates:
    def test_finds_bare_number_at_top(self):
        assert 25 in page_number_candidates("25\nName of\nEntity")

    def test_finds_number_inside_short_header(self):
        assert 41 in page_number_candidates("INDIA\nINTERNATIONAL MONETARY FUND 41\nbody")

    def test_ignores_numbers_in_long_prose_lines(self):
        prose = "x" * 90 + " 42 " + "y" * 90
        assert 42 not in page_number_candidates(prose)

    def test_rejects_implausibly_large_numbers(self):
        # A magnitude filter only: DINs and amounts are out of range.
        assert 5127 not in page_number_candidates("5127\nAnnual Report")

    def test_years_are_candidates_and_are_rejected_later(self):
        """A year is small enough to look like a page number, so the magnitude
        filter keeps it. It is the offset vote that discards it — see
        test_distractor_loses_to_the_run."""
        assert 2024 in page_number_candidates("2024\nAnnual Report")

    def test_empty_page_yields_nothing(self):
        assert page_number_candidates("   \n\n  ") == []


class TestInference:
    def test_simple_run(self):
        pages = [f"{n}\nbody text here" for n in range(7, 17)]
        labels = infer_printed_pages(pages)
        assert [l.label for l in labels] == [str(n) for n in range(7, 17)]

    def test_distractor_loses_to_the_run(self):
        # Every page also carries a constant "2024" and a random table value.
        pages = [f"{n}\n2024\n{n * 37}\nbody" for n in range(7, 17)]
        labels = infer_printed_pages(pages)
        assert [l.label for l in labels] == [str(n) for n in range(7, 17)]

    def test_section_jump_is_absorbed(self):
        first = [f"{n}\nbody" for n in range(10, 22)]
        second = [f"{n}\nbody" for n in range(90, 102)]
        labels = infer_printed_pages(first + second)
        got = [l.label if l else None for l in labels]
        assert got[:12] == [str(n) for n in range(10, 22)]
        assert got[12:] == [str(n) for n in range(90, 102)]

    def test_two_up_spread_is_detected(self):
        # One sheet shows a facing pair: 10-11, 12-13, ...
        pages = [f"header\n{2 * i + 10}\n{2 * i + 11}\nbody" for i in range(12)]
        labels = infer_printed_pages(pages)
        assert labels[0].label == "10-11"
        assert labels[5].label == "20-21"

    def test_pages_without_numbers_return_none(self):
        pages = ["no digits at all here" for _ in range(8)]
        assert all(l is None for l in infer_printed_pages(pages))

    def test_never_emits_a_page_below_one(self):
        pages = ["1\nbody", "2\nbody", "3\nbody"]
        labels = infer_printed_pages(pages)
        assert all(l is None or l.primary >= 1 for l in labels)


@requires_corpus
class TestAgainstRealDocuments:
    """Ground truth read by eye from the PDFs. If these drift, the citations we
    hand a reviewer are wrong, which is worse than having no citation."""

    # (filename, pdf page, expected printed label)
    GROUND_TRUTH = [
        (PROSPECTUS, 6, "25"),
        (PROSPECTUS, 22, "97"),
        (PROSPECTUS, 46, "216"),
        (PROSPECTUS, 30, "105"),
        (ANNUAL_REPORT, 6, "10-11"),
        (ANNUAL_REPORT, 21, "40-41"),
        (ANNUAL_REPORT, 22, "42-43"),
        (ANNUAL_REPORT, 46, "90-91"),
        (ANNUAL_REPORT, 61, "120-121"),
        (ECONOMIC_SURVEY, 6, "3"),
        (ECONOMIC_SURVEY, 22, "19"),
        (ECONOMIC_SURVEY, 46, "88"),
        (RBI_ANNUAL, 6, "1"),
        (RBI_ANNUAL, 22, "17"),
        (RBI_ANNUAL, 46, "41"),
        (IMF_ARTICLE_IV, 22, "17"),
        (IMF_ARTICLE_IV, 46, "41"),
    ]

    @pytest.mark.parametrize("filename,pdf_page,expected", GROUND_TRUTH)
    def test_printed_page_matches_the_document(self, conn, filename, pdf_page, expected):
        row = page_row(conn, filename, pdf_page)
        assert row["printed_page"] == expected

    # Coverage floors, not exact counts: tuning the heuristic should be free to
    # improve these, and only a regression should fail the suite.
    COVERAGE_FLOORS = [
        (PROSPECTUS, 100, 95),
        (ANNUAL_REPORT, 100, 95),
        (ECONOMIC_SURVEY, 89, 80),
        (RBI_ANNUAL, 100, 90),
        (IMF_ARTICLE_IV, 95, 85),
    ]

    @pytest.mark.parametrize("filename,total,floor", COVERAGE_FLOORS)
    def test_resolution_coverage(self, conn, filename, total, floor):
        row = conn.execute(
            """SELECT COUNT(*) AS n FROM pages p JOIN documents d ON d.id = p.doc_id
                WHERE d.filename = ? AND p.printed_page IS NOT NULL""",
            (filename,),
        ).fetchone()
        assert row["n"] >= floor, f"only {row['n']}/{total} printed pages resolved"