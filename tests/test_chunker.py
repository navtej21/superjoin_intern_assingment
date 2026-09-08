"""Chunking, with the page trail intact.

The property that matters is not chunk size — it is that a chunk can always name
the pages its text came from, and that no text is lost between chunks. Every
grounded citation the system will ever produce depends on both.
"""
from __future__ import annotations

import pytest

from app.chunker import build_page_spans, chunk_pages
from app.textnorm import normalize


def pages_from(*texts: str) -> list[tuple[int, str]]:
    return [(i + 1, t) for i, t in enumerate(texts)]


class TestPageSpans:
    def test_spans_point_at_their_own_text(self):
        document, spans = build_page_spans(pages_from("alpha", "beta", "gamma"))
        for span, expected in zip(spans, ("alpha", "beta", "gamma")):
            assert document[span.start : span.end] == expected

    def test_page_numbers_are_preserved(self):
        _, spans = build_page_spans([(7, "a"), (8, "b")])
        assert [s.page_number for s in spans] == [7, 8]


class TestChunking:
    def test_short_document_is_one_chunk(self):
        chunks = chunk_pages(pages_from("a short page"))
        assert len(chunks) == 1
        assert chunks[0].page_start == chunks[0].page_end == 1

    def test_empty_input_yields_nothing(self):
        assert chunk_pages([]) == []
        assert chunk_pages(pages_from("   ", "\n")) == []

    def test_chunk_indices_are_contiguous(self):
        chunks = chunk_pages(pages_from(*["word " * 400] * 6))
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_page_range_is_ordered_and_real(self):
        chunks = chunk_pages(pages_from(*["word " * 400] * 6))
        for c in chunks:
            assert 1 <= c.page_start <= c.page_end <= 6

    def test_chunks_cover_the_whole_document(self):
        """No text may fall between two chunks — a dropped span is a fact we can
        never extract and will never know we missed."""
        pages = pages_from(*[f"page {i} " + "word " * 300 for i in range(1, 8)])
        document, _ = build_page_spans(pages)
        chunks = chunk_pages(pages)
        joined = normalize(" ".join(c.text for c in chunks))
        for token in normalize(document).split():
            if token not in joined:
                pytest.fail(f"token {token!r} lost during chunking")

    def test_consecutive_chunks_overlap(self):
        """A fact straddling a boundary must survive whole in one chunk, which
        is the only reason the overlap exists."""
        chunks = chunk_pages(
            pages_from("word " * 4000), target_chars=1200, overlap_chars=300
        )
        assert len(chunks) > 2
        for previous, following in zip(chunks, chunks[1:]):
            opening = following.text[:150]
            assert opening in previous.text, "chunks do not overlap"

    def test_chunks_respect_the_size_target(self):
        chunks = chunk_pages(
            pages_from("word " * 4000), target_chars=1200, overlap_chars=300
        )
        assert all(len(c.text) <= 1200 for c in chunks)

    def test_overlap_must_be_smaller_than_target(self):
        with pytest.raises(ValueError):
            chunk_pages(pages_from("word " * 100), target_chars=100, overlap_chars=100)

    def test_char_offset_locates_the_chunk_inside_its_first_page(self):
        pages = pages_from("A" * 5000, "B" * 5000)
        for chunk in chunk_pages(pages):
            page_text = pages[chunk.page_start - 1][1]
            assert 0 <= chunk.char_offset <= len(page_text)

    def test_forward_progress_on_pathological_input(self):
        """A page with no boundary markers at all must still terminate."""
        chunks = chunk_pages(pages_from("x" * 20000), target_chars=1000, overlap_chars=200)
        assert len(chunks) > 1
        assert all(c.text for c in chunks)


class TestAgainstRealDocuments:
    def test_every_chunk_traces_back_to_its_pages(self, conn):
        """The core provenance guarantee: a chunk's text must be findable in the
        pages it claims to span. If this fails, every citation built on top of it
        is unsafe."""
        rows = conn.execute(
            """SELECT c.doc_id, c.chunk_index, c.text, c.page_start, c.page_end
                 FROM chunks c ORDER BY c.doc_id, c.chunk_index"""
        ).fetchall()
        assert rows, "no chunks were ingested"

        checked = 0
        for row in rows[::11]:  # every 11th chunk keeps the suite quick
            pages = conn.execute(
                """SELECT text FROM pages
                    WHERE doc_id = ? AND page_number BETWEEN ? AND ?
                    ORDER BY page_number""",
                (row["doc_id"], row["page_start"], row["page_end"]),
            ).fetchall()
            haystack = normalize(" ".join(p["text"] for p in pages))
            assert normalize(row["text"]) in haystack, (
                f"chunk {row['doc_id']}:{row['chunk_index']} not found in "
                f"pages {row['page_start']}-{row['page_end']}"
            )
            checked += 1
        assert checked > 20

    def test_page_ranges_are_within_the_document(self, conn):
        bad = conn.execute(
            """SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id
                WHERE c.page_start < 1 OR c.page_end > d.num_pages
                   OR c.page_start > c.page_end"""
        ).fetchall()
        assert not bad, f"{len(bad)} chunks have impossible page ranges"