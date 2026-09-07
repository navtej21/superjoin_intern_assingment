"""Page-aware overlapping chunker.

Turns a document's per-page text into overlapping windows sized for a
single LLM call, while keeping each window anchored to the exact page(s)
and character offset it came from. That anchor is what makes the Phase B
grounding check possible: verifying a quote the LLM returns is actually
present on the source page, not just plausible-looking.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from config import CHUNK_OVERLAP_CHARS, CHUNK_TARGET_CHARS
from ingestion.ingest import PageText

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    # tiktoken fetches its BPE ranks from a hardcoded Microsoft blob URL on
    # first use (not vendored with the package). A blocked or flaky network
    # — a locked-down campus/corporate proxy, for example — would otherwise
    # take down the whole chunker for a field that's purely informational.
    # Fall back to a standard ~4-chars-per-token estimate for English prose
    # instead of letting this be a hard dependency.
    _ENCODING = None


def _count_tokens(text: str) -> int:
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return max(1, len(text) // 4)


@dataclass
class Chunk:
    id: str
    doc_id: str
    chunk_index: int
    page_start: int
    page_end: int
    char_offset: int  # offset into page_start's own text
    text: str
    token_count: int
    content_hash: str


def _page_boundaries(pages: list[PageText]) -> list[tuple[int, int, int]]:
    """(page_number, start_offset, end_offset) in the concatenated
    full-text coordinate space, for every page, including empty ones —
    keeps offsets contiguous so lookups never skip a page."""
    boundaries = []
    cursor = 0
    for p in pages:
        start = cursor
        cursor += len(p.text)
        boundaries.append((p.page_number, start, cursor))
    return boundaries


def _page_for_offset(boundaries: list[tuple[int, int, int]], offset: int) -> tuple[int, int]:
    """(page_number, offset_into_that_page's_own_text) for a position in
    the full concatenated text."""
    for page_number, start, end in boundaries:
        if start <= offset < end:
            return page_number, offset - start
    # offset lands exactly at end-of-document (chunk_end == len(full_text))
    last_page_number, last_start, last_end = boundaries[-1]
    return last_page_number, last_end - last_start


def chunk_document(doc_id: str, pages: list[PageText]) -> list[Chunk]:
    boundaries = _page_boundaries(pages)
    full_text = "".join(p.text for p in pages)
    n = len(full_text)

    chunks: list[Chunk] = []
    chunk_index = 0
    chunk_start = 0

    while chunk_start < n:
        raw_end = min(chunk_start + CHUNK_TARGET_CHARS, n)

        # snap the cut point back to the nearest whitespace so we don't
        # split a word — but never snap past raw_end or before chunk_start
        chunk_end = raw_end
        if raw_end < n:
            snap = full_text.rfind(" ", chunk_start, raw_end)
            if snap == -1:
                snap = full_text.rfind("\n", chunk_start, raw_end)
            if snap > chunk_start:
                chunk_end = snap

        raw_chunk = full_text[chunk_start:chunk_end]
        chunk_text = raw_chunk.strip()

        if chunk_text:
            # char_offset must point at the *stored* text's real start, not
            # the pre-strip window start — leading whitespace trimmed by
            # .strip() would otherwise leave char_offset pointing 1-2 chars
            # too early, silently breaking any later verbatim-quote lookup
            # that trusts this offset.
            lstrip_amount = len(raw_chunk) - len(raw_chunk.lstrip())
            true_start = chunk_start + lstrip_amount
            true_end = true_start + len(chunk_text)  # exclusive

            page_start, char_offset = _page_for_offset(boundaries, true_start)
            page_end, _ = _page_for_offset(boundaries, max(true_end - 1, true_start))

            chunks.append(
                Chunk(
                    id=f"{doc_id}:{chunk_index}",
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    page_start=page_start,
                    page_end=page_end,
                    char_offset=char_offset,
                    text=chunk_text,
                    token_count=_count_tokens(chunk_text),
                    content_hash=hashlib.sha256(chunk_text.encode()).hexdigest(),
                )
            )
            chunk_index += 1

        if chunk_end >= n:
            break

        # overlap the next window; always make forward progress even if
        # overlap is misconfigured to be >= the chunk size
        chunk_start = max(chunk_end - CHUNK_OVERLAP_CHARS, chunk_start + 1)

    return chunks


def persist_chunks(conn, chunks: list[Chunk]) -> None:
    conn.executemany(
        """INSERT INTO chunks
           (id, doc_id, chunk_index, page_start, page_end, char_offset, text, token_count, content_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (c.id, c.doc_id, c.chunk_index, c.page_start, c.page_end,
             c.char_offset, c.text, c.token_count, c.content_hash)
            for c in chunks
        ],
    )