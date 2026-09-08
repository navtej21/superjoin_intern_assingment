"""Split a document into overlapping chunks that never lose their page trail.

Chunks exist because a 100-page filing does not fit in one model call. But the
moment we concatenate pages, we risk losing the thing the whole system is built
on: the ability to say *which page* a claim came from. So every chunk records
the page range it spans, and its offset inside the first of those pages. A quote
recovered from a chunk can then be checked against the small set of pages that
chunk actually covers, rather than against the whole document.

Sizing is in characters, not tokens. Token counts are model-specific, and tying
chunk boundaries to one provider's tokenizer would mean re-chunking — and
re-paying for extraction — on a provider switch. Characters are stable, and the
budget only needs to be approximately right.

Boundaries prefer paragraph breaks, then line breaks, then sentence ends, then a
hard cut. This matters more than it looks: a fact split across two chunks is a
fact neither chunk can state completely, and a paragraph break is the cheapest
available signal that a thought has finished. The overlap between consecutive
chunks is the backstop for the cases where it happens anyway.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import CHUNK_OVERLAP_CHARS, CHUNK_TARGET_CHARS
from app.textnorm import estimate_tokens

# The separator inserted between pages when they are concatenated. A blank line
# is also a paragraph boundary, so page edges become preferred split points for
# free — which is what we want, since a chunk that stops at a page edge has the
# tidiest possible provenance.
PAGE_SEPARATOR = "\n\n"

# How far back from the target size we will look for a nicer boundary. Too small
# and we almost never find one; too large and chunks vary wildly in size.
BOUNDARY_SEARCH_CHARS = 600

# Boundary markers in descending order of preference.
_BOUNDARY_PATTERNS = ("\n\n", "\n", ". ", "; ")


@dataclass(frozen=True)
class PageSpan:
    """Where one page's text sits inside the concatenated document."""

    page_number: int
    start: int
    end: int


@dataclass(frozen=True)
class Chunk:
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    char_offset: int  # offset into page_start's own text
    token_count: int


def build_page_spans(pages: list[tuple[int, str]]) -> tuple[str, list[PageSpan]]:
    """Concatenate pages, returning the joined text and each page's span in it.

    `pages` is a list of (page_number, text) in document order.
    """
    parts: list[str] = []
    spans: list[PageSpan] = []
    cursor = 0
    for position, (page_number, text) in enumerate(pages):
        if position > 0:
            parts.append(PAGE_SEPARATOR)
            cursor += len(PAGE_SEPARATOR)
        parts.append(text)
        spans.append(PageSpan(page_number, cursor, cursor + len(text)))
        cursor += len(text)
    return "".join(parts), spans


def _find_boundary(text: str, hard_end: int) -> int:
    """Best split point at or before `hard_end`, preferring structural breaks."""
    window_start = max(0, hard_end - BOUNDARY_SEARCH_CHARS)
    for marker in _BOUNDARY_PATTERNS:
        found = text.rfind(marker, window_start, hard_end)
        if found > window_start:
            return found + len(marker)
    return hard_end


def _pages_covering(spans: list[PageSpan], start: int, end: int) -> list[PageSpan]:
    """Pages whose text overlaps the half-open interval [start, end)."""
    covering = [s for s in spans if s.start < end and s.end > start]
    if covering:
        return covering
    # A chunk that lands entirely inside a page separator has no overlap. Fall
    # back to the last page that began before it, so provenance is never empty.
    earlier = [s for s in spans if s.start <= start]
    return [earlier[-1]] if earlier else [spans[0]]


def chunk_pages(
    pages: list[tuple[int, str]],
    target_chars: int = CHUNK_TARGET_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[Chunk]:
    """Chunk a document's pages, preserving the page range behind each chunk.

    Pages with no text are kept in the span map — they still occupy a position
    in the document — but contribute nothing to the text, so they never appear
    as the sole page behind a chunk.
    """
    if not pages:
        return []
    if overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be smaller than target_chars")

    document, spans = build_page_spans(pages)
    if not document.strip():
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    length = len(document)

    while start < length:
        hard_end = min(start + target_chars, length)
        end = hard_end if hard_end >= length else _find_boundary(document, hard_end)
        if end <= start:  # no boundary found in range; take the hard cut
            end = hard_end

        text = document[start:end]
        if text.strip():
            covering = _pages_covering(spans, start, end)
            first = covering[0]
            chunks.append(
                Chunk(
                    chunk_index=index,
                    text=text,
                    page_start=first.page_number,
                    page_end=covering[-1].page_number,
                    char_offset=max(0, start - first.start),
                    token_count=estimate_tokens(text),
                )
            )
            index += 1

        if end >= length:
            break
        # Step back by the overlap so a fact straddling a boundary appears whole
        # in at least one chunk. Guaranteed forward progress: overlap < target.
        start = max(start + 1, end - overlap_chars)

    return chunks