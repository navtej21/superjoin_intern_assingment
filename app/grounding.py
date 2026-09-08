"""The grounding gate: prove a quote is really in the document.

This is the smallest module in the project and the one that decides whether
anything else can be believed. A language model asked to extract facts will
occasionally produce a quote that is *nearly* what the page says — a rounded
figure, a tidied phrase, a merged sentence. Those are the dangerous outputs,
because they read as evidence while being subtly invented.

So no fact enters the store until its quote has been located on a page the
chunk actually covered. A fact that fails this check is not silently dropped
either — it is recorded as ungrounded, because "the model claimed this and we
could not verify it" is itself a finding, and it is the honest raw material for
the extraction-failure case the brief asks us to demonstrate.

Matching is normalised (see textnorm) but never fuzzy on content. Whitespace,
ligatures, quotation marks and dashes may differ; digits and letters may not. A
tolerant match on a *number* would defeat the entire purpose.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.textnorm import normalize_for_match

# A quote shorter than this is not evidence. Single figures ("6.5") appear on
# dozens of pages and would "ground" against any of them, which is worse than no
# citation at all because it looks like provenance.
MIN_QUOTE_CHARS = 12


@dataclass(frozen=True)
class GroundingResult:
    """Outcome of checking one quote against the pages a chunk covered."""

    grounded: bool
    page_number: int | None       # PDF page the quote was found on
    printed_page: str | None      # what the document prints there
    reason: str                   # why it failed, for the failure log

    @property
    def ok(self) -> bool:
        return self.grounded


def quote_on_page(quote: str, page_text: str) -> bool:
    """True if `quote` appears in `page_text` once presentational differences
    are normalised away."""
    if not quote or not page_text:
        return False
    return normalize_for_match(quote) in normalize_for_match(page_text)


def verify_quote(quote: str, pages: list[tuple[int, str | None, str]]) -> GroundingResult:
    """Locate a quote among the pages a chunk spanned.

    `pages` is a list of (page_number, printed_page, text) in page order — the
    output of ``db.pages_in_range``. The search is confined to these pages on
    purpose: a chunk covering pages 22-24 has no business grounding a quote on
    page 90, and allowing it would let a model's misattribution pass unnoticed.

    Returns the first match in page order. Repeated boilerplate (a running
    header, a recurring table label) can appear on several pages; the earliest
    page in the chunk's own range is the most defensible attribution available
    without layout analysis, and the ambiguity is noted rather than hidden.
    """
    cleaned = (quote or "").strip()
    if len(cleaned) < MIN_QUOTE_CHARS:
        return GroundingResult(False, None, None, "quote too short to be evidence")
    if not pages:
        return GroundingResult(False, None, None, "no pages in chunk range")

    for page_number, printed_page, text in pages:
        if quote_on_page(cleaned, text):
            return GroundingResult(True, page_number, printed_page, "")

    return GroundingResult(False, None, None, "quote not found on any page in range")