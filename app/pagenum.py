"""Recover the page number *printed inside* a document, which is not the same
thing as the PDF's page index.

Why this exists: curated document excerpts can jump between retained
sections, so a citation of "page 22" is ambiguous — it means one page to our
extractor and a different page to a human holding the original filing. We
recover both, so a reviewer can check our evidence against the real document.

Why it is done this way: across different documents the printed number can
appear in different places — the first line of the page, the second line
after a running header, embedded in a header string, several lines up from
the bottom, or not at all on some pages. Hard-coding any one of those would be
a document-specific rule, which breaks the moment a reviewer supplies their
own PDF with its own layout.

So rather than asking "where is the number", we lean on the one property
every paginated document has: **printed numbers advance in step with the
pages.** For a candidate `v` on PDF page `i`, the implied offset `v - k*i` is
constant across a run of pages, where `k` is how many printed pages one PDF
sheet carries. Real page numbers therefore cluster on a single offset; stray
integers (table cells, years, footnote markers) do not, because nothing makes
them increment in step.

Two wrinkles handled generically rather than by special case:

* **Sheet factor.** A 2-up scanned or exported spread shows a facing pair,
  "42" and "43", on one PDF page. Assuming k=1 would only explain half of
  that. We try each small k and keep whichever explains the document better —
  model selection over one integer, not a rule about any specific file.
* **Section jumps.** An excerpt that keeps pages 26-37 then jumps to 94-120
  starts a new offset run at the seam. A *windowed* vote absorbs this for
  free — that is why the vote is local rather than global.

Where the evidence is genuinely weak we return None rather than guess. A
wrong printed page in a citation is worse than an absent one.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# A printed page number lives on a short line. Long lines are body prose, and
# an integer inside one is far likelier to be data than pagination.
MAX_CANDIDATE_LINE_LEN = 70

# Non-empty lines to inspect at each end of the page. Six reaches a page
# number that sits above a multi-line running footer.
EDGE_LINES = 6

# Printed page numbers are small. Rejects years, amounts and ID numbers
# without knowing anything about the document.
MAX_PAGE_LABEL = 3000

# Half-width of the voting window, in pages. Wide enough that a real run
# outvotes coincidence, narrow enough that a section jump only disturbs the
# pages either side of the seam.
VOTE_WINDOW = 6

# Printed pages carried by one PDF sheet. 1 is a normal page; 2 is a scanned
# or exported facing-page spread. Beyond 2 is vanishingly rare.
SHEET_FACTORS = (1, 2)

# A candidate must agree with this many other pages in its window to be trusted.
MIN_SUPPORT = 2

_INT_TOKEN = re.compile(r"(?<!\d)(\d{1,4})(?!\d)")


@dataclass(frozen=True)
class PageLabel:
    """A printed-page citation for one PDF page.

    `label` is what a human should be told ("43", or "42-43" for a spread).
    `primary` is the highest printed number on the sheet, kept separately so
    callers can sort or range-check without parsing the label.
    """

    label: str
    primary: int


def page_number_candidates(page_text: str) -> list[int]:
    """Every integer that could plausibly be this page's printed number.

    Deliberately over-generates. Precision comes from the offset vote, not
    from being clever here: a candidate that is not a page number will not
    sit in an incrementing run, so it loses on its own.
    """
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    if not lines:
        return []

    edge = lines[:EDGE_LINES] + lines[-EDGE_LINES:]
    found: list[int] = []
    for line in edge:
        if len(line) > MAX_CANDIDATE_LINE_LEN:
            continue
        for match in _INT_TOKEN.finditer(line):
            value = int(match.group(1))
            if 0 < value <= MAX_PAGE_LABEL and value not in found:
                found.append(value)
    return found


def _candidate_offsets(value: int, index: int, k: int) -> set[int]:
    """Base offsets a candidate is consistent with.

    We vote in *offset* space, keyed on the lowest printed number the sheet
    carries. A sheet showing "42 43" yields the same base offset whichever of
    the two numbers the text extractor happened to surface, so pages where
    only one of the pair survives still reinforce the same hypothesis instead
    of splitting the vote between two adjacent offsets.
    """
    return {value - k * index - d for d in range(k)}


def _infer_for_factor(
    candidates: list[list[int]], k: int
) -> tuple[list[int | None], int]:
    """Resolve base offsets assuming k printed pages per PDF sheet.

    Returns the per-page base offset (or None) and a coverage score, so
    callers can compare one sheet factor against another.
    """
    per_page: list[set[int]] = [
        set().union(*(_candidate_offsets(v, i, k) for v in cands)) if cands else set()
        for i, cands in enumerate(candidates)
    ]

    picks: list[int | None] = []
    for i, offsets in enumerate(per_page):
        if not offsets:
            picks.append(None)
            continue

        lo = max(0, i - VOTE_WINDOW)
        hi = min(len(per_page), i + VOTE_WINDOW + 1)
        votes: Counter[int] = Counter()
        for j in range(lo, hi):
            if j != i:
                votes.update(per_page[j])

        def local_fit(offset: int, index: int = i) -> int:
            """How many of this page's own candidates the offset explains.

            On a spread showing "42 43", base offset 42 accounts for both
            numbers while 41 or 43 accounts for only one. Neighbouring pages
            often can't separate those hypotheses by themselves — every page
            in the run votes for all of them equally — so the page's own
            evidence is what has to break the tie.
            """
            low = k * index + offset
            return sum(1 for v in candidates[index] if low <= v <= low + k - 1)

        best = max(offsets, key=lambda o: (votes.get(o, 0), local_fit(o), -abs(o)))
        picks.append(best if votes.get(best, 0) >= MIN_SUPPORT else None)

    # A pick must agree with an immediate neighbour. This removes stray hits
    # on slide decks, where a lone integer can coincidentally match the
    # offset of a distant page.
    confirmed: list[int | None] = []
    for i, offset in enumerate(picks):
        if offset is None:
            confirmed.append(None)
            continue
        neighbours = [picks[j] for j in (i - 1, i + 1) if 0 <= j < len(picks)]
        confirmed.append(offset if offset in neighbours else None)

    return confirmed, sum(1 for o in confirmed if o is not None)

def infer_printed_pages(page_texts: list[str]) -> list[PageLabel | None]:
    """Map each PDF page (by list position) to its printed-page citation.

    Returns a list the same length as `page_texts`. None means no candidate
    on that page agreed with its neighbours — an honest gap, not a guess.
    """
    candidates = [page_number_candidates(t) for t in page_texts]

    best_factor, best_picks, best_score = 1, [None] * len(page_texts), -1
    for k in SHEET_FACTORS:
        picks, score = _infer_for_factor(candidates, k)
        # A higher factor must clearly beat a lower one, not merely tie: k=2
        # can always mimic k=1 on half the pages, so we require real gain
        # before accepting the more complicated explanation.
        if score > best_score * 1.2:
            best_factor, best_picks, best_score = k, picks, score

    labels: list[PageLabel | None] = []
    for i, offset in enumerate(best_picks):
        if offset is None:
            labels.append(None)
            continue
        low = best_factor * i + offset
        high = low + best_factor - 1
        if low < 1:
            labels.append(None)
            continue
        label = str(low) if best_factor == 1 else f"{low}-{high}"
        labels.append(PageLabel(label=label, primary=high))
    return labels