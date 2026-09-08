"""The Fact schema: the one contract every later phase agrees to.

The brief forbids relying on hard-coded facts, filenames, schemas, or
document-specific rules, and says a reviewer may test with their own PDFs. That
rules out a fixed vocabulary of metric names ("revenue_from_operations") or
entity types baked into code — those are exactly the document-specific rules
the brief warns against, and they would silently fail to fire on a filing that
uses different words for the same idea.

So what's fixed here is the *shape* of a fact, not its *content*. Every fact
names what it is about (entity), what is being stated (metric, in whatever
words the source document and the model settle on), what kind of claim it is
(numeric or semantic — the brief's own split), and the dimensions that make two
facts comparable later: unit, period, basis, status. The values that fill those
slots are proposed by the model at extraction time from the actual document
text, never chosen from a list written in advance. Canonicalisation (Phase C)
clusters those free-text values after the fact; it does not constrain them
before it. Anything a model attaches beyond the named slots survives in
`dimensions` rather than being dropped, which is what lets an unseen kind of
qualifier show up without a schema migration.

Every fact also carries its own evidence — quote, document, chunk — because a
fact that cannot point at its source is not one this system will assert. This
module only checks that a candidate is *shaped* like a fact (required fields
present, quote long enough to be evidence, fact_type is one of the two the
brief defines). Whether the quote is *actually* on the page it claims is a
separate question, answered by app.grounding once a chunk's page range is
known — see `attach_grounding` below.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any

from app.grounding import GroundingResult

# A quote shorter than this cannot be evidence — mirrors
# grounding.MIN_QUOTE_CHARS, enforced again here so a Fact can never exist
# without it even if a caller skips the grounding gate by mistake.
MIN_QUOTE_CHARS = 12

# The brief's own split. Kept as the one closed vocabulary in this module,
# because it is a distinction the assignment itself defines, not one we
# invented — everything else (entity, metric, basis, status, unit) stays open.
FACT_TYPES = ("numeric", "semantic")

REQUIRED_STRING_FIELDS = ("entity", "metric", "fact_type", "quote")

# Named slots a model may fill. Anything else in the model's JSON is kept, not
# discarded — see `dimensions` on Fact.
_KNOWN_FIELDS = {
    "entity", "metric", "fact_type", "value", "unit", "period_label",
    "period_start", "period_end", "basis", "status", "quote", "confidence",
}


class FactValidationError(ValueError):
    """A candidate fact is missing something required or malformed.

    Raised per-candidate and always caught by the caller (see
    `parse_llm_facts`), never allowed to abort a whole chunk's extraction. A
    rejection is itself a finding — it is the raw material for the
    assignment's required extraction-failure case, so the reason string is
    kept, not just the fact of failure.
    """


@dataclass
class Fact:
    id: str
    doc_id: str
    chunk_id: str

    entity: str        # what/who the fact is about, as the model names it
    metric: str         # what is being stated, in the source document's words
    fact_type: str       # "numeric" | "semantic"
    quote: str
    confidence: float

    value: str | None = None          # kept as text; Phase C parses/normalises it
    unit: str | None = None            # e.g. "INR million", "percent" — as stated
    period_label: str | None = None    # e.g. "FY24", "Q2 FY25" — as stated
    period_start: str | None = None    # ISO date; filled in by Phase C normalisation
    period_end: str | None = None
    basis: str | None = None           # e.g. "standalone", "consolidated" — open
    status: str | None = None          # e.g. "actual", "projection", "restated" — open

    # Filled in only once app.grounding has confirmed the quote's page — never
    # guessed, never left implicit. A Fact with grounded=False must not reach
    # the facts table; see extract.py.
    page_number: int | None = None
    printed_page: str | None = None
    grounded: bool = False

    dimensions: dict[str, Any] = field(default_factory=dict)
    # Anything the model attached that doesn't fit a named slot above — the
    # mechanism that lets a new kind of qualifier appear on an unseen document
    # without a code change.


def _stable_id(doc_id: str, chunk_id: str, entity: str, metric: str, quote: str) -> str:
    """Deterministic id so re-extracting an unchanged chunk reproduces the same
    fact id instead of a duplicate row — the fact-level counterpart to
    chunk-level caching on content_hash (see app.ingest)."""
    basis = "|".join(
        (doc_id, chunk_id, entity.strip().lower(), metric.strip().lower(), quote.strip())
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:20]


def _clean_optional(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def parse_llm_fact(raw: dict[str, Any], *, doc_id: str, chunk_id: str) -> Fact:
    """Build one Fact from a JSON object the extraction model returned.

    Checks shape only: required fields present and non-empty, fact_type is one
    of the two the brief defines, quote is long enough to be evidence,
    confidence is a number in [0, 1]. Does not check that the quote is real —
    that is app.grounding's job, applied after this parses cleanly.
    """
    if not isinstance(raw, dict):
        raise FactValidationError(f"fact is not a JSON object: {raw!r}")

    for name in REQUIRED_STRING_FIELDS:
        value = raw.get(name)
        if not isinstance(value, str) or not value.strip():
            raise FactValidationError(f"missing or empty required field: {name!r}")

    fact_type = raw["fact_type"].strip().lower()
    if fact_type not in FACT_TYPES:
        raise FactValidationError(
            f"fact_type must be one of {FACT_TYPES}, got {raw['fact_type']!r}"
        )

    quote = raw["quote"].strip()
    if len(quote) < MIN_QUOTE_CHARS:
        raise FactValidationError(f"quote too short to be evidence: {quote!r}")

    confidence_raw = raw.get("confidence", 0.5)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        raise FactValidationError(f"confidence is not a number: {confidence_raw!r}")
    confidence = min(1.0, max(0.0, confidence))

    entity = raw["entity"].strip()
    metric = raw["metric"].strip()

    return Fact(
        id=_stable_id(doc_id, chunk_id, entity, metric, quote),
        doc_id=doc_id,
        chunk_id=chunk_id,
        entity=entity,
        metric=metric,
        fact_type=fact_type,
        quote=quote,
        confidence=confidence,
        value=_clean_optional(raw, "value"),
        unit=_clean_optional(raw, "unit"),
        period_label=_clean_optional(raw, "period_label"),
        period_start=_clean_optional(raw, "period_start"),
        period_end=_clean_optional(raw, "period_end"),
        basis=_clean_optional(raw, "basis"),
        status=_clean_optional(raw, "status"),
        dimensions={k: v for k, v in raw.items() if k not in _KNOWN_FIELDS},
    )


def parse_llm_facts(
    raw_facts: list[Any], *, doc_id: str, chunk_id: str
) -> tuple[list[Fact], list[dict[str, Any]]]:
    """Parse every fact object a model returned for one chunk.

    Returns (facts, rejections). One malformed fact must never discard the
    other nineteen in the same response, so failures are collected rather than
    raised past this function.
    """
    facts: list[Fact] = []
    rejections: list[dict[str, Any]] = []
    for raw in raw_facts:
        try:
            facts.append(parse_llm_fact(raw, doc_id=doc_id, chunk_id=chunk_id))
        except FactValidationError as exc:
            rejections.append({"raw": raw, "reason": str(exc)})
    return facts, rejections


def attach_grounding(fact: Fact, result: GroundingResult) -> Fact:
    """Apply a grounding verdict to a parsed fact, returning a new Fact.

    Only a grounded result may set grounded=True. Calling this with a failed
    GroundingResult is a caller error — the failure belongs in the rejection
    log (see app.extract), not attached to a Fact that will look grounded.
    """
    if not result.ok:
        raise ValueError("attach_grounding called with a failed GroundingResult; "
                          "record this as a rejection instead of attaching it")
    return replace(
        fact,
        page_number=result.page_number,
        printed_page=result.printed_page,
        grounded=True,
    )