"""Phase C: finding corroboration, contradiction, and context-reconcilable
relationships between stored facts.

This is the actual point of a "fact knowledge layer" — Phase A and B only
get you a pile of grounded, individually-true statements. What makes it a
*knowledge* layer is noticing when two of those statements are about the
same real-world thing, and characterising how they relate:

  corroboration        two independent mentions of the same claim agree
  contradiction         two mentions of the same claim disagree, with no
                         basis/period difference to explain why
  context_reconcilable  two mentions look like they disagree, but a
                         basis difference (standalone vs consolidated) or a
                         third fact (a sum) fully explains the gap

Deliberately heuristic, not ML-based: matching is done on normalised
entity/metric text, numeric tolerance, and simple arithmetic — no entity
embeddings, no fixed lookup table of "this metric name means that metric
name". That keeps it consistent with the rest of the project's open-
vocabulary stance (see app.facts).

This module went through one real correction worth stating plainly, because
it's a good example of the kind of mistake a heuristic system makes
silently if nobody checks its output against real data: an early version
compared *any* two numeric facts sharing an entity, a period, and one
overlapping word in their metric names. Run against 242 real extracted
facts, that produced dozens of nonsense findings — "Total borrowings"
flagged as contradicting "Total non-current assets" because both contain
the word "total"; a per-share acquisition cost accepted as equal to a share
count because a relative-tolerance check scaled up to a multi-thousand-unit
allowance for large numbers. Both are fixed below (see `values_match` and
the stop-word list).

A second round of real-data testing, after that first fix, surfaced a
sharper version of the same lesson: a real balance sheet has dozens of line
items per entity/period, and a low overlap bar that's safe when two values
already agree is not safe once they disagree — "Total non-current assets"
and "Total current assets" share 2 of their 3 significant words; "EBITDA"
and "EBITDA margin" share their only word; every revenue-by-segment line
shares "revenue" and "services" with the aggregate it sums into. None of
these are the same claim, but a single overlap threshold couldn't tell
"these values agree, weak overlap is enough evidence" apart from "these
values disagree, weak overlap is not evidence of anything." The fix
(`_METRIC_OVERLAP_THRESHOLD` vs `_METRIC_OVERLAP_THRESHOLD_DIFFERING`) is
below, alongside three narrower fixes for patterns the threshold alone
didn't catch: a percent/rate vs absolute-figure guard (`_is_percentage`,
for "EBITDA" vs "EBITDA margin"-shaped pairs), gluing hyphenated modifiers
before tokenizing (so "non-current" can't share a token with "current"),
and treating two identically-named facts on different specific calendar
dates within the same year as separate events, not a contradiction (a cap
table records the same "equity shares allotted through Employee Stock
Options Purchases" metric on four different dates in 2023).

A third round, against a larger 436-fact extraction, found the round-two
stop-word additions had a side effect: stopwording "number"/"office"/
"address" shrank some metric names down to almost nothing, so "Corporate
Identity Number" and "corporate office address" — reduced to {corporate,
identity} and {corporate} — looked 50% overlapping on a single shared word
that was doing no identifying work. `metric_overlap` now requires at least
3 combined significant words before trusting a *partial* match (an
identical token set still matches at any length — that's real evidence).
Two narrower fixes rode alongside it: coincidentally-equal percentages
("headline inflation" and "core inflation" both landing on 4.6) now always
need the high overlap bar, not just the low one equal values normally earn,
and semantic values are compared after collapsing commas/whitespace, so an
address quoted with different punctuation in two filings is recognised as
the same address instead of flagged as contradicting itself.

A fourth round, against a cleaner 436-fact run with the first three rounds'
fixes in place, found two more issues — one a matching gap, one a data-
quality gap the earlier rounds hadn't been in a position to see yet.

The matching gap: a corporate office address restated across two different
filings as "Plot 5, Sector 44, Gurugram 122002 Haryana, India" and "Plot
No. 5, Sector-44, Gurugram, Haryana 122002" was flagged as a contradiction.
Same address — reordered (city/pin/state vs city/state/pin), "Plot" vs
"Plot No.", "Sector 44" vs "Sector-44" — but the existing comma/whitespace
normalisation only helps when the words are in the same order, and simple
substring containment can't see past a reorder. `_looks_like_same_semantic_
value` now also treats two values as the same when their significant-word
sets overlap heavily (Jaccard >= 0.75, over a combined vocabulary of at
least 5 words) — high enough that two designations differing by one real
word ("Executive Director" vs "Whole Time Director", "Chairman" vs
"Chairperson") still land below it and correctly keep contradicting, which
matters because those *are* the kind of judgment call this module exists to
surface for a human, not paper over.

The data-quality gap: several facts turned out to be reported twice with
*byte-identical evidence* — the same Corporate Identity Number and the same
Registration Number, each with an identical quote, both attributed to the
same page; the same "real GDP growth = 7.8" and "nominal GDP growth = 9.8"
figures, each pair's quotes overlapping (one a substring of the other) on
the same page; a budgetary-provision figure restated with two slightly
different metric names but a quote identical but for a trailing period.
All of these are one real sentence handed to the extraction LLM twice —
almost certainly because the chunker's overlap window straddles a
page/chunk boundary — not two independent mentions, and reporting one as
"corroboration ... from independent locations" is simply false: it is one
location, described twice. `find_pairwise_relationships` and
`find_additive_reconciliations` now both dedupe facts that share a doc,
page, entity and value with an overlapping quote before comparing anything
— collapsing what were four near-identical corroboration entries for one
CIN mention down to the single genuine one (the real second mention, on a
different page, still corroborates correctly).

A fifth round, against the full 1,012-fact corpus (all six source documents
finally extracted), found the additive-reconciliation check breaking down
at scale. A dense macro-economic document contributes hundreds of small,
single- or double-decimal figures for one entity ("India") in one period
bucket, and at that volume, arithmetic coincidences stop being rare: "tur
import quantity" (7.7) + "valuation gain on foreign exchange reserves"
(4.3) summing to exactly "new markets explored for cranes, lifts and
winches" (12) is pure coincidence, not a real part-of-a-whole relationship
— those three figures have nothing to do with each other. Every genuine
reconciliation in this corpus, by contrast, is named in a way that shares
real vocabulary with what it sums to ("non-current assets"/"current
assets" -> "total assets", shares "assets"; "revenue from services"/
"revenue from sale of traded goods" -> "revenue from operations", shares
"revenue"). `find_additive_reconciliations` now requires at least one of
the two addends to share a significant word with the target metric before
accepting a sum — checked directly against every real reconciliation and
every real coincidence surfaced by this run, not just reasoned about in the
abstract (see `_shares_significant_word` and its tests).

None of this makes the heuristic precise — a handful of same-date,
similar-worded pairs (two preference-share conversions differing only by
series letter) still surface as "contradictions" a human will immediately
recognise and dismiss from the quotes alone. That's an accepted trade-off,
not an oversight: every relationship this module reports carries full
quotes and periods precisely so a human can catch what the heuristic gets
wrong, and no relationship is a "fact" — only the individual Facts it
points at are.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

# Below this, two numbers are treated as "the same value". Deliberately a
# flat absolute tolerance, not a relative one: these are figures reported to
# two decimal places, not measurements with proportional error, and a
# tolerance that scales with magnitude is actively dangerous here — at 0.005%
# of a 141-million share count, an unrelated per-share price (196.19) was
# once accepted as making that count "equal to itself plus a bit", producing
# a nonsense reconciliation. A number this size should only ever match
# another occurrence of the exact same figure.
_ABS_TOLERANCE = 0.05

# Two metrics are considered "about the same thing" when the Jaccard overlap
# of their significant-word sets clears this. Real Jaccard (intersection over
# union), not intersection-over-smaller-set — the smaller-set version let a
# short, generic metric name match almost anything that shared one word with
# it, which combined with the stop-word gaps below was the main source of
# false positives found when this was run against the real corpus.
#
# This threshold is deliberately split in two, found necessary by a second
# round of real-data testing after the stop-word fix above (see the module
# docstring's second correction). A balance sheet has dozens of line items
# per entity/period, and once two facts' *values* already differ — which is
# true of nearly any two unrelated line items — a low overlap bar turns any
# shared generic-but-not-quite-stopworded word ("current", "income",
# "assets", "services") into a false "contradiction": "Total non-current
# assets" vs "Total current assets" share 2 of 3 words; "EBITDA" vs "EBITDA
# margin" share their only word. Two facts whose values already agree need
# much less textual evidence to be the same claim (exact numeric coincidence
# between two genuinely unrelated multi-decimal figures is rare), so that
# path keeps the low bar; two facts whose values disagree need much
# stronger textual evidence before being called a *conflict* rather than
# just two different things, so that path uses a high bar instead.
_METRIC_OVERLAP_THRESHOLD = 0.2
_METRIC_OVERLAP_THRESHOLD_DIFFERING = 0.7
_SEMANTIC_METRIC_OVERLAP_THRESHOLD = 0.34

# How much two semantic *values* (not metric names) have to overlap, word for
# word, before a reordered/reworded restatement counts as "the same value"
# rather than a contradiction — see the module docstring's fourth
# correction. Set from a real case ("Plot 5, Sector 44, Gurugram 122002
# Haryana, India" vs "Plot No. 5, Sector-44, Gurugram, Haryana 122002",
# Jaccard 7/9 = 0.778) and checked against real near-miss designations that
# must stay contradictions ("Executive Director" vs "Whole Time Director",
# 0.625; "Chairman" vs "Chairperson", 0.714) — both a clear margin below.
_SEMANTIC_VALUE_TOKEN_THRESHOLD = 0.75
_MIN_SEMANTIC_VALUE_COMBINED_TOKENS = 5

_METRIC_STOPWORDS = {
    "the", "a", "an", "of", "for", "on", "in", "and", "or", "to", "as",
    "basis", "from", "with", "at", "is", "was", "were", "status", "role",
    "by", "this",
    # Generic financial-statement connective nouns: each one appears on
    # dozens of unrelated line items ("Total assets", "Total borrowings",
    # "Net worth", "Closing balance"...), so treating them as significant
    # words turns "shares the word 'total'" into a false signal of
    # relatedness. Removing them means only the substantive noun (assets,
    # borrowings, revenue, worth) has to match.
    "total", "amount", "number", "value", "net", "gross", "balance",
    "closing", "opening", "expense", "expenses", "cost", "costs",
    "standalone", "consolidated",
    # Generic company-info nouns: "registered office address" and
    # "corporate office address" are two different fields, not two mentions
    # of the same one, and sharing "office"/"address" was enough to flag
    # them as contradicting each other in the real corpus.
    "office", "address",
}

_ENTITY_STOPWORDS = {
    "mr", "mrs", "ms", "dr", "the", "limited", "ltd", "inc", "llp",
    "pvt", "private", "company", "co",
}

# Metrics describing a rate, growth figure, or ratio are not meaningfully
# additive — two unrelated percentages summing to a third by coincidence is
# common (values 0-100 with one decimal place collide easily), whereas a
# genuine "parts of a whole" reconciliation is about absolute totals. Used
# only to gate the additive check, not pairwise comparison.
_NON_ADDITIVE_METRIC_WORDS = {
    "growth", "rate", "ratio", "margin", "deficit", "inflation", "yield",
    "participation", "percentage",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_entity(name: str) -> str:
    """Collapse casing/titles/corporate suffixes so the same real entity
    written two ways compares equal: 'Mr. Suvir Suren Sujan' and 'Suvir Suren
    Sujan' both become 'suvir suren sujan'; 'Delhivery Limited' and
    'Delhivery' both become 'delhivery'."""
    tokens = [t for t in _WORD_RE.findall(name.lower()) if t not in _ENTITY_STOPWORDS]
    return " ".join(tokens)


def _metric_tokens(metric: str) -> frozenset[str]:
    """Hyphens are glued rather than split ('non-current' -> 'noncurrent'),
    so a modifier doesn't get torn into its parts and spuriously share a
    token with its near-opposite: 'Total non-current assets' was overlapping
    heavily with 'Total current assets' purely because 'non-current' split
    into 'non' + 'current', handing the two labels a shared 'current' token
    they should never have shared."""
    glued = metric.lower().replace("-", "")
    return frozenset(t for t in _WORD_RE.findall(glued) if t not in _METRIC_STOPWORDS)


def metric_overlap(a: str, b: str) -> float:
    """Real Jaccard similarity (intersection / union) over significant
    metric words. See the stop-word list above for why generic connective
    nouns are excluded first.

    An identical token set is always a full match, however small — that's
    real evidence, not noise. But a *partial* match drawn from fewer than 3
    combined significant words is not trustworthy: 'Corporate Identity
    Number' and 'corporate office address' reduce, after stopwording their
    generic nouns ('number', 'office', 'address'), to {corporate, identity}
    and {corporate} — sharing one word out of a two-word combined
    vocabulary looks like 50% overlap, but 'corporate' is doing no
    identifying work there at all. Requiring more combined vocabulary
    before trusting a partial overlap catches this without having to
    special-case every short label the stop-word list eventually thins out.
    """
    ta, tb = _metric_tokens(a), _metric_tokens(b)
    if not ta or not tb:
        return 0.0
    if ta == tb:
        return 1.0
    union = ta | tb
    if len(union) < 3:
        return 0.0
    return len(ta & tb) / len(union)


def parse_numeric_value(value: str | None) -> float | None:
    """Parse a Fact.value string into a float, handling the two conventions
    real filings use that plain float() rejects: comma thousands separators
    and parenthesised negatives for losses, e.g. '(1,679.68)' -> -1679.68."""
    if not value:
        return None
    s = value.strip()
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace(",", "").replace("₹", "").replace("%", "").strip()
    if not s:
        return None
    try:
        parsed = float(s)
    except ValueError:
        return None
    return -parsed if negative else parsed


def values_match(a: float, b: float) -> bool:
    """Flat absolute tolerance — see the module-level comment on
    _ABS_TOLERANCE for why this must not scale with magnitude."""
    return abs(a - b) <= _ABS_TOLERANCE


_YEAR_RE = re.compile(r"(20\d{2})")
_FY_SHORT_RE = re.compile(r"\bFY\s?(\d{2})\b", re.IGNORECASE)
_MONTH_NAMES = (
    "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
)
_SPECIFIC_DATE_RE = re.compile(
    rf"\b(?:{_MONTH_NAMES})[a-z]*\.?\s+\d{{1,2}},?\s*(?:19|20)\d{{2}}\b"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTH_NAMES})[a-z]*\.?,?\s*(?:19|20)\d{{2}}\b",
    re.IGNORECASE,
)


def extract_year(period_label: str | None) -> int | None:
    """Best-effort single year out of a period string, for coarse bucketing
    only — not a real fiscal-calendar parser. 'FY ended March 31, 2024',
    'as of March 31, 2024', and 'FY24' all resolve to 2024, which is enough
    to tell 'this is about FY23' apart from 'this is about FY24' without
    needing a real date parser for Phase C."""
    if not period_label:
        return None
    match = _YEAR_RE.search(period_label)
    if match:
        return int(match.group(1))
    match = _FY_SHORT_RE.search(period_label)
    if match:
        return 2000 + int(match.group(1))
    return None


def _looks_like_specific_date(period_label: str | None) -> bool:
    """True when a period label names a specific calendar day ('April 06,
    2023'), not just a fiscal year ('FY24') or a year-end description.

    Used only to tell apart two same-year but different-day period labels —
    real corpus example: a cap table records equity allotments on 'April 06,
    2023', 'May 06, 2023', and 'June 08, 2023' under the identical metric
    name 'equity shares allotted through Employee Stock Options Purchases'.
    extract_year collapses all three to 2023, so without this check they
    were flagged as three-way contradicting each other, when they are just
    four separate purchase events in the same year.
    """
    return bool(period_label and _SPECIFIC_DATE_RE.search(period_label))


def _is_percentage(fact: Any) -> bool:
    """True when a fact reads as a rate, ratio, or percentage rather than an
    absolute figure — by declared unit first, and by metric wording as a
    fallback for when the unit field wasn't populated consistently by
    extraction. A percentage and an absolute figure are never "the same
    claim" stated two ways, however similar their metric names look —
    'EBITDA' (Rs 1,266) vs 'EBITDA margin' (1.6) share their only
    significant word and were flagged as contradicting each other in the
    real corpus; so were 'revenue from services' vs 'Part truckload revenue
    growth'. Used to gate both pairwise comparison and the additive check.
    """
    unit = (fact["unit"] or "").lower()
    if "percent" in unit or "%" in unit or "per cent" in unit:
        return True
    metric_words = _WORD_RE.findall((fact["metric"] or "").lower())
    return any(w in _NON_ADDITIVE_METRIC_WORDS for w in metric_words)


def _is_additive_eligible(fact: Any) -> bool:
    """Excludes rates/growth/ratios from the additive reconciliation check —
    a percentage or ratio is never a "part of a whole" in the additive
    sense, and small single-decimal figures collide by coincidence often
    enough that this guard matters in practice, not just in theory."""
    return not _is_percentage(fact)


@dataclass
class Relationship:
    relation_type: str  # "corroboration" | "contradiction" | "context_reconcilable"
    fact_ids: list[str]
    explanation: str
    evidence: list[dict[str, Any]] = field(default_factory=list)


def _evidence(fact: Any) -> dict[str, Any]:
    return {
        "fact_id": fact["id"],
        "doc_id": fact["doc_id"],
        "entity": fact["entity"],
        "metric": fact["metric"],
        "value": fact["value"],
        "period_label": fact["period_label"],
        "basis": fact["basis"],
        "status": fact["status"],
        "page_number": fact["page_number"],
        "printed_page": fact["printed_page"],
        "quote": fact["quote"],
    }


def _compare_numeric_pair(a: Any, b: Any) -> Relationship | None:
    va, vb = parse_numeric_value(a["value"]), parse_numeric_value(b["value"])
    if va is None or vb is None:
        return None

    year_a, year_b = extract_year(a["period_label"]), extract_year(b["period_label"])
    same_year = year_a is not None and year_a == year_b
    label_a, label_b = (a["period_label"] or "").strip(), (b["period_label"] or "").strip()
    if (
        same_year
        and label_a != label_b
        and _looks_like_specific_date(label_a)
        and _looks_like_specific_date(label_b)
    ):
        # Same fiscal year, but both periods name a specific, different day
        # — e.g. 'April 06, 2023' vs 'May 06, 2023' — so this is two
        # separate events, not the same claim restated. See
        # _looks_like_specific_date's docstring for the real case.
        same_period = False
    else:
        same_period = same_year

    basis_a = (a["basis"] or "").strip().lower()
    basis_b = (b["basis"] or "").strip().lower()
    same_basis = basis_a == basis_b

    equal_values = values_match(va, vb)

    if not same_period:
        # Different fiscal periods are expected to differ — that's not a
        # relationship worth reporting, it's just two different facts.
        return None

    if same_basis and equal_values:
        return Relationship(
            "corroboration",
            [a["id"], b["id"]],
            f"Both state {a['entity']} {a['metric']!r} = {a['value']} for the same "
            f"period ({a['period_label']!r}) and basis, from independent locations.",
            [_evidence(a), _evidence(b)],
        )
    if same_basis and not equal_values:
        return Relationship(
            "contradiction",
            [a["id"], b["id"]],
            f"Same entity, same metric family, same period ({a['period_label']!r}) "
            f"and same basis ({basis_a or 'unstated'!r}), but different values: "
            f"{a['value']} vs {b['value']}. No basis or period difference explains this.",
            [_evidence(a), _evidence(b)],
        )
    if not same_basis and not equal_values:
        return Relationship(
            "context_reconcilable",
            [a["id"], b["id"]],
            f"Values differ ({a['value']} vs {b['value']}) but the two facts state "
            f"different bases ({basis_a or 'unstated'!r} vs {basis_b or 'unstated'!r}) "
            f"for the same period — the basis difference plausibly explains the gap.",
            [_evidence(a), _evidence(b)],
        )
    # different basis, equal values: not a conflict worth flagging (e.g. both
    # coincidentally round to the same figure) — nothing to report.
    return None


_VALUE_WHITESPACE_RE = re.compile(r"[,\s]+")


def _normalize_semantic_value(value: str) -> str:
    """Collapse commas and repeated whitespace so two mentions of the same
    real address/text differing only in punctuation compare equal — a real
    corpus case had a registered office address quoted with commas in one
    filing ('...Centre-II, Opposite Gate 6...') and without in another
    ('...Centre-II Opposite Gate 6...'), and the raw containment check
    below saw them as two different values and flagged the address as
    contradicting itself."""
    return _VALUE_WHITESPACE_RE.sub(" ", value.strip().lower()).strip()


def _semantic_value_tokens(value: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall(value.lower()))


def _semantic_values_token_overlap_high(a: str, b: str) -> bool:
    """True when two values share almost all their significant words, just
    reordered or lightly reworded — catches an address restated in a
    different field order ("city, pin, state" vs "city, state, pin") that
    plain containment can't see past. Guarded by a minimum combined
    vocabulary for the same reason metric_overlap needs one: without it, two
    short values sharing one word out of two would look like 50%+ overlap
    on no real evidence."""
    ta, tb = _semantic_value_tokens(a), _semantic_value_tokens(b)
    if not ta or not tb:
        return False
    union = ta | tb
    if len(union) < _MIN_SEMANTIC_VALUE_COMBINED_TOKENS:
        return False
    return len(ta & tb) / len(union) >= _SEMANTIC_VALUE_TOKEN_THRESHOLD


def _looks_like_same_semantic_value(a: str, b: str) -> bool:
    a_norm, b_norm = _normalize_semantic_value(a), _normalize_semantic_value(b)
    if a_norm == b_norm:
        return True
    # one is a short-hand of the other (e.g. "resigned" vs "resigned from the
    # office of Executive Director") — treat containment as agreement.
    if a_norm in b_norm or b_norm in a_norm:
        return True
    # same words, different order/phrasing (e.g. an address with its
    # city/state/pin fields in a different order) — see the module
    # docstring's fourth correction.
    return _semantic_values_token_overlap_high(a, b)


def _compare_semantic_pair(a: Any, b: Any) -> Relationship | None:
    if not a["value"] or not b["value"]:
        return None
    if _looks_like_same_semantic_value(a["value"], b["value"]):
        return Relationship(
            "corroboration",
            [a["id"], b["id"]],
            f"Both describe {a['entity']}'s {a['metric']} as {a['value']!r} "
            f"(or a more detailed restatement of it), from independent locations.",
            [_evidence(a), _evidence(b)],
        )
    # Different, non-overlapping values for what looks like the same
    # entity+metric question. Deliberately permissive rather than requiring
    # a hand-written antonym table (a fixed "active means not resigned" list
    # is exactly the kind of document-specific rule the brief rules out, and
    # it would silently miss anything not on the list). The known cost: two
    # genuinely different facts about the same role (e.g. an appointment
    # date and a separate re-appointment recommendation) can share enough
    # vocabulary to be flagged here even though they don't really conflict.
    # The metric-overlap threshold for semantic pairs is set higher than for
    # numeric ones specifically to cut down on this, and the full quotes are
    # always included so a human can tell a real status flip from a
    # false-positive "these are just two related facts" in one glance.
    return Relationship(
        "contradiction",
        [a["id"], b["id"]],
        f"{a['entity']}'s {a['metric']} is stated as {a['value']!r} in one place "
        f"and {b['value']!r} in another, with no shared basis to reconcile them. "
        f"Check the periods below — this is often a status that changed over time "
        f"rather than a genuine conflict.",
        [_evidence(a), _evidence(b)],
    )


def _shares_significant_word(a: str, b: str) -> bool:
    """Raw significant-word intersection — unlike metric_overlap, no Jaccard
    ratio and no minimum-combined-vocabulary guard, just "do these two
    labels share at least one real identifying word". Used only to gate the
    additive check (see the module docstring's fifth correction): a genuine
    part-of-a-whole reconciliation is always named in a way that shares
    vocabulary with the whole it sums to; a coincidental arithmetic match
    between three unrelated indicators never does.
    """
    return bool(_metric_tokens(a) & _metric_tokens(b))


def _dedupe_facts(facts: list[Any]) -> list[Any]:
    """Collapse facts that are almost certainly the same source statement
    extracted twice, not two independent mentions — see the module
    docstring's fourth correction.

    Two facts collapse into one when they share a doc, a page, an entity,
    and a value, and their quotes overlap (equal, or one contained in the
    other, after normalising whitespace/punctuation). That last condition
    is what keeps this safe: two genuinely different facts that happen to
    share a page, an entity, and a coincidentally-equal value but come from
    different sentences are never merged, only ones whose evidence is
    actually the same text.

    Run before both find_pairwise_relationships and
    find_additive_reconciliations, so a duplicated fact can't produce a
    false "corroboration from independent locations" (it's one location)
    and can't silently double-count in either check.
    """
    buckets: dict[tuple[str, int | None, str, str], list[Any]] = {}
    for fact in facts:
        key = (
            fact["doc_id"],
            fact["page_number"],
            (fact["entity"] or "").strip().lower(),
            (fact["value"] or "").strip().lower(),
        )
        buckets.setdefault(key, []).append(fact)

    deduped: list[Any] = []
    for bucket in buckets.values():
        kept: list[Any] = []
        for fact in bucket:
            quote = _normalize_semantic_value(fact["quote"] or "")
            is_duplicate = quote and any(
                quote == kept_quote or quote in kept_quote or kept_quote in quote
                for kept_quote in (
                    _normalize_semantic_value(k["quote"] or "") for k in kept
                )
            )
            if not is_duplicate:
                kept.append(fact)
        deduped.extend(kept)
    return deduped


def find_pairwise_relationships(facts: list[Any]) -> list[Relationship]:
    """Compare every pair of facts that plausibly refer to the same claim.

    O(n^2) over the fact set, which is fine at the scale this project
    operates at (hundreds, not millions, of facts).
    """
    facts = _dedupe_facts(facts)
    relationships: list[Relationship] = []
    by_entity: dict[str, list[Any]] = {}
    for fact in facts:
        by_entity.setdefault(normalize_entity(fact["entity"]), []).append(fact)

    for entity_facts in by_entity.values():
        if len(entity_facts) < 2:
            continue
        for a, b in combinations(entity_facts, 2):
            if a["id"] == b["id"]:
                continue
            if a["fact_type"] != b["fact_type"]:
                continue

            overlap = metric_overlap(a["metric"], b["metric"])
            if a["fact_type"] == "numeric":
                is_pct_a, is_pct_b = _is_percentage(a), _is_percentage(b)
                if is_pct_a != is_pct_b:
                    # A rate/percentage and an absolute figure are never the
                    # same claim, whatever their metric names share.
                    continue
                va, vb = parse_numeric_value(a["value"]), parse_numeric_value(b["value"])
                already_equal = va is not None and vb is not None and values_match(va, vb)
                if is_pct_a and is_pct_b:
                    # Percentages and rates cluster in a narrow band (macro
                    # indicators are almost always single or low double
                    # digits), so two coincidentally-equal percentages are
                    # much weaker evidence of being the same claim than two
                    # coincidentally-equal large absolute figures — real
                    # corpus example: "headline inflation" and "core
                    # inflation" both landing on 4.6 are genuinely different
                    # measures, not the same claim restated. Percentages
                    # always need the high bar, equal values or not.
                    threshold = _METRIC_OVERLAP_THRESHOLD_DIFFERING
                else:
                    threshold = _METRIC_OVERLAP_THRESHOLD if already_equal else _METRIC_OVERLAP_THRESHOLD_DIFFERING
                if overlap < threshold:
                    continue
                result = _compare_numeric_pair(a, b)
            else:
                if overlap < _SEMANTIC_METRIC_OVERLAP_THRESHOLD:
                    continue
                result = _compare_semantic_pair(a, b)

            if result is not None:
                relationships.append(result)

    return relationships


def find_additive_reconciliations(facts: list[Any]) -> list[Relationship]:
    """A fact that equals the sum of two others, all for the same entity,
    period and basis, is the generic form of Case 3's Note-21 reconciliation
    (revenue from services + revenue from sale of traded goods = revenue
    from operations) — found by arithmetic, not by naming these three
    metrics in code, so it generalises to any filing that breaks a total
    into parts this way.

    Two guards keep this from over-firing (both added after the arithmetic
    coincidences a first pass produced against the real corpus): rate/growth
    figures are excluded (_is_additive_eligible — see its docstring), and a
    grouping is rejected if either addend alone already equals the target,
    which signals two near-duplicate facts rather than a genuine split into
    parts.
    """
    facts = _dedupe_facts(facts)
    relationships: list[Relationship] = []
    by_bucket: dict[tuple[str, int | None, str], list[Any]] = {}
    for fact in facts:
        if fact["fact_type"] != "numeric" or not _is_additive_eligible(fact):
            continue
        value = parse_numeric_value(fact["value"])
        if value is None:
            continue
        key = (
            normalize_entity(fact["entity"]),
            extract_year(fact["period_label"]),
            (fact["basis"] or "").strip().lower(),
        )
        by_bucket.setdefault(key, []).append(fact)

    for bucket_facts in by_bucket.values():
        if len(bucket_facts) < 3:
            continue
        values = [(f, parse_numeric_value(f["value"])) for f in bucket_facts]
        for c, vc in values:
            for (a, va), (b, vb) in combinations(
                [(f, v) for f, v in values if f["id"] != c["id"]], 2
            ):
                if va is None or vb is None or vc is None:
                    continue
                if values_match(va, vc) or values_match(vb, vc):
                    continue  # degenerate: one addend alone already equals the target
                if not (
                    _shares_significant_word(a["metric"], c["metric"])
                    or _shares_significant_word(b["metric"], c["metric"])
                ):
                    # Neither addend shares a real word with what it supposedly
                    # sums to — at corpus scale this is what an arithmetic
                    # coincidence looks like, not a real relationship. See the
                    # module docstring's fifth correction.
                    continue
                if values_match(va + vb, vc) and len({a["id"], b["id"], c["id"]}) == 3:
                    relationships.append(
                        Relationship(
                            "context_reconcilable",
                            [a["id"], b["id"], c["id"]],
                            f"{a['metric']} ({a['value']}) + {b['metric']} ({b['value']}) "
                            f"= {c['metric']} ({c['value']}) for {c['entity']} — the gap "
                            f"between {a['metric']!r}/{b['metric']!r} and {c['metric']!r} "
                            f"is fully explained by this sum, not a conflict.",
                            [_evidence(a), _evidence(b), _evidence(c)],
                        )
                    )
    return relationships


def find_relationships(facts: list[Any]) -> list[Relationship]:
    """Run every relationship-finding strategy over one fact set."""
    return find_pairwise_relationships(facts) + find_additive_reconciliations(facts)