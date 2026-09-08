"""Text normalisation shared by chunking and quote grounding.

Grounding compares a model-produced quote against the page it claims to come
from. That comparison cannot be exact-string: PDF text extraction introduces
differences the model has no way to reproduce and no reason to — ligatures
(``ﬁ`` for ``fi``), non-breaking and thin spaces around figures, soft hyphens,
curly quotation marks, en/em dashes standing in for hyphens, and line wrapping
that turns a single sentence into three lines.

So both sides are normalised into one flattened form before matching. The rule
is that normalisation may only remove *presentational* difference. It must never
touch a digit, a letter, or a decimal point — if it did, a grounding check could
pass on a figure the document does not state, which is the one failure this
whole layer exists to prevent.

The original quote is always stored unmodified. Normalisation exists for the
comparison, not for the record.
"""
from __future__ import annotations

import re
import unicodedata

# Characters that carry no meaning for our purposes and vary by extractor.
_ZERO_WIDTH = dict.fromkeys(
    map(ord, "​‌‍⁠﻿­"),  # incl. soft hyphen
    None,
)

# Quotation marks and dashes have many Unicode spellings; fold each family to
# one ASCII representative so a model's straight quote matches a PDF's curly one.
_PUNCT_FOLD = {
    ord("‘"): "'", ord("’"): "'", ord("‚"): "'", ord("‛"): "'",
    ord("“"): '"', ord("”"): '"', ord("„"): '"', ord("‟"): '"',
    ord("‐"): "-", ord("‑"): "-", ord("‒"): "-", ord("–"): "-",
    ord("—"): "-", ord("―"): "-", ord("−"): "-",
    ord(" "): " ", ord(" "): " ", ord(" "): " ", ord(" "): " ",
    ord(" "): " ", ord(" "): " ",
}

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Flatten presentational variation, preserving every character that carries
    meaning.

    NFKC is what expands ligatures and folds full-width forms. It is applied
    first so the explicit tables below operate on a settled representation.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ZERO_WIDTH)
    text = text.translate(_PUNCT_FOLD)
    return _WHITESPACE.sub(" ", text).strip()


def normalize_for_match(text: str) -> str:
    """Normalisation for quote lookup: case-folded on top of :func:`normalize`.

    Case is dropped because extraction of a heading or a table label may differ
    in case from the body text a model echoes back. Digits and separators are
    untouched, so this cannot make a wrong number look right.
    """
    return normalize(text).casefold()


def estimate_tokens(text: str) -> int:
    """Rough token count, used only for chunk budgeting and cost reporting.

    Deliberately not a real tokenizer. Tokenizers are model-specific, and a
    dependency on one would tie chunk boundaries to a provider we might swap.
    Four characters per token is close enough for sizing decisions, and nothing
    downstream depends on it being exact.
    """
    return max(1, len(text) // 4)