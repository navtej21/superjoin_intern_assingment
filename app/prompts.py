"""The extraction prompt: turns one chunk of document text into candidate facts.

Every constraint here traces back to a specific requirement, not a preference:

- No fixed metric or entity vocabulary is suggested anywhere in this prompt.
  The brief bans hard-coded, document-specific schemas, so the model is asked
  to describe what a fact is about in the document's own words, not to
  classify it into categories we wrote in advance.
- The quote instruction is the most heavily emphasised part of the prompt on
  purpose. Everything downstream — grounding, comparison, the reviewer's
  trust in the system — depends on that string being a real, unedited
  substring of the source text. A model asked politely for a quote will still
  round a number or tidy a sentence unless told, explicitly, that doing so
  gets the fact thrown away.
- The worked example is a fictional company with invented figures, never a
  real sentence from the starter corpus. Anchoring the few-shot example to
  Delhivery or the IMF would bias the model toward extracting facts that look
  like *this* corpus, which cuts against the brief's requirement to
  generalise to a reviewer's own PDFs.
- fact_type is the one closed field, because "numerical or semantic" is the
  brief's own split, not one we invented.

PROMPT_VERSION is the cache key alongside each chunk's content hash (see
app.llm). Bump it whenever the wording below changes meaningfully, so a
cached response from an older prompt is never mistaken for one that reflects
the current instructions.
"""
from __future__ import annotations

PROMPT_VERSION = "v1"

_INSTRUCTIONS = """You are a meticulous fact-extraction engine. You will be given a page or several \
consecutive pages of text taken from one PDF document — a company filing, an annual \
report, an economic report, or similar. Extract atomic, checkable facts stated in \
this text. Nothing else.

WHAT COUNTS AS A FACT
A fact is one specific, checkable claim: a number attached to a metric and (usually) \
a time period ("numeric"), or a stated status, role, relationship, or identifying \
detail about a named entity ("semantic"). Do not extract vague statements, opinions, \
or marketing language. Do not extract a table header or column label on its own — \
extract the value it labels. Never invent a fact that is not explicitly stated in \
the text below.

Extract between 3 and 20 of the most significant, distinct facts in this text. Prefer \
facts that state a quantity, a date-bound period, a named entity's role or status, or \
a named entity's identifying details (address, registration number) — these are the \
facts most likely to be checked against other documents later. If the text contains \
no extractable facts (a cover page, a table of contents, a pure image caption), \
return an empty array.

OUTPUT FORMAT
Return a single JSON array. Each element is an object with exactly these keys:

  "entity"        the specific person, organization, product, or named thing the \
                   fact is about. Not the document itself.
  "metric"        what is being stated about the entity, described in language drawn \
                   from the text itself (e.g. "revenue from operations", "board \
                   membership status", "registered office address"). Describe it the \
                   way the document describes it — do not force it into a category.
  "fact_type"     exactly "numeric" or "semantic".
  "value"         the stated value, as text. For a numeric fact, the number as \
                   written (e.g. "81,415.38"). For a semantic fact, the stated status \
                   or descriptor (e.g. "resigned", "active", "Gurugram, Haryana").
  "unit"          the unit as stated, if any (e.g. "INR million", "per cent", \
                   "₹ crore"). Omit the key or use null if not applicable.
  "period_label"  the time period as stated, if any (e.g. "FY24", "Q2 FY25", "as of \
                   March 31, 2024"). Omit the key or use null if not applicable.
  "basis"         a qualifying basis if the text states one (e.g. "standalone", \
                   "consolidated", "pro forma"). Omit the key or use null if the text \
                   does not distinguish one.
  "status"        whether the text marks the fact as actual, projected, estimated, \
                   restated, or provisional. Omit the key or use null if the text \
                   does not say.
  "quote"         a VERBATIM substring copied character-for-character from the \
                   document text below, long enough to stand as evidence on its own \
                   (aim for 15-40 words). This is the single most important field. Do \
                   not paraphrase. Do not round a number. Do not correct spelling or \
                   reformat punctuation. Copy exactly, including any line breaks \
                   collapsed to spaces. A fact whose quote does not appear verbatim in \
                   the source text will be discarded even if the fact itself is true.
  "confidence"    your confidence, from 0.0 to 1.0, that this is a genuine, atomic, \
                   well-grounded fact.

EXAMPLE (illustrative only — a fictional company, invented figures. Do not reuse \
these entities or values; extract only what the actual document text below states.)

Text: "Acme Freight Ltd's standalone revenue for FY24 was ₹412.6 crore, up from \
₹365.1 crore in FY23. Ravi Sharma, appointed Independent Director in 2019, resigned \
from the Board effective 1 June 2024."

Output:
[
  {
    "entity": "Acme Freight Ltd",
    "metric": "revenue, standalone basis",
    "fact_type": "numeric",
    "value": "412.6",
    "unit": "INR crore",
    "period_label": "FY24",
    "basis": "standalone",
    "status": "actual",
    "quote": "Acme Freight Ltd's standalone revenue for FY24 was ₹412.6 crore",
    "confidence": 0.95
  },
  {
    "entity": "Ravi Sharma",
    "metric": "board membership status",
    "fact_type": "semantic",
    "value": "resigned",
    "period_label": "effective 1 June 2024",
    "status": "actual",
    "quote": "Ravi Sharma, appointed Independent Director in 2019, resigned from the Board effective 1 June 2024",
    "confidence": 0.9
  }
]

Now extract facts from the following document text. Return ONLY the JSON array — \
no prose before or after it, no markdown code fence.

---DOCUMENT TEXT START---
{chunk_text}
---DOCUMENT TEXT END---"""

_PLACEHOLDER = "{chunk_text}"


def build_extraction_prompt(chunk_text: str) -> str:
    """Insert the chunk text into the prompt template.

    Deliberately a plain substring replace, not str.format(): the template
    above contains a worked JSON example full of literal '{' and '}'
    characters, which .format() would try to interpret as fields of its own
    and raise a KeyError on the first one it hit ('{"entity"' etc). A single
    known placeholder token sidesteps that entirely.
    """
    return _INSTRUCTIONS.replace(_PLACEHOLDER, chunk_text)