# Fact Knowledge Layer

A system that reads a corpus of PDFs, extracts individually-grounded
facts (a number or a status, tied to the exact quote and page it came
from), and finds where those facts corroborate, contradict, or can be
reconciled with context — across documents, not just within one.

Built for the Superjoin VIT 2026 Engineering Intern hiring assignment,
against six real filings: Delhivery's IPO prospectus (2022), its FY24
annual report, its Q4 FY24 earnings deck, India's Economic Survey
2024-25, the RBI's Annual Report 2024-25, and the IMF's 2025 India
Article IV report.

## Demo Of The Product

[Watch the video](https://drive.google.com/file/d/1dG03SM00RsnzZyJutbwymn4SKVQdac-V/view?usp=drive_link)

## 1. Approach & Architecture

The pipeline runs in four phases, each one a thin, independently
runnable module rather than one monolithic script:

**Phase A — Ingestion** (`app/ingest.py`, `app/chunker.py`). Each PDF is
parsed page by page (PyMuPDF), with the *printed* page number (the one
on the page itself, which is often not the same as the PDF's physical
page index) recovered separately from the physical index, so a fact can
cite both. Pages are then split into overlapping chunks sized for the
extraction LLM's context window. Ingestion is content-hash deduplicated,
so re-running it, or ingesting the same file twice under different
names, is free.

**Phase B — Extraction & Grounding** (`app/extract.py`,
`app/grounding.py`, `app/facts.py`). Each chunk is handed to an LLM
(Claude) with a prompt asking for facts as `(entity, metric, value,
period, basis, quote)` tuples — deliberately open-vocabulary: there is
no fixed schema of allowed metric names, entities, or bases, because a
hard-coded lookup table would silently fail on the next document that
uses different wording. Every candidate fact is then *grounded*:
its `quote` must appear verbatim (whitespace-normalised) on a page the
source chunk actually covered, or it's rejected, not silently dropped
into the facts table. Rejected candidates are kept in
`extraction_rejections`, not discarded — a system that shows what it
couldn't verify is more trustworthy than one that only ever shows
successes.

**Phase C — Relationship Detection** (`app/adjudicate.py`). Every pair
of facts sharing a normalised entity is compared: matching metric names
(Jaccard word-overlap, not a fixed synonym table) and matching periods
decide whether two facts are "about the same claim"; matching values
then decide whether that claim is corroborated or contradicted. A
separate pass looks for *arithmetic* relationships — a fact that equals
the sum of two others for the same entity/period/basis — found by
addition, not by naming specific line items, so it generalises to any
filing that breaks a total into parts. This module is heuristic by
design (see §3) and was hardened through five rounds of fixes driven
entirely by real extraction output, documented in full in its own
module docstring.

**Phase D — API** (`app/api.py`). A read-only FastAPI layer over the
populated database — `/documents`, `/facts`, `/relationships`,
`/rejections` — so a reviewer can explore the results through
`/docs` (Swagger UI) without touching a terminal. Relationships are
computed live on each request rather than stored, since the schema has
no relations table and `find_relationships` is fast enough at this
corpus size that persisting the output would only be a cache with no
real benefit.

Storage is SQLite (`app/db.py`) — a single file, so `pip install &&
uvicorn` is the entire setup story for a reviewer, no service to
provision.

## 2. Setup & How to Run

```bash
git clone https://github.com/navtej21/superjoin_intern_assingment.git
cd superjoin_intern_assingment
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

`data/fact_layer.db` ships in this repo with all six documents already
ingested (Phase A: pages and chunks, `python -m app.pipeline stats` to
confirm), so cloning gets you the corpus without touching a PDF
parser. Facts and relationships (Phases B/C) are not pre-populated —
they cost real API calls, so they're left for you to run with your own
key rather than baked in. Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-api03-***
```

```bash
# Phase A — ingest all PDFs in the corpus (free, local)
python -m app.pipeline ingest data/corpus/

# see what's in the database so far
python -m app.pipeline stats

# Phase B — extract facts (this is the only step that costs API calls)
python -m app.pipeline extract                # all documents
python -m app.pipeline extract --doc-id <id>  # just one, e.g. to retry a failed doc

# Phase C — find relationships (free, local, re-runnable any time)
python -m app.pipeline relationships
```

Either way, serve the API with:

```bash
uvicorn app.api:app --reload
# then open http://127.0.0.1:8000/docs
```

Run the test suite with:

```bash
pytest -q
```

## 3. Design Decisions & Trade-offs

**Heuristic, not ML.** Matching is done on normalised entity/metric
text, numeric tolerance, and simple arithmetic — no entity embeddings,
no fixed lookup table of "this metric name means that metric name."
That's consistent with the open-vocabulary stance in Phase B: a lookup
table of allowed metrics is exactly the kind of hard-coded schema the
brief rules out.

**Asymmetric overlap thresholds.** Two facts whose *values already
agree* need much less textual similarity to be treated as the same
claim than two facts whose values *disagree* need before being called a
conflict. This split exists because a real balance sheet has dozens of
line items per entity/period — "Total non-current assets" and "Total
current assets" share 2 of their 3 significant words but are not the
same claim, so a single shared threshold either missed real
corroborations or manufactured false contradictions depending on where
it was set.

**Absolute, not relative, numeric tolerance.** A relative tolerance
scales with magnitude, which is dangerous at large numbers — a real
false positive had a ~141-million-scale share count accepted as "equal"
to itself plus an unrelated 196.19 per-share price, because 0.005% of
141 million swallowed the difference. Figures here are compared with a
flat ±0.05 tolerance instead.

**Five rounds of real-data hardening.** Rather than reasoning about
edge cases in the abstract, `app/adjudicate.py` was run against
successively larger real extractions (242 → 436 → 1,012 facts) and
fixed against what actually broke each time: generic connective words
("total", "expense") falsely linking unrelated line items; percentage
and absolute figures being compared as if they were the same kind of
number; an address restated with reordered fields being flagged as
contradicting itself; the same sentence extracted twice by the LLM
across a chunk-overlap boundary being reported as "corroboration from
independent locations" when it's one location; and, at the largest
scale, dense macro-economic data producing coincidental arithmetic
matches between unrelated indicators. Each round is documented in the
module's own docstring with the real numbers that motivated it, and
pinned with a regression test so later rounds can't re-break it.

**Relationships computed live, not persisted.** No relations table in
the schema — computing on request means the API can never serve a
stale answer relative to the facts table, and at this corpus size
(hundreds to low thousands of facts) the computation is fast enough
that caching it would add complexity for no measurable benefit.

## 4. Known Limitations

- **Grounding checks the quote against the page, not the value against
  its own quote.** A fact can have a `value` that doesn't actually
  appear in its stored `quote` (a real example: a "net payroll
  additions = 131" fact whose quote text doesn't contain "131"
  anywhere) — the LLM extracted the number correctly from context but
  attached the wrong sentence as evidence. Grounding verifies the quote
  is real; it doesn't yet cross-check the value against it.
- **Templated metric names can produce low-value but technically-correct
  corroborations.** Two facts named by the same boilerplate phrasing
  for different subjects — e.g. "new markets explored for cranes, lifts
  and winches" and "new markets explored for office equipment," both
  equal to 12 — share enough vocabulary to corroborate despite naming
  different product categories. Tightening the relevant threshold to
  exclude this would also exclude genuine low-overlap corroborations
  found in the same corpus, so this is left as a documented trade-off
  rather than "fixed."
- **Period matching is year-granularity, not a real fiscal calendar.**
  A GDP growth *projection* for "2025-26" and a *realized* print for
  "2025Q2" can land in the same coarse year bucket and get flagged as
  contradicting, when they're actually different kinds of claims about
  different sub-periods. Some facts sharing a generic metric label
  across genuinely different sub-categories (e.g. rankings across
  different export sectors) can be compared as if they were the same
  claim for the same reason — every relationship carries full quotes
  and periods specifically so a human reviewer can dismiss this kind of
  case at a glance.
- **A handful of extraction candidates use an ellipsis to compress a
  table row** (e.g. "Service EBITDA (₹ million)... 9,414"), which can
  never verbatim-match the source PDF text. Grounding correctly rejects
  these, but a tighter extraction prompt could avoid paying for the API
  call in the first place.

None of this is presented as "solved" — every relationship this system
reports carries its full source quotes precisely so a human can catch
what the heuristic gets wrong. No relationship is itself a fact; only
the individual, independently-grounded Facts it points at are.

## 5. Example Results

Run against the full corpus — all six documents, 1,012 grounded facts —
`python -m app.pipeline relationships` finds real corroboration,
contradiction, and context-reconcilable relationships. One example of
each required pattern:

**Corroboration** — the same fact, stated independently in two places:

> Delhivery Limited's Corporate Identity Number = `U63090DL2011PLC221234`,
> stated on the prospectus cover page and again in its signature
> section — two different pages, byte-identical claim.

**Contradiction** — the same claim, genuinely in conflict:

> Kapil Bharati's board designation is stated as "Executive Director and
> Chief Technology Officer" in the 2022 prospectus and "Whole Time
> Director and Chief Technology Officer" in the FY24 annual report — a
> real title change between filings, not a data error.

**Context-reconcilable — arithmetic.** A total explained by summing its
named parts, found by addition rather than by hard-coding these line
items:

> Total non-current assets (₹40,934.01M) + Total current assets
> (₹43,360.82M) = Total assets (₹84,294.83M).

(The same pattern independently fires several more times in this
corpus — total liabilities, the IPO's fresh-issue + offer-for-sale =
total offer size, finance income totals, and two separate forex-reserve
reconciliations — demonstrating the check generalises rather than being
tuned to one filing.)

**Context-reconcilable — basis difference.** Two values differ, but a
stated accounting basis explains the gap:

> Revenue from operations is ₹74,540.82M on a standalone basis and
> ₹81,415.38M on a consolidated basis for FY24 — the same period, a
> different scope, not a conflict.

The full set of relationships, with every fact's exact quote and page
number, is browsable at `/relationships` once the API is running
(`uvicorn app.api:app --reload`, then `/docs`), or via
`python -m app.pipeline relationships` directly from the database.
