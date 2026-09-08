"""Chunk-level extraction, wired end to end except the network call.

Uses the real ingested corpus (see conftest) and hand-written response
strings standing in for what the model would return: real quotes from the
corpus for the positive cases, and a deliberately altered quote for the
negative case that must be rejected. This exercises the entire Phase B
pipeline — parsing, shape validation, grounding, persistence, caching —
without spending a token or needing an API key. Only app.llm.call_claude
itself is untested here; see scripts/smoke_test_extraction.py for that part.
"""
from __future__ import annotations

import json

from app.extract import extract_facts_for_chunk, persist_outcome, run_extraction
from tests.conftest import ANNUAL_REPORT, doc_id_for, requires_corpus

pytestmark = requires_corpus


def chunk_covering(conn, filename, pdf_page):
    doc_id = doc_id_for(conn, filename)
    return conn.execute(
        """SELECT * FROM chunks WHERE doc_id = ? AND page_start <= ? AND page_end >= ?
           ORDER BY chunk_index LIMIT 1""",
        (doc_id, pdf_page, pdf_page),
    ).fetchone()


class TestExtractFactsForChunk:
    def test_real_quote_grounds_and_is_kept(self, conn):
        """The Sujan resignation, verbatim, exactly as a correct model
        response would state it. Must ground and produce one fact."""
        chunk = chunk_covering(conn, ANNUAL_REPORT, 24)
        assert chunk is not None

        response = json.dumps([{
            "entity": "Suvir Suren Sujan",
            "metric": "board membership status",
            "fact_type": "semantic",
            "value": "resigned",
            "quote": "resigned from the Board with effect from August 24, 2023",
            "confidence": 0.92,
        }])

        outcome = extract_facts_for_chunk(conn, chunk, response)
        assert len(outcome.facts) == 1
        assert outcome.facts[0].grounded is True
        assert outcome.facts[0].page_number == 24
        assert outcome.facts[0].printed_page == "46-47"
        assert outcome.rejections == []

    def test_fabricated_quote_is_rejected_not_stored(self, conn):
        """A plausible but altered quote — wrong date — must fail grounding.
        This is the entire point of the gate: a fact that reads correctly
        must still be provably real, not just plausible."""
        chunk = chunk_covering(conn, ANNUAL_REPORT, 24)

        response = json.dumps([{
            "entity": "Suvir Suren Sujan",
            "metric": "board membership status",
            "fact_type": "semantic",
            "value": "resigned",
            "quote": "resigned from the Board with effect from September 24, 2023",
            "confidence": 0.92,
        }])

        outcome = extract_facts_for_chunk(conn, chunk, response)
        assert outcome.facts == []
        assert len(outcome.rejections) == 1
        assert outcome.rejections[0]["stage"] == "grounding"

    def test_malformed_fact_in_batch_does_not_block_the_good_one(self, conn):
        chunk = chunk_covering(conn, ANNUAL_REPORT, 24)
        response = json.dumps([
            {
                "entity": "Suvir Suren Sujan", "metric": "board membership status",
                "fact_type": "bogus_type",
                "quote": "resigned from the Board with effect from August 24, 2023",
            },
            {
                "entity": "Suvir Suren Sujan", "metric": "board membership status",
                "fact_type": "semantic", "value": "resigned",
                "quote": "resigned from the Board with effect from August 24, 2023",
                "confidence": 0.9,
            },
        ])
        outcome = extract_facts_for_chunk(conn, chunk, response)
        assert len(outcome.facts) == 1
        assert len(outcome.rejections) == 1
        assert outcome.rejections[0]["stage"] == "shape"

    def test_unparseable_response_is_recorded_as_a_parse_rejection(self, conn):
        chunk = chunk_covering(conn, ANNUAL_REPORT, 24)
        outcome = extract_facts_for_chunk(conn, chunk, "I couldn't find any facts here.")
        assert outcome.facts == []
        assert outcome.rejections[0]["stage"] == "parse"

    def test_two_facts_from_the_revenue_split_ground_independently(self, conn):
        """Standalone and consolidated revenue on the same page — the
        classic basis-mismatch case (Case 3's setup). Both quotes are real
        and distinct; both must ground to the same page."""
        chunk = chunk_covering(conn, ANNUAL_REPORT, 22)
        response = json.dumps([
            {
                "entity": "Delhivery Limited",
                "metric": "revenue from operations, standalone basis",
                "fact_type": "numeric", "value": "74540.82", "unit": "INR million",
                "period_label": "FY24", "basis": "standalone", "status": "actual",
                "quote": "the revenue from operations on standalone basis for FY24 stood at",
                "confidence": 0.9,
            },
            {
                "entity": "Delhivery Limited",
                "metric": "revenue from operations, consolidated basis",
                "fact_type": "numeric", "value": "81415.38", "unit": "INR million",
                "period_label": "FY24", "basis": "consolidated", "status": "actual",
                "quote": "the revenue from operations on consolidated basis for FY24 stood at",
                "confidence": 0.9,
            },
        ])
        outcome = extract_facts_for_chunk(conn, chunk, response)
        assert len(outcome.facts) == 2
        assert {f.basis for f in outcome.facts} == {"standalone", "consolidated"}
        assert all(f.page_number == 22 for f in outcome.facts)

    def test_extra_model_attached_fields_survive_in_dimensions(self, conn):
        """The schema-evolution path, exercised against a real chunk: a field
        we never named must still reach the database, not be dropped."""
        chunk = chunk_covering(conn, ANNUAL_REPORT, 24)
        response = json.dumps([{
            "entity": "Suvir Suren Sujan", "metric": "board membership status",
            "fact_type": "semantic", "value": "resigned",
            "quote": "resigned from the Board with effect from August 24, 2023",
            "confidence": 0.9,
            "reason_given": "pre-occupation and other commitments",
        }])
        outcome = extract_facts_for_chunk(conn, chunk, response)
        assert outcome.facts[0].dimensions == {
            "reason_given": "pre-occupation and other commitments"
        }


class TestPersistOutcome:
    def test_facts_and_rejections_are_both_written(self, conn):
        chunk = chunk_covering(conn, ANNUAL_REPORT, 24)
        response = json.dumps([
            {
                "entity": "Suvir Suren Sujan", "metric": "board membership status",
                "fact_type": "semantic", "value": "resigned",
                "quote": "resigned from the Board with effect from August 24, 2023",
                "confidence": 0.92,
            },
            {
                "entity": "Suvir Suren Sujan", "metric": "board membership status",
                "fact_type": "semantic", "value": "resigned",
                "quote": "resigned effective some other date entirely",
                "confidence": 0.5,
            },
        ])
        outcome = extract_facts_for_chunk(conn, chunk, response)
        persist_outcome(conn, chunk["doc_id"], chunk["id"], outcome)
        conn.commit()

        stored = conn.execute(
            "SELECT * FROM facts WHERE id = ?", (outcome.facts[0].id,)
        ).fetchone()
        assert stored is not None
        assert stored["entity"] == "Suvir Suren Sujan"
        assert stored["page_number"] == 24
        assert stored["printed_page"] == "46-47"

        rejected = conn.execute(
            "SELECT * FROM extraction_rejections WHERE chunk_id = ?", (chunk["id"],)
        ).fetchall()
        assert len(rejected) == 1
        assert rejected[0]["stage"] == "grounding"


class TestRunExtraction:
    def test_uses_an_injected_model_and_caches_by_chunk_hash(self, conn):
        """No network involved: call_model is a plain function under our
        control, so this proves the orchestration itself — caching,
        persistence, counting — is correct independent of what a real model
        returns."""
        chunk = chunk_covering(conn, ANNUAL_REPORT, 24)
        calls = []

        def fake_model(chunk_text: str) -> str:
            calls.append(chunk_text)
            return json.dumps([{
                "entity": "Suvir Suren Sujan", "metric": "board membership status",
                "fact_type": "semantic", "value": "resigned",
                "quote": "resigned from the Board with effect from August 24, 2023",
                "confidence": 0.9,
            }])

        result = run_extraction(conn, [chunk], fake_model)
        assert result == {"chunks": 1, "calls_made": 1, "facts": 1, "rejections": 0}
        assert len(calls) == 1

        # Second run over the same chunk must hit the cache, not the model.
        result2 = run_extraction(conn, [chunk], fake_model)
        assert result2["calls_made"] == 0
        assert result2["facts"] == 1
        assert len(calls) == 1  # the fake was not called again

    def test_progress_callback_fires_once_per_chunk(self, conn):
        chunk = chunk_covering(conn, ANNUAL_REPORT, 22)
        seen = []
        run_extraction(
            conn, [chunk], lambda _: "[]", on_progress=lambda i, n: seen.append((i, n))
        )
        assert seen == [(1, 1)]