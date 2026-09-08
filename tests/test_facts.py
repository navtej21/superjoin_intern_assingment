"""Fact schema validation — no LLM, no network, no API key.

This is pure shape-checking against JSON that looks like what the extraction
model will return. It uses real corpus content (the Sujan resignation, the
Delhivery revenue split) so the test data doubles as documentation of what a
real extracted fact looks like.
"""
from __future__ import annotations

import pytest

from app.facts import (
    Fact,
    FactValidationError,
    attach_grounding,
    parse_llm_fact,
    parse_llm_facts,
)
from app.grounding import GroundingResult

NUMERIC_FACT = {
    "entity": "Delhivery Limited",
    "metric": "revenue from operations, consolidated basis",
    "fact_type": "numeric",
    "value": "81415.38",
    "unit": "INR million",
    "period_label": "FY24",
    "basis": "consolidated",
    "status": "actual",
    "quote": "the revenue from operations on consolidated basis for FY24 stood at ₹ 81,415.38 million",
    "confidence": 0.95,
}

SEMANTIC_FACT = {
    "entity": "Suvir Suren Sujan",
    "metric": "board membership status",
    "fact_type": "semantic",
    "value": "resigned",
    "quote": "resigned from the Board with effect from August 24, 2023",
    "confidence": 0.9,
}


class TestParseValidFact:
    def test_numeric_fact_parses(self):
        fact = parse_llm_fact(NUMERIC_FACT, doc_id="doc1", chunk_id="doc1:4")
        assert fact.entity == "Delhivery Limited"
        assert fact.fact_type == "numeric"
        assert fact.value == "81415.38"
        assert fact.basis == "consolidated"
        assert fact.grounded is False
        assert fact.page_number is None

    def test_semantic_fact_parses(self):
        fact = parse_llm_fact(SEMANTIC_FACT, doc_id="doc2", chunk_id="doc2:9")
        assert fact.fact_type == "semantic"
        assert fact.value == "resigned"
        assert fact.unit is None  # not applicable to this fact, correctly absent

    def test_id_is_deterministic(self):
        a = parse_llm_fact(NUMERIC_FACT, doc_id="doc1", chunk_id="doc1:4")
        b = parse_llm_fact(NUMERIC_FACT, doc_id="doc1", chunk_id="doc1:4")
        assert a.id == b.id

    def test_id_differs_for_different_quotes(self):
        other = {**NUMERIC_FACT, "quote": NUMERIC_FACT["quote"] + " (restated)"}
        a = parse_llm_fact(NUMERIC_FACT, doc_id="doc1", chunk_id="doc1:4")
        b = parse_llm_fact(other, doc_id="doc1", chunk_id="doc1:4")
        assert a.id != b.id

    def test_unknown_extra_fields_are_preserved_not_dropped(self):
        """The schema-evolution mechanism: a model-attached qualifier we never
        named survives instead of being silently discarded."""
        raw = {**NUMERIC_FACT, "currency": "INR", "restated_in": "FY25 AR"}
        fact = parse_llm_fact(raw, doc_id="doc1", chunk_id="doc1:4")
        assert fact.dimensions == {"currency": "INR", "restated_in": "FY25 AR"}

    def test_confidence_missing_defaults_to_midpoint(self):
        raw = {k: v for k, v in NUMERIC_FACT.items() if k != "confidence"}
        fact = parse_llm_fact(raw, doc_id="doc1", chunk_id="doc1:4")
        assert fact.confidence == 0.5

    def test_confidence_is_clamped_to_unit_interval(self):
        raw = {**NUMERIC_FACT, "confidence": 1.7}
        assert parse_llm_fact(raw, doc_id="d", chunk_id="c").confidence == 1.0
        raw = {**NUMERIC_FACT, "confidence": -0.3}
        assert parse_llm_fact(raw, doc_id="d", chunk_id="c").confidence == 0.0

    def test_blank_optional_field_becomes_none_not_empty_string(self):
        raw = {**NUMERIC_FACT, "unit": "   "}
        assert parse_llm_fact(raw, doc_id="d", chunk_id="c").unit is None


class TestRejections:
    @pytest.mark.parametrize("missing", ["entity", "metric", "fact_type", "quote"])
    def test_missing_required_field_is_rejected(self, missing):
        raw = {k: v for k, v in NUMERIC_FACT.items() if k != missing}
        with pytest.raises(FactValidationError, match=missing):
            parse_llm_fact(raw, doc_id="d", chunk_id="c")

    def test_empty_string_required_field_is_rejected(self):
        raw = {**NUMERIC_FACT, "entity": "   "}
        with pytest.raises(FactValidationError):
            parse_llm_fact(raw, doc_id="d", chunk_id="c")

    def test_invalid_fact_type_is_rejected(self):
        raw = {**NUMERIC_FACT, "fact_type": "financial"}
        with pytest.raises(FactValidationError, match="fact_type"):
            parse_llm_fact(raw, doc_id="d", chunk_id="c")

    def test_short_quote_is_rejected(self):
        """'6.5' would ground against dozens of pages — see app.grounding. The
        same floor applies here so a Fact can never be built without real
        evidence, independent of whether grounding runs at all."""
        raw = {**NUMERIC_FACT, "quote": "6.5%"}
        with pytest.raises(FactValidationError, match="too short"):
            parse_llm_fact(raw, doc_id="d", chunk_id="c")

    def test_non_numeric_confidence_is_rejected(self):
        raw = {**NUMERIC_FACT, "confidence": "very sure"}
        with pytest.raises(FactValidationError, match="confidence"):
            parse_llm_fact(raw, doc_id="d", chunk_id="c")

    def test_non_object_fact_is_rejected(self):
        with pytest.raises(FactValidationError):
            parse_llm_fact("not a dict", doc_id="d", chunk_id="c")


class TestParseBatch:
    def test_batch_of_two_valid_facts(self):
        facts, rejections = parse_llm_facts(
            [NUMERIC_FACT, SEMANTIC_FACT], doc_id="d", chunk_id="c"
        )
        assert len(facts) == 2
        assert rejections == []

    def test_one_bad_fact_does_not_discard_the_rest(self):
        broken = {**NUMERIC_FACT, "fact_type": "nonsense"}
        facts, rejections = parse_llm_facts(
            [NUMERIC_FACT, broken, SEMANTIC_FACT], doc_id="d", chunk_id="c"
        )
        assert len(facts) == 2
        assert len(rejections) == 1
        assert rejections[0]["raw"] == broken
        assert "fact_type" in rejections[0]["reason"]

    def test_empty_batch(self):
        facts, rejections = parse_llm_facts([], doc_id="d", chunk_id="c")
        assert facts == [] and rejections == []


class TestAttachGrounding:
    def test_successful_grounding_fills_page_and_sets_flag(self):
        fact = parse_llm_fact(NUMERIC_FACT, doc_id="d", chunk_id="c")
        result = GroundingResult(grounded=True, page_number=22, printed_page="42-43", reason="")
        grounded = attach_grounding(fact, result)

        assert grounded.grounded is True
        assert grounded.page_number == 22
        assert grounded.printed_page == "42-43"
        # Original fact is untouched — attach_grounding returns a new object.
        assert fact.grounded is False

    def test_failed_grounding_result_cannot_be_attached(self):
        fact = parse_llm_fact(NUMERIC_FACT, doc_id="d", chunk_id="c")
        failed = GroundingResult(grounded=False, page_number=None, printed_page=None,
                                  reason="quote not found on any page in range")
        with pytest.raises(ValueError, match="record this as a rejection"):
            attach_grounding(fact, failed)