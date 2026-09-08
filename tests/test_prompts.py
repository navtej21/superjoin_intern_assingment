"""Sanity checks on the extraction prompt itself.

This isn't where the real coverage lives — that's tests/test_extract.py, which
exercises real model-shaped responses against real corpus chunks. What belongs
here is narrower: the prompt-building function must actually embed the chunk
text untouched, must not leak a real corpus entity into the few-shot example
(which would bias every extraction toward this corpus, defeating the point of
an open-vocabulary prompt), and PROMPT_VERSION must exist as a stable string,
since app.llm keys its cache on it.
"""
from __future__ import annotations

from app.prompts import PROMPT_VERSION, build_extraction_prompt


class TestBuildExtractionPrompt:
    def test_chunk_text_is_embedded_verbatim(self):
        chunk = "Revenue from services stood at ₹81,415.38 million in FY24."
        prompt = build_extraction_prompt(chunk)
        assert chunk in prompt

    def test_asks_for_a_json_array_only(self):
        prompt = build_extraction_prompt("some text")
        assert "JSON array" in prompt
        assert "no markdown code fence" in prompt

    def test_fact_type_is_the_only_closed_field(self):
        prompt = build_extraction_prompt("some text")
        assert '"numeric" or "semantic"' in prompt

    def test_quote_field_demands_verbatim_copying(self):
        prompt = build_extraction_prompt("some text")
        assert "VERBATIM" in prompt
        assert "discarded" in prompt

    def test_example_does_not_reuse_real_corpus_entities(self):
        """The few-shot example must stay fictional — Delhivery, Sujan, IGI
        Airport etc. must never appear here, or every extraction gets biased
        toward this corpus specifically."""
        prompt = build_extraction_prompt("some text")
        for real_entity in ("Delhivery", "Sujan", "Barasia", "IGI Airport"):
            assert real_entity not in prompt

    def test_prompt_version_is_a_stable_string(self):
        assert isinstance(PROMPT_VERSION, str)
        assert PROMPT_VERSION

    def test_different_chunks_produce_different_prompts(self):
        assert build_extraction_prompt("alpha") != build_extraction_prompt("beta")