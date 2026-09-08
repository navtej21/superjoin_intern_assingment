"""JSON extraction from a model's raw text response — no network, no key.

Covers the ways a real model response reliably differs from clean JSON even
when explicitly told not to: wrapped in a code fence, prefixed with a
sentence, followed by one. What this must never do is guess past genuinely
broken output — a response that can't be parsed is a real extraction
failure and should surface as one, not be silently patched into something
that looks fine but might not be what the model meant.
"""
from __future__ import annotations

import pytest

from app.llm import extract_json_array

CLEAN = (
    '[{"entity": "Acme", "metric": "revenue", "fact_type": "numeric", '
    '"quote": "revenue was 100 million", "confidence": 0.9}]'
)


class TestExtractJsonArray:
    def test_clean_json_parses(self):
        result = extract_json_array(CLEAN)
        assert len(result) == 1
        assert result[0]["entity"] == "Acme"

    def test_json_code_fence_with_language_tag_is_stripped(self):
        wrapped = f"```json\n{CLEAN}\n```"
        assert extract_json_array(wrapped) == extract_json_array(CLEAN)

    def test_bare_code_fence_without_language_tag(self):
        wrapped = f"```\n{CLEAN}\n```"
        assert extract_json_array(wrapped) == extract_json_array(CLEAN)

    def test_leading_prose_is_ignored(self):
        wrapped = f"Here are the facts I found in this document:\n\n{CLEAN}"
        assert extract_json_array(wrapped) == extract_json_array(CLEAN)

    def test_trailing_prose_is_ignored(self):
        wrapped = f"{CLEAN}\n\nLet me know if you'd like more detail."
        assert extract_json_array(wrapped) == extract_json_array(CLEAN)

    def test_leading_and_trailing_prose_together(self):
        wrapped = f"Sure, here you go:\n\n{CLEAN}\n\nHope that helps!"
        assert extract_json_array(wrapped) == extract_json_array(CLEAN)

    def test_empty_array_is_valid(self):
        """A page with nothing extractable — a cover page, a table of
        contents — must return cleanly, not be treated as an error."""
        assert extract_json_array("[]") == []

    def test_multiple_facts_parse_in_order(self):
        two = (
            '[{"entity": "A", "metric": "m1", "fact_type": "numeric", '
            '"quote": "first quote here is long enough", "confidence": 0.8},'
            '{"entity": "B", "metric": "m2", "fact_type": "semantic", '
            '"quote": "second quote here is long enough", "confidence": 0.7}]'
        )
        result = extract_json_array(two)
        assert [r["entity"] for r in result] == ["A", "B"]

    def test_no_array_at_all_raises(self):
        with pytest.raises(ValueError, match="no JSON array"):
            extract_json_array("I could not find any facts in this text.")

    def test_malformed_json_raises_rather_than_guessing(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            extract_json_array('[{"entity": "Acme", "metric": }]')

    def test_object_instead_of_array_raises(self):
        with pytest.raises(ValueError, match="expected a JSON array"):
            extract_json_array('{"entity": "Acme"}')

    def test_truncated_response_raises(self):
        """A response cut off mid-generation (hit max_tokens) must fail
        loudly, not silently parse whatever prefix happens to be valid."""
        with pytest.raises(ValueError):
            extract_json_array('[{"entity": "Acme", "metric": "revenue and')