"""Text normalisation — the foundation grounding depends on.

Two directions matter and they are not symmetric. Normalising too little means
a real quote fails to match its source over a curly quote or a line wrap. But
normalising too much — touching digits, merging distinct numbers — would let a
grounding check pass on a figure the document never stated, which is the one
failure this entire project is built to prevent. Most of the tests below are
aimed at the second, more dangerous direction.
"""
from __future__ import annotations

from app.textnorm import estimate_tokens, normalize, normalize_for_match


class TestNormalize:
    def test_collapses_line_wrapping(self):
        assert normalize("revenue from\noperations   grew") == "revenue from operations grew"

    def test_folds_curly_quotes_and_dashes(self):
        assert normalize("“Delhivery” – FY24") == '"Delhivery" - FY24'

    def test_expands_ligatures(self):
        assert "financial" in normalize("ﬁnancial")

    def test_strips_soft_hyphens_and_zero_width_chars(self):
        assert normalize("oper­ations​") == "operations"

    def test_collapses_various_unicode_spaces(self):
        assert normalize("74,540.82\u00a0million") == "74,540.82 million"

    def test_preserves_digits_and_decimals(self):
        # The one thing normalisation must never touch.
        assert "81,415.38" in normalize("81,415.38")

    def test_does_not_merge_distinct_numbers(self):
        assert normalize("72,236") != normalize("72,253")

    def test_strips_leading_and_trailing_whitespace(self):
        assert normalize("   padded text   ") == "padded text"

    def test_empty_string_stays_empty(self):
        assert normalize("") == ""


class TestNormalizeForMatch:
    def test_is_case_insensitive(self):
        assert normalize_for_match("REVENUE") == normalize_for_match("revenue")

    def test_still_preserves_digits(self):
        assert normalize_for_match("74,540.82") == "74,540.82"

    def test_case_difference_does_not_hide_a_number_difference(self):
        assert normalize_for_match("Revenue 100") != normalize_for_match("revenue 200")


class TestEstimateTokens:
    def test_roughly_four_chars_per_token(self):
        assert estimate_tokens("a" * 400) == 100

    def test_never_returns_zero_for_nonempty_text(self):
        assert estimate_tokens("hi") >= 1