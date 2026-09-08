"""The grounding gate.

Two failure directions matter and they are not symmetric. A false negative costs
us a fact. A false positive lets an invented figure into the store wearing a
citation, which is the failure the whole system is supposed to make impossible.
The tests below spend most of their attention on the second.
"""
from __future__ import annotations

from app.grounding import MIN_QUOTE_CHARS, quote_on_page, verify_quote
from app.textnorm import normalize, normalize_for_match


class TestNormalisation:
    def test_collapses_line_wrapping(self):
        assert normalize("revenue from\noperations   grew") == "revenue from operations grew"

    def test_folds_curly_quotes_and_dashes(self):
        assert normalize("“Delhivery” – FY24") == '"Delhivery" - FY24'

    def test_expands_ligatures(self):
        assert "financial" in normalize("ﬁnancial")

    def test_strips_soft_hyphens_and_zero_width(self):
        assert normalize("oper­ations​") == "operations"

    def test_preserves_digits_and_decimals(self):
        # The one thing normalisation must never touch.
        assert "81,415.38" in normalize("81,415.38")
        assert normalize_for_match("74,540.82") == "74,540.82"

    def test_does_not_merge_distinct_numbers(self):
        assert normalize_for_match("72,236") != normalize_for_match("72,253")


class TestQuoteMatching:
    PAGE = (
        "y The revenue from operations on standalone basis for FY24 stood at\n"
        "₹ 74,540.82 million as against ₹66,586.61 million for FY23"
    )

    def test_exact_quote_matches(self):
        assert quote_on_page("revenue from operations on standalone basis", self.PAGE)

    def test_quote_spanning_a_line_break_matches(self):
        assert quote_on_page("for FY24 stood at ₹ 74,540.82 million", self.PAGE)

    def test_case_difference_matches(self):
        assert quote_on_page("REVENUE FROM OPERATIONS ON STANDALONE", self.PAGE)

    def test_altered_figure_does_not_match(self):
        """The critical negative: a plausible but wrong number must fail."""
        assert not quote_on_page("stood at ₹ 74,540.83 million", self.PAGE)

    def test_rounded_figure_does_not_match(self):
        assert not quote_on_page("stood at ₹ 74,540 million", self.PAGE)

    def test_paraphrase_does_not_match(self):
        assert not quote_on_page("standalone revenue for FY24 was 74,540.82", self.PAGE)

    def test_empty_inputs_do_not_match(self):
        assert not quote_on_page("", self.PAGE)
        assert not quote_on_page("anything", "")


class TestVerifyQuote:
    PAGES = [
        (22, "42-43", "Revenue from Operations 74,540.82 66,586.61 81,415.38"),
        (23, "44-45", "Other Income 4,753.49 3,311.74"),
    ]

    def test_returns_the_page_it_was_found_on(self):
        result = verify_quote("Revenue from Operations 74,540.82", self.PAGES)
        assert result.ok
        assert result.page_number == 22
        assert result.printed_page == "42-43"

    def test_finds_quotes_on_later_pages_in_range(self):
        result = verify_quote("Other Income 4,753.49", self.PAGES)
        assert result.ok and result.page_number == 23

    def test_quote_absent_from_range_is_rejected(self):
        result = verify_quote("Revenue from sale of traded goods 16.54", self.PAGES)
        assert not result.ok
        assert "not found" in result.reason

    def test_short_quote_is_rejected_as_evidence(self):
        """'6.5' appears on dozens of pages; grounding it anywhere is worse than
        admitting we have no citation."""
        result = verify_quote("6.5", self.PAGES)
        assert not result.ok
        assert len("6.5") < MIN_QUOTE_CHARS
        assert "too short" in result.reason

    def test_empty_page_range_is_rejected(self):
        result = verify_quote("Revenue from Operations 74,540.82", [])
        assert not result.ok
        assert "no pages" in result.reason