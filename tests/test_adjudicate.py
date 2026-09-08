"""Phase C: relationship detection between facts.

Fixtures are built from the same real cases the rest of this project is
anchored on (see tests/conftest.py's EVIDENCE table and the module docstring
in app.adjudicate) so a passing test here means a passing test against the
real corpus's numbers, not just synthetic ones.
"""
from __future__ import annotations

from app.adjudicate import (
    extract_year,
    find_additive_reconciliations,
    find_pairwise_relationships,
    find_relationships,
    metric_overlap,
    normalize_entity,
    parse_numeric_value,
    values_match,
)
from app.adjudicate import _shares_significant_word


def fact(**kwargs) -> dict:
    """A minimal Fact-shaped dict — every field find_relationships reads,
    defaulted, so a test only needs to name the fields it cares about."""
    base = {
        "id": kwargs.get("id", "f0"),
        "doc_id": "doc1",
        "chunk_id": "doc1:0",
        "entity": "Delhivery Limited",
        "metric": "revenue",
        "fact_type": "numeric",
        "value": "100",
        "unit": None,
        "period_label": "FY24",
        "basis": None,
        "status": None,
        "page_number": 1,
        "printed_page": "1",
        "quote": "revenue was 100",
    }
    base.update(kwargs)
    return base


class TestNormalizeEntity:
    def test_strips_titles_and_corporate_suffixes(self):
        assert normalize_entity("Mr. Suvir Suren Sujan") == "suvir suren sujan"
        assert normalize_entity("Suvir Suren Sujan") == "suvir suren sujan"
        assert normalize_entity("Delhivery Limited") == "delhivery"
        assert normalize_entity("Delhivery") == "delhivery"

    def test_different_entities_stay_different(self):
        assert normalize_entity("Sandeep Kumar Barasia") != normalize_entity("Suvir Suren Sujan")


class TestMetricOverlap:
    def test_identical_metrics_overlap_fully(self):
        assert metric_overlap("revenue from operations", "revenue from operations") == 1.0

    def test_differently_worded_but_related_metrics_overlap_partially(self):
        overlap = metric_overlap(
            "revenue from operations, standalone basis", "revenue from services"
        )
        assert overlap > 0

    def test_unrelated_metrics_do_not_overlap(self):
        assert metric_overlap("revenue from operations", "board membership status") == 0.0

    def test_empty_metric_never_overlaps(self):
        assert metric_overlap("", "revenue") == 0.0


class TestParseNumericValue:
    def test_plain_number(self):
        assert parse_numeric_value("81415.38") == 81415.38

    def test_comma_thousands_separator(self):
        assert parse_numeric_value("81,415.38") == 81415.38

    def test_parenthesised_value_is_negative(self):
        assert parse_numeric_value("(1,679.68)") == -1679.68

    def test_currency_symbol_and_percent_are_stripped(self):
        assert parse_numeric_value("₹74,540.82") == 74540.82
        assert parse_numeric_value("75%") == 75.0

    def test_non_numeric_returns_none(self):
        assert parse_numeric_value("resigned") is None
        assert parse_numeric_value(None) is None
        assert parse_numeric_value("") is None


class TestValuesMatch:
    def test_exact_match(self):
        assert values_match(81415.38, 81415.38)

    def test_within_tolerance(self):
        assert values_match(81415.38, 81415.39)

    def test_outside_tolerance(self):
        assert not values_match(74540.82, 81415.38)


class TestExtractYear:
    def test_four_digit_year_in_prose(self):
        assert extract_year("FY ended March 31, 2024") == 2024

    def test_fy_shorthand(self):
        assert extract_year("FY24") == 2024
        assert extract_year("FY23") == 2023

    def test_no_year_present(self):
        assert extract_year("as stated") is None
        assert extract_year(None) is None


class TestCorroboration:
    def test_same_entity_metric_period_basis_value_from_different_chunks(self):
        """Case 1: 'Revenue from services' ₹81,415.38 stated identically in
        three places in the real corpus."""
        a = fact(id="a", chunk_id="doc1:4", page_number=4, metric="revenue from services",
                  value="81,415.38", period_label="FY24", basis="consolidated")
        b = fact(id="b", chunk_id="doc1:22", page_number=22, metric="revenue from services",
                  value="81,415.38", period_label="FY24", basis="consolidated")
        rels = find_pairwise_relationships([a, b])
        assert len(rels) == 1
        assert rels[0].relation_type == "corroboration"
        assert set(rels[0].fact_ids) == {"a", "b"}

    def test_semantic_facts_corroborate_on_matching_status(self):
        a = fact(id="a", fact_type="semantic", entity="Suvir Suren Sujan",
                  metric="board membership status", value="resigned",
                  quote="resigned from the Board with effect from August 24, 2023")
        b = fact(id="b", fact_type="semantic", entity="Mr. Suvir Suren Sujan",
                  metric="board membership status",
                  value="resigned from the Board with effect from August 24, 2023",
                  quote="resigned from the Board with effect from August 24, 2023")
        rels = find_pairwise_relationships([a, b])
        assert len(rels) == 1
        assert rels[0].relation_type == "corroboration"


class TestContradiction:
    def test_same_period_and_basis_but_different_numeric_values(self):
        a = fact(id="a", metric="revenue from operations", value="81415.38",
                  period_label="FY24", basis="consolidated")
        b = fact(id="b", metric="revenue from operations", value="79000.00",
                  period_label="FY24", basis="consolidated")
        rels = find_pairwise_relationships([a, b])
        assert len(rels) == 1
        assert rels[0].relation_type == "contradiction"

    def test_semantic_status_flip_is_flagged(self):
        """Case 2: active director in one filing, resigned in another."""
        a = fact(id="a", fact_type="semantic", entity="Suvir Suren Sujan",
                  metric="board membership status", value="active",
                  period_label="2022")
        b = fact(id="b", fact_type="semantic", entity="Suvir Suren Sujan",
                  metric="board membership status", value="resigned",
                  period_label="FY24")
        rels = find_pairwise_relationships([a, b])
        assert len(rels) == 1
        assert rels[0].relation_type == "contradiction"
        # Both periods must be visible in the evidence — this is what lets a
        # human tell "a status that changed over time" from a real conflict.
        periods = {e["period_label"] for e in rels[0].evidence}
        assert periods == {"2022", "FY24"}


class TestContextReconcilable:
    def test_basis_difference_explains_value_difference(self):
        """FY24 standalone vs consolidated revenue on the same page."""
        a = fact(id="a", metric="revenue from operations, standalone basis",
                  value="74540.82", period_label="FY24", basis="standalone")
        b = fact(id="b", metric="revenue from operations, consolidated basis",
                  value="81415.38", period_label="FY24", basis="consolidated")
        rels = find_pairwise_relationships([a, b])
        assert len(rels) == 1
        assert rels[0].relation_type == "context_reconcilable"

    def test_additive_reconciliation_across_three_facts(self):
        """Case 3, generalised: Note 21 explains a 16.54 gap by addition,
        found here by arithmetic rather than by naming these three metrics."""
        services = fact(id="services", metric="revenue from services",
                          value="72236.47", period_label="FY23", basis="consolidated")
        traded_goods = fact(id="traded_goods", metric="revenue from sale of traded goods",
                              value="16.54", period_label="FY23", basis="consolidated")
        operations = fact(id="operations", metric="revenue from operations",
                            value="72253.01", period_label="FY23", basis="consolidated")
        rels = find_additive_reconciliations([services, traded_goods, operations])
        assert len(rels) == 1
        assert rels[0].relation_type == "context_reconcilable"
        assert set(rels[0].fact_ids) == {"services", "traded_goods", "operations"}

    def test_additive_check_ignores_unrelated_entities(self):
        a = fact(id="a", entity="Delhivery Limited", value="10", period_label="FY24", basis="standalone")
        b = fact(id="b", entity="Delhivery Limited", value="20", period_label="FY24", basis="standalone")
        c = fact(id="c", entity="Some Other Company", value="30", period_label="FY24", basis="standalone")
        assert find_additive_reconciliations([a, b, c]) == []


class TestNoRelationship:
    def test_different_entities_are_never_compared(self):
        a = fact(id="a", entity="Delhivery Limited")
        b = fact(id="b", entity="Kotak Mahindra Capital Company Limited")
        assert find_pairwise_relationships([a, b]) == []

    def test_unrelated_metrics_are_never_compared(self):
        a = fact(id="a", metric="revenue from operations")
        b = fact(id="b", metric="board membership status", fact_type="semantic", value="resigned")
        assert find_pairwise_relationships([a, b]) == []

    def test_different_periods_are_not_flagged(self):
        """A metric naturally has a different value in a different year —
        that's not a conflict, it's just two different facts."""
        a = fact(id="a", value="100", period_label="FY23")
        b = fact(id="b", value="150", period_label="FY24")
        assert find_pairwise_relationships([a, b]) == []

    def test_numeric_and_semantic_facts_are_never_compared(self):
        a = fact(id="a", fact_type="numeric", metric="revenue", value="100")
        a2 = fact(id="a2", fact_type="semantic", metric="revenue commentary", value="grew")
        assert find_pairwise_relationships([a, a2]) == []

    def test_different_basis_with_matching_values_is_not_flagged(self):
        """Coincidentally-equal figures under different bases aren't a
        finding worth reporting."""
        a = fact(id="a", value="100", basis="standalone", period_label="FY24")
        b = fact(id="b", value="100", basis="consolidated", period_label="FY24")
        assert find_pairwise_relationships([a, b]) == []


class TestFindRelationships:
    def test_combines_pairwise_and_additive(self):
        a = fact(id="a", metric="revenue from operations, standalone basis",
                  value="74540.82", period_label="FY24", basis="standalone")
        b = fact(id="b", metric="revenue from operations, consolidated basis",
                  value="81415.38", period_label="FY24", basis="consolidated")
        services = fact(id="services", metric="revenue from services",
                          value="72236.47", period_label="FY23", basis="consolidated")
        traded_goods = fact(id="traded_goods", metric="revenue from sale of traded goods",
                              value="16.54", period_label="FY23", basis="consolidated")
        operations = fact(id="operations", metric="revenue from operations",
                            value="72253.01", period_label="FY23", basis="consolidated")
        rels = find_relationships([a, b, services, traded_goods, operations])
        types = {r.relation_type for r in rels}
        assert "context_reconcilable" in types
        assert len(rels) >= 2


class TestRealDataRegressions:
    """Pinned directly to the false positives the first version produced when
    run against the user's real 242-fact extraction — see the module
    docstring in app.adjudicate. Each test here failed under the pre-fix
    version and must pass now."""

    def test_large_share_count_does_not_loosely_match_a_per_share_price(self):
        """The exact real false positive: 'average cost of acquisition per
        equity share' (196.19) + 'number of equity shares held' (141,593,300)
        was accepted as reconciling to 'number of equity shares as at the
        date of this Prospectus' (141,593,300) — not because 196.19 is
        negligible next to 141 million, but because the old relative
        tolerance (0.005% of the target) turned a ~141-million-scale number
        into something a stray 196.19 could nudge across the equality check.
        A flat absolute tolerance must reject this outright."""
        price = fact(id="price", metric="average cost of acquisition per equity share",
                      value="196.19", period_label="FY24", basis="standalone")
        held = fact(id="held", metric="number of equity shares held",
                     value="141593300", period_label="FY24", basis="standalone")
        asof = fact(id="asof", metric="number of equity shares as at the date of this prospectus",
                     value="141593300", period_label="FY24", basis="standalone")
        assert find_additive_reconciliations([price, held, asof]) == []

    def test_generic_connector_word_alone_does_not_link_unrelated_metrics(self):
        """Real false positive: 'Total borrowings' vs 'Total non-current
        assets' shared only the word 'total' — now stopworded — and nothing
        else, so they must not overlap at all."""
        assert metric_overlap("Total borrowings", "Total non-current assets") == 0.0

    def test_shared_expense_word_does_not_link_unrelated_line_items(self):
        """Real false positive: 'Employee benefit expense' flagged as
        contradicting 'Depreciation and amortisation expense' purely via the
        shared generic noun 'expense'."""
        assert metric_overlap("Employee benefit expense", "Depreciation and amortisation expense") == 0.0

    def test_generic_word_overlap_does_not_produce_a_pairwise_finding(self):
        a = fact(id="a", metric="Total borrowings", value="500", period_label="FY24", basis="consolidated")
        b = fact(id="b", metric="Total non-current assets", value="900", period_label="FY24", basis="consolidated")
        assert find_pairwise_relationships([a, b]) == []

    def test_additive_check_skips_percentage_and_rate_metrics(self):
        """Real false positive: 'merchandise exports to the U.S. as share of
        GDP' (2.2) + 'central government deficit' (4.9) coincidentally summed
        to 'investment growth' (7.1) — three unrelated single-decimal
        percentages/rates for the same country and year. All three must be
        excluded from the additive bucket by their metric wording."""
        gdp_share = fact(id="gdp_share", entity="India",
                          metric="merchandise exports to the U.S. as share of GDP",
                          value="2.2", unit="percent", period_label="2025", basis=None)
        deficit = fact(id="deficit", entity="India",
                        metric="central government deficit", value="4.9",
                        unit="percent of GDP", period_label="2025", basis=None)
        growth = fact(id="growth", entity="India", metric="investment growth",
                       value="7.1", unit="percent", period_label="2025", basis=None)
        assert find_additive_reconciliations([gdp_share, deficit, growth]) == []

    def test_additive_check_rejects_grouping_where_an_addend_already_equals_the_target(self):
        """Degenerate case: one 'addend' is really just a duplicate of the
        target fact (same value restated), not a genuine part-of-a-whole
        split. Adding a near-zero third figure must not manufacture a
        reconciliation out of what are really two copies of the same fact."""
        duplicate = fact(id="duplicate", metric="revenue from operations",
                          value="72253.00", period_label="FY23", basis="consolidated")
        near_zero = fact(id="near_zero", metric="rounding adjustment",
                          value="0.01", period_label="FY23", basis="consolidated")
        target = fact(id="target", metric="total revenue",
                       value="72253.01", period_label="FY23", basis="consolidated")
        assert find_additive_reconciliations([duplicate, near_zero, target]) == []


class TestRealDataRegressionsRoundTwo:
    """A second round of false positives, found by running the round-one fix
    (TestRealDataRegressions above) against the user's real 242-fact output.
    Fixing the generic-connector-word problem exposed a sharper version of
    the same lesson: two genuinely different balance-sheet line items can
    still share most of their significant words. See the module docstring's
    second correction in app.adjudicate for the full write-up."""

    def test_balance_sheet_sibling_totals_do_not_contradict_each_other(self):
        """'Total non-current assets' and 'Total current assets' are two
        different parts of a balance sheet, not two statements of the same
        figure — they used to share 2 of 3 significant words."""
        a = fact(id="a", metric="Total non-current assets", value="40934.01",
                  period_label="As at December 31, 2021")
        b = fact(id="b", metric="Total current assets", value="43360.82",
                  period_label="As at December 31, 2021")
        assert find_pairwise_relationships([a, b]) == []

    def test_hyphenated_modifier_does_not_share_a_token_with_its_opposite(self):
        """'non-current' must not tokenize down to a bare 'current' that
        collides with a plain 'current' label."""
        assert metric_overlap("Investments (current)", "Borrowings (non-current)") == 0.0

    def test_absolute_figure_never_matches_a_percentage_metric(self):
        """'EBITDA' (an absolute Rs figure) and 'EBITDA margin' (a
        percentage) share their only significant word but are never the
        same claim — one measures money, the other a ratio."""
        a = fact(id="a", metric="EBITDA", value="1266", period_label="FY24")
        b = fact(id="b", metric="EBITDA margin", value="1.6", unit="%", period_label="FY24")
        assert find_pairwise_relationships([a, b]) == []

    def test_two_percentage_metrics_still_need_high_overlap(self):
        """Both sides being percentages doesn't exempt them from the overlap
        bar: 'tonnage growth' and 'revenue growth' are different metrics
        that happen to both be percentages for the same product line."""
        a = fact(id="a", metric="Part truckload tonnage growth", value="30",
                  unit="%", period_label="FY24")
        b = fact(id="b", metric="Part truckload revenue growth", value="31",
                  unit="%", period_label="FY24")
        assert find_pairwise_relationships([a, b]) == []

    def test_revenue_total_does_not_contradict_its_own_segment_breakdown(self):
        """An aggregate figure and one of the line items that sums into it
        share enough vocabulary ('revenue', 'services') to look related, but
        are not the same claim — this is the additive check's job, not the
        pairwise contradiction check's."""
        a = fact(id="a", metric="Revenue from services", value="81415", period_label="FY24")
        b = fact(id="b", metric="revenue from Express Parcel services", value="50765.87",
                  period_label="FY24")
        assert find_pairwise_relationships([a, b]) == []

    def test_registered_and_corporate_office_address_are_different_fields(self):
        """A company's registered office and corporate office are two
        different addresses, not two mentions of one — sharing
        'office'/'address' should not link them."""
        a = fact(id="a", fact_type="semantic", metric="registered office address",
                  value="N24-N34, Air Cargo Logistics Centre-II", period_label="p")
        b = fact(id="b", fact_type="semantic", metric="corporate office address",
                  value="Plot 5, Sector 44, Gurugram", period_label="p")
        assert find_pairwise_relationships([a, b]) == []

    def test_same_metric_on_different_specific_dates_within_one_year_is_not_a_conflict(self):
        """A cap table records four separate ESOP-purchase allotments across
        2023 under the identical metric name — same year, different exact
        dates, four different events, not a three-way contradiction."""
        a = fact(id="a", metric="equity shares allotted through Employee Stock Options Purchases",
                  value="158855", period_label="April 06, 2023")
        b = fact(id="b", metric="equity shares allotted through Employee Stock Options Purchases",
                  value="385739", period_label="May 06, 2023")
        assert find_pairwise_relationships([a, b]) == []

    def test_genuine_cases_survive_the_second_round_of_fixes(self):
        """The four required-case patterns must still fire after all of the
        above — this round of fixes must only remove noise, not signal."""
        # Case 1: corroboration
        a = fact(id="a", chunk_id="d:4", page_number=4, metric="revenue from services",
                  value="81,415.38", period_label="FY24", basis="consolidated")
        b = fact(id="b", chunk_id="d:22", page_number=22, metric="revenue from services",
                  value="81,415.38", period_label="FY24", basis="consolidated")
        rels = find_pairwise_relationships([a, b])
        assert len(rels) == 1 and rels[0].relation_type == "corroboration"

        # Case 2: semantic status flip
        c = fact(id="c", fact_type="semantic", entity="Suvir Suren Sujan",
                  metric="board membership status", value="active", period_label="2022")
        d = fact(id="d", fact_type="semantic", entity="Suvir Suren Sujan",
                  metric="board membership status", value="resigned", period_label="FY24")
        rels = find_pairwise_relationships([c, d])
        assert len(rels) == 1 and rels[0].relation_type == "contradiction"

        # Case 3: additive reconciliation (Note 21 pattern)
        services = fact(id="services", metric="revenue from services", value="72236.47",
                         period_label="FY23", basis="consolidated")
        traded_goods = fact(id="traded_goods", metric="revenue from sale of traded goods",
                             value="16.54", period_label="FY23", basis="consolidated")
        operations = fact(id="operations", metric="revenue from operations", value="72253.01",
                           period_label="FY23", basis="consolidated")
        rels = find_additive_reconciliations([services, traded_goods, operations])
        assert len(rels) == 1 and rels[0].relation_type == "context_reconcilable"

        # basis-difference reconciliation (standalone vs consolidated revenue)
        e = fact(id="e", metric="revenue from operations, standalone basis", value="74540.82",
                  period_label="FY24", basis="standalone")
        f = fact(id="f", metric="revenue from operations, consolidated basis", value="81415.38",
                  period_label="FY24", basis="consolidated")
        rels = find_pairwise_relationships([e, f])
        assert len(rels) == 1 and rels[0].relation_type == "context_reconcilable"


class TestRealDataRegressionsRoundThree:
    """A third round of false positives, found by running the round-two fix
    against a larger, 436-fact real extraction. Two of the round-two fixes
    (stopwording 'number'/'office'/'address') shrank some metric names down
    far enough that a single shared word looked like a strong match; a third
    is a plain value-normalization gap. See the module docstring's third
    correction in app.adjudicate."""

    def test_short_labels_do_not_match_on_a_single_shared_word(self):
        """'Corporate Identity Number' and 'corporate office address' reduce,
        after stopwording their generic nouns, to {corporate, identity} and
        {corporate} — one shared word out of a two-word combined vocabulary
        is not real evidence these are the same field."""
        assert metric_overlap("Corporate Identity Number", "corporate office address") == 0.0

    def test_identical_significant_words_always_match_regardless_of_length(self):
        """The short-vocabulary guard must not swallow genuine short-label
        matches — an identical token set is real evidence at any length."""
        assert metric_overlap("EBITDA", "EBITDA") == 1.0

    def test_cin_does_not_contradict_a_company_office_address(self):
        a = fact(id="a", fact_type="semantic", metric="Corporate Identity Number",
                  value="U63090DL2011PLC221234", period_label="p")
        b = fact(id="b", fact_type="semantic", metric="corporate office address",
                  value="Plot 5, Sector 44, Gurugram", period_label="p")
        assert find_pairwise_relationships([a, b]) == []

    def test_same_address_with_different_comma_placement_corroborates(self):
        """Real false positive: a registered office address quoted with
        commas in one filing and without in another was flagged as
        contradicting itself. Punctuation-insensitive comparison must treat
        it as the same address."""
        a = fact(id="a", fact_type="semantic", metric="registered office address",
                  value="N24-N34, S24-S34, Air Cargo Logistics Centre-II, Opposite Gate 6, "
                        "New Delhi 110037 Delhi, India", period_label="p")
        b = fact(id="b", fact_type="semantic", metric="registered office address",
                  value="N24-N34, S24-S34 Air Cargo Logistics Centre-II Opposite Gate 6 "
                        "New Delhi 110037 Delhi, India", period_label="p")
        rels = find_pairwise_relationships([a, b])
        assert len(rels) == 1
        assert rels[0].relation_type == "corroboration"

    def test_coincidentally_equal_percentages_still_need_high_overlap(self):
        """Real false positive: 'headline inflation' and 'core inflation'
        both landed on 4.6 in the same report — genuinely different
        measures, not the same claim, however strong 'equal values' is as a
        signal for large absolute figures."""
        a = fact(id="a", entity="India", metric="headline inflation", value="4.6",
                  unit="percent", period_label="2024-25")
        b = fact(id="b", entity="India", metric="core inflation", value="4.6",
                  unit="percent", period_label="2024-25")
        assert find_pairwise_relationships([a, b]) == []

    def test_genuine_cases_survive_the_third_round_of_fixes(self):
        """The four required-case patterns must still fire after all of the
        above — this round of fixes must only remove noise, not signal."""
        a = fact(id="a", chunk_id="d:4", page_number=4, metric="revenue from services",
                  value="81,415.38", period_label="FY24", basis="consolidated")
        b = fact(id="b", chunk_id="d:22", page_number=22, metric="revenue from services",
                  value="81,415.38", period_label="FY24", basis="consolidated")
        rels = find_pairwise_relationships([a, b])
        assert len(rels) == 1 and rels[0].relation_type == "corroboration"

        c = fact(id="c", fact_type="semantic", entity="Suvir Suren Sujan",
                  metric="board membership status", value="active", period_label="2022")
        d = fact(id="d", fact_type="semantic", entity="Suvir Suren Sujan",
                  metric="board membership status", value="resigned", period_label="FY24")
        rels = find_pairwise_relationships([c, d])
        assert len(rels) == 1 and rels[0].relation_type == "contradiction"

        services = fact(id="services", metric="revenue from services", value="72236.47",
                         period_label="FY23", basis="consolidated")
        traded_goods = fact(id="traded_goods", metric="revenue from sale of traded goods",
                             value="16.54", period_label="FY23", basis="consolidated")
        operations = fact(id="operations", metric="revenue from operations", value="72253.01",
                           period_label="FY23", basis="consolidated")
        rels = find_additive_reconciliations([services, traded_goods, operations])
        assert len(rels) == 1 and rels[0].relation_type == "context_reconcilable"

        e = fact(id="e", metric="revenue from operations, standalone basis", value="74540.82",
                  period_label="FY24", basis="standalone")
        f = fact(id="f", metric="revenue from operations, consolidated basis", value="81415.38",
                  period_label="FY24", basis="consolidated")
        rels = find_pairwise_relationships([e, f])
        assert len(rels) == 1 and rels[0].relation_type == "context_reconcilable"


class TestRealDataRegressionsRoundFour:
    """A fourth round, run against a cleaner 436-fact extraction with the
    first three rounds' fixes in place, found a matching gap and a
    data-quality gap the earlier rounds hadn't surfaced yet. See the module
    docstring's fourth correction in app.adjudicate."""

    def test_duplicate_extraction_with_byte_identical_quote_is_not_double_reported(self):
        """Real false positive: 'Corporate Identity Number' and
        'Registration Number' were each reported twice on the same page,
        with byte-identical quotes — one real sentence handed to the
        extraction LLM twice, not two independent mentions."""
        a = fact(id="a", doc_id="d1", page_number=30, fact_type="semantic",
                  metric="Corporate Identity Number", value="U63090DL2011PLC221234",
                  quote="Corporate Identity Number: U63090DL2011PLC221234", period_label="p")
        b = fact(id="b", doc_id="d1", page_number=30, fact_type="semantic",
                  metric="Corporate Identity Number", value="U63090DL2011PLC221234",
                  quote="Corporate Identity Number: U63090DL2011PLC221234", period_label="p")
        assert find_pairwise_relationships([a, b]) == []

    def test_duplicate_extraction_with_one_quote_a_substring_of_the_other_is_not_double_reported(self):
        """Real false positive: 'real GDP growth' = 7.8 was reported twice
        from the same page — one quote a short excerpt, the other the full
        sentence it sits inside. Still one location, one mention."""
        a = fact(id="a", doc_id="d2", page_number=10, entity="India", metric="real GDP growth",
                  value="7.8", unit="percent", period_label="2025Q2",
                  quote="real GDP growth of 7.8 percent in 2025Q2")
        b = fact(id="b", doc_id="d2", page_number=10, entity="India", metric="real GDP growth",
                  value="7.8", unit="percent", period_label="2025Q2",
                  quote="Strong private consumption has continued this fiscal year, "
                        "underpinning real GDP growth of 7.8 percent in 2025Q2.")
        assert find_pairwise_relationships([a, b]) == []

    def test_same_fact_on_a_genuinely_different_page_still_corroborates(self):
        """The dedupe guard must not swallow a real second mention — the
        same CIN on the cover page and again on the signature page is
        exactly the corroboration this module exists to find."""
        cover = fact(id="cover", doc_id="d1", page_number=1, fact_type="semantic",
                      metric="Corporate Identity Number", value="U63090DL2011PLC221234",
                      quote="CORPORATE IDENTITY NUMBER: U63090DL2011PLC221234", period_label="p")
        signature = fact(id="signature", doc_id="d1", page_number=30, fact_type="semantic",
                          metric="Corporate Identity Number", value="U63090DL2011PLC221234",
                          quote="Corporate Identity Number: U63090DL2011PLC221234", period_label="p")
        rels = find_pairwise_relationships([cover, signature])
        assert len(rels) == 1
        assert rels[0].relation_type == "corroboration"

    def test_address_with_fields_in_a_different_order_still_corroborates(self):
        """Real false positive: a corporate office address restated in two
        filings with its city/pin/state fields reordered ('Gurugram 122002
        Haryana, India' vs 'Gurugram, Haryana 122002') and light wording
        differences ('Plot' vs 'Plot No.', 'Sector 44' vs 'Sector-44') was
        flagged as a contradiction. Same address, just reordered."""
        a = fact(id="a", doc_id="d1", page_number=1, fact_type="semantic",
                  metric="corporate office address",
                  value="Plot 5, Sector 44, Gurugram 122002 Haryana, India",
                  quote="Plot 5, Sector 44, Gurugram 122002 Haryana, India", period_label="p")
        b = fact(id="b", doc_id="d1", page_number=31, fact_type="semantic",
                  metric="corporate office address",
                  value="Plot No. 5, Sector-44, Gurugram, Haryana 122002",
                  quote="Corporate Office: Plot No. 5, Sector-44, Gurugram, Haryana 122002",
                  period_label="p")
        rels = find_pairwise_relationships([a, b])
        assert len(rels) == 1
        assert rels[0].relation_type == "corroboration"

    def test_designations_differing_by_one_real_word_still_contradict(self):
        """Guard against the reordering fix above being too generous:
        'Executive Director' vs 'Whole Time Director' (Jaccard 0.625) and
        'Chairman' vs 'Chairperson' (Jaccard 0.714) are real, reportable
        differences, not the same value reordered, and must stay below the
        threshold that let the address case through (0.778)."""
        a = fact(id="a", doc_id="d1", page_number=1, fact_type="semantic",
                  entity="Kapil Bharati", metric="board designation",
                  value="Executive Director and Chief Technology Officer",
                  quote="Kapil Bharati Executive Director and Chief Technology Officer",
                  period_label="p")
        b = fact(id="b", doc_id="d1", page_number=2, fact_type="semantic",
                  entity="Kapil Bharati", metric="board designation",
                  value="Whole Time Director and Chief Technology Officer",
                  quote="Mr. Kapil Bharati Whole Time Director and Chief Technology Officer",
                  period_label="p")
        rels = find_pairwise_relationships([a, b])
        assert len(rels) == 1 and rels[0].relation_type == "contradiction"

        c = fact(id="c", doc_id="d1", page_number=1, fact_type="semantic",
                  entity="Deepak Kapoor", metric="board designation",
                  value="Chairman and Non-Executive Independent Director",
                  quote="Deepak Kapoor Chairman and Non-Executive Independent Director",
                  period_label="p")
        d = fact(id="d", doc_id="d1", page_number=2, fact_type="semantic",
                  entity="Deepak Kapoor", metric="board designation",
                  value="Chairperson and Non-Executive Independent Director",
                  quote="Mr. Deepak Kapoor Chairperson and Non-Executive Independent Director",
                  period_label="p")
        rels = find_pairwise_relationships([c, d])
        assert len(rels) == 1 and rels[0].relation_type == "contradiction"

    def test_dedupe_does_not_merge_coincidentally_equal_values_from_different_sentences(self):
        """Two different facts that happen to share a doc, page, entity and
        value but come from unrelated sentences must not be collapsed —
        only overlapping quotes signal a real duplicate extraction."""
        a = fact(id="a", doc_id="d1", page_number=5, entity="Delhivery Limited",
                  metric="opening balance", value="100", period_label="FY24",
                  basis="standalone", quote="opening balance carried forward was 100")
        b = fact(id="b", doc_id="d1", page_number=5, entity="Delhivery Limited",
                  metric="rounding adjustment", value="100", period_label="FY24",
                  basis="standalone", quote="a one-time rounding adjustment of 100 was booked")
        # different metrics entirely, so this is never compared as a pair in
        # the first place — the point here is that _dedupe_facts must leave
        # both facts in the list rather than silently dropping one.
        from app.adjudicate import _dedupe_facts
        assert len(_dedupe_facts([a, b])) == 2

    def test_genuine_cases_survive_the_fourth_round_of_fixes(self):
        """The four required-case patterns must still fire after all of the
        above — this round of fixes must only remove noise, not signal."""
        a = fact(id="a", chunk_id="d:4", page_number=4, metric="revenue from services",
                  value="81,415.38", period_label="FY24", basis="consolidated")
        b = fact(id="b", chunk_id="d:22", page_number=22, metric="revenue from services",
                  value="81,415.38", period_label="FY24", basis="consolidated")
        rels = find_pairwise_relationships([a, b])
        assert len(rels) == 1 and rels[0].relation_type == "corroboration"

        c = fact(id="c", fact_type="semantic", entity="Suvir Suren Sujan",
                  metric="board membership status", value="active", period_label="2022")
        d = fact(id="d", fact_type="semantic", entity="Suvir Suren Sujan",
                  metric="board membership status", value="resigned", period_label="FY24")
        rels = find_pairwise_relationships([c, d])
        assert len(rels) == 1 and rels[0].relation_type == "contradiction"

        services = fact(id="services", metric="revenue from services", value="72236.47",
                         period_label="FY23", basis="consolidated")
        traded_goods = fact(id="traded_goods", metric="revenue from sale of traded goods",
                             value="16.54", period_label="FY23", basis="consolidated")
        operations = fact(id="operations", metric="revenue from operations", value="72253.01",
                           period_label="FY23", basis="consolidated")
        rels = find_additive_reconciliations([services, traded_goods, operations])
        assert len(rels) == 1 and rels[0].relation_type == "context_reconcilable"

        e = fact(id="e", metric="revenue from operations, standalone basis", value="74540.82",
                  period_label="FY24", basis="standalone")
        f = fact(id="f", metric="revenue from operations, consolidated basis", value="81415.38",
                  period_label="FY24", basis="consolidated")
        rels = find_pairwise_relationships([e, f])
        assert len(rels) == 1 and rels[0].relation_type == "context_reconcilable"


class TestRealDataRegressionsRoundFive:
    """A fifth round, run against the full 1,012-fact corpus (all six source
    documents extracted), found the additive check breaking down at scale —
    see the module docstring's fifth correction in app.adjudicate."""

    def test_shares_significant_word_is_true_for_real_reconciliation_pairs(self):
        assert _shares_significant_word("Total non-current assets", "Total assets")
        assert _shares_significant_word("fresh issue size", "total offer size")
        assert _shares_significant_word("total finance income", "grand total other income")
        assert _shares_significant_word("revenue from services", "revenue from operations")

    def test_shares_significant_word_is_false_for_unrelated_metrics(self):
        assert not _shares_significant_word("net services receipts", "net payroll additions")
        assert not _shares_significant_word(
            "valuation gain on foreign exchange reserves",
            "new markets explored for cranes, lifts and winches",
        )

    def test_unrelated_indicators_do_not_reconcile_by_arithmetic_coincidence(self):
        """Real false positive: 'tur import quantity' (7.7) + 'valuation
        gain on foreign exchange reserves' (4.3) summed to exactly 12,
        matching an unrelated fact about export markets for construction
        equipment. Three genuinely unrelated economic indicators for the
        same entity/period/basis bucket, linked only by coincidence."""
        valuation_gain = fact(id="valuation_gain", entity="India",
                               metric="valuation gain on foreign exchange reserves",
                               value="4.3", period_label="2024", basis=None)
        tur_import = fact(id="tur_import", entity="India", metric="tur import quantity",
                           value="7.7", period_label="2024", basis=None)
        new_markets = fact(id="new_markets", entity="India",
                            metric="new markets explored for cranes, lifts and winches",
                            value="12", period_label="2024", basis=None)
        assert find_additive_reconciliations([valuation_gain, tur_import, new_markets]) == []

    def test_unrelated_indicators_do_not_reconcile_second_real_example(self):
        """Second real false positive from the same run: 'net services
        receipts' (120.1) + 'import cover' (10.9) summed to exactly 131,
        coincidentally matching a differently-worded payroll figure."""
        receipts = fact(id="receipts", entity="India", metric="net services receipts",
                         value="120.1", period_label="2024", basis=None)
        cover = fact(id="cover", entity="India", metric="import cover",
                     value="10.9", period_label="2024", basis=None)
        payroll = fact(id="payroll", entity="India", metric="net payroll additions",
                       value="131", period_label="2024", basis=None)
        assert find_additive_reconciliations([receipts, cover, payroll]) == []

    def test_genuine_reconciliation_with_differently_worded_addends_still_fires(self):
        """Guard against the round-five fix being too strict: a real
        reconciliation from this same corpus where only ONE addend shares a
        word with the target ('reserves') must still fire — the fix
        requires at least one shared word, not both."""
        bop_surplus = fact(id="bop_surplus", entity="India",
                            metric="balance of payments surplus",
                            value="23.9", period_label="FY25", basis=None)
        valuation_gain = fact(id="valuation_gain", entity="India",
                               metric="valuation gain on foreign exchange reserves",
                               value="35.5", period_label="FY25", basis=None)
        forex_increase = fact(id="forex_increase", entity="India",
                               metric="forex reserves increase",
                               value="59.4", period_label="FY25", basis=None)
        rels = find_additive_reconciliations([bop_surplus, valuation_gain, forex_increase])
        assert len(rels) == 1
        assert rels[0].relation_type == "context_reconcilable"

    def test_genuine_cases_survive_the_fifth_round_of_fixes(self):
        """The four required-case patterns must still fire after all of the
        above — this round of fixes must only remove noise, not signal."""
        a = fact(id="a", chunk_id="d:4", page_number=4, metric="revenue from services",
                  value="81,415.38", period_label="FY24", basis="consolidated")
        b = fact(id="b", chunk_id="d:22", page_number=22, metric="revenue from services",
                  value="81,415.38", period_label="FY24", basis="consolidated")
        rels = find_pairwise_relationships([a, b])
        assert len(rels) == 1 and rels[0].relation_type == "corroboration"

        c = fact(id="c", fact_type="semantic", entity="Suvir Suren Sujan",
                  metric="board membership status", value="active", period_label="2022")
        d = fact(id="d", fact_type="semantic", entity="Suvir Suren Sujan",
                  metric="board membership status", value="resigned", period_label="FY24")
        rels = find_pairwise_relationships([c, d])
        assert len(rels) == 1 and rels[0].relation_type == "contradiction"

        services = fact(id="services", metric="revenue from services", value="72236.47",
                         period_label="FY23", basis="consolidated")
        traded_goods = fact(id="traded_goods", metric="revenue from sale of traded goods",
                             value="16.54", period_label="FY23", basis="consolidated")
        operations = fact(id="operations", metric="revenue from operations", value="72253.01",
                           period_label="FY23", basis="consolidated")
        rels = find_additive_reconciliations([services, traded_goods, operations])
        assert len(rels) == 1 and rels[0].relation_type == "context_reconcilable"

        e = fact(id="e", metric="revenue from operations, standalone basis", value="74540.82",
                  period_label="FY24", basis="standalone")
        f = fact(id="f", metric="revenue from operations, consolidated basis", value="81415.38",
                  period_label="FY24", basis="consolidated")
        rels = find_pairwise_relationships([e, f])
        assert len(rels) == 1 and rels[0].relation_type == "context_reconcilable"