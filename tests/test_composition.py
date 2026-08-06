"""Unit tests for verified_cost_router.data_prep.composition."""

from __future__ import annotations

from verified_cost_router.data_prep.composition import (
    categorize,
    compute_composition,
    render_composition_report,
)


def test_categorize_matches_code_keyword():
    assert categorize("Can you debug this Python function?") == "code"


def test_categorize_matches_creative_writing():
    assert categorize("Write a short story about a dragon") == "creative_writing"


def test_categorize_falls_back_to_other():
    assert categorize("blorp zibble wooo") == "other"


def test_categorize_first_matching_category_wins():
    # Contains both a "code" keyword ("function") and a "general_qa"
    # keyword ("what is"); "code" is earlier in CATEGORY_KEYWORDS.
    assert categorize("what is a function in programming?") == "code"


def test_compute_composition_counts_each_category():
    queries = [
        "debug this python script",
        "write a poem about autumn",
        "debug this python script",
    ]
    counts = compute_composition(queries)
    assert counts["code"] == 2
    assert counts["creative_writing"] == 1


def test_render_composition_report_includes_share_column():
    counts = compute_composition(["debug this", "debug that", "write a poem"])
    report = render_composition_report(counts, sample_size=3)
    assert "| category | count | share |" in report
    assert "66.7%" in report


def test_render_composition_report_handles_empty_sample():
    report = render_composition_report(compute_composition([]), sample_size=0)
    assert "| category | count | share |" in report
