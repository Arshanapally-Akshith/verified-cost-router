"""Unit tests for verified_cost_router.dashboard.data."""

from __future__ import annotations

import math

from verified_cost_router.dashboard.data import (
    baseline_summary_table,
    cumulative_cache_hit_rate,
    cumulative_cost_by_baseline,
    path_distribution,
    quality_spot_check_table,
)
from verified_cost_router.eval.baselines import BaselineResult
from verified_cost_router.eval.cache_eval import CacheEvalResult
from verified_cost_router.eval.quality_eval import QualitySpotCheckItem, QualitySpotCheckResult
from verified_cost_router.eval.report import EvalReport
from verified_cost_router.eval.router_eval import RouterEvalResult
from verified_cost_router.eval.verifier_eval import CacheVerifierEvalResult, RouteVerifierEvalResult


def _result(query, cost, calls, cache, strong, path):
    return BaselineResult(
        query=query, response="a", cost_usd=cost, llm_call_count=calls,
        served_from_cache=cache, used_strong_model=strong, path_taken=path,
    )


def _report(baseline_raw_results, quality_items=()):
    return EvalReport(
        cache_eval=CacheEvalResult(outcomes=()),
        router_eval=RouterEvalResult(outcomes=()),
        cache_verifier_eval=CacheVerifierEvalResult(reached_verifier=(), skipped=()),
        route_verifier_eval=RouteVerifierEvalResult(misrouted=(), correctly_routed_count=0),
        baseline_raw_results=baseline_raw_results,
        quality_spot_check=QualitySpotCheckResult(items=quality_items),
    )


def test_baseline_summary_table_has_one_row_per_baseline_in_order():
    report = _report(
        {
            "no_system": (_result("q1", 0.01, 1, False, True, "no_system"),),
            "cache_router_no_verifier": (_result("q1", 0.0, 0, True, False, "cache_hit"),),
            "full_system": (_result("q1", 0.005, 2, False, True, "router-70B"),),
        }
    )

    table = baseline_summary_table(report)

    assert list(table.index) == ["no_system", "cache_router_no_verifier", "full_system"]
    assert table.loc["no_system", "mean_cost_usd"] == 0.01
    assert table.loc["cache_router_no_verifier", "cache_hit_rate"] == 1.0


def test_baseline_summary_table_handles_missing_baseline():
    report = _report({"no_system": (_result("q1", 0.01, 1, False, True, "no_system"),)})
    table = baseline_summary_table(report)
    assert table.loc["full_system", "queries"] == 0


def test_cumulative_cost_by_baseline_accumulates_in_order():
    report = _report(
        {
            "no_system": (
                _result("q1", 0.01, 1, False, True, "no_system"),
                _result("q2", 0.02, 1, False, True, "no_system"),
            ),
        }
    )

    df = cumulative_cost_by_baseline(report)

    assert list(df["no_system"]) == [0.01, 0.03]
    assert list(df.index) == [1, 2]


def test_cumulative_cost_by_baseline_pads_shorter_series_with_nan_not_repeated_value():
    report = _report(
        {
            "no_system": (
                _result("q1", 0.01, 1, False, True, "no_system"),
                _result("q2", 0.01, 1, False, True, "no_system"),
            ),
            "full_system": (_result("q1", 0.005, 2, False, True, "router-70B"),),  # one query skipped
        }
    )

    df = cumulative_cost_by_baseline(report)

    assert df["no_system"].tolist() == [0.01, 0.02]
    assert df["full_system"].iloc[0] == 0.005
    assert math.isnan(df["full_system"].iloc[1])  # gap, not a repeated 0.005


def test_cumulative_cache_hit_rate_tracks_running_rate():
    report = _report(
        {
            "full_system": (
                _result("q1", 0.0, 0, True, False, "cache-hit"),
                _result("q2", 0.005, 2, False, True, "router-70B"),
                _result("q3", 0.0, 1, True, False, "cache-hit-verified"),
            ),
        }
    )

    df = cumulative_cache_hit_rate(report)

    assert df["full_system"].tolist() == [1.0, 0.5, 2 / 3]


def test_cumulative_cache_hit_rate_excludes_no_system():
    report = _report({"no_system": (_result("q1", 0.01, 1, False, True, "no_system"),)})
    df = cumulative_cache_hit_rate(report)
    assert "no_system" not in df.columns


def test_path_distribution_counts_by_path():
    report = _report(
        {
            "full_system": (
                _result("q1", 0.0, 0, True, False, "cache-hit"),
                _result("q2", 0.005, 2, False, True, "router-70B"),
                _result("q3", 0.005, 2, False, True, "router-70B"),
            ),
        }
    )

    dist = path_distribution(report)

    assert dist.loc["cache-hit", "count"] == 1
    assert dist.loc["router-70B", "count"] == 2


def test_path_distribution_defaults_to_full_system():
    report = _report(
        {
            "cache_router_no_verifier": (_result("q1", 0.0, 0, True, False, "cache_hit"),),
            "full_system": (_result("q1", 0.0, 0, True, False, "cache-hit"),),
        }
    )
    dist = path_distribution(report)
    assert list(dist.index) == ["cache-hit"]


def test_path_distribution_can_target_a_different_baseline():
    report = _report({"cache_router_no_verifier": (_result("q1", 0.0, 0, True, False, "cache_hit"),)})
    dist = path_distribution(report, baseline="cache_router_no_verifier")
    assert list(dist.index) == ["cache_hit"]


def test_quality_spot_check_table_has_expected_columns():
    report = _report({}, quality_items=(QualitySpotCheckItem("q1", "served", "reference", "comparable"),))
    table = quality_spot_check_table(report)
    assert list(table.columns) == ["query", "served_response", "reference_response", "verdict"]
    assert table.iloc[0]["verdict"] == "comparable"


def test_quality_spot_check_table_handles_empty_items():
    report = _report({})
    table = quality_spot_check_table(report)
    assert len(table) == 0
    assert "verdict" in table.columns
