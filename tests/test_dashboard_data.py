"""Unit tests for verified_cost_router.dashboard.data."""

from __future__ import annotations

import math

from verified_cost_router.dashboard.data import (
    baseline_summary_table,
    cache_reuse_path_distribution,
    compute_cache_reuse_metrics,
    compute_headline_metrics,
    cumulative_cache_hit_rate,
    cumulative_cost_by_baseline,
    path_distribution,
    quality_spot_check_table,
)
from verified_cost_router.eval.baselines import BaselineResult
from verified_cost_router.eval.cache_eval import CacheEvalResult, CachePairOutcome
from verified_cost_router.eval.quality_eval import QualitySpotCheckItem, QualitySpotCheckResult
from verified_cost_router.eval.report import BaselineSummary, EvalReport
from verified_cost_router.eval.router_eval import ComplexityItemOutcome, RouterEvalResult
from verified_cost_router.eval.verifier_eval import (
    CacheVerifierEvalResult,
    CacheVerifierOutcome,
    RouteVerifierEvalResult,
    RouteVerifierOutcome,
)


def _result(query, cost, calls, cache, strong, path):
    return BaselineResult(
        query=query, response="a", cost_usd=cost, llm_call_count=calls,
        served_from_cache=cache, used_strong_model=strong, path_taken=path,
    )


def _report(baseline_raw_results, quality_items=(), cache_reuse_raw_results=None):
    return EvalReport(
        cache_eval=CacheEvalResult(outcomes=()),
        router_eval=RouterEvalResult(outcomes=()),
        cache_verifier_eval=CacheVerifierEvalResult(reached_verifier=(), skipped=()),
        route_verifier_eval=RouteVerifierEvalResult(misrouted=(), correctly_routed_count=0),
        baseline_raw_results=baseline_raw_results,
        quality_spot_check=QualitySpotCheckResult(items=quality_items),
        cache_reuse_raw_results=cache_reuse_raw_results or {},
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


def test_baseline_summary_table_uses_stored_summaries_for_legacy_reports():
    """A pre-Phase-6 report has no per-query baseline_raw_results, but its
    real aggregate numbers must still show -- not zeros -- via
    stored_baseline_summaries (see EvalReport.baseline_summaries)."""
    report = EvalReport(
        cache_eval=CacheEvalResult(outcomes=()),
        router_eval=RouterEvalResult(outcomes=()),
        cache_verifier_eval=CacheVerifierEvalResult(reached_verifier=(), skipped=()),
        route_verifier_eval=RouteVerifierEvalResult(misrouted=(), correctly_routed_count=0),
        baseline_raw_results={},
        quality_spot_check=QualitySpotCheckResult(items=()),
        stored_baseline_summaries=(
            BaselineSummary("no_system", 30, 0.014812, 0.000494, 1.0, 0.0),
            BaselineSummary("cache_router_no_verifier", 24, 0.012797, 0.000533, 2.0, 0.0),
            BaselineSummary("full_system", 24, 0.011773, 0.000491, 2.12, 0.0),
        ),
    )

    table = baseline_summary_table(report)

    assert table.loc["no_system", "mean_cost_usd"] == 0.000494
    assert table.loc["full_system", "queries"] == 24
    # No per-query data exists for a legacy report -- charts must degrade
    # to empty, not raise or fabricate points.
    assert cumulative_cost_by_baseline(report).empty
    assert path_distribution(report).empty


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


def _reuse_raw_results():
    return {
        "no_system": (
            _result("qa1", 0.004, 1, False, True, "no_system"),
            _result("qb1", 0.004, 1, False, True, "no_system"),
        ),
        "full_system": (
            _result("qa1", 0.002, 2, False, False, "router-8B"),
            _result("qb1", 0.0, 0, True, False, "cache-hit"),
        ),
    }


def test_cache_reuse_path_distribution_counts_full_system_paths_only():
    report = _report({}, cache_reuse_raw_results=_reuse_raw_results())

    dist = cache_reuse_path_distribution(report)

    assert dist.loc["router-8B", "count"] == 1
    assert dist.loc["cache-hit", "count"] == 1
    # no_system's path_taken is always "no_system" -- not part of this view.
    assert "no_system" not in dist.index


def test_cache_reuse_path_distribution_independent_of_baseline_raw_results():
    """The cache-reuse benchmark is a separate stream -- it must never
    read from or leak into the 3-baseline natural-replay data."""
    report = _report(
        {"full_system": (_result("q1", 0.0, 0, True, False, "cache-hit"),)},
        cache_reuse_raw_results={
            "no_system": (_result("qa1", 0.004, 1, False, True, "no_system"),),
            "full_system": (_result("qa1", 0.002, 2, False, False, "router-8B"),),
        },
    )

    reuse_dist = cache_reuse_path_distribution(report)
    baseline_dist = path_distribution(report)

    assert list(reuse_dist.index) == ["router-8B"]
    assert list(baseline_dist.index) == ["cache-hit"]


def test_compute_cache_reuse_metrics_computes_savings_and_hit_rate():
    report = _report({}, cache_reuse_raw_results=_reuse_raw_results())

    m = compute_cache_reuse_metrics(report)

    assert m.pair_count == 1  # 2 full_system queries / 2
    assert m.no_system_mean_cost_usd == 0.004
    assert m.full_system_mean_cost_usd == 0.001
    assert m.savings_pct == (0.004 - 0.001) / 0.004
    assert m.full_system_cache_hit_rate == 0.5


def test_compute_cache_reuse_metrics_handles_absent_benchmark():
    report = _report({})

    m = compute_cache_reuse_metrics(report)

    assert m.pair_count == 0
    assert m.no_system_mean_cost_usd == 0.0
    assert m.full_system_mean_cost_usd == 0.0
    assert m.savings_pct == 0.0


def test_compute_headline_metrics_matches_underlying_eval_results():
    report = EvalReport(
        cache_eval=CacheEvalResult(
            outcomes=(
                CachePairOutcome("dup1", "true_duplicate", True, True, "high_confidence_hit"),
                CachePairOutcome("nm1", "near_miss", False, True, "risky_hit"),
            )
        ),
        router_eval=RouterEvalResult(
            outcomes=(
                ComplexityItemOutcome("cx1", "q1", "complex", True),
                ComplexityItemOutcome("cx2", "q2", "simple", False),
            )
        ),
        cache_verifier_eval=CacheVerifierEvalResult(
            reached_verifier=(CacheVerifierOutcome("nm1", "near_miss", "risky_hit", "fail", True),),
            skipped=(),
        ),
        route_verifier_eval=RouteVerifierEvalResult(
            misrouted=(RouteVerifierOutcome("cx2", "fail", True),), correctly_routed_count=1
        ),
        baseline_raw_results={
            "no_system": (_result("q1", 0.01, 1, False, True, "no_system"),),
            "cache_router_no_verifier": (_result("q1", 0.005, 2, False, False, "router_cheap"),),
            "full_system": (_result("q1", 0.008, 2, False, True, "router-70B"),),
        },
        quality_spot_check=QualitySpotCheckResult(
            items=(QualitySpotCheckItem("q1", "served", "reference", "comparable"),)
        ),
    )

    m = compute_headline_metrics(report)

    assert m.cache_precision == report.cache_eval.precision
    assert m.cache_recall == report.cache_eval.recall
    assert m.router_complex_recall == 0.5
    assert m.cache_verifier_near_miss_catch_rate == 1.0
    assert m.route_verifier_catch_rate == 1.0
    assert m.route_verifier_misrouted_count == 1
    assert m.quality_comparable_rate == 1.0
    assert m.quality_sample_size == 1
    assert m.no_system_mean_cost_usd == 0.01
    assert m.full_system_mean_cost_usd == 0.008
    assert m.cost_savings_pct == (0.01 - 0.008) / 0.01
    assert m.natural_replay_cache_hit_rate == 0.0


def test_compute_headline_metrics_uses_stored_summaries_for_legacy_reports():
    report = EvalReport(
        cache_eval=CacheEvalResult(outcomes=()),
        router_eval=RouterEvalResult(outcomes=()),
        cache_verifier_eval=CacheVerifierEvalResult(reached_verifier=(), skipped=()),
        route_verifier_eval=RouteVerifierEvalResult(misrouted=(), correctly_routed_count=0),
        baseline_raw_results={},
        quality_spot_check=QualitySpotCheckResult(items=()),
        stored_baseline_summaries=(
            BaselineSummary("no_system", 30, 0.014812, 0.000494, 1.0, 0.0),
            BaselineSummary("cache_router_no_verifier", 24, 0.012797, 0.000533, 2.0, 0.0),
            BaselineSummary("full_system", 24, 0.011773, 0.000491, 2.12, 0.0),
        ),
    )

    m = compute_headline_metrics(report)

    assert m.no_system_mean_cost_usd == 0.000494
    assert m.full_system_mean_cost_usd == 0.000491
    assert m.natural_replay_cache_hit_rate == 0.0
