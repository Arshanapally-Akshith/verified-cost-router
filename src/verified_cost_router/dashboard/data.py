"""Pure data-transformation layer for the Streamlit dashboard
(ARCHITECTURE.md 4.6): cost with/without the system, cache hit rate
over time, verifier precision/recall, path distribution, and the
quality-regression spot check.

No Streamlit import here -- keeps this testable with plain pytest;
dashboard/app.py is the thin rendering layer that calls these functions
and reads the report via verified_cost_router.eval.report.load_eval_report,
reusing the Phase 5 dataclasses (and their precision/recall/catch-rate
properties) rather than re-deriving that logic from raw JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from verified_cost_router.eval.baselines import BaselineResult
from verified_cost_router.eval.report import BASELINE_NAMES, FULL_SYSTEM, NO_SYSTEM, EvalReport

# Baselines with a real cache -- no_system has none, so it's excluded
# from the cache-hit-rate-over-time view.
CACHE_BASELINE_NAMES = ("cache_router_no_verifier", "full_system")


@dataclass(frozen=True)
class HeadlineMetrics:
    """The handful of numbers that summarize the whole report -- rendered
    as the "Results at a glance" strip at the top of the dashboard, so a
    visitor doesn't have to scroll through every section to find them."""

    cache_precision: float
    cache_recall: float
    router_complex_recall: float
    cache_verifier_near_miss_catch_rate: float
    route_verifier_catch_rate: float
    route_verifier_misrouted_count: int
    quality_comparable_rate: float
    quality_sample_size: int
    no_system_mean_cost_usd: float
    full_system_mean_cost_usd: float
    cost_savings_pct: float
    natural_replay_cache_hit_rate: float


def compute_headline_metrics(report: EvalReport) -> HeadlineMetrics:
    summaries = {s.name: s for s in report.baseline_summaries}
    no_system = summaries["no_system"]
    full_system = summaries["full_system"]
    savings = (
        (no_system.mean_cost_usd - full_system.mean_cost_usd) / no_system.mean_cost_usd
        if no_system.mean_cost_usd
        else 0.0
    )
    return HeadlineMetrics(
        cache_precision=report.cache_eval.precision,
        cache_recall=report.cache_eval.recall,
        router_complex_recall=report.router_eval.complex_recall,
        cache_verifier_near_miss_catch_rate=report.cache_verifier_eval.near_miss_catch_rate,
        route_verifier_catch_rate=report.route_verifier_eval.catch_rate,
        route_verifier_misrouted_count=len(report.route_verifier_eval.misrouted),
        quality_comparable_rate=report.quality_spot_check.comparable_rate,
        quality_sample_size=len(report.quality_spot_check.items),
        no_system_mean_cost_usd=no_system.mean_cost_usd,
        full_system_mean_cost_usd=full_system.mean_cost_usd,
        cost_savings_pct=savings,
        natural_replay_cache_hit_rate=full_system.cache_hit_rate,
    )


def baseline_summary_table(report: EvalReport) -> pd.DataFrame:
    """One row per baseline: query count, mean/total cost, mean LLM calls, cache hit rate."""
    rows = [
        {
            "baseline": summary.name,
            "queries": summary.query_count,
            "mean_cost_usd": summary.mean_cost_usd,
            "total_cost_usd": summary.total_cost_usd,
            "mean_llm_calls": summary.mean_llm_calls,
            "cache_hit_rate": summary.cache_hit_rate,
        }
        for summary in report.baseline_summaries
    ]
    return pd.DataFrame(rows).set_index("baseline")


def cumulative_cost_by_baseline(report: EvalReport) -> pd.DataFrame:
    """Running total cost per query index, one column per baseline -- the
    "cost with vs. without the system, replayed over sampled traffic"
    view (ARCHITECTURE.md 4.6).

    Baselines can end up with different lengths (a query that fails
    classification is skipped independently per baseline -- see
    scripts/run_eval.py's `_run_over_queries`); shorter columns are left
    as NaN beyond their length rather than padded with a repeated last
    value, so the chart shows a gap instead of implying no further cost
    was incurred.
    """
    series = {}
    for name in BASELINE_NAMES:
        results = report.baseline_raw_results.get(name, ())
        running = 0.0
        values = []
        for result in results:
            running += result.cost_usd
            values.append(running)
        series[name] = pd.Series(values, index=range(1, len(values) + 1))
    df = pd.DataFrame(series)
    df.index.name = "query #"
    return df


def cumulative_cache_hit_rate(report: EvalReport) -> pd.DataFrame:
    """Running cache hit rate per query index, for the baselines that
    have a real cache."""
    series = {}
    for name in CACHE_BASELINE_NAMES:
        results = report.baseline_raw_results.get(name, ())
        hits = 0
        values = []
        for index, result in enumerate(results, start=1):
            if result.served_from_cache:
                hits += 1
            values.append(hits / index)
        series[name] = pd.Series(values, index=range(1, len(values) + 1))
    df = pd.DataFrame(series)
    df.index.name = "query #"
    return df


def _count_by_path(results: Sequence[BaselineResult]) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.path_taken] = counts.get(result.path_taken, 0) + 1
    return pd.DataFrame({"count": counts})


def path_distribution(report: EvalReport, baseline: str = "full_system") -> pd.DataFrame:
    """Count of requests per path_taken for one baseline -- "how much
    traffic each branch handled" (ARCHITECTURE.md 4.6). Defaults to
    full_system, the only baseline with the real 5-way named taxonomy."""
    return _count_by_path(report.baseline_raw_results.get(baseline, ()))


def cache_reuse_path_distribution(report: EvalReport) -> pd.DataFrame:
    """Count of requests per path_taken in the cache-reuse benchmark's
    full_system stream (see eval.cache_reuse_benchmark) -- separate from
    path_distribution() above, which only ever looks at the 3
    ARCHITECTURE.md section 6 baselines. no_system has no path variety
    (every request is "no_system"), so this only covers full_system."""
    return _count_by_path(report.cache_reuse_raw_results.get(FULL_SYSTEM, ()))


@dataclass(frozen=True)
class CacheReuseMetrics:
    """no_system vs full_system on the cache-reuse benchmark's own
    workload -- the same shape as the natural-replay cost comparison in
    HeadlineMetrics, scoped to this separate, synthetic benchmark.

    population_size/sample_pct/seed document how pair_count was arrived
    at (a seeded percentage sample, not a hardcoded count) so the sample
    is auditable rather than a bare number.
    """

    pair_count: int
    population_size: int
    sample_pct: float
    seed: int
    no_system_mean_cost_usd: float
    full_system_mean_cost_usd: float
    savings_pct: float
    full_system_cache_hit_rate: float


def compute_cache_reuse_metrics(report: EvalReport) -> CacheReuseMetrics:
    summaries = report.cache_reuse_summaries
    no_system, full_system = summaries[NO_SYSTEM], summaries[FULL_SYSTEM]
    return CacheReuseMetrics(
        pair_count=full_system.query_count // 2,
        population_size=report.cache_reuse_population_size,
        sample_pct=report.cache_reuse_sample_pct,
        seed=report.cache_reuse_seed,
        no_system_mean_cost_usd=no_system.mean_cost_usd,
        full_system_mean_cost_usd=full_system.mean_cost_usd,
        savings_pct=report.cache_reuse_savings_pct,
        full_system_cache_hit_rate=full_system.cache_hit_rate,
    )


def quality_spot_check_table(report: EvalReport) -> pd.DataFrame:
    """One row per spot-checked response: query, served vs. reference answer, verdict."""
    items = report.quality_spot_check.items
    if not items:
        return pd.DataFrame(columns=["query", "served_response", "reference_response", "verdict"])
    return pd.DataFrame(
        [
            {
                "query": item.query,
                "served_response": item.served_response,
                "reference_response": item.reference_response,
                "verdict": item.verdict,
            }
            for item in items
        ]
    )
