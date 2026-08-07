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

import pandas as pd

from verified_cost_router.eval.report import BASELINE_NAMES, EvalReport

# Baselines with a real cache -- no_system has none, so it's excluded
# from the cache-hit-rate-over-time view.
CACHE_BASELINE_NAMES = ("cache_router_no_verifier", "full_system")


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


def path_distribution(report: EvalReport, baseline: str = "full_system") -> pd.DataFrame:
    """Count of requests per path_taken for one baseline -- "how much
    traffic each branch handled" (ARCHITECTURE.md 4.6). Defaults to
    full_system, the only baseline with the real 5-way named taxonomy."""
    counts: dict[str, int] = {}
    for result in report.baseline_raw_results.get(baseline, ()):
        counts[result.path_taken] = counts.get(result.path_taken, 0) + 1
    return pd.DataFrame({"count": counts})


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
