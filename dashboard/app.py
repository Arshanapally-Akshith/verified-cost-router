"""Streamlit dashboard (ARCHITECTURE.md 4.6): reads data/eval_report.json
(scripts/run_eval.py's output, Phase 5) and displays cost with vs.
without the system, cache hit rate over time, cache/router/verifier
precision-recall-catch-rate, path distribution, and the
quality-regression spot check.

Run with:
    streamlit run dashboard/app.py

This file is UI rendering only -- all data loading/transformation lives
in verified_cost_router.dashboard.data, which is plain Python (testable
without Streamlit) reusing the Phase 5 eval dataclasses directly.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from verified_cost_router.dashboard.data import (
    baseline_summary_table,
    cumulative_cache_hit_rate,
    cumulative_cost_by_baseline,
    path_distribution,
    quality_spot_check_table,
)
from verified_cost_router.eval.report import load_eval_report

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = REPO_ROOT / "data" / "eval_report.json"

st.set_page_config(page_title="Verified Cost Router -- Eval Dashboard", layout="wide")


def main() -> None:
    st.title("Verified Cost Router -- Eval Dashboard")
    st.caption(
        "Reads data/eval_report.json, produced by `python scripts/run_eval.py` (Phase 5). "
        "Re-run that script for fresh numbers; this page only displays what it wrote."
    )

    report_path = Path(st.sidebar.text_input("Eval report path", value=str(DEFAULT_REPORT_PATH)))
    if not report_path.exists():
        st.error(f"No eval report found at `{report_path}`. Run `python scripts/run_eval.py` first.")
        return

    report = load_eval_report(report_path)

    # --- Cost with vs. without the system --------------------------------
    st.header("Cost with vs. without the system")
    summary = baseline_summary_table(report)
    st.dataframe(
        summary.style.format(
            {
                "mean_cost_usd": "${:.6f}",
                "total_cost_usd": "${:.6f}",
                "mean_llm_calls": "{:.2f}",
                "cache_hit_rate": "{:.1%}",
            }
        ),
        use_container_width=True,
    )

    no_system_cost = summary.loc["no_system", "mean_cost_usd"]
    full_system_cost = summary.loc["full_system", "mean_cost_usd"]
    savings = (no_system_cost - full_system_cost) / no_system_cost if no_system_cost else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("no_system mean cost/query", f"${no_system_cost:.6f}")
    col2.metric("full_system mean cost/query", f"${full_system_cost:.6f}", delta=f"{-savings:.1%}")
    col3.metric("Replayed queries (per baseline)", int(summary["queries"].max()))

    st.subheader("Cumulative cost, replayed over sampled traffic")
    cost_df = cumulative_cost_by_baseline(report)
    if not cost_df.empty:
        st.line_chart(cost_df)
    else:
        st.info("No baseline query data in this report.")

    # --- Cache hit rate over time -----------------------------------------
    st.header("Cache hit rate over time")
    hit_rate_df = cumulative_cache_hit_rate(report)
    if not hit_rate_df.empty:
        st.line_chart(hit_rate_df)
    else:
        st.info("No cache-enabled baseline data in this report.")

    # --- Path distribution --------------------------------------------------
    st.header("Path distribution (full system)")
    st.caption("How much replayed traffic each of the 5 named paths handled.")
    dist = path_distribution(report)
    if not dist.empty:
        st.bar_chart(dist)
    else:
        st.info("No full_system query data in this report.")

    # --- Cache / router / verifier quality on the labeled adversarial set --
    st.header("Cache, router, and verifier quality (labeled adversarial set)")
    left, right = st.columns(2)
    with left:
        st.subheader("Cache layer")
        st.metric("Precision", f"{report.cache_eval.precision:.1%}")
        st.metric("Recall", f"{report.cache_eval.recall:.1%}")
        st.caption(f"{len(report.cache_eval.outcomes)} labeled pairs evaluated.")

        st.subheader("Router")
        st.metric("Complex-recall (adversarial)", f"{report.router_eval.complex_recall:.1%}")
        st.caption(
            f"{len(report.router_eval.outcomes)} complexity-mislabeled items evaluated "
            "(all genuinely complex, worded simply)."
        )

    with right:
        st.subheader("Verifier: bad cache-hit catch rate")
        st.metric("Near-miss catch rate", f"{report.cache_verifier_eval.near_miss_catch_rate:.1%}")
        st.metric("True-duplicate pass rate", f"{report.cache_verifier_eval.true_duplicate_pass_rate:.1%}")
        st.caption(
            f"{len(report.cache_verifier_eval.reached_verifier)} pairs reached the verifier "
            f"({report.cache_verifier_eval.near_miss_high_confidence_leaks} near-miss pairs leaked "
            "past it entirely, into an unverified high-confidence auto-serve)."
        )

        st.subheader("Verifier: bad route catch rate")
        st.metric("Route catch rate", f"{report.route_verifier_eval.catch_rate:.1%}")
        st.caption(
            f"{len(report.route_verifier_eval.misrouted)} items the router misrouted to the cheap "
            f"model ({report.route_verifier_eval.correctly_routed_count} were correctly routed complex)."
        )

    # --- Quality-regression spot check -------------------------------------
    st.header("Quality-regression spot check")
    st.caption(
        "A small sample of non-strong-model responses, judged by the strong model against "
        "what it would have produced itself -- to show cost savings aren't silently costing quality."
    )
    st.metric("Comparable-to-strong-model rate", f"{report.quality_spot_check.comparable_rate:.1%}")
    spot_check_df = quality_spot_check_table(report)
    if not spot_check_df.empty:
        st.dataframe(spot_check_df, use_container_width=True)
    else:
        st.info("No responses were spot-checked in this report.")


if __name__ == "__main__":
    main()
