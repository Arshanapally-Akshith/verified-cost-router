"""Streamlit dashboard (ARCHITECTURE.md 4.6): reads data/eval_report.json
(scripts/run_eval.py's output, Phase 5) and displays cost with vs.
without the system, cache hit rate over time, cache/router/verifier
precision-recall-catch-rate, path distribution, the cache-reuse
benchmark, and the quality-regression spot check.

Run with:
    streamlit run dashboard/app.py

This file is UI rendering only -- all data loading/transformation lives
in verified_cost_router.dashboard.data, which is plain Python (testable
without Streamlit) reusing the Phase 5 eval dataclasses directly.

Always reads data/eval_report.json relative to the repo root -- no
user-editable path, so the deployed app can't leak (or require) a local
filesystem path.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

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
from verified_cost_router.eval.report import FULL_SYSTEM, load_eval_report

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "data" / "eval_report.json"

st.set_page_config(page_title="Verified Cost Router -- Eval Dashboard", layout="wide")


def main() -> None:
    st.title("Verified Cost Router -- Eval Dashboard")
    st.caption(
        "Reads data/eval_report.json, produced by `python scripts/run_eval.py`. "
        "Re-run that script for fresh numbers; this page only displays what it wrote."
    )

    if not REPORT_PATH.exists():
        st.error(f"No eval report found at `{REPORT_PATH}`. Run `python scripts/run_eval.py` first.")
        return

    report = load_eval_report(REPORT_PATH)
    m = compute_headline_metrics(report)

    # --- Results at a glance ------------------------------------------------
    st.header("Results at a glance")
    row1 = st.columns(3)
    row1[0].metric("Cache precision / recall", f"{m.cache_precision:.0%} / {m.cache_recall:.0%}")
    row1[1].metric("Router complex-recall", f"{m.router_complex_recall:.0%}")
    row1[2].metric("Verifier catch rate -- bad cache hit", f"{m.cache_verifier_near_miss_catch_rate:.0%}")

    row2 = st.columns(3)
    row2[0].metric(
        "Verifier catch rate -- bad route",
        f"{m.route_verifier_catch_rate:.0%}",
        help=f"n={m.route_verifier_misrouted_count} misrouted item(s) in this run -- too small to generalize from.",
    )
    row2[1].metric(
        "Quality spot check: comparable to strong model",
        f"{m.quality_comparable_rate:.0%}",
        help=f"n={m.quality_sample_size} response(s) spot-checked -- a small sample, read as directional only.",
    )
    row2[2].metric(
        "Cost vs. no-system baseline",
        f"{m.cost_savings_pct:+.1%}",
        help="See the caption under 'Cost with vs. without the system' below for why this number is small.",
    )
    st.caption(
        f"⚠️ The {m.cost_savings_pct:+.1%} cost figure comes from a natural, unfiltered "
        f"replay sample with a **{m.natural_replay_cache_hit_rate:.0%} cache hit rate** -- "
        "no query in that sample was semantically similar enough to another to hit the "
        "cache, so this number reflects router/verifier efficiency alone, not caching. "
        "It is not a caching benchmark; see the baseline comparison below for detail."
    )

    if not report.baseline_raw_results:
        st.info(
            "This report predates per-query baseline tracking, so the cumulative-cost, "
            "cache-hit-rate-over-time, and path-distribution charts below have no data to "
            "show. The baseline comparison table and all cache/router/verifier/quality "
            "numbers are real, from this report. Re-run `python scripts/run_eval.py` to "
            "populate per-query detail."
        )

    # --- Cost with vs. without the system --------------------------------
    st.header("Cost with vs. without the system")
    st.caption(
        "Natural replay traffic, no repeated or paraphrased queries by construction -- "
        f"cache hit rate was {m.natural_replay_cache_hit_rate:.0%} in this run, so the cost "
        "delta below is driven by routing (cheap vs. strong model) and verification, not "
        "caching. A workload with genuine repeat traffic would give caching more to do."
    )
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

    # --- Cache-reuse benchmark (synthetic, separate from the baselines above) --
    st.header("Cache-reuse benchmark (synthetic)")
    if report.cache_reuse_raw_results.get(FULL_SYSTEM):
        reuse = compute_cache_reuse_metrics(report)
        st.caption(
            f"Uses {reuse.pair_count} seeded, hand-verified true-duplicate pairs from the same "
            "labeled adversarial eval set the cache precision/recall numbers above use -- not new "
            "or cherry-picked data. Each pair is asked as two queries (the original, then its "
            "paraphrase), through **both** no_system and full_system, so a real cost-savings "
            "number can be computed the same way as the baseline comparison above -- but on a "
            "workload that actually has cache reuse in it. full_system uses a fresh, isolated "
            "cache **per pair**, so one pair's cached answer can never affect another pair's "
            "result. This is a deliberately repetitive, illustrative workload, **not** "
            "representative of the natural replay traffic above.\n\n"
            f"**Sampling:** {reuse.pair_count} of {reuse.population_size} available true_duplicate "
            f"pairs ({reuse.sample_pct:.0%}, seed={reuse.seed})."
        )
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("no_system mean cost/query", f"${reuse.no_system_mean_cost_usd:.6f}")
        rc2.metric(
            "full_system mean cost/query",
            f"${reuse.full_system_mean_cost_usd:.6f}",
            delta=f"{-reuse.savings_pct:.1%}",
        )
        rc3.metric("Cache-reuse savings", f"{reuse.savings_pct:+.1%}")
        rc4.metric("full_system cache hit rate", f"{reuse.full_system_cache_hit_rate:.0%}")
        reuse_dist = cache_reuse_path_distribution(report)
        if not reuse_dist.empty:
            st.bar_chart(reuse_dist)
    else:
        st.info(
            "Not present in this report. Run `python scripts/run_eval.py` "
            "(cache-reuse benchmark runs by default, `--cache-reuse-sample-pct 0` disables it) "
            "to populate this section."
        )

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
            f"model ({report.route_verifier_eval.correctly_routed_count} were correctly routed complex) "
            "-- a catch rate this small a sample is directional, not conclusive."
        )

    # --- Quality-regression spot check -------------------------------------
    st.header("Quality-regression spot check")
    st.caption(
        f"n={m.quality_sample_size} non-strong-model response(s), judged by the strong model "
        "against what it would have produced itself -- to show cost savings aren't silently "
        f"costing quality. A sample this small (n={m.quality_sample_size}) is directional, not "
        "statistically conclusive; treat the rate below as a spot check, not a guarantee."
    )
    st.metric("Comparable-to-strong-model rate", f"{report.quality_spot_check.comparable_rate:.1%}")
    spot_check_df = quality_spot_check_table(report)
    if not spot_check_df.empty:
        st.dataframe(spot_check_df, use_container_width=True)
    else:
        st.info("No responses were spot-checked in this report.")


if __name__ == "__main__":
    main()
