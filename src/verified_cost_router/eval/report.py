"""Aggregates every Phase 5 eval result into one report, rendered as
JSON (machine-readable, for Phase 6's dashboard) and markdown (human
summary) -- BUILD.md section 5: "Eval harness produces the precision/
recall + cost numbers... from a single script run."

Stores the raw per-query BaselineResult sequences (not just aggregate
summaries): the dashboard (Phase 6) needs query-by-query granularity
for "cache hit rate over time" and "path distribution"
(ARCHITECTURE.md 4.6), which a pre-aggregated summary can't provide.
Summaries are derived from the raw results, not stored separately, so
the two can never drift apart.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from verified_cost_router.eval.baselines import BaselineResult
from verified_cost_router.eval.cache_eval import CacheEvalResult, CachePairOutcome
from verified_cost_router.eval.quality_eval import QualitySpotCheckItem, QualitySpotCheckResult
from verified_cost_router.eval.router_eval import ComplexityItemOutcome, RouterEvalResult
from verified_cost_router.eval.verifier_eval import (
    CacheVerifierEvalResult,
    CacheVerifierOutcome,
    RouteVerifierEvalResult,
    RouteVerifierOutcome,
    SkippedCachePair,
)

# Canonical baseline names/order (ARCHITECTURE.md section 6).
NO_SYSTEM = "no_system"
CACHE_ROUTER_NO_VERIFIER = "cache_router_no_verifier"
FULL_SYSTEM = "full_system"
BASELINE_NAMES: tuple[str, str, str] = (NO_SYSTEM, CACHE_ROUTER_NO_VERIFIER, FULL_SYSTEM)


@dataclass(frozen=True)
class BaselineSummary:
    name: str
    query_count: int
    total_cost_usd: float
    mean_cost_usd: float
    mean_llm_calls: float
    cache_hit_rate: float


def summarize_baseline(name: str, results: Sequence[BaselineResult]) -> BaselineSummary:
    n = len(results)
    if n == 0:
        return BaselineSummary(
            name=name, query_count=0, total_cost_usd=0.0, mean_cost_usd=0.0, mean_llm_calls=0.0, cache_hit_rate=0.0
        )
    total_cost = sum(r.cost_usd for r in results)
    return BaselineSummary(
        name=name,
        query_count=n,
        total_cost_usd=total_cost,
        mean_cost_usd=total_cost / n,
        mean_llm_calls=sum(r.llm_call_count for r in results) / n,
        cache_hit_rate=sum(1 for r in results if r.served_from_cache) / n,
    )


@dataclass(frozen=True)
class EvalReport:
    cache_eval: CacheEvalResult
    router_eval: RouterEvalResult
    cache_verifier_eval: CacheVerifierEvalResult
    route_verifier_eval: RouteVerifierEvalResult
    # Keyed by BASELINE_NAMES (no_system / cache_router_no_verifier / full_system).
    baseline_raw_results: dict[str, tuple[BaselineResult, ...]]
    quality_spot_check: QualitySpotCheckResult
    # Set only by load_eval_report() when reading a pre-Phase-6 report that
    # has aggregate baseline_summaries but no per-query baseline_raw_results
    # (the dashboard's real summary table still needs real numbers to show;
    # see load_eval_report's docstring). None for every report produced by
    # a live run -- baseline_summaries always derives from baseline_raw_results
    # in that case, same as before this field existed.
    stored_baseline_summaries: tuple[BaselineSummary, ...] | None = None
    # Optional, separate from the 3 ARCHITECTURE.md section 6 baselines above:
    # no_system vs full_system on a small, explicitly synthetic workload
    # built from repeated/paraphrased queries (see eval.cache_reuse_benchmark),
    # so a real cost-savings number can be computed for a workload that
    # actually has cache reuse in it -- unlike the natural replay sample,
    # which saw a 0% cache hit rate. Keyed like baseline_raw_results
    # (NO_SYSTEM / FULL_SYSTEM) but stored in a separate dict, so it can
    # never be merged into or mistaken for the natural-replay baselines.
    # Empty for every report that predates this benchmark.
    cache_reuse_raw_results: dict[str, tuple[BaselineResult, ...]] = field(default_factory=dict)

    @property
    def baseline_summaries(self) -> tuple[BaselineSummary, ...]:
        if self.stored_baseline_summaries is not None:
            return self.stored_baseline_summaries
        return tuple(summarize_baseline(name, self.baseline_raw_results.get(name, ())) for name in BASELINE_NAMES)

    @property
    def cache_reuse_summaries(self) -> dict[str, BaselineSummary]:
        return {
            name: summarize_baseline(name, self.cache_reuse_raw_results.get(name, ()))
            for name in (NO_SYSTEM, FULL_SYSTEM)
        }

    @property
    def cache_reuse_savings_pct(self) -> float:
        """(no_system - full_system) / no_system mean cost, on the
        cache-reuse benchmark's own workload -- the same computation
        to_markdown() already does for the natural-replay baselines
        above, just scoped to this separate comparison."""
        summaries = self.cache_reuse_summaries
        no_system, full_system = summaries[NO_SYSTEM], summaries[FULL_SYSTEM]
        if not no_system.mean_cost_usd:
            return 0.0
        return (no_system.mean_cost_usd - full_system.mean_cost_usd) / no_system.mean_cost_usd

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_markdown(self) -> str:
        summaries = {summary.name: summary for summary in self.baseline_summaries}
        no_system, cache_router, full_system = (summaries[name] for name in BASELINE_NAMES)
        savings_vs_no_system = (
            (no_system.mean_cost_usd - full_system.mean_cost_usd) / no_system.mean_cost_usd
            if no_system.mean_cost_usd
            else 0.0
        )
        verifier_cost_delta = full_system.mean_cost_usd - cache_router.mean_cost_usd

        lines = [
            "# Eval report\n",
            "## Cache precision/recall (labeled cache_pairs, BUILD.md section 4)\n",
            f"- precision: {self.cache_eval.precision:.1%}",
            f"- recall: {self.cache_eval.recall:.1%}",
            f"- pairs evaluated: {len(self.cache_eval.outcomes)}\n",
            "## Router accuracy (labeled complexity_items)\n",
            f"- complex-recall on adversarial complexity-mislabeled queries: {self.router_eval.complex_recall:.1%}",
            f"- items evaluated: {len(self.router_eval.outcomes)}\n",
            "## Verifier catch rate\n",
            f"- bad cache-hit catch rate (near_miss pairs correctly failed): {self.cache_verifier_eval.near_miss_catch_rate:.1%}",
            f"- good cache-hit pass rate (true_duplicate pairs correctly passed): {self.cache_verifier_eval.true_duplicate_pass_rate:.1%}",
            f"- pairs that reached the verifier: {len(self.cache_verifier_eval.reached_verifier)}"
            f" (skipped, no_match: {sum(1 for s in self.cache_verifier_eval.skipped if s.reason == 'no_match')};"
            f" skipped, leaked into high_confidence: "
            f"{sum(1 for s in self.cache_verifier_eval.skipped if s.reason == 'high_confidence_hit')},"
            f" of which near_miss: {self.cache_verifier_eval.near_miss_high_confidence_leaks})",
            f"- bad route catch rate (misrouted items correctly failed): {self.route_verifier_eval.catch_rate:.1%}",
            f"- items router misrouted to cheap model: {len(self.route_verifier_eval.misrouted)}"
            f" (router correctly routed: {self.route_verifier_eval.correctly_routed_count})\n",
            "## Baseline comparison (ARCHITECTURE.md section 6, replayed traffic)\n",
            "| baseline | queries | mean cost/query (USD) | total cost (USD) | mean LLM calls | cache hit rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for summary in self.baseline_summaries:
            lines.append(
                f"| {summary.name} | {summary.query_count} | {summary.mean_cost_usd:.6f} | "
                f"{summary.total_cost_usd:.6f} | {summary.mean_llm_calls:.2f} | {summary.cache_hit_rate:.1%} |"
            )
        lines += [
            "",
            f"- full-system savings vs. no-system baseline: {savings_vs_no_system:.1%}",
            f"- verifier's added cost per query (full_system - cache_router_no_verifier): "
            f"${verifier_cost_delta:.6f}\n",
            "## Quality-regression spot check\n",
            f"- comparable-to-strong-model rate: {self.quality_spot_check.comparable_rate:.1%}",
            f"- responses spot-checked: {len(self.quality_spot_check.items)}\n",
        ]
        if self.cache_reuse_raw_results:
            reuse = self.cache_reuse_summaries
            reuse_no_system, reuse_full_system = reuse[NO_SYSTEM], reuse[FULL_SYSTEM]
            lines += [
                "## Cache-reuse benchmark (synthetic, illustrative -- see eval.cache_reuse_benchmark)\n",
                "Seeded true_duplicate pairs from the labeled adversarial eval set, each asked as two "
                "queries (original, then paraphrase); full_system uses a fresh, isolated cache per pair "
                "so no pair's cached answer can affect another pair's result.\n",
                "| system | queries | mean cost/query (USD) | cache hit rate |",
                "|---|---:|---:|---:|",
                f"| no_system | {reuse_no_system.query_count} | {reuse_no_system.mean_cost_usd:.6f} | "
                f"{reuse_no_system.cache_hit_rate:.1%} |",
                f"| full_system | {reuse_full_system.query_count} | {reuse_full_system.mean_cost_usd:.6f} | "
                f"{reuse_full_system.cache_hit_rate:.1%} |",
                "",
                f"- cache-reuse savings vs. no-system: {self.cache_reuse_savings_pct:.1%}\n",
            ]
        return "\n".join(lines)

    def write(self, json_path: Path, markdown_path: Path) -> None:
        json_path.write_text(self.to_json() + "\n", encoding="utf-8")
        markdown_path.write_text(self.to_markdown() + "\n", encoding="utf-8")


def load_eval_report(path: Path) -> EvalReport:
    """Reconstruct an EvalReport from a JSON file written by `EvalReport.write`.

    The counterpart to `to_json`/`write`: rebuilds the real dataclasses
    (not plain dicts) so consumers -- namely the Phase 6 dashboard --
    get the same computed properties (precision, recall, catch rates,
    baseline_summaries) as the process that originally produced the
    report, instead of re-deriving that logic separately.

    Also reads pre-Phase-6 reports (`baseline_summaries`, no
    `baseline_raw_results`) without raising: `baseline_raw_results` comes
    back empty -- there's no per-query data in that schema to reconstruct,
    so per-query views (cumulative cost/hit-rate, path distribution)
    correctly render as "no data" rather than fabricating it -- but the
    real aggregate numbers the old report already computed are preserved
    via `stored_baseline_summaries` so the baseline comparison table still
    shows actual figures instead of zeros.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    cache_eval = CacheEvalResult(outcomes=tuple(CachePairOutcome(**o) for o in raw["cache_eval"]["outcomes"]))
    router_eval = RouterEvalResult(
        outcomes=tuple(ComplexityItemOutcome(**o) for o in raw["router_eval"]["outcomes"])
    )
    cache_verifier_eval = CacheVerifierEvalResult(
        reached_verifier=tuple(CacheVerifierOutcome(**o) for o in raw["cache_verifier_eval"]["reached_verifier"]),
        skipped=tuple(SkippedCachePair(**s) for s in raw["cache_verifier_eval"]["skipped"]),
    )
    route_verifier_eval = RouteVerifierEvalResult(
        misrouted=tuple(RouteVerifierOutcome(**o) for o in raw["route_verifier_eval"]["misrouted"]),
        correctly_routed_count=raw["route_verifier_eval"]["correctly_routed_count"],
    )
    quality_spot_check = QualitySpotCheckResult(
        items=tuple(QualitySpotCheckItem(**i) for i in raw["quality_spot_check"]["items"])
    )

    if "baseline_raw_results" in raw:
        baseline_raw_results = {
            name: tuple(BaselineResult(**r) for r in results)
            for name, results in raw["baseline_raw_results"].items()
        }
        stored_baseline_summaries = None
    else:
        baseline_raw_results = {}
        summaries_by_name = {s["name"]: BaselineSummary(**s) for s in raw.get("baseline_summaries", [])}
        stored_baseline_summaries = tuple(
            summaries_by_name.get(name, summarize_baseline(name, ())) for name in BASELINE_NAMES
        )

    cache_reuse_raw_results = {
        name: tuple(BaselineResult(**r) for r in results)
        for name, results in raw.get("cache_reuse_raw_results", {}).items()
    }

    return EvalReport(
        cache_eval=cache_eval,
        router_eval=router_eval,
        cache_verifier_eval=cache_verifier_eval,
        route_verifier_eval=route_verifier_eval,
        baseline_raw_results=baseline_raw_results,
        quality_spot_check=quality_spot_check,
        stored_baseline_summaries=stored_baseline_summaries,
        cache_reuse_raw_results=cache_reuse_raw_results,
    )
