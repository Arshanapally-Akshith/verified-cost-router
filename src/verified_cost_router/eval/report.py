"""Aggregates every Phase 5 eval result into one report, rendered as
JSON (machine-readable, for Phase 6's dashboard) and markdown (human
summary) -- BUILD.md section 5: "Eval harness produces the precision/
recall + cost numbers... from a single script run."
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from verified_cost_router.eval.baselines import BaselineResult
from verified_cost_router.eval.cache_eval import CacheEvalResult
from verified_cost_router.eval.quality_eval import QualitySpotCheckResult
from verified_cost_router.eval.router_eval import RouterEvalResult
from verified_cost_router.eval.verifier_eval import CacheVerifierEvalResult, RouteVerifierEvalResult


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
        return BaselineSummary(name=name, query_count=0, total_cost_usd=0.0, mean_cost_usd=0.0, mean_llm_calls=0.0, cache_hit_rate=0.0)
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
    # Must be exactly (no_system, cache_router_no_verifier, full_system), in
    # that order (ARCHITECTURE.md section 6) -- to_markdown() unpacks them
    # positionally to compute the savings/verifier-cost deltas below.
    baseline_summaries: tuple[BaselineSummary, BaselineSummary, BaselineSummary]
    quality_spot_check: QualitySpotCheckResult

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_markdown(self) -> str:
        no_system, cache_router, full_system = self.baseline_summaries
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
        return "\n".join(lines)

    def write(self, json_path: Path, markdown_path: Path) -> None:
        json_path.write_text(self.to_json() + "\n", encoding="utf-8")
        markdown_path.write_text(self.to_markdown() + "\n", encoding="utf-8")
