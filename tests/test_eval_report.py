"""Unit tests for verified_cost_router.eval.report."""

from __future__ import annotations

import json
from pathlib import Path

from verified_cost_router.eval.baselines import BaselineResult
from verified_cost_router.eval.cache_eval import CacheEvalResult, CachePairOutcome
from verified_cost_router.eval.quality_eval import QualitySpotCheckItem, QualitySpotCheckResult
from verified_cost_router.eval.report import EvalReport, summarize_baseline
from verified_cost_router.eval.router_eval import ComplexityItemOutcome, RouterEvalResult
from verified_cost_router.eval.verifier_eval import (
    CacheVerifierEvalResult,
    CacheVerifierOutcome,
    RouteVerifierEvalResult,
)


def _sample_report() -> EvalReport:
    cache_eval = CacheEvalResult(
        outcomes=(
            CachePairOutcome("dup1", "true_duplicate", True, True, "high_confidence_hit"),
            CachePairOutcome("nm1", "near_miss", False, False, "no_match"),
        )
    )
    router_eval = RouterEvalResult(outcomes=(ComplexityItemOutcome("cx1", "q1", "complex", True),))
    cache_verifier_eval = CacheVerifierEvalResult(
        reached_verifier=(CacheVerifierOutcome("nm2", "near_miss", "risky_hit", "fail", True),),
        skipped=(),
    )
    route_verifier_eval = RouteVerifierEvalResult(misrouted=(), correctly_routed_count=1)

    no_system = summarize_baseline("no_system", [BaselineResult("q", "a", 0.001, 1, False, True)])
    cache_router = summarize_baseline("cache_router_no_verifier", [BaselineResult("q", "a", 0.0005, 1, False, False)])
    full_system = summarize_baseline("full_system", [BaselineResult("q", "a", 0.0007, 2, False, False)])

    quality = QualitySpotCheckResult(
        items=(QualitySpotCheckItem("q1", "served", "reference", "comparable"),)
    )

    return EvalReport(
        cache_eval=cache_eval,
        router_eval=router_eval,
        cache_verifier_eval=cache_verifier_eval,
        route_verifier_eval=route_verifier_eval,
        baseline_summaries=(no_system, cache_router, full_system),
        quality_spot_check=quality,
    )


def test_summarize_baseline_computes_totals_and_means():
    results = [
        BaselineResult("q1", "a1", 0.01, 2, False, True),
        BaselineResult("q2", "a2", 0.02, 1, True, False),
    ]
    summary = summarize_baseline("test", results)

    assert summary.query_count == 2
    assert summary.total_cost_usd == 0.03
    assert summary.mean_cost_usd == 0.015
    assert summary.mean_llm_calls == 1.5
    assert summary.cache_hit_rate == 0.5


def test_summarize_baseline_handles_empty_results():
    summary = summarize_baseline("empty", [])
    assert summary.query_count == 0
    assert summary.total_cost_usd == 0.0
    assert summary.cache_hit_rate == 0.0


def test_to_json_round_trips_through_json_loads():
    report = _sample_report()
    parsed = json.loads(report.to_json())

    assert parsed["cache_eval"]["outcomes"][0]["pair_id"] == "dup1"
    assert len(parsed["baseline_summaries"]) == 3
    assert parsed["quality_spot_check"]["items"][0]["verdict"] == "comparable"


def test_to_markdown_includes_key_sections_and_numbers():
    report = _sample_report()
    markdown = report.to_markdown()

    assert "# Eval report" in markdown
    assert "Cache precision/recall" in markdown
    assert "Router accuracy" in markdown
    assert "Verifier catch rate" in markdown
    assert "Baseline comparison" in markdown
    assert "Quality-regression spot check" in markdown
    assert "no_system" in markdown
    assert "full_system" in markdown


def test_write_creates_both_files(tmp_path: Path):
    report = _sample_report()
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    report.write(json_path, md_path)

    assert json_path.exists()
    assert md_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["router_eval"]["outcomes"][0]["item_id"] == "cx1"
