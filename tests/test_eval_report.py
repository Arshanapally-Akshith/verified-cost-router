"""Unit tests for verified_cost_router.eval.report."""

from __future__ import annotations

import json
from pathlib import Path

from verified_cost_router.eval.baselines import BaselineResult
from verified_cost_router.eval.cache_eval import CacheEvalResult, CachePairOutcome
from verified_cost_router.eval.quality_eval import QualitySpotCheckItem, QualitySpotCheckResult
from verified_cost_router.eval.report import EvalReport, load_eval_report, summarize_baseline
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

    baseline_raw_results = {
        "no_system": (BaselineResult("q", "a", 0.001, 1, False, True, "no_system"),),
        "cache_router_no_verifier": (BaselineResult("q", "a", 0.0005, 1, False, False, "router_cheap"),),
        "full_system": (BaselineResult("q", "a", 0.0007, 2, False, False, "router-8B"),),
    }

    quality = QualitySpotCheckResult(
        items=(QualitySpotCheckItem("q1", "served", "reference", "comparable"),)
    )

    return EvalReport(
        cache_eval=cache_eval,
        router_eval=router_eval,
        cache_verifier_eval=cache_verifier_eval,
        route_verifier_eval=route_verifier_eval,
        baseline_raw_results=baseline_raw_results,
        quality_spot_check=quality,
    )


def test_summarize_baseline_computes_totals_and_means():
    results = [
        BaselineResult("q1", "a1", 0.01, 2, False, True, "no_system"),
        BaselineResult("q2", "a2", 0.02, 1, True, False, "cache_hit"),
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
    assert set(parsed["baseline_raw_results"].keys()) == {"no_system", "cache_router_no_verifier", "full_system"}
    assert parsed["baseline_raw_results"]["no_system"][0]["query"] == "q"
    assert parsed["quality_spot_check"]["items"][0]["verdict"] == "comparable"


def test_baseline_summaries_is_derived_from_raw_results_not_stored_separately():
    report = _sample_report()
    summaries = {s.name: s for s in report.baseline_summaries}

    assert summaries["no_system"].query_count == 1
    assert summaries["no_system"].mean_cost_usd == 0.001
    assert summaries["full_system"].mean_llm_calls == 2


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


def test_load_eval_report_round_trips_write(tmp_path: Path):
    original = _sample_report()
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    original.write(json_path, md_path)

    loaded = load_eval_report(json_path)

    assert loaded.cache_eval == original.cache_eval
    assert loaded.router_eval == original.router_eval
    assert loaded.cache_verifier_eval == original.cache_verifier_eval
    assert loaded.route_verifier_eval == original.route_verifier_eval
    assert loaded.baseline_raw_results == original.baseline_raw_results
    assert loaded.quality_spot_check == original.quality_spot_check
    # Computed properties must still work after reconstruction, not just
    # the stored fields -- this is the whole point of round-tripping into
    # real dataclasses instead of leaving the dashboard with raw dicts.
    assert loaded.cache_eval.precision == original.cache_eval.precision
    assert loaded.baseline_summaries == original.baseline_summaries
    assert loaded.to_markdown() == original.to_markdown()
