"""Unit tests for verified_cost_router.eval.report."""

from __future__ import annotations

import json
from dataclasses import asdict
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


def _sample_report(
    cache_reuse_raw_results: dict | None = None,
    cache_reuse_population_size: int = 0,
    cache_reuse_sample_pct: float = 0.0,
    cache_reuse_seed: int = 0,
) -> EvalReport:
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
        cache_reuse_raw_results=cache_reuse_raw_results or {},
        cache_reuse_population_size=cache_reuse_population_size,
        cache_reuse_sample_pct=cache_reuse_sample_pct,
        cache_reuse_seed=cache_reuse_seed,
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


def test_load_eval_report_reads_pre_phase6_schema_without_raising(tmp_path: Path):
    """Regression test: a report written before Phase 6 has aggregate
    `baseline_summaries` instead of per-query `baseline_raw_results`.
    load_eval_report must not KeyError on it -- the dashboard needs to
    keep working against reports that predate per-query tracking."""
    original = _sample_report()
    raw = json.loads(original.to_json())
    del raw["baseline_raw_results"]
    raw["baseline_summaries"] = [asdict(s) for s in original.baseline_summaries]
    json_path = tmp_path / "legacy_report.json"
    json_path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = load_eval_report(json_path)

    # No per-query data exists in this schema -- must come back empty, not fabricated.
    assert loaded.baseline_raw_results == {}
    # But the real aggregate numbers the old report already computed are preserved.
    assert loaded.baseline_summaries == original.baseline_summaries
    # Every other section is unaffected by the legacy baseline schema.
    assert loaded.cache_eval == original.cache_eval
    assert loaded.router_eval == original.router_eval
    assert loaded.quality_spot_check == original.quality_spot_check


def test_cache_reuse_raw_results_defaults_to_empty_and_stays_out_of_baselines():
    """Reports without a cache-reuse benchmark (every report before this
    feature existed) must keep working unchanged -- default empty, and
    never merged into baseline_raw_results / baseline_summaries."""
    report = _sample_report()

    assert report.cache_reuse_raw_results == {}
    assert report.cache_reuse_summaries["no_system"].query_count == 0
    assert report.cache_reuse_summaries["full_system"].query_count == 0
    assert report.cache_reuse_savings_pct == 0.0
    assert {s.name for s in report.baseline_summaries} == {"no_system", "cache_router_no_verifier", "full_system"}


def _reuse_results():
    return {
        "no_system": (
            BaselineResult("qa", "a", 0.004, 1, False, True, "no_system"),
            BaselineResult("qb", "a", 0.004, 1, False, True, "no_system"),
        ),
        "full_system": (
            BaselineResult("qa", "a", 0.002, 2, False, False, "router-8B"),
            BaselineResult("qb", "a", 0.0, 0, True, False, "cache-hit"),
        ),
    }


def test_cache_reuse_summaries_computed_per_system():
    report = _sample_report(cache_reuse_raw_results=_reuse_results())

    summaries = report.cache_reuse_summaries

    assert summaries["no_system"].query_count == 2
    assert summaries["no_system"].mean_cost_usd == 0.004
    assert summaries["full_system"].query_count == 2
    assert summaries["full_system"].cache_hit_rate == 0.5
    assert summaries["full_system"].mean_cost_usd == 0.001


def test_cache_reuse_savings_pct_matches_no_system_vs_full_system():
    report = _sample_report(cache_reuse_raw_results=_reuse_results())

    # no_system mean = 0.004, full_system mean = 0.001 -> 75% cheaper.
    assert report.cache_reuse_savings_pct == (0.004 - 0.001) / 0.004


def test_cache_reuse_savings_pct_is_zero_when_no_system_has_no_cost():
    report = _sample_report()  # empty cache_reuse_raw_results -> no_system mean cost is 0
    assert report.cache_reuse_savings_pct == 0.0


def test_to_json_round_trips_cache_reuse_raw_results():
    report = _sample_report(cache_reuse_raw_results=_reuse_results())

    parsed = json.loads(report.to_json())

    assert set(parsed["cache_reuse_raw_results"].keys()) == {"no_system", "full_system"}
    assert parsed["cache_reuse_raw_results"]["no_system"][0]["query"] == "qa"


def test_load_eval_report_round_trips_cache_reuse_raw_results(tmp_path: Path):
    original = _sample_report(cache_reuse_raw_results=_reuse_results())
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    original.write(json_path, md_path)

    loaded = load_eval_report(json_path)

    assert loaded.cache_reuse_raw_results == original.cache_reuse_raw_results
    assert loaded.cache_reuse_summaries == original.cache_reuse_summaries
    assert loaded.cache_reuse_savings_pct == original.cache_reuse_savings_pct


def test_load_eval_report_defaults_cache_reuse_raw_results_when_absent(tmp_path: Path):
    """A report from before this feature existed has no cache_reuse_raw_results
    key at all -- must load as empty, not raise."""
    original = _sample_report()
    raw = json.loads(original.to_json())
    del raw["cache_reuse_raw_results"]
    json_path = tmp_path / "report.json"
    json_path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = load_eval_report(json_path)

    assert loaded.cache_reuse_raw_results == {}


def test_to_markdown_includes_cache_reuse_section_only_when_present():
    without = _sample_report()
    assert "Cache-reuse benchmark" not in without.to_markdown()

    with_reuse = _sample_report(cache_reuse_raw_results=_reuse_results())
    markdown = with_reuse.to_markdown()
    assert "Cache-reuse benchmark" in markdown
    assert "cache-reuse savings vs. no-system: 75.0%" in markdown


def test_to_markdown_documents_sampling_provenance():
    report = _sample_report(
        cache_reuse_raw_results=_reuse_results(),
        cache_reuse_population_size=50,
        cache_reuse_sample_pct=0.2,
        cache_reuse_seed=42,
    )
    markdown = report.to_markdown()

    assert "population (true_duplicate pairs available): 50" in markdown
    assert "sample percentage: 20%" in markdown
    assert "seed: 42" in markdown
    assert "sample size: 1 pairs (2 queries)" in markdown


def test_cache_reuse_pair_count_derived_from_raw_results():
    report = _sample_report(cache_reuse_raw_results=_reuse_results())
    assert report.cache_reuse_pair_count == 1  # 2 queries in full_system / 2


def test_load_eval_report_round_trips_sampling_provenance(tmp_path: Path):
    original = _sample_report(
        cache_reuse_raw_results=_reuse_results(),
        cache_reuse_population_size=50,
        cache_reuse_sample_pct=0.2,
        cache_reuse_seed=42,
    )
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    original.write(json_path, md_path)

    loaded = load_eval_report(json_path)

    assert loaded.cache_reuse_population_size == 50
    assert loaded.cache_reuse_sample_pct == 0.2
    assert loaded.cache_reuse_seed == 42


def test_load_eval_report_defaults_sampling_provenance_when_absent(tmp_path: Path):
    """A report from before percentage-based sampling has no
    cache_reuse_population_size/sample_pct/seed keys -- must load with
    the neutral 0 defaults, not raise."""
    original = _sample_report(cache_reuse_raw_results=_reuse_results())
    raw = json.loads(original.to_json())
    del raw["cache_reuse_population_size"]
    del raw["cache_reuse_sample_pct"]
    del raw["cache_reuse_seed"]
    json_path = tmp_path / "report.json"
    json_path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = load_eval_report(json_path)

    assert loaded.cache_reuse_population_size == 0
    assert loaded.cache_reuse_sample_pct == 0.0
    assert loaded.cache_reuse_seed == 0
