"""Unit tests for scripts/run_eval.py's cache-reuse orchestration
(run_cache_reuse_step, run_cache_reuse_only).

scripts/ isn't a package (it's standalone CLI entry points, not part of
the installed src/ package -- see pyproject.toml's packages.find scoped
to "src"), so the module is loaded directly by file path rather than
via a normal import.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest
from fakes import FakeChatCompletionClient, FakeClassifier, FakeEmbedder, FakeVerifier

from verified_cost_router.cache.thresholds import CacheThresholds
from verified_cost_router.config import GroqSettings
from verified_cost_router.data_prep.adversarial_eval import AdversarialEvalSet, CachePair
from verified_cost_router.eval.baselines import BaselineResult
from verified_cost_router.eval.cache_eval import CacheEvalResult, CachePairOutcome
from verified_cost_router.eval.quality_eval import QualitySpotCheckResult
from verified_cost_router.eval.report import EvalReport, load_eval_report
from verified_cost_router.eval.router_eval import RouterEvalResult
from verified_cost_router.eval.verifier_eval import CacheVerifierEvalResult, RouteVerifierEvalResult

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("run_eval_script", REPO_ROOT / "scripts" / "run_eval.py")
run_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_eval)

THRESHOLDS = CacheThresholds(high_confidence=0.9, risky=0.7)
# run_cache_reuse_step (scripts/run_eval.py) always prices with
# DEFAULT_PRICING, matching production (real Groq model ids) -- so these
# must be real DEFAULT_PRICING keys, not arbitrary fake names, or cost
# estimation raises KeyError.
GROQ_SETTINGS = GroqSettings(
    api_key="test-key", cheap_model="llama-3.1-8b-instant", strong_model="llama-3.3-70b-versatile"
)


def _pairs(n: int) -> tuple[CachePair, ...]:
    return tuple(
        CachePair(
            id=f"dup{i}",
            category="true_duplicate",
            query_a=f"dup{i} query a",
            query_b=f"dup{i} query b",
            expect_cache_hit=True,
            rationale="test fixture",
        )
        for i in range(n)
    )


def _eval_set(n_pairs: int) -> AdversarialEvalSet:
    return AdversarialEvalSet(cache_pairs=_pairs(n_pairs), complexity_items=())


def _classifier() -> FakeClassifier:
    return FakeClassifier(label="simple", model=GROQ_SETTINGS.cheap_model)


def _verifier() -> FakeVerifier:
    return FakeVerifier(model=GROQ_SETTINGS.cheap_model)


def _sample_report() -> EvalReport:
    return EvalReport(
        cache_eval=CacheEvalResult(
            outcomes=(CachePairOutcome("dup1", "true_duplicate", True, True, "high_confidence_hit"),)
        ),
        router_eval=RouterEvalResult(outcomes=()),
        cache_verifier_eval=CacheVerifierEvalResult(reached_verifier=(), skipped=()),
        route_verifier_eval=RouteVerifierEvalResult(misrouted=(), correctly_routed_count=0),
        baseline_raw_results={
            "no_system": (BaselineResult("q", "a", 0.001, 1, False, True, "no_system"),),
            "cache_router_no_verifier": (),
            "full_system": (),
        },
        quality_spot_check=QualitySpotCheckResult(items=()),
    )


# --- run_cache_reuse_step ------------------------------------------------------


def test_run_cache_reuse_step_returns_empty_dict_when_n_pairs_is_zero():
    result = run_eval.run_cache_reuse_step(
        _eval_set(3), 0, 42, FakeEmbedder(), THRESHOLDS, _classifier(), _verifier(),
        FakeChatCompletionClient(next_content="answer"), GROQ_SETTINGS,
    )
    assert result == {}


def test_run_cache_reuse_step_returns_no_system_and_full_system_keyed_results():
    result = run_eval.run_cache_reuse_step(
        _eval_set(3), 2, 42, FakeEmbedder(), THRESHOLDS, _classifier(), _verifier(),
        FakeChatCompletionClient(next_content="answer"), GROQ_SETTINGS,
    )

    assert set(result.keys()) == {"no_system", "full_system"}
    assert len(result["no_system"]) == 4  # 2 pairs x 2 queries
    assert len(result["full_system"]) == 4
    assert all(r.path_taken == "no_system" for r in result["no_system"])


# --- run_cache_reuse_only --------------------------------------------------


def _args(tmp_path: Path, cache_reuse_pairs: int = 2, seed: int = 42) -> argparse.Namespace:
    return argparse.Namespace(
        cache_reuse_pairs=cache_reuse_pairs,
        report_json=tmp_path / "eval_report.json",
        report_md=tmp_path / "eval_report.md",
        seed=seed,
    )


def test_run_cache_reuse_only_exits_if_pairs_not_positive(tmp_path: Path):
    args = _args(tmp_path, cache_reuse_pairs=0)

    with pytest.raises(SystemExit, match="cache-reuse-pairs"):
        run_eval.run_cache_reuse_only(
            args, _eval_set(3), FakeEmbedder(), THRESHOLDS, _classifier(), _verifier(),
            FakeChatCompletionClient(next_content="answer"), GROQ_SETTINGS,
        )


def test_run_cache_reuse_only_exits_if_report_json_missing(tmp_path: Path):
    args = _args(tmp_path)  # report_json points at a file that doesn't exist yet

    with pytest.raises(SystemExit, match="requires an existing report"):
        run_eval.run_cache_reuse_only(
            args, _eval_set(3), FakeEmbedder(), THRESHOLDS, _classifier(), _verifier(),
            FakeChatCompletionClient(next_content="answer"), GROQ_SETTINGS,
        )


def test_run_cache_reuse_only_preserves_every_other_section_unchanged(tmp_path: Path):
    original = _sample_report()
    args = _args(tmp_path, cache_reuse_pairs=2)
    original.write(args.report_json, args.report_md)

    run_eval.run_cache_reuse_only(
        args, _eval_set(3), FakeEmbedder(), THRESHOLDS, _classifier(), _verifier(),
        FakeChatCompletionClient(next_content="answer"), GROQ_SETTINGS,
    )

    updated = load_eval_report(args.report_json)
    # Everything except cache_reuse_raw_results is carried over byte-for-byte.
    assert updated.cache_eval == original.cache_eval
    assert updated.router_eval == original.router_eval
    assert updated.cache_verifier_eval == original.cache_verifier_eval
    assert updated.route_verifier_eval == original.route_verifier_eval
    assert updated.baseline_raw_results == original.baseline_raw_results
    assert updated.quality_spot_check == original.quality_spot_check
    # The cache-reuse benchmark itself was actually run and written.
    assert set(updated.cache_reuse_raw_results.keys()) == {"no_system", "full_system"}
    assert len(updated.cache_reuse_raw_results["no_system"]) == 4


def test_run_cache_reuse_only_does_not_touch_report_md_content_of_other_sections(tmp_path: Path):
    original = _sample_report()
    args = _args(tmp_path, cache_reuse_pairs=2)
    original.write(args.report_json, args.report_md)

    run_eval.run_cache_reuse_only(
        args, _eval_set(3), FakeEmbedder(), THRESHOLDS, _classifier(), _verifier(),
        FakeChatCompletionClient(next_content="answer"), GROQ_SETTINGS,
    )

    markdown = args.report_md.read_text(encoding="utf-8")
    assert "Cache-reuse benchmark" in markdown
    assert "Cache precision/recall" in markdown  # untouched sections still render
