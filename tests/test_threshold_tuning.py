"""Unit tests for verified_cost_router.cache.threshold_tuning.

evaluate_candidate/sweep_thresholds are pure functions over precomputed
similarities, so their scoring/selection logic is tested with crafted
similarity dicts -- no embedder needed. compute_pair_similarities is
tested separately with a ScriptedEmbedder for exact control.
"""

from __future__ import annotations

import pytest
from fakes import ScriptedEmbedder, unit_vectors_with_similarity

from verified_cost_router.cache.thresholds import CacheThresholds
from verified_cost_router.cache.threshold_tuning import (
    compute_pair_similarities,
    evaluate_candidate,
    sweep_thresholds,
)
from verified_cost_router.data_prep.adversarial_eval import CachePair


def _pair(id_: str, category: str) -> CachePair:
    return CachePair(
        id=id_,
        category=category,
        query_a=f"{id_}-a",
        query_b=f"{id_}-b",
        expect_cache_hit=(category == "true_duplicate"),
        rationale="r",
    )


def test_compute_pair_similarities_uses_dot_product_of_normalized_vectors():
    vec_a, vec_b = unit_vectors_with_similarity(0.42)
    pair = _pair("p1", "true_duplicate")
    embedder = ScriptedEmbedder({pair.query_a: vec_a, pair.query_b: vec_b}, dim=2)

    similarities = compute_pair_similarities([pair], embedder)

    assert similarities["p1"] == pytest.approx(0.42, abs=1e-5)


def test_compute_pair_similarities_empty_input():
    embedder = ScriptedEmbedder({}, dim=2)
    assert compute_pair_similarities([], embedder) == {}


def test_evaluate_candidate_precision_and_recall():
    pairs = [
        _pair("dup1", "true_duplicate"),  # sim 0.95 -> high_confidence_hit
        _pair("dup2", "true_duplicate"),  # sim 0.80 -> risky_hit
        _pair("dup3", "true_duplicate"),  # sim 0.50 -> no_match (missed)
        _pair("nm1", "near_miss"),  # sim 0.95 -> high_confidence_hit (a leak!)
        _pair("nm2", "near_miss"),  # sim 0.60 -> no_match
    ]
    similarities = {"dup1": 0.95, "dup2": 0.8, "dup3": 0.5, "nm1": 0.95, "nm2": 0.6}
    thresholds = CacheThresholds(high_confidence=0.9, risky=0.7)

    evaluation = evaluate_candidate(pairs, similarities, thresholds)

    assert evaluation.high_confidence_precision == pytest.approx(0.5)  # dup1 TP, nm1 FP
    assert evaluation.recall == pytest.approx(2 / 3)  # dup1, dup2 detected out of 3
    assert evaluation.near_miss_leaks == 1
    assert evaluation.true_duplicate_missed == 1


def test_evaluate_candidate_precision_defaults_to_one_with_no_high_confidence_hits():
    pairs = [_pair("dup1", "true_duplicate")]
    similarities = {"dup1": 0.5}
    thresholds = CacheThresholds(high_confidence=0.9, risky=0.3)

    evaluation = evaluate_candidate(pairs, similarities, thresholds)

    assert evaluation.high_confidence_precision == 1.0
    assert evaluation.recall == 1.0


def test_sweep_thresholds_prefers_higher_precision_meeting_recall_floor():
    # dup1/dup2 sit just under 0.90; nm1 sits at 0.93. A high_confidence
    # cutoff of 0.95 keeps nm1 out of the auto-serve tier (precision 1.0);
    # a cutoff of 0.90 lets it leak in (precision 0.0). Both meet recall=1.0.
    pairs = [
        _pair("dup1", "true_duplicate"),
        _pair("dup2", "true_duplicate"),
        _pair("nm1", "near_miss"),
    ]
    similarities = {"dup1": 0.86, "dup2": 0.85, "nm1": 0.93}
    grid = (0.80, 0.90, 0.95)

    evaluations = sweep_thresholds(pairs, similarities, min_recall=1.0, grid=grid)

    best = evaluations[0]
    assert best.thresholds.high_confidence == 0.95
    assert best.thresholds.risky == 0.80
    assert best.high_confidence_precision == 1.0


def test_sweep_thresholds_raises_when_no_candidate_meets_min_recall():
    pairs = [_pair("dup1", "true_duplicate")]
    similarities = {"dup1": 0.5}
    with pytest.raises(ValueError, match="minimum recall"):
        sweep_thresholds(pairs, similarities, min_recall=1.0, grid=(0.9, 0.95))
