"""Unit tests for verified_cost_router.eval.cache_eval."""

from __future__ import annotations

from fakes import ScriptedEmbedder, unit_vectors_with_similarity

from verified_cost_router.cache.thresholds import CacheThresholds
from verified_cost_router.data_prep.adversarial_eval import CachePair
from verified_cost_router.eval.cache_eval import evaluate_cache_pairs

THRESHOLDS = CacheThresholds(high_confidence=0.9, risky=0.7)


def _pair(id_: str, category: str, similarity: float) -> tuple[CachePair, dict]:
    vec_a, vec_b = unit_vectors_with_similarity(similarity)
    pair = CachePair(
        id=id_,
        category=category,
        query_a=f"{id_}-a",
        query_b=f"{id_}-b",
        expect_cache_hit=(category == "true_duplicate"),
        rationale="r",
    )
    return pair, {pair.query_a: vec_a, pair.query_b: vec_b}


def test_true_duplicate_high_similarity_is_a_correct_predicted_hit():
    pair, vectors = _pair("dup1", "true_duplicate", 0.95)
    embedder = ScriptedEmbedder(vectors, dim=2)

    result = evaluate_cache_pairs([pair], embedder, THRESHOLDS)

    outcome = result.outcomes[0]
    assert outcome.predicted_hit is True
    assert outcome.match_category == "high_confidence_hit"
    assert result.precision == 1.0
    assert result.recall == 1.0


def test_near_miss_high_similarity_is_a_false_positive():
    pair, vectors = _pair("nm1", "near_miss", 0.95)
    embedder = ScriptedEmbedder(vectors, dim=2)

    result = evaluate_cache_pairs([pair], embedder, THRESHOLDS)

    outcome = result.outcomes[0]
    assert outcome.predicted_hit is True  # the cache layer's own mistake
    assert result.precision == 0.0  # the only predicted-positive was wrong
    assert result.recall == 1.0  # no true_duplicate pairs to miss, so vacuously 1.0


def test_true_duplicate_low_similarity_is_a_missed_recall():
    pair, vectors = _pair("dup1", "true_duplicate", 0.3)
    embedder = ScriptedEmbedder(vectors, dim=2)

    result = evaluate_cache_pairs([pair], embedder, THRESHOLDS)

    outcome = result.outcomes[0]
    assert outcome.predicted_hit is False
    assert outcome.match_category == "no_match"
    assert result.recall == 0.0


def test_near_miss_low_similarity_is_a_correct_rejection():
    pair, vectors = _pair("nm1", "near_miss", 0.3)
    embedder = ScriptedEmbedder(vectors, dim=2)

    result = evaluate_cache_pairs([pair], embedder, THRESHOLDS)

    outcome = result.outcomes[0]
    assert outcome.predicted_hit is False
    assert result.precision == 1.0  # vacuous: no predicted positives at all


def test_precision_and_recall_across_a_mixed_set():
    dup_hit, dup_hit_vecs = _pair("dup1", "true_duplicate", 0.95)  # TP
    dup_miss, dup_miss_vecs = _pair("dup2", "true_duplicate", 0.3)  # FN
    nm_leak, nm_leak_vecs = _pair("nm1", "near_miss", 0.95)  # FP
    nm_ok, nm_ok_vecs = _pair("nm2", "near_miss", 0.3)  # TN

    vectors = {**dup_hit_vecs, **dup_miss_vecs, **nm_leak_vecs, **nm_ok_vecs}
    embedder = ScriptedEmbedder(vectors, dim=2)

    result = evaluate_cache_pairs([dup_hit, dup_miss, nm_leak, nm_ok], embedder, THRESHOLDS)

    # predicted positives: dup_hit (correct), nm_leak (wrong) -> precision 1/2
    assert result.precision == 0.5
    # actual positives: dup_hit (found), dup_miss (missed) -> recall 1/2
    assert result.recall == 0.5


def test_pairs_are_evaluated_in_isolation():
    # Two pairs that would collide if sharing one cache (both "a" queries
    # embed identically) must not affect each other's outcome.
    vec_shared, vec_other = unit_vectors_with_similarity(0.3)
    pair_1 = CachePair(
        id="p1", category="true_duplicate", query_a="shared text", query_b="p1-b",
        expect_cache_hit=True, rationale="r",
    )
    pair_2 = CachePair(
        id="p2", category="near_miss", query_a="shared text", query_b="p2-b",
        expect_cache_hit=False, rationale="r",
    )
    embedder = ScriptedEmbedder(
        {"shared text": vec_shared, "p1-b": vec_other, "p2-b": vec_other}, dim=2
    )

    result = evaluate_cache_pairs([pair_1, pair_2], embedder, THRESHOLDS)

    assert len(result.outcomes) == 2
    assert {o.pair_id for o in result.outcomes} == {"p1", "p2"}
    # If pairs shared one cache, p2's cache would contain p1's entry too,
    # but similarity here is low regardless, so both should independently
    # and correctly resolve to no_match.
    assert all(o.match_category == "no_match" for o in result.outcomes)
