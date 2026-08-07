"""Unit tests for verified_cost_router.eval.verifier_eval."""

from __future__ import annotations

from fakes import (
    FakeChatCompletionClient,
    ScriptedClassifier,
    ScriptedEmbedder,
    ScriptedVerifier,
    unit_vectors_with_similarity,
)

from verified_cost_router.cache.thresholds import CacheThresholds
from verified_cost_router.data_prep.adversarial_eval import CachePair, ComplexityItem
from verified_cost_router.eval.verifier_eval import evaluate_cache_verifier, evaluate_route_verifier

THRESHOLDS = CacheThresholds(high_confidence=0.9, risky=0.7)


def _cache_pair(id_: str, category: str, similarity: float):
    vec_a, vec_b = unit_vectors_with_similarity(similarity)
    pair = CachePair(
        id=id_, category=category, query_a=f"{id_}-a", query_b=f"{id_}-b",
        expect_cache_hit=(category == "true_duplicate"), rationale="r",
    )
    return pair, {pair.query_a: vec_a, pair.query_b: vec_b}


# --- evaluate_cache_verifier -------------------------------------------------


def test_near_miss_in_risky_band_correctly_flagged_fail_counts_as_caught():
    pair, vectors = _cache_pair("nm1", "near_miss", 0.8)  # risky band
    embedder = ScriptedEmbedder(vectors, dim=2)
    verifier = ScriptedVerifier(cache_hit_labels={("nm1-b", "nm1-a"): "fail"})

    result = evaluate_cache_verifier([pair], embedder, THRESHOLDS, verifier)

    assert len(result.reached_verifier) == 1
    assert result.reached_verifier[0].correctly_flagged is True
    assert result.near_miss_catch_rate == 1.0


def test_near_miss_in_risky_band_wrongly_passed_is_not_caught():
    pair, vectors = _cache_pair("nm1", "near_miss", 0.8)
    embedder = ScriptedEmbedder(vectors, dim=2)
    verifier = ScriptedVerifier(cache_hit_labels={("nm1-b", "nm1-a"): "pass"})

    result = evaluate_cache_verifier([pair], embedder, THRESHOLDS, verifier)

    assert result.reached_verifier[0].correctly_flagged is False
    assert result.near_miss_catch_rate == 0.0


def test_true_duplicate_in_risky_band_correctly_passed():
    pair, vectors = _cache_pair("dup1", "true_duplicate", 0.8)
    embedder = ScriptedEmbedder(vectors, dim=2)
    verifier = ScriptedVerifier(cache_hit_labels={("dup1-b", "dup1-a"): "pass"})

    result = evaluate_cache_verifier([pair], embedder, THRESHOLDS, verifier)

    assert result.reached_verifier[0].correctly_flagged is True
    assert result.true_duplicate_pass_rate == 1.0


def test_no_match_pairs_are_skipped_not_counted_as_failures():
    pair, vectors = _cache_pair("nm1", "near_miss", 0.3)  # below risky
    embedder = ScriptedEmbedder(vectors, dim=2)
    verifier = ScriptedVerifier()

    result = evaluate_cache_verifier([pair], embedder, THRESHOLDS, verifier)

    assert result.reached_verifier == ()
    assert [s.pair_id for s in result.skipped] == ["nm1"]
    assert result.skipped[0].reason == "no_match"
    assert verifier.cache_hit_calls == []  # verifier never invoked
    assert result.near_miss_catch_rate == 1.0  # vacuous: nothing reached it
    assert result.near_miss_high_confidence_leaks == 0


def test_high_confidence_near_miss_is_skipped_as_a_leak_not_a_verifier_failure():
    pair, vectors = _cache_pair("nm1", "near_miss", 0.95)  # above high_confidence
    embedder = ScriptedEmbedder(vectors, dim=2)
    verifier = ScriptedVerifier()

    result = evaluate_cache_verifier([pair], embedder, THRESHOLDS, verifier)

    assert result.reached_verifier == ()
    assert [s.pair_id for s in result.skipped] == ["nm1"]
    assert result.skipped[0].reason == "high_confidence_hit"
    assert result.near_miss_high_confidence_leaks == 1
    assert verifier.cache_hit_calls == []


def test_high_confidence_true_duplicate_is_skipped_but_not_counted_as_a_leak():
    pair, vectors = _cache_pair("dup1", "true_duplicate", 0.95)
    embedder = ScriptedEmbedder(vectors, dim=2)
    verifier = ScriptedVerifier()

    result = evaluate_cache_verifier([pair], embedder, THRESHOLDS, verifier)

    # A true_duplicate landing in high_confidence_hit is a correct, silent
    # auto-serve -- it must still be accounted for in `skipped` (this used
    # to be silently dropped entirely), but not counted as a "leak" (that
    # term is reserved for near_miss pairs slipping past verification).
    assert result.reached_verifier == ()
    assert [s.pair_id for s in result.skipped] == ["dup1"]
    assert result.skipped[0].reason == "high_confidence_hit"
    assert result.near_miss_high_confidence_leaks == 0


def test_every_pair_is_accounted_for_exactly_once():
    pairs_and_vectors = [
        _cache_pair("dup_high", "true_duplicate", 0.95),
        _cache_pair("dup_risky", "true_duplicate", 0.8),
        _cache_pair("dup_miss", "true_duplicate", 0.3),
        _cache_pair("nm_high", "near_miss", 0.95),
        _cache_pair("nm_risky", "near_miss", 0.8),
        _cache_pair("nm_miss", "near_miss", 0.3),
    ]
    pairs = [p for p, _ in pairs_and_vectors]
    vectors = {}
    for _, v in pairs_and_vectors:
        vectors.update(v)
    embedder = ScriptedEmbedder(vectors, dim=2)
    verifier = ScriptedVerifier()

    result = evaluate_cache_verifier(pairs, embedder, THRESHOLDS, verifier)

    accounted_ids = {o.pair_id for o in result.reached_verifier} | {s.pair_id for s in result.skipped}
    assert accounted_ids == {p.id for p in pairs}
    assert len(result.reached_verifier) + len(result.skipped) == len(pairs)


# --- evaluate_route_verifier --------------------------------------------------


def _complexity_item(id_: str, query: str) -> ComplexityItem:
    return ComplexityItem(id=id_, category="complexity_mislabeled", query=query, true_complexity="complex", rationale="r")


def test_correctly_routed_item_never_reaches_generation_or_verifier():
    items = [_complexity_item("cx1", "q1")]
    classifier = ScriptedClassifier({"q1": "complex"})
    groq_client = FakeChatCompletionClient(next_content="should not be called")
    verifier = ScriptedVerifier()

    result = evaluate_route_verifier(items, classifier, groq_client, "cheap-model", verifier)

    assert result.misrouted == ()
    assert result.correctly_routed_count == 1
    assert groq_client.call_count == 0
    assert verifier.output_calls == []
    assert result.catch_rate == 1.0  # vacuous: nothing was misrouted


def test_misrouted_item_correctly_flagged_by_verifier_counts_as_caught():
    items = [_complexity_item("cx1", "q1")]
    classifier = ScriptedClassifier({"q1": "simple"})
    groq_client = FakeChatCompletionClient(next_content="a cheap answer")
    verifier = ScriptedVerifier(output_labels={"q1": "fail"})

    result = evaluate_route_verifier(items, classifier, groq_client, "cheap-model", verifier)

    assert len(result.misrouted) == 1
    assert result.misrouted[0].correctly_flagged is True
    assert result.catch_rate == 1.0
    assert groq_client.last_model == "cheap-model"


def test_misrouted_item_wrongly_passed_is_not_caught():
    items = [_complexity_item("cx1", "q1")]
    classifier = ScriptedClassifier({"q1": "simple"})
    groq_client = FakeChatCompletionClient(next_content="a cheap answer")
    verifier = ScriptedVerifier(output_labels={"q1": "pass"})

    result = evaluate_route_verifier(items, classifier, groq_client, "cheap-model", verifier)

    assert result.misrouted[0].correctly_flagged is False
    assert result.catch_rate == 0.0


def test_mixed_routing_outcomes_computed_correctly():
    items = [_complexity_item("cx1", "q1"), _complexity_item("cx2", "q2"), _complexity_item("cx3", "q3")]
    classifier = ScriptedClassifier({"q1": "complex", "q2": "simple", "q3": "simple"})
    groq_client = FakeChatCompletionClient(next_content="answer")
    verifier = ScriptedVerifier(output_labels={"q2": "fail", "q3": "pass"})

    result = evaluate_route_verifier(items, classifier, groq_client, "cheap-model", verifier)

    assert result.correctly_routed_count == 1
    assert len(result.misrouted) == 2
    assert result.catch_rate == 0.5
