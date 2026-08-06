"""Unit tests for verified_cost_router.cache.semantic_cache.SemanticCache.

Uses ScriptedEmbedder/FakeEmbedder test doubles (tests/fakes.py) so
similarity values and cosine-similarity mechanics are fully controlled
without loading a real embedding model.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fakes import FakeEmbedder, ScriptedEmbedder, unit_vectors_with_similarity

from verified_cost_router.cache.semantic_cache import SemanticCache
from verified_cost_router.cache.thresholds import CacheThresholds

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
THRESHOLDS = CacheThresholds(high_confidence=0.9, risky=0.7)


def _clock_at(t: datetime):
    return lambda: t


def test_lookup_on_empty_cache_is_no_match():
    cache = SemanticCache(FakeEmbedder(), THRESHOLDS)
    result = cache.lookup("anything")
    assert result.category == "no_match"
    assert result.match is None
    assert result.similarity is None


def test_high_similarity_lookup_is_high_confidence_hit():
    vec_a, vec_b = unit_vectors_with_similarity(0.95)
    embedder = ScriptedEmbedder({"original": vec_a, "duplicate": vec_b}, dim=2)
    cache = SemanticCache(embedder, THRESHOLDS)

    cache.put("original", "the answer")
    result = cache.lookup("duplicate")

    assert result.category == "high_confidence_hit"
    assert result.match is not None
    assert result.match.response == "the answer"
    assert result.similarity == pytest.approx(0.95, abs=1e-4)


def test_mid_similarity_lookup_is_risky_hit():
    vec_a, vec_b = unit_vectors_with_similarity(0.8)
    embedder = ScriptedEmbedder({"original": vec_a, "near_miss": vec_b}, dim=2)
    cache = SemanticCache(embedder, THRESHOLDS)

    cache.put("original", "the answer")
    result = cache.lookup("near_miss")

    assert result.category == "risky_hit"
    assert result.match is not None


def test_low_similarity_lookup_is_no_match_and_returns_no_entry():
    vec_a, vec_b = unit_vectors_with_similarity(0.5)
    embedder = ScriptedEmbedder({"original": vec_a, "unrelated": vec_b}, dim=2)
    cache = SemanticCache(embedder, THRESHOLDS)

    cache.put("original", "the answer")
    result = cache.lookup("unrelated")

    assert result.category == "no_match"
    assert result.match is None


def test_put_returns_unique_ids():
    cache = SemanticCache(FakeEmbedder(), THRESHOLDS)
    id_a = cache.put("q1", "a1")
    id_b = cache.put("q2", "a2")
    assert id_a != id_b
    assert len(cache) == 2


def test_expired_entry_is_purged_and_not_returned():
    embedder = FakeEmbedder()
    clock_state = {"now": _T0}
    cache = SemanticCache(
        embedder, THRESHOLDS, default_ttl_seconds=60, clock=lambda: clock_state["now"]
    )
    cache.put("same text", "cached answer", ttl_seconds=60)

    clock_state["now"] = _T0 + timedelta(seconds=30)
    result = cache.lookup("same text")
    assert result.category == "high_confidence_hit"  # identical text -> similarity 1.0

    clock_state["now"] = _T0 + timedelta(seconds=61)
    result = cache.lookup("same text")
    assert result.category == "no_match"
    assert len(cache) == 0


def test_per_entry_ttl_overrides_default():
    embedder = FakeEmbedder()
    clock_state = {"now": _T0}
    cache = SemanticCache(
        embedder, THRESHOLDS, default_ttl_seconds=3600, clock=lambda: clock_state["now"]
    )
    cache.put("short-lived", "answer", ttl_seconds=10)

    clock_state["now"] = _T0 + timedelta(seconds=11)
    result = cache.lookup("short-lived")
    assert result.category == "no_match"


def test_best_match_selected_among_multiple_entries():
    query = "query"
    close_vec, far_vec = unit_vectors_with_similarity(0.5)
    query_vec = close_vec
    embedder = ScriptedEmbedder(
        {"far": far_vec, "close": close_vec, query: query_vec}, dim=2
    )
    cache = SemanticCache(embedder, THRESHOLDS)
    cache.put("far", "far answer")
    cache.put("close", "close answer")

    result = cache.lookup(query)

    assert result.match is not None
    assert result.match.response == "close answer"
