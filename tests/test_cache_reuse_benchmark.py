"""Unit tests for verified_cost_router.eval.cache_reuse_benchmark."""

from __future__ import annotations

from fakes import (
    FakeChatCompletionClient,
    FakeClassifier,
    FakeVerifier,
    ScriptedEmbedder,
    unit_vectors_with_similarity,
)

from verified_cost_router.cache.thresholds import CacheThresholds
from verified_cost_router.data_prep.adversarial_eval import CachePair
from verified_cost_router.eval.cache_reuse_benchmark import (
    build_cache_reuse_queries,
    run_cache_reuse_benchmark,
    select_cache_reuse_pairs,
)
from verified_cost_router.llm.groq_client import ChatCompletionResult, GroqAPIError
from verified_cost_router.pipeline.request_log import ModelPricing

THRESHOLDS = CacheThresholds(high_confidence=0.9, risky=0.7)
CHEAP_MODEL = "fake-cheap-model"
STRONG_MODEL = "fake-strong-model"
PRICING = {
    CHEAP_MODEL: ModelPricing(input_per_million=0.05, output_per_million=0.08),
    STRONG_MODEL: ModelPricing(input_per_million=0.59, output_per_million=0.79),
}


def _pair(id_: str, category: str) -> CachePair:
    return CachePair(
        id=id_,
        category=category,
        query_a=f"{id_} query a",
        query_b=f"{id_} query b",
        expect_cache_hit=(category == "true_duplicate"),
        rationale="test fixture",
    )


# --- select_cache_reuse_pairs / build_cache_reuse_queries (unchanged) --------


def test_select_cache_reuse_pairs_only_returns_true_duplicates():
    pairs = [_pair("dup1", "true_duplicate"), _pair("nm1", "near_miss"), _pair("dup2", "true_duplicate")]

    selected = select_cache_reuse_pairs(pairs, n=2, seed=0)

    assert len(selected) == 2
    assert all(p.category == "true_duplicate" for p in selected)


def test_select_cache_reuse_pairs_is_deterministic_for_a_given_seed():
    pairs = [_pair(f"dup{i}", "true_duplicate") for i in range(10)]

    first = select_cache_reuse_pairs(pairs, n=4, seed=7)
    second = select_cache_reuse_pairs(pairs, n=4, seed=7)

    assert first == second


def test_select_cache_reuse_pairs_different_seeds_can_select_differently():
    pairs = [_pair(f"dup{i}", "true_duplicate") for i in range(20)]

    a = select_cache_reuse_pairs(pairs, n=5, seed=1)
    b = select_cache_reuse_pairs(pairs, n=5, seed=2)

    assert a != b


def test_select_cache_reuse_pairs_caps_at_available_true_duplicates():
    pairs = [_pair("dup1", "true_duplicate"), _pair("nm1", "near_miss")]

    selected = select_cache_reuse_pairs(pairs, n=10, seed=0)

    assert len(selected) == 1


def test_build_cache_reuse_queries_alternates_query_a_then_query_b_per_pair():
    pairs = [_pair("dup1", "true_duplicate"), _pair("dup2", "true_duplicate")]

    queries = build_cache_reuse_queries(pairs)

    assert queries == [
        "dup1 query a",
        "dup1 query b",
        "dup2 query a",
        "dup2 query b",
    ]


# --- run_cache_reuse_benchmark ------------------------------------------------


def _isolation_pairs_and_embedder():
    """Two pairs whose query_a texts are DIFFERENT strings but embed to
    the EXACT SAME vector as each other (and each pair's own query_b
    paraphrases at 0.95 similarity) -- so if the cache were shared across
    pairs, pair 2's query_a would incorrectly hit against pair 1's
    already-cached query_a. Isolation must prevent that."""
    vec_a, vec_b = unit_vectors_with_similarity(0.95)
    embedder = ScriptedEmbedder(
        {"a1 text": vec_a, "b1 text": vec_b, "a2 text": vec_a, "b2 text": vec_b}, dim=2
    )
    pairs = [
        CachePair("dup1", "true_duplicate", "a1 text", "b1 text", True, "test"),
        CachePair("dup2", "true_duplicate", "a2 text", "b2 text", True, "test"),
    ]
    return pairs, embedder


def test_run_cache_reuse_benchmark_isolates_cache_per_pair():
    pairs, embedder = _isolation_pairs_and_embedder()

    result = run_cache_reuse_benchmark(
        pairs,
        embedder=embedder,
        thresholds=THRESHOLDS,
        classifier=FakeClassifier(label="simple"),
        verifier=FakeVerifier(),
        groq_client=FakeChatCompletionClient(next_content="answer"),
        cheap_model=CHEAP_MODEL,
        strong_model=STRONG_MODEL,
        pricing=PRICING,
    )

    full = result.full_system_raw_results
    assert [r.query for r in full] == ["a1 text", "b1 text", "a2 text", "b2 text"]
    assert full[0].served_from_cache is False  # pair 1, query_a: fresh cache, miss
    assert full[1].served_from_cache is True  # pair 1, query_b: paraphrase, hits pair 1's own cache
    # Pair 2's query_a embeds identically to pair 1's already-cached query_a --
    # a shared cache would incorrectly hit here. Isolation must force a miss.
    assert full[2].served_from_cache is False
    assert full[3].served_from_cache is True  # pair 2, query_b: hits pair 2's own cache


def test_run_cache_reuse_benchmark_runs_no_system_and_full_system_on_same_queries():
    pairs, embedder = _isolation_pairs_and_embedder()

    result = run_cache_reuse_benchmark(
        pairs,
        embedder=embedder,
        thresholds=THRESHOLDS,
        classifier=FakeClassifier(label="simple"),
        verifier=FakeVerifier(),
        groq_client=FakeChatCompletionClient(next_content="answer"),
        cheap_model=CHEAP_MODEL,
        strong_model=STRONG_MODEL,
        pricing=PRICING,
    )

    assert result.pair_count == 2
    assert [r.query for r in result.no_system_raw_results] == ["a1 text", "b1 text", "a2 text", "b2 text"]
    assert all(r.path_taken == "no_system" for r in result.no_system_raw_results)
    assert all(r.used_strong_model for r in result.no_system_raw_results)
    # no_system never caches -- every query costs the same, regardless of
    # the paraphrase structure that lets full_system serve some for free.
    assert all(r.served_from_cache is False for r in result.no_system_raw_results)


def test_run_cache_reuse_benchmark_skips_one_bad_query_without_aborting():
    pairs, embedder = _isolation_pairs_and_embedder()

    class _RaisingForOneQueryClient:
        """Raises for any chat_completion whose prompt mentions "a2 text"
        (both run_no_system and full_system's generate_cheap node send
        the query verbatim as the user message) -- everything else
        succeeds normally, matching FakeChatCompletionClient's behavior."""

        def chat_completion(self, model, messages, *, temperature=0.0, max_tokens=None):
            if "a2 text" in messages[-1].content:
                raise GroqAPIError("simulated failure")
            return ChatCompletionResult(content="answer", model=model, prompt_tokens=1, completion_tokens=1)

    result = run_cache_reuse_benchmark(
        pairs,
        embedder=embedder,
        thresholds=THRESHOLDS,
        classifier=FakeClassifier(label="simple"),
        verifier=FakeVerifier(),
        groq_client=_RaisingForOneQueryClient(),
        cheap_model=CHEAP_MODEL,
        strong_model=STRONG_MODEL,
        pricing=PRICING,
    )

    assert len(result.no_system_raw_results) == 3
    assert "a2 text" not in [r.query for r in result.no_system_raw_results]
    assert len(result.full_system_raw_results) == 3
    assert "a2 text" not in [r.query for r in result.full_system_raw_results]
