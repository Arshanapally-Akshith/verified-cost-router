"""Unit tests for verified_cost_router.eval.baselines."""

from __future__ import annotations

from fakes import (
    FakeChatCompletionClient,
    FakeClassifier,
    FakeEmbedder,
    FakeVerifier,
    ScriptedClassifier,
    ScriptedEmbedder,
    unit_vectors_with_similarity,
)

from verified_cost_router.cache.semantic_cache import SemanticCache
from verified_cost_router.cache.thresholds import CacheThresholds
from verified_cost_router.eval.baselines import run_cache_router_no_verifier, run_full_system, run_no_system
from verified_cost_router.graph import build_pipeline_graph
from verified_cost_router.pipeline.nodes import PipelineNodes
from verified_cost_router.pipeline.request_log import ModelPricing
from verified_cost_router.router.classifier import ComplexityClassifier

THRESHOLDS = CacheThresholds(high_confidence=0.9, risky=0.7)
CHEAP_MODEL = "fake-cheap-model"
STRONG_MODEL = "fake-strong-model"
PRICING = {
    CHEAP_MODEL: ModelPricing(input_per_million=0.05, output_per_million=0.08),
    STRONG_MODEL: ModelPricing(input_per_million=0.59, output_per_million=0.79),
}


# --- run_no_system ------------------------------------------------------------


def test_run_no_system_always_uses_strong_model():
    groq_client = FakeChatCompletionClient(next_content="strong answer")
    result = run_no_system("q", groq_client, STRONG_MODEL, PRICING)

    assert result.response == "strong answer"
    assert result.llm_call_count == 1
    assert result.served_from_cache is False
    assert groq_client.last_model == STRONG_MODEL
    assert result.cost_usd > 0
    assert result.path_taken == "no_system"


# --- run_cache_router_no_verifier ----------------------------------------------


def test_high_confidence_cache_hit_is_served_free_with_no_llm_calls():
    cache = SemanticCache(FakeEmbedder(), THRESHOLDS)
    cache.put("original", "cached answer")
    classifier = ComplexityClassifier(FakeChatCompletionClient(next_content="simple"), model=CHEAP_MODEL)
    groq_client = FakeChatCompletionClient(next_content="should not be used")

    result = run_cache_router_no_verifier(
        "original", cache, classifier, groq_client, CHEAP_MODEL, STRONG_MODEL, PRICING
    )

    assert result.response == "cached answer"
    assert result.served_from_cache is True
    assert result.llm_call_count == 0
    assert result.cost_usd == 0.0
    assert groq_client.call_count == 0
    assert result.path_taken == "cache_hit"


def test_risky_hit_is_not_served_falls_through_to_router():
    # A "risky_hit" similarity can't be told apart from a miss without a
    # verifier -- this baseline has none, so it must fall through.
    vec_a, vec_b = unit_vectors_with_similarity(0.8)  # risky band
    embedder = ScriptedEmbedder({"cached query": vec_a, "new query": vec_b}, dim=2)
    cache = SemanticCache(embedder, THRESHOLDS)
    cache.put("cached query", "cached answer")

    classifier = ScriptedClassifier({"new query": "simple"})
    groq_client = FakeChatCompletionClient(next_content="fresh answer")

    result = run_cache_router_no_verifier(
        "new query", cache, classifier, groq_client, CHEAP_MODEL, STRONG_MODEL, PRICING
    )

    assert result.response == "fresh answer"
    assert result.served_from_cache is False
    assert groq_client.call_count == 1
    assert result.path_taken == "router_cheap"


def test_cache_miss_simple_uses_cheap_model_and_writes_cache():
    cache = SemanticCache(FakeEmbedder(), THRESHOLDS)
    classifier = ScriptedClassifier({"q": "simple"})
    groq_client = FakeChatCompletionClient(next_content="cheap answer")

    result = run_cache_router_no_verifier("q", cache, classifier, groq_client, CHEAP_MODEL, STRONG_MODEL, PRICING)

    assert result.response == "cheap answer"
    assert groq_client.last_model == CHEAP_MODEL
    assert result.llm_call_count == 2  # classify + generate
    assert len(cache) == 1


def test_cache_miss_complex_uses_strong_model():
    cache = SemanticCache(FakeEmbedder(), THRESHOLDS)
    classifier = ScriptedClassifier({"q": "complex"})
    groq_client = FakeChatCompletionClient(next_content="strong answer")

    result = run_cache_router_no_verifier("q", cache, classifier, groq_client, CHEAP_MODEL, STRONG_MODEL, PRICING)

    assert result.response == "strong answer"
    assert groq_client.last_model == STRONG_MODEL
    assert result.path_taken == "router_strong"


# --- run_full_system -----------------------------------------------------------


def test_run_full_system_reuses_the_real_pipeline():
    cache = SemanticCache(FakeEmbedder(), THRESHOLDS)
    nodes = PipelineNodes(
        cache=cache,
        classifier=FakeClassifier(label="simple"),
        verifier=FakeVerifier(output_label="pass"),
        groq_client=FakeChatCompletionClient(next_content="answer"),
        cheap_model=CHEAP_MODEL,
        strong_model=STRONG_MODEL,
        pricing=PRICING,
    )
    app = build_pipeline_graph(nodes)

    result = run_full_system("q", app, PRICING)

    assert result.response == "answer"
    assert result.served_from_cache is False
    assert result.llm_call_count == 3  # classify + generate_cheap + verify_output
    assert result.cost_usd > 0
    assert result.path_taken == "router-8B"


def test_run_full_system_marks_cache_hit_as_served_from_cache():
    vec_cached, vec_query = unit_vectors_with_similarity(0.95)
    embedder = ScriptedEmbedder({"cached": vec_cached, "query": vec_query}, dim=2)
    cache = SemanticCache(embedder, THRESHOLDS)
    cache.put("cached", "cached answer")

    nodes = PipelineNodes(
        cache=cache,
        classifier=FakeClassifier(label="simple"),
        verifier=FakeVerifier(),
        groq_client=FakeChatCompletionClient(next_content="should not be used"),
        cheap_model=CHEAP_MODEL,
        strong_model=STRONG_MODEL,
        pricing=PRICING,
    )
    app = build_pipeline_graph(nodes)

    result = run_full_system("query", app, PRICING)

    assert result.response == "cached answer"
    assert result.served_from_cache is True
    assert result.llm_call_count == 0
    assert result.path_taken == "cache-hit"
