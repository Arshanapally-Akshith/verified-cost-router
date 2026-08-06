"""Tests for verified_cost_router.pipeline.nodes.PipelineNodes.

Two layers:
- Node-level unit tests call individual PipelineNodes methods directly
  with a hand-built state dict, focused on log_and_cache_write's
  cache-write-only-on-fresh-generation and cost/latency logic.
- Full graph integration tests run build_pipeline_graph(nodes).invoke()
  end-to-end, covering the same 5 named paths from ARCHITECTURE.md
  section 2, using a real SemanticCache (Phase 2) wired to fake
  embedders/classifier/verifier/groq client so similarity and every
  LLM verdict are fully controlled and offline.
"""

from __future__ import annotations

import time

from fakes import FakeChatCompletionClient, FakeClassifier, FakeEmbedder, FakeVerifier, ScriptedEmbedder, unit_vectors_with_similarity

from verified_cost_router.cache.semantic_cache import SemanticCache
from verified_cost_router.cache.thresholds import CacheThresholds
from verified_cost_router.graph import build_pipeline_graph
from verified_cost_router.pipeline.nodes import PipelineNodes
from verified_cost_router.pipeline.request_log import ModelPricing, RequestLog

THRESHOLDS = CacheThresholds(high_confidence=0.9, risky=0.7)
CHEAP_MODEL = "fake-cheap-model"
STRONG_MODEL = "fake-strong-model"
FAKE_PRICING = {
    CHEAP_MODEL: ModelPricing(input_per_million=0.05, output_per_million=0.08),
    STRONG_MODEL: ModelPricing(input_per_million=0.59, output_per_million=0.79),
}


class _SpyLogger:
    def __init__(self) -> None:
        self.entries: list[RequestLog] = []

    def log(self, entry: RequestLog) -> None:
        self.entries.append(entry)


def _make_nodes(
    cache: SemanticCache | None = None,
    classifier: FakeClassifier | None = None,
    verifier: FakeVerifier | None = None,
    groq_client: FakeChatCompletionClient | None = None,
    request_logger: _SpyLogger | None = None,
) -> PipelineNodes:
    # NOTE: `cache or default` would be wrong here -- an empty SemanticCache
    # has len() == 0 and no __bool__, so Python treats it as falsy and the
    # `or` would silently discard a deliberately-empty cache passed by a
    # caller. Use explicit `is None` checks for every optional dependency.
    return PipelineNodes(
        cache=cache if cache is not None else SemanticCache(FakeEmbedder(), THRESHOLDS),
        classifier=classifier if classifier is not None else FakeClassifier(label="simple"),
        verifier=verifier if verifier is not None else FakeVerifier(),
        groq_client=groq_client if groq_client is not None else FakeChatCompletionClient(next_content="generated"),
        cheap_model=CHEAP_MODEL,
        strong_model=STRONG_MODEL,
        request_logger=request_logger,
        pricing=FAKE_PRICING,
    )


# --- Node-level unit tests -------------------------------------------------


def test_cache_check_sets_started_at_and_no_match_on_empty_cache():
    nodes = _make_nodes()
    updates = nodes.cache_check({"query": "hello"})
    assert updates["cache_result"] == "no_match"
    assert "started_at" in updates
    assert updates["visited"] == ["cache_check"]
    assert "cache_match_response" not in updates


def test_router_records_llm_call_usage():
    classifier = FakeClassifier(label="complex", model="usage-model")
    nodes = _make_nodes(classifier=classifier)
    updates = nodes.router({"query": "hello"})
    assert updates["route"] == "complex"
    [call] = updates["llm_calls"]
    assert call.purpose == "classify"
    assert call.model == "usage-model"


def test_generate_cheap_uses_cheap_model():
    groq_client = FakeChatCompletionClient(next_content="cheap output")
    nodes = _make_nodes(groq_client=groq_client)
    updates = nodes.generate_cheap({"query": "hello"})
    assert updates["generation"] == "cheap output"
    assert groq_client.last_model == CHEAP_MODEL
    assert updates["llm_calls"][0].purpose == "generate_cheap"


def test_generate_strong_uses_strong_model():
    groq_client = FakeChatCompletionClient(next_content="strong output")
    nodes = _make_nodes(groq_client=groq_client)
    updates = nodes.generate_strong({"query": "hello"})
    assert updates["generation"] == "strong output"
    assert groq_client.last_model == STRONG_MODEL
    assert updates["llm_calls"][0].purpose == "generate_strong"


def test_log_and_cache_write_serves_cache_match_without_writing_to_cache():
    cache = SemanticCache(FakeEmbedder(), THRESHOLDS)
    cache.put("original", "cached answer")
    assert len(cache) == 1
    nodes = _make_nodes(cache=cache)

    state = {
        "query": "original",
        "cache_result": "high_confidence_hit",
        "cache_match_response": "cached answer",
        "started_at": time.monotonic(),
        "llm_calls": [],
    }
    updates = nodes.log_and_cache_write(state)

    assert updates["response"] == "cached answer"
    assert len(cache) == 1  # unchanged -- no fresh generation to write


def test_log_and_cache_write_writes_fresh_generation_to_cache():
    cache = SemanticCache(FakeEmbedder(), THRESHOLDS)
    nodes = _make_nodes(cache=cache)

    state = {
        "query": "a brand new query",
        "cache_result": "no_match",
        "route": "simple",
        "verifier_output_result": "pass",
        "generation": "a fresh answer",
        "started_at": time.monotonic(),
        "llm_calls": [],
    }
    updates = nodes.log_and_cache_write(state)

    assert updates["response"] == "a fresh answer"
    assert len(cache) == 1
    result = cache.lookup("a brand new query")
    assert result.match is not None
    assert result.match.response == "a fresh answer"


def test_log_and_cache_write_logs_path_latency_and_cost():
    spy = _SpyLogger()
    nodes = _make_nodes(request_logger=spy)

    state = {
        "query": "q",
        "cache_result": "no_match",
        "route": "complex",
        "generation": "answer",
        "started_at": time.monotonic(),
        "llm_calls": [],
    }
    nodes.log_and_cache_write(state)

    assert len(spy.entries) == 1
    entry = spy.entries[0]
    assert entry.path_taken == "router-70B"
    assert entry.latency_ms >= 0
    assert entry.cost_usd == 0.0  # no llm_calls recorded in this synthetic state


# --- Full graph integration tests ------------------------------------------


def test_cache_hit_serves_directly_with_no_llm_calls():
    vec_cached, vec_query = unit_vectors_with_similarity(0.95)
    embedder = ScriptedEmbedder({"cached query": vec_cached, "new query": vec_query}, dim=2)
    cache = SemanticCache(embedder, THRESHOLDS)
    cache.put("cached query", "cached answer")

    classifier = FakeClassifier(label="simple")
    verifier = FakeVerifier()
    groq_client = FakeChatCompletionClient(next_content="should not be used")
    nodes = _make_nodes(cache=cache, classifier=classifier, verifier=verifier, groq_client=groq_client)

    app = build_pipeline_graph(nodes)
    result = app.invoke({"query": "new query"})

    assert result["response"] == "cached answer"
    assert result["visited"] == ["cache_check", "log_and_cache_write"]
    assert classifier.calls == []
    assert verifier.cache_hit_calls == [] and verifier.output_calls == []
    assert groq_client.call_count == 0
    assert len(cache) == 1


def test_cache_hit_verified_serves_from_cache_without_rewriting():
    vec_cached, vec_query = unit_vectors_with_similarity(0.8)
    embedder = ScriptedEmbedder({"cached query b": vec_cached, "new query b": vec_query}, dim=2)
    cache = SemanticCache(embedder, THRESHOLDS)
    cache.put("cached query b", "cached answer b")

    verifier = FakeVerifier(cache_hit_label="pass")
    nodes = _make_nodes(cache=cache, verifier=verifier)

    app = build_pipeline_graph(nodes)
    result = app.invoke({"query": "new query b"})

    assert result["response"] == "cached answer b"
    assert result["visited"] == ["cache_check", "verifier_cache", "log_and_cache_write"]
    assert verifier.cache_hit_calls == [("new query b", "cached query b", "cached answer b")]
    assert len(cache) == 1  # not rewritten


def test_risky_hit_verify_fail_falls_back_to_router_and_writes_cache():
    vec_cached, vec_query = unit_vectors_with_similarity(0.8)
    embedder = ScriptedEmbedder({"cached query c": vec_cached, "new query c": vec_query}, dim=2)
    cache = SemanticCache(embedder, THRESHOLDS)
    cache.put("cached query c", "cached answer c")

    classifier = FakeClassifier(label="simple")
    verifier = FakeVerifier(cache_hit_label="fail", output_label="pass")
    groq_client = FakeChatCompletionClient(next_content="fresh cheap answer")
    nodes = _make_nodes(cache=cache, classifier=classifier, verifier=verifier, groq_client=groq_client)

    app = build_pipeline_graph(nodes)
    result = app.invoke({"query": "new query c"})

    assert result["response"] == "fresh cheap answer"
    assert result["visited"] == [
        "cache_check",
        "verifier_cache",
        "router",
        "generate_cheap",
        "verifier_output",
        "log_and_cache_write",
    ]
    assert len(cache) == 2  # fresh generation written alongside the original entry
    assert groq_client.call_count == 1
    assert groq_client.last_model == CHEAP_MODEL


def test_no_match_complex_routes_straight_to_strong_model():
    cache = SemanticCache(FakeEmbedder(), THRESHOLDS)
    classifier = FakeClassifier(label="complex")
    verifier = FakeVerifier()
    groq_client = FakeChatCompletionClient(next_content="fresh strong answer")
    nodes = _make_nodes(cache=cache, classifier=classifier, verifier=verifier, groq_client=groq_client)

    app = build_pipeline_graph(nodes)
    result = app.invoke({"query": "some complex query"})

    assert result["response"] == "fresh strong answer"
    assert result["visited"] == ["cache_check", "router", "generate_strong", "log_and_cache_write"]
    assert verifier.cache_hit_calls == [] and verifier.output_calls == []
    assert groq_client.call_count == 1
    assert groq_client.last_model == STRONG_MODEL
    assert len(cache) == 1  # fresh generation was written back


def test_no_match_simple_verified_pass_serves_cheap_output():
    classifier = FakeClassifier(label="simple")
    verifier = FakeVerifier(output_label="pass")
    groq_client = FakeChatCompletionClient(next_content="fresh cheap answer 2")
    nodes = _make_nodes(classifier=classifier, verifier=verifier, groq_client=groq_client)

    app = build_pipeline_graph(nodes)
    result = app.invoke({"query": "some simple query"})

    assert result["response"] == "fresh cheap answer 2"
    assert result["visited"] == [
        "cache_check",
        "router",
        "generate_cheap",
        "verifier_output",
        "log_and_cache_write",
    ]
    assert groq_client.call_count == 1
    assert groq_client.last_model == CHEAP_MODEL


def test_no_match_simple_verified_fail_escalates_to_strong_model():
    classifier = FakeClassifier(label="simple")
    verifier = FakeVerifier(output_label="fail")
    groq_client = FakeChatCompletionClient(next_content="escalated answer")
    nodes = _make_nodes(classifier=classifier, verifier=verifier, groq_client=groq_client)

    app = build_pipeline_graph(nodes)
    result = app.invoke({"query": "some tricky-simple query"})

    assert result["response"] == "escalated answer"
    assert result["visited"] == [
        "cache_check",
        "router",
        "generate_cheap",
        "verifier_output",
        "generate_strong",
        "log_and_cache_write",
    ]
    assert groq_client.call_count == 2
    assert groq_client.last_model == STRONG_MODEL  # the escalated (final) call
    purposes = [call.purpose for call in result["llm_calls"]]
    assert purposes == ["classify", "generate_cheap", "verify_output", "generate_strong"]


def test_llm_calls_accumulate_across_the_whole_request():
    vec_cached, vec_query = unit_vectors_with_similarity(0.8)
    embedder = ScriptedEmbedder({"cached": vec_cached, "query": vec_query}, dim=2)
    cache = SemanticCache(embedder, THRESHOLDS)
    cache.put("cached", "cached answer")

    classifier = FakeClassifier(label="simple")
    verifier = FakeVerifier(cache_hit_label="fail", output_label="pass")
    groq_client = FakeChatCompletionClient(next_content="answer")
    nodes = _make_nodes(cache=cache, classifier=classifier, verifier=verifier, groq_client=groq_client)

    app = build_pipeline_graph(nodes)
    result = app.invoke({"query": "query"})

    purposes = [call.purpose for call in result["llm_calls"]]
    assert purposes == ["verify_cache", "classify", "generate_cheap", "verify_output"]
