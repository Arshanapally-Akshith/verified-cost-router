"""Real LangGraph node implementations (Phase 4).

Each method here has the exact `(state: GraphState) -> dict` signature
LangGraph nodes require, matching Phase 0's stub nodes.py one-for-one --
graph.py's topology and conditional-edge routing functions are unchanged
(they only ever read state fields, never care which module set them).
This module is orchestration glue: it delegates to the real cache
(Phase 2), classifier (Phase 3), and verifier/generation (Phase 4)
components, and does no business logic of its own beyond wiring.

The Phase 0 stub module (verified_cost_router.nodes) is untouched and
still backs graph.build_graph() for the topology-proof tests.
"""

from __future__ import annotations

import time
from typing import Mapping

from verified_cost_router.cache.semantic_cache import SemanticCache
from verified_cost_router.llm.generation import generate
from verified_cost_router.llm.groq_client import ChatCompletionClient
from verified_cost_router.pipeline.request_log import (
    DEFAULT_PRICING,
    ModelPricing,
    RequestLog,
    RequestLogger,
    determine_path_taken,
    estimate_cost_usd,
    is_fresh_generation_path,
)
from verified_cost_router.router.classifier import ComplexityClassifier
from verified_cost_router.state import GraphState, LlmCallUsage
from verified_cost_router.verifier.verifier import Verifier


class PipelineNodes:
    """Bundles the real components each node needs; graph.build_pipeline_graph
    registers these bound methods directly as LangGraph nodes."""

    def __init__(
        self,
        cache: SemanticCache,
        classifier: ComplexityClassifier,
        verifier: Verifier,
        groq_client: ChatCompletionClient,
        cheap_model: str,
        strong_model: str,
        request_logger: RequestLogger | None = None,
        pricing: Mapping[str, ModelPricing] = DEFAULT_PRICING,
    ) -> None:
        self._cache = cache
        self._classifier = classifier
        self._verifier = verifier
        self._groq_client = groq_client
        self._cheap_model = cheap_model
        self._strong_model = strong_model
        self._request_logger = request_logger or RequestLogger()
        self._pricing = pricing

    def cache_check(self, state: GraphState) -> dict:
        started_at = time.monotonic()
        result = self._cache.lookup(state["query"])
        updates: dict = {
            "cache_result": result.category,
            "started_at": started_at,
            "visited": ["cache_check"],
        }
        if result.match is not None:
            updates["cache_match_query"] = result.match.prompt
            updates["cache_match_response"] = result.match.response
            updates["cache_similarity"] = result.similarity
        return updates

    def verifier_cache(self, state: GraphState) -> dict:
        outcome = self._verifier.verify_cache_hit(
            query=state["query"],
            cached_query=state.get("cache_match_query", ""),
            cached_response=state.get("cache_match_response", ""),
        )
        return {
            "verifier_cache_result": outcome.label,
            "llm_calls": [
                LlmCallUsage("verify_cache", outcome.model, outcome.prompt_tokens, outcome.completion_tokens)
            ],
            "visited": ["verifier_cache"],
        }

    def router(self, state: GraphState) -> dict:
        result = self._classifier.classify_with_usage(state["query"])
        return {
            "route": result.label,
            "llm_calls": [
                LlmCallUsage("classify", result.model, result.prompt_tokens, result.completion_tokens)
            ],
            "visited": ["router"],
        }

    def generate_cheap(self, state: GraphState) -> dict:
        result = generate(self._groq_client, self._cheap_model, state["query"])
        return {
            "generation": result.content,
            "llm_calls": [
                LlmCallUsage("generate_cheap", result.model, result.prompt_tokens, result.completion_tokens)
            ],
            "visited": ["generate_cheap"],
        }

    def generate_strong(self, state: GraphState) -> dict:
        result = generate(self._groq_client, self._strong_model, state["query"])
        return {
            "generation": result.content,
            "llm_calls": [
                LlmCallUsage("generate_strong", result.model, result.prompt_tokens, result.completion_tokens)
            ],
            "visited": ["generate_strong"],
        }

    def verifier_output(self, state: GraphState) -> dict:
        outcome = self._verifier.verify_output(query=state["query"], output=state["generation"])
        return {
            "verifier_output_result": outcome.label,
            "llm_calls": [
                LlmCallUsage("verify_output", outcome.model, outcome.prompt_tokens, outcome.completion_tokens)
            ],
            "visited": ["verifier_output"],
        }

    def log_and_cache_write(self, state: GraphState) -> dict:
        path_taken = determine_path_taken(state)
        response = (
            state["cache_match_response"]
            if path_taken in ("cache-hit", "cache-hit-verified")
            else state["generation"]
        )

        if is_fresh_generation_path(path_taken):
            self._cache.put(state["query"], response)

        llm_calls = tuple(state.get("llm_calls", ()))
        latency_ms = (time.monotonic() - state["started_at"]) * 1000
        cost_usd = estimate_cost_usd(llm_calls, pricing=self._pricing)

        self._request_logger.log(
            RequestLog(
                query=state["query"],
                path_taken=path_taken,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                llm_calls=llm_calls,
            )
        )
        return {"response": response, "visited": ["log_and_cache_write"]}
