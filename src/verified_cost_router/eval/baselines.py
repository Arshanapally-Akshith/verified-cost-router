"""Cost/quality baseline runners (ARCHITECTURE.md section 6): three
pipeline variants replayed over the same traffic sample so their costs
are directly comparable, and the verifier's own contribution (the
delta between baselines 2 and 3) is isolated rather than assumed.

1. no_system: every query goes straight to the strong model -- the
   cost/quality ceiling.
2. cache_router_no_verifier: cache + router, verifier nodes removed.
   Without a verifier, a risky_hit can't be safely auto-served (that's
   the whole reason the risky band exists), so it's treated like a
   miss and falls through to the router -- the same thing a verifier
   "fail" would have done. Likewise a cheap-model output has nothing to
   catch and escalate a bad answer, so it's served as-is.
3. full_system: the real Phase 4 pipeline, unchanged -- reuses
   build_pipeline_graph()/PipelineNodes directly rather than
   duplicating the topology.

Baselines 1 and 2 are simple, non-branching sequences with no need for
a second LangGraph state machine -- plain functions are clearer here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from langgraph.graph.state import CompiledStateGraph

from verified_cost_router.cache.semantic_cache import SemanticCache
from verified_cost_router.llm.generation import generate
from verified_cost_router.llm.groq_client import ChatCompletionClient
from verified_cost_router.pipeline.request_log import (
    ModelPricing,
    determine_path_taken,
    estimate_cost_usd,
    is_fresh_generation_path,
)
from verified_cost_router.router.classifier import ComplexityClassifier
from verified_cost_router.state import LlmCallUsage


@dataclass(frozen=True)
class BaselineResult:
    """One query's outcome under one baseline: enough to compare cost
    across baselines, feed the quality spot check, and (Phase 6) chart
    path distribution / cache-hit-rate-over-time on the dashboard.

    `used_strong_model` marks responses that already came from the
    strong model -- the quality spot check (ARCHITECTURE.md 4.6)
    excludes these, since comparing a strong-model answer against a
    fresh strong-model reference isn't an informative quality check.

    `path_taken` is baseline-specific: "no_system" for baseline 1;
    "cache_hit" / "router_cheap" / "router_strong" for baseline 2 (it
    has no verifier, so no escalation path exists); the real 5-way
    ARCHITECTURE.md taxonomy (e.g. "router-8B-escalated") for baseline 3.
    """

    query: str
    response: str
    cost_usd: float
    llm_call_count: int
    served_from_cache: bool
    used_strong_model: bool
    path_taken: str


def run_no_system(
    query: str, groq_client: ChatCompletionClient, strong_model: str, pricing: Mapping[str, ModelPricing]
) -> BaselineResult:
    """Baseline 1: always the strong model, no cache, no routing."""
    result = generate(groq_client, strong_model, query)
    call = LlmCallUsage("generate_strong", result.model, result.prompt_tokens, result.completion_tokens)
    return BaselineResult(
        query=query,
        response=result.content,
        cost_usd=estimate_cost_usd([call], pricing=pricing),
        llm_call_count=1,
        served_from_cache=False,
        used_strong_model=True,
        path_taken="no_system",
    )


def run_cache_router_no_verifier(
    query: str,
    cache: SemanticCache,
    classifier: ComplexityClassifier,
    groq_client: ChatCompletionClient,
    cheap_model: str,
    strong_model: str,
    pricing: Mapping[str, ModelPricing],
) -> BaselineResult:
    """Baseline 2: cache + router, no verifier. Only a high_confidence_hit
    is safe to auto-serve without a verifier; everything else (including
    a risky_hit) falls through to the router."""
    lookup = cache.lookup(query)
    if lookup.category == "high_confidence_hit":
        assert lookup.match is not None
        return BaselineResult(
            query=query,
            response=lookup.match.response,
            cost_usd=0.0,
            llm_call_count=0,
            served_from_cache=True,
            used_strong_model=False,
            path_taken="cache_hit",
        )

    calls: list[LlmCallUsage] = []
    classification = classifier.classify_with_usage(query)
    calls.append(
        LlmCallUsage("classify", classification.model, classification.prompt_tokens, classification.completion_tokens)
    )

    is_complex = classification.label != "simple"
    model = strong_model if is_complex else cheap_model
    purpose = "generate_strong" if is_complex else "generate_cheap"
    generation = generate(groq_client, model, query)
    calls.append(LlmCallUsage(purpose, generation.model, generation.prompt_tokens, generation.completion_tokens))

    cache.put(query, generation.content)
    return BaselineResult(
        query=query,
        response=generation.content,
        cost_usd=estimate_cost_usd(calls, pricing=pricing),
        llm_call_count=len(calls),
        served_from_cache=False,
        used_strong_model=is_complex,
        path_taken="router_strong" if is_complex else "router_cheap",
    )


def run_full_system(query: str, app: CompiledStateGraph, pricing: Mapping[str, ModelPricing]) -> BaselineResult:
    """Baseline 3: the real pipeline (build_pipeline_graph output), unchanged."""
    result = app.invoke({"query": query})
    llm_calls = result.get("llm_calls", ())
    path_taken = determine_path_taken(result)
    return BaselineResult(
        query=query,
        response=result["response"],
        cost_usd=estimate_cost_usd(llm_calls, pricing=pricing),
        llm_call_count=len(llm_calls),
        served_from_cache=not is_fresh_generation_path(path_taken),
        # router-8B-escalated's *served* content is the escalated
        # generate_strong call's output, not the discarded cheap draft.
        used_strong_model=(path_taken in ("router-70B", "router-8B-escalated")),
        path_taken=path_taken,
    )
