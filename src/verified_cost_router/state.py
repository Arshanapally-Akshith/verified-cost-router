"""Shared state schema passed between LangGraph nodes.

Mirrors the pipeline in ARCHITECTURE.md section 2. Fields beyond ``query``
are populated as the query moves through the graph. Phase 0 populated
these with pass-through stubs; Phase 4's real nodes (pipeline/nodes.py)
populate them with real cache/router/verifier/generation results, using
the exact same field names and shape.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, List, Literal, TypedDict


@dataclass(frozen=True)
class LlmCallUsage:
    """Token usage from one real LLM call made while processing a request.

    Accumulated in GraphState so log_and_cache_write can price the
    request's *total* Groq usage (classification + verification +
    generation), not just the final generation call.
    """

    purpose: Literal["classify", "verify_cache", "verify_output", "generate_cheap", "generate_strong"]
    model: str
    prompt_tokens: int
    completion_tokens: int


class GraphState(TypedDict, total=False):
    query: str

    # Set by cache_check (entry point): wall-clock start, for latency logging.
    started_at: float

    # Set by cache_check.
    cache_result: Literal["no_match", "risky_hit", "high_confidence_hit"]
    # Set by cache_check when cache_result != "no_match".
    cache_match_query: str
    cache_match_response: str
    cache_similarity: float

    # Set by verifier_cache (only reached on a risky cache hit).
    verifier_cache_result: Literal["pass", "fail"]

    # Set by router.
    route: Literal["simple", "complex"]

    # Set by generate_cheap / generate_strong.
    generation: str

    # Set by verifier_output (only reached after the cheap model generates).
    verifier_output_result: Literal["pass", "fail"]

    # Set by log_and_cache_write: the response actually served to the user
    # (either the cache match, or a fresh generation).
    response: str

    # Token usage from every real LLM call made for this request, appended
    # to by router/verifier_cache/verifier_output/generate_cheap/generate_strong.
    llm_calls: Annotated[List[LlmCallUsage], operator.add]

    # Node names, appended to by every node, in execution order.
    visited: Annotated[List[str], operator.add]
