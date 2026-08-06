"""Per-request logging: which path a request took, how long it took, and
a hypothetical paid-tier cost estimate (ARCHITECTURE.md 4.4-4.5).

The free tier costs $0, so "cost saved" only means something once every
request's token usage is converted to Groq's published paid-tier
per-token rates -- that conversion is what this module does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from verified_cost_router.state import GraphState, LlmCallUsage

PathTaken = Literal["cache-hit", "cache-hit-verified", "router-8B", "router-8B-escalated", "router-70B"]

_FRESH_GENERATION_PATHS: frozenset[PathTaken] = frozenset({"router-8B", "router-8B-escalated", "router-70B"})


@dataclass(frozen=True)
class ModelPricing:
    """Groq published paid-tier price, USD per 1M tokens."""

    input_per_million: float
    output_per_million: float


# Groq on-demand (paid-tier) pricing, USD per 1M tokens, checked against
# console.groq.com pricing in 2026-08. Update here if Groq changes rates --
# nothing else in the codebase hardcodes these numbers.
DEFAULT_PRICING: Mapping[str, ModelPricing] = {
    "llama-3.1-8b-instant": ModelPricing(input_per_million=0.05, output_per_million=0.08),
    "llama-3.3-70b-versatile": ModelPricing(input_per_million=0.59, output_per_million=0.79),
}


@dataclass(frozen=True)
class RequestLog:
    """One completed request's summary, as ARCHITECTURE.md 4.5 specifies:
    path taken, latency, and a token-cost estimate."""

    query: str
    path_taken: PathTaken
    latency_ms: float
    cost_usd: float
    llm_calls: tuple[LlmCallUsage, ...]


def is_fresh_generation_path(path_taken: PathTaken) -> bool:
    """Whether `path_taken` produced a brand-new generation (vs. serving
    an existing cache entry) -- these are the paths log_and_cache_write
    writes back to the cache."""
    return path_taken in _FRESH_GENERATION_PATHS


def determine_path_taken(state: GraphState) -> PathTaken:
    """Classify a completed request into ARCHITECTURE.md's 5 named paths:
    cache-hit / cache-hit-verified / router-8B / router-8B-escalated / router-70B.

    Raises ValueError if `state` doesn't represent a completed request
    (missing fields for the path it appears to be on) -- a bug elsewhere
    in the graph, not a case to silently paper over.
    """
    cache_result = state.get("cache_result")
    if cache_result == "high_confidence_hit":
        return "cache-hit"
    if cache_result == "risky_hit" and state.get("verifier_cache_result") == "pass":
        return "cache-hit-verified"

    route = state.get("route")
    if route == "complex":
        return "router-70B"
    if route == "simple":
        return "router-8B-escalated" if state.get("verifier_output_result") == "fail" else "router-8B"

    raise ValueError(f"could not determine path_taken from state: {state!r}")


def estimate_cost_usd(
    llm_calls: Sequence[LlmCallUsage], pricing: Mapping[str, ModelPricing] = DEFAULT_PRICING
) -> float:
    """Hypothetical paid-tier cost, in USD, of every real LLM call made for one request.

    Raises KeyError if a call used a model with no configured pricing --
    fail loudly rather than silently under-report cost.
    """
    total = 0.0
    for call in llm_calls:
        rates = pricing.get(call.model)
        if rates is None:
            raise KeyError(f"no pricing configured for model {call.model!r}")
        total += (call.prompt_tokens / 1_000_000) * rates.input_per_million
        total += (call.completion_tokens / 1_000_000) * rates.output_per_million
    return total


class RequestLogger:
    """Emits one structured log line per completed request."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def log(self, entry: RequestLog) -> None:
        self._logger.info(
            "path=%s latency_ms=%.1f cost_usd=%.6f llm_calls=%d query=%r",
            entry.path_taken,
            entry.latency_ms,
            entry.cost_usd,
            len(entry.llm_calls),
            entry.query,
        )
