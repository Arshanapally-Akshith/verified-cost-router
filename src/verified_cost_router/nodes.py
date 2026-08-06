"""Phase 0 stub nodes.

Each function here stands in for real logic that later phases implement
(embedding + vector search, LLM-based verification, prompt-based routing,
actual Groq calls). For now a node does two things only: it logs that it
ran, and it either passes through a decision already present in the input
state (so tests can drive the graph down any branch) or falls back to a
fixed default. No cache, no LLM calls, no embeddings.
"""

from __future__ import annotations

import logging

from verified_cost_router.state import GraphState

logger = logging.getLogger(__name__)


def _log_visit(node_name: str, state: GraphState) -> None:
    logger.info("node=%s query=%r", node_name, state.get("query"))


def cache_check(state: GraphState) -> dict:
    _log_visit("cache_check", state)
    return {
        "cache_result": state.get("cache_result", "no_match"),
        "visited": ["cache_check"],
    }


def verifier_cache(state: GraphState) -> dict:
    _log_visit("verifier_cache", state)
    return {
        "verifier_cache_result": state.get("verifier_cache_result", "pass"),
        "visited": ["verifier_cache"],
    }


def router(state: GraphState) -> dict:
    _log_visit("router", state)
    return {
        "route": state.get("route", "simple"),
        "visited": ["router"],
    }


def generate_cheap(state: GraphState) -> dict:
    _log_visit("generate_cheap", state)
    return {
        "generation": "[stub response from Groq 8B]",
        "visited": ["generate_cheap"],
    }


def generate_strong(state: GraphState) -> dict:
    _log_visit("generate_strong", state)
    return {
        "generation": "[stub response from Groq 70B]",
        "visited": ["generate_strong"],
    }


def verifier_output(state: GraphState) -> dict:
    _log_visit("verifier_output", state)
    return {
        "verifier_output_result": state.get("verifier_output_result", "pass"),
        "visited": ["verifier_output"],
    }


def log_and_cache_write(state: GraphState) -> dict:
    _log_visit("log_and_cache_write", state)
    return {"visited": ["log_and_cache_write"]}
