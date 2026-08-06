"""Shared state schema passed between LangGraph nodes.

Mirrors the pipeline in ARCHITECTURE.md section 2. Fields beyond ``query``
are populated as the query moves through the graph; later phases attach
real values (embeddings, generations, etc.) to the same shape.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Literal, TypedDict


class GraphState(TypedDict, total=False):
    query: str

    # Set by cache_check.
    cache_result: Literal["no_match", "risky_hit", "high_confidence_hit"]

    # Set by verifier_cache (only reached on a risky cache hit).
    verifier_cache_result: Literal["pass", "fail"]

    # Set by router.
    route: Literal["simple", "complex"]

    # Set by generate_cheap / generate_strong.
    generation: str

    # Set by verifier_output (only reached after the cheap model generates).
    verifier_output_result: Literal["pass", "fail"]

    # Node names, appended to by every node, in execution order.
    visited: Annotated[List[str], operator.add]
