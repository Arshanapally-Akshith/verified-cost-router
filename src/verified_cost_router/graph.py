"""LangGraph wiring for the verified cost router pipeline.

Graph shape mirrors ARCHITECTURE.md section 2:

    cache_check --no_match--------------------------> router
    cache_check --risky_hit-----------> verifier_cache
    cache_check --high_confidence_hit-----------------------------> log_and_cache_write
    verifier_cache --fail--> router
    verifier_cache --pass----------------------------------------> log_and_cache_write
    router --simple--> generate_cheap --> verifier_output
    router --complex-> generate_strong ---------------------------> log_and_cache_write
    verifier_output --pass--------------------------------------->  log_and_cache_write
    verifier_output --fail (escalate)--> generate_strong

Node implementations are Phase 0 stubs (see nodes.py); this module only
owns the graph shape and the conditional-edge routing logic, which is
real and will not change when the stubs are replaced with real logic.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from verified_cost_router import nodes
from verified_cost_router.state import GraphState


def _route_after_cache_check(state: GraphState) -> str:
    result = state["cache_result"]
    if result == "no_match":
        return "router"
    if result == "risky_hit":
        return "verifier_cache"
    return "log_and_cache_write"  # high_confidence_hit


def _route_after_verifier_cache(state: GraphState) -> str:
    return "log_and_cache_write" if state["verifier_cache_result"] == "pass" else "router"


def _route_after_router(state: GraphState) -> str:
    return "generate_cheap" if state["route"] == "simple" else "generate_strong"


def _route_after_verifier_output(state: GraphState) -> str:
    return "log_and_cache_write" if state["verifier_output_result"] == "pass" else "generate_strong"


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("cache_check", nodes.cache_check)
    graph.add_node("verifier_cache", nodes.verifier_cache)
    graph.add_node("router", nodes.router)
    graph.add_node("generate_cheap", nodes.generate_cheap)
    graph.add_node("generate_strong", nodes.generate_strong)
    graph.add_node("verifier_output", nodes.verifier_output)
    graph.add_node("log_and_cache_write", nodes.log_and_cache_write)

    graph.set_entry_point("cache_check")

    graph.add_conditional_edges(
        "cache_check",
        _route_after_cache_check,
        {
            "router": "router",
            "verifier_cache": "verifier_cache",
            "log_and_cache_write": "log_and_cache_write",
        },
    )
    graph.add_conditional_edges(
        "verifier_cache",
        _route_after_verifier_cache,
        {"log_and_cache_write": "log_and_cache_write", "router": "router"},
    )
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {"generate_cheap": "generate_cheap", "generate_strong": "generate_strong"},
    )
    graph.add_edge("generate_cheap", "verifier_output")
    graph.add_edge("generate_strong", "log_and_cache_write")
    graph.add_conditional_edges(
        "verifier_output",
        _route_after_verifier_output,
        {"log_and_cache_write": "log_and_cache_write", "generate_strong": "generate_strong"},
    )

    graph.add_edge("log_and_cache_write", END)

    return graph.compile()
