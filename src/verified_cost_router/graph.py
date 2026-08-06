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

This module owns only the graph shape and the conditional-edge routing
logic -- it never cares which module supplies node behavior, only that
`state["cache_result"]` etc. are set once the corresponding node has run.
That's what makes two callers possible from the same topology:

- build_graph(): wires the Phase 0 stub nodes (verified_cost_router.nodes).
  Kept unchanged for tests/test_graph_skeleton.py, which proves the graph
  shape/edges without any real cache, router, or verifier logic.
- build_pipeline_graph(nodes): wires a pipeline.nodes.PipelineNodes
  instance -- the real Phase 4 pipeline, identical topology.
"""

from __future__ import annotations

from typing import Protocol

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from verified_cost_router import nodes as stub_nodes
from verified_cost_router.state import GraphState


class NodeProvider(Protocol):
    """Structural shape both the stub `nodes` module and PipelineNodes satisfy."""

    def cache_check(self, state: GraphState) -> dict: ...
    def verifier_cache(self, state: GraphState) -> dict: ...
    def router(self, state: GraphState) -> dict: ...
    def generate_cheap(self, state: GraphState) -> dict: ...
    def generate_strong(self, state: GraphState) -> dict: ...
    def verifier_output(self, state: GraphState) -> dict: ...
    def log_and_cache_write(self, state: GraphState) -> dict: ...


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


def _wire(node_provider: NodeProvider) -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("cache_check", node_provider.cache_check)
    graph.add_node("verifier_cache", node_provider.verifier_cache)
    graph.add_node("router", node_provider.router)
    graph.add_node("generate_cheap", node_provider.generate_cheap)
    graph.add_node("generate_strong", node_provider.generate_strong)
    graph.add_node("verifier_output", node_provider.verifier_output)
    graph.add_node("log_and_cache_write", node_provider.log_and_cache_write)

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

    return graph


def build_graph() -> CompiledStateGraph:
    """The Phase 0 walking-skeleton graph: stub nodes only. Proves the
    topology/edges in tests/test_graph_skeleton.py -- not for real use,
    see build_pipeline_graph()."""
    return _wire(stub_nodes).compile()


def build_pipeline_graph(nodes: NodeProvider) -> CompiledStateGraph:
    """The real Phase 4 pipeline: identical topology to build_graph(),
    wired to real cache/router/verifier/generation nodes instead of stubs.

    `nodes` is typically a pipeline.nodes.PipelineNodes built via
    pipeline.dependencies.build_pipeline_nodes().
    """
    return _wire(nodes).compile()
