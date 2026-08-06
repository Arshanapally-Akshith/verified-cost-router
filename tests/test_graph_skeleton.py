"""Phase 0 acceptance test: the graph compiles and every edge in
ARCHITECTURE.md section 2 is reachable and lands on log_and_cache_write.

Stub nodes pass through whatever decision fields are seeded in the input
state (see nodes.py), so each branch can be driven deterministically by
seeding the relevant field(s) in the initial state.
"""

from __future__ import annotations

import pytest

from verified_cost_router.graph import build_graph


@pytest.fixture(scope="module")
def app():
    return build_graph()


def test_high_confidence_cache_hit_serves_directly(app):
    result = app.invoke({"query": "q", "cache_result": "high_confidence_hit"})
    assert result["visited"] == ["cache_check", "log_and_cache_write"]


def test_risky_hit_verified_pass_serves_from_cache(app):
    result = app.invoke(
        {"query": "q", "cache_result": "risky_hit", "verifier_cache_result": "pass"}
    )
    assert result["visited"] == ["cache_check", "verifier_cache", "log_and_cache_write"]


def test_risky_hit_verified_fail_falls_back_to_router(app):
    result = app.invoke(
        {
            "query": "q",
            "cache_result": "risky_hit",
            "verifier_cache_result": "fail",
            "route": "simple",
            "verifier_output_result": "pass",
        }
    )
    assert result["visited"] == [
        "cache_check",
        "verifier_cache",
        "router",
        "generate_cheap",
        "verifier_output",
        "log_and_cache_write",
    ]


def test_no_match_complex_route_goes_straight_to_strong_model(app):
    result = app.invoke({"query": "q", "cache_result": "no_match", "route": "complex"})
    assert result["visited"] == [
        "cache_check",
        "router",
        "generate_strong",
        "log_and_cache_write",
    ]


def test_no_match_simple_route_verified_pass_serves_cheap_output(app):
    result = app.invoke(
        {
            "query": "q",
            "cache_result": "no_match",
            "route": "simple",
            "verifier_output_result": "pass",
        }
    )
    assert result["visited"] == [
        "cache_check",
        "router",
        "generate_cheap",
        "verifier_output",
        "log_and_cache_write",
    ]


def test_no_match_simple_route_verified_fail_escalates_to_strong_model(app):
    result = app.invoke(
        {
            "query": "q",
            "cache_result": "no_match",
            "route": "simple",
            "verifier_output_result": "fail",
        }
    )
    assert result["visited"] == [
        "cache_check",
        "router",
        "generate_cheap",
        "verifier_output",
        "generate_strong",
        "log_and_cache_write",
    ]


def test_default_state_runs_end_to_end_without_seeding(app):
    result = app.invoke({"query": "no overrides supplied"})
    assert result["visited"][0] == "cache_check"
    assert result["visited"][-1] == "log_and_cache_write"
