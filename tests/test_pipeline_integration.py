"""Real end-to-end integration test for the Phase 4 pipeline.

Skipped automatically when GROQ_API_KEY isn't set. Every node's own
logic is covered offline in test_pipeline_nodes.py with fakes; this test
exists to catch real integration drift across cache + router + verifier
+ generation + Groq all wired together, which no combination of fakes
can fully guarantee -- run it once GROQ_API_KEY is set to verify against
the live API and a real embedding model.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from verified_cost_router.config import load_groq_settings
from verified_cost_router.graph import build_pipeline_graph
from verified_cost_router.pipeline.dependencies import build_pipeline_nodes, load_cache_thresholds

pytestmark = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY", "").strip(),
    reason="GROQ_API_KEY not set; skipping real pipeline integration test",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS_PATH = REPO_ROOT / "data" / "cache_thresholds.json"


@pytest.fixture(scope="module")
def app():
    settings = load_groq_settings()
    thresholds = load_cache_thresholds(THRESHOLDS_PATH)
    nodes = build_pipeline_nodes(settings, thresholds)
    return build_pipeline_graph(nodes)


def test_simple_query_produces_a_response(app):
    result = app.invoke({"query": "What is the capital of Japan?"})

    assert result["response"]
    assert result["visited"][0] == "cache_check"
    assert result["visited"][-1] == "log_and_cache_write"
    assert result["route"] in ("simple", "complex")


def test_repeating_the_same_query_hits_the_cache(app):
    query = "What is the boiling point of water at sea level, in Celsius?"
    first = app.invoke({"query": query})
    second = app.invoke({"query": query})

    assert second["cache_result"] in ("risky_hit", "high_confidence_hit")
    assert second["response"]
    assert first["response"]
