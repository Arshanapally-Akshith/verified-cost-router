"""Real end-to-end integration test for the Phase 5 eval harness pieces.

Skipped automatically when GROQ_API_KEY isn't set. Every eval function's
own logic is covered offline in test_cache_eval.py / test_router_eval.py
/ test_verifier_eval.py / test_baselines.py / test_quality_eval.py with
fakes; this test exists to catch real integration drift across the
actual cache, router, verifier, and generation components working
together on a tiny slice of real data -- run it once GROQ_API_KEY is
set to verify against the live API.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from verified_cost_router.cache.embeddings import SentenceTransformerEmbedder
from verified_cost_router.cache.semantic_cache import SemanticCache
from verified_cost_router.config import load_groq_settings
from verified_cost_router.data_prep.adversarial_eval import load_adversarial_eval_set
from verified_cost_router.eval.baselines import run_cache_router_no_verifier, run_full_system, run_no_system
from verified_cost_router.eval.cache_eval import evaluate_cache_pairs
from verified_cost_router.eval.quality_eval import spot_check_quality
from verified_cost_router.eval.router_eval import evaluate_complexity_items
from verified_cost_router.eval.verifier_eval import evaluate_cache_verifier
from verified_cost_router.graph import build_pipeline_graph
from verified_cost_router.llm.groq_client import GroqClient
from verified_cost_router.pipeline.dependencies import load_cache_thresholds
from verified_cost_router.pipeline.nodes import PipelineNodes
from verified_cost_router.pipeline.request_log import DEFAULT_PRICING
from verified_cost_router.router.classifier import ComplexityClassifier
from verified_cost_router.verifier.verifier import Verifier

pytestmark = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY", "").strip(),
    reason="GROQ_API_KEY not set; skipping real eval harness integration test",
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def components() -> dict:
    settings = load_groq_settings()
    thresholds = load_cache_thresholds(REPO_ROOT / "data" / "cache_thresholds.json")
    eval_set = load_adversarial_eval_set(REPO_ROOT / "data" / "adversarial_eval_set.json")
    embedder = SentenceTransformerEmbedder()
    groq_client = GroqClient(api_key=settings.api_key)
    classifier = ComplexityClassifier(groq_client, model=settings.cheap_model)
    verifier = Verifier(groq_client, model=settings.cheap_model)
    return {
        "settings": settings,
        "thresholds": thresholds,
        "eval_set": eval_set,
        "embedder": embedder,
        "groq_client": groq_client,
        "classifier": classifier,
        "verifier": verifier,
    }


def test_cache_eval_runs_against_a_small_real_slice(components: dict):
    pairs = components["eval_set"].cache_pairs[:6]
    result = evaluate_cache_pairs(pairs, components["embedder"], components["thresholds"])
    assert len(result.outcomes) == 6
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0


def test_router_eval_runs_against_a_small_real_slice(components: dict):
    items = components["eval_set"].complexity_items[:3]
    result = evaluate_complexity_items(items, components["classifier"])
    assert len(result.outcomes) == 3


def test_cache_verifier_eval_runs_against_a_small_real_slice(components: dict):
    pairs = components["eval_set"].cache_pairs[:6]
    result = evaluate_cache_verifier(pairs, components["embedder"], components["thresholds"], components["verifier"])
    assert len(result.reached_verifier) + len(result.skipped) == 6


def test_baselines_run_over_two_real_queries(components: dict):
    queries = ["What is the capital of Japan?", "How do I reverse a string in Python?"]
    settings = components["settings"]

    no_system = [
        run_no_system(q, components["groq_client"], settings.strong_model, DEFAULT_PRICING) for q in queries
    ]
    assert all(r.response for r in no_system)

    cache_2 = SemanticCache(components["embedder"], components["thresholds"])
    cache_router = [
        run_cache_router_no_verifier(
            q,
            cache_2,
            components["classifier"],
            components["groq_client"],
            settings.cheap_model,
            settings.strong_model,
            DEFAULT_PRICING,
        )
        for q in queries
    ]
    assert all(r.response for r in cache_router)

    cache_3 = SemanticCache(components["embedder"], components["thresholds"])
    nodes = PipelineNodes(
        cache=cache_3,
        classifier=components["classifier"],
        verifier=components["verifier"],
        groq_client=components["groq_client"],
        cheap_model=settings.cheap_model,
        strong_model=settings.strong_model,
    )
    app = build_pipeline_graph(nodes)
    full_system = [run_full_system(q, app, DEFAULT_PRICING) for q in queries]
    assert all(r.response for r in full_system)


def test_quality_spot_check_runs_on_one_real_response(components: dict):
    served = [("What is the capital of Japan?", "Tokyo is the capital of Japan.")]
    result = spot_check_quality(
        served, components["groq_client"], components["settings"].strong_model, 1, random.Random(0)
    )
    assert len(result.items) == 1
    assert result.items[0].verdict in ("comparable", "worse")
