"""Composition root for the real pipeline: builds the cache, classifier,
verifier, and Groq client from resolved settings, and bundles them into
a PipelineNodes instance for graph.build_pipeline_graph.

Deliberately takes settings/thresholds as explicit arguments rather than
reading files or the environment itself, so it stays unit-testable
without real network/model/API access -- callers (main.py) that do want
the "load everything from disk and env" convenience use
load_cache_thresholds() plus config.load_groq_settings() explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

from verified_cost_router.cache.embeddings import SentenceTransformerEmbedder
from verified_cost_router.cache.semantic_cache import SemanticCache
from verified_cost_router.cache.thresholds import CacheThresholds
from verified_cost_router.config import GroqSettings
from verified_cost_router.llm.groq_client import GroqClient
from verified_cost_router.pipeline.nodes import PipelineNodes
from verified_cost_router.router.classifier import ComplexityClassifier
from verified_cost_router.verifier.verifier import Verifier


def load_cache_thresholds(path: Path) -> CacheThresholds:
    """Load the (high_confidence, risky) cutoffs tuned in Phase 2
    (data/cache_thresholds.json, produced by scripts/tune_cache_thresholds.py)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return CacheThresholds(high_confidence=data["high_confidence"], risky=data["risky"])


def build_pipeline_nodes(groq_settings: GroqSettings, cache_thresholds: CacheThresholds) -> PipelineNodes:
    """Construct every real component and bundle them into a PipelineNodes.

    The verifier runs on the cheap-tier model by design (ARCHITECTURE.md
    4.2: verification is a lighter task than generation), same as the
    router's classifier -- both share one GroqClient instance.
    """
    embedder = SentenceTransformerEmbedder()
    cache = SemanticCache(embedder, cache_thresholds)

    groq_client = GroqClient(api_key=groq_settings.api_key)
    classifier = ComplexityClassifier(groq_client, model=groq_settings.cheap_model)
    verifier = Verifier(groq_client, model=groq_settings.cheap_model)

    return PipelineNodes(
        cache=cache,
        classifier=classifier,
        verifier=verifier,
        groq_client=groq_client,
        cheap_model=groq_settings.cheap_model,
        strong_model=groq_settings.strong_model,
    )
