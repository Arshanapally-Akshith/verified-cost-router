"""Tests for verified_cost_router.pipeline.dependencies.

build_pipeline_nodes constructs real components (SentenceTransformerEmbedder,
GroqClient, ComplexityClassifier, Verifier), but none of their
constructors make a network/API call -- that only happens on first
embed()/chat_completion() use -- so this is safe and fast to exercise
directly rather than mocking, and doubles as a check that the wiring
itself (which settings go where) is correct.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verified_cost_router.cache.semantic_cache import SemanticCache
from verified_cost_router.cache.thresholds import CacheThresholds
from verified_cost_router.config import GroqSettings
from verified_cost_router.pipeline.dependencies import build_pipeline_nodes, load_cache_thresholds
from verified_cost_router.pipeline.nodes import PipelineNodes
from verified_cost_router.router.classifier import ComplexityClassifier
from verified_cost_router.verifier.verifier import Verifier


def test_load_cache_thresholds_reads_high_confidence_and_risky(tmp_path: Path):
    path = tmp_path / "cache_thresholds.json"
    path.write_text(json.dumps({"high_confidence": 0.86, "risky": 0.5, "extra_field": "ignored"}), encoding="utf-8")

    thresholds = load_cache_thresholds(path)

    assert thresholds == CacheThresholds(high_confidence=0.86, risky=0.5)


def test_build_pipeline_nodes_wires_correct_models_and_types():
    settings = GroqSettings(api_key="test-key", cheap_model="cheap-x", strong_model="strong-y")
    thresholds = CacheThresholds(high_confidence=0.9, risky=0.7)

    nodes = build_pipeline_nodes(settings, thresholds)

    assert isinstance(nodes, PipelineNodes)
    assert isinstance(nodes._cache, SemanticCache)  # noqa: SLF001 -- structural wiring check
    assert isinstance(nodes._classifier, ComplexityClassifier)  # noqa: SLF001
    assert isinstance(nodes._verifier, Verifier)  # noqa: SLF001
    assert nodes._cheap_model == "cheap-x"  # noqa: SLF001
    assert nodes._strong_model == "strong-y"  # noqa: SLF001


def test_build_pipeline_nodes_classifier_and_verifier_use_cheap_model():
    settings = GroqSettings(api_key="test-key", cheap_model="cheap-x", strong_model="strong-y")
    thresholds = CacheThresholds(high_confidence=0.9, risky=0.7)

    nodes = build_pipeline_nodes(settings, thresholds)

    assert nodes._classifier._model == "cheap-x"  # noqa: SLF001
    assert nodes._verifier._model == "cheap-x"  # noqa: SLF001 -- ARCHITECTURE.md 4.2: verifier stays cheap
