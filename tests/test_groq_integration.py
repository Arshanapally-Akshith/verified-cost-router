"""Real-API integration test for the Groq client + router classifier.

Skipped automatically when GROQ_API_KEY isn't set (e.g. in a fresh clone
before the key is added to .env). Everything else in the suite covers
this logic with fakes; this test exists to catch real integration drift
(wrong endpoint, wrong payload shape, a renamed/retired model) that a
mock can't -- run it once GROQ_API_KEY is set to actually verify against
the live API.
"""

from __future__ import annotations

import os

import pytest

from verified_cost_router.config import load_groq_settings
from verified_cost_router.llm.groq_client import GroqClient
from verified_cost_router.router.classifier import ComplexityClassifier

pytestmark = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY", "").strip(),
    reason="GROQ_API_KEY not set; skipping real Groq API integration test",
)


@pytest.fixture(scope="module")
def classifier() -> ComplexityClassifier:
    settings = load_groq_settings()
    client = GroqClient(api_key=settings.api_key)
    return ComplexityClassifier(client, model=settings.cheap_model)


def test_classifies_an_obviously_simple_query(classifier: ComplexityClassifier):
    assert classifier.classify("What is the capital of Japan?") == "simple"


def test_classifies_an_obviously_complex_query(classifier: ComplexityClassifier):
    label = classifier.classify(
        "Is it safe to mix bleach and ammonia in an enclosed bathroom while cleaning?"
    )
    assert label == "complex"
