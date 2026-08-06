"""Unit tests for verified_cost_router.router.classifier.ComplexityClassifier.

Uses a fake ChatCompletionClient (tests/fakes.py) so the prompt-building
and response-parsing logic is tested without any real model call.
"""

from __future__ import annotations

import pytest
from fakes import FakeChatCompletionClient

from verified_cost_router.router.classifier import ClassificationError, ComplexityClassifier


def test_classify_returns_simple_for_clean_simple_response():
    client = FakeChatCompletionClient(next_content="simple")
    classifier = ComplexityClassifier(client, model="test-model")
    assert classifier.classify("What is the capital of France?") == "simple"


def test_classify_returns_complex_for_clean_complex_response():
    client = FakeChatCompletionClient(next_content="complex")
    classifier = ComplexityClassifier(client, model="test-model")
    assert classifier.classify("Is it safe to mix bleach and ammonia?") == "complex"


@pytest.mark.parametrize(
    "raw_response, expected",
    [
        ("Simple", "simple"),
        ("  simple.  ", "simple"),
        ("SIMPLE", "simple"),
        ("Complex.", "complex"),
        ("  COMPLEX\n", "complex"),
    ],
)
def test_classify_is_robust_to_case_and_punctuation(raw_response: str, expected: str):
    client = FakeChatCompletionClient(next_content=raw_response)
    classifier = ComplexityClassifier(client, model="test-model")
    assert classifier.classify("some query") == expected


def test_classify_resolves_response_containing_both_words_by_first_occurrence():
    client = FakeChatCompletionClient(next_content="This is not simple, it is actually complex.")
    classifier = ComplexityClassifier(client, model="test-model")
    assert classifier.classify("some query") == "simple"


def test_classify_raises_on_unparseable_response():
    client = FakeChatCompletionClient(next_content="banana")
    classifier = ComplexityClassifier(client, model="test-model")
    with pytest.raises(ClassificationError, match="banana"):
        classifier.classify("some query")


def test_classify_sends_system_prompt_few_shot_examples_and_query():
    client = FakeChatCompletionClient(next_content="simple")
    classifier = ComplexityClassifier(client, model="test-model")

    classifier.classify("What's the airspeed velocity of an unladen swallow?")

    sent = client.last_messages
    assert sent is not None
    assert sent[0].role == "system"
    assert "simple" in sent[0].content.lower() and "complex" in sent[0].content.lower()
    # few-shot pairs are (user, assistant) turns sandwiched between system and the final query
    assert sent[1].role == "user"
    assert sent[2].role == "assistant"
    assert sent[-1].role == "user"
    assert sent[-1].content == "What's the airspeed velocity of an unladen swallow?"


def test_classify_passes_model_and_temperature_to_client():
    client = FakeChatCompletionClient(next_content="simple")
    classifier = ComplexityClassifier(client, model="my-model", temperature=0.3)

    classifier.classify("q")

    assert client.last_model == "my-model"
    assert client.last_temperature == 0.3
