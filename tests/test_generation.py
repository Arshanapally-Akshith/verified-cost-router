"""Unit tests for verified_cost_router.llm.generation.generate."""

from __future__ import annotations

from fakes import FakeChatCompletionClient

from verified_cost_router.llm.generation import generate


def test_generate_returns_client_result():
    client = FakeChatCompletionClient(next_content="Paris is the capital of France.")
    result = generate(client, model="my-model", query="What is the capital of France?")
    assert result.content == "Paris is the capital of France."
    assert result.model == "my-model"


def test_generate_sends_system_prompt_and_query():
    client = FakeChatCompletionClient(next_content="answer")
    generate(client, model="my-model", query="What is the capital of France?")

    sent = client.last_messages
    assert sent is not None
    assert sent[0].role == "system"
    assert sent[1].role == "user"
    assert sent[1].content == "What is the capital of France?"


def test_generate_passes_model_and_temperature():
    client = FakeChatCompletionClient(next_content="answer")
    generate(client, model="my-model", query="q", temperature=0.7)

    assert client.last_model == "my-model"
    assert client.last_temperature == 0.7


def test_generate_default_temperature():
    client = FakeChatCompletionClient(next_content="answer")
    generate(client, model="my-model", query="q")

    assert client.last_temperature == 0.2
