"""Unit tests for verified_cost_router.verifier.verifier.Verifier.

Uses a fake ChatCompletionClient (tests/fakes.py) so prompt-building and
response-parsing are tested without any real model call.
"""

from __future__ import annotations

import pytest
from fakes import FakeChatCompletionClient

from verified_cost_router.verifier.verifier import VerificationError, VerificationOutcome, Verifier


def test_verify_cache_hit_returns_pass():
    client = FakeChatCompletionClient(next_content="pass")
    verifier = Verifier(client, model="test-model")
    outcome = verifier.verify_cache_hit(
        query="How do I lower my blood pressure?",
        cached_query="How do I lower my blood pressure naturally?",
        cached_response="Reduce sodium, exercise regularly, manage stress.",
    )
    assert outcome.label == "pass"


def test_verify_cache_hit_returns_fail():
    client = FakeChatCompletionClient(next_content="fail")
    verifier = Verifier(client, model="test-model")
    outcome = verifier.verify_cache_hit(
        query="How do I raise my blood pressure?",
        cached_query="How do I lower my blood pressure naturally?",
        cached_response="Reduce sodium, exercise regularly, manage stress.",
    )
    assert outcome.label == "fail"


def test_verify_output_returns_pass():
    client = FakeChatCompletionClient(next_content="pass")
    verifier = Verifier(client, model="test-model")
    outcome = verifier.verify_output(query="What is 2+2?", output="4")
    assert outcome.label == "pass"


def test_verify_output_returns_fail():
    client = FakeChatCompletionClient(next_content="fail")
    verifier = Verifier(client, model="test-model")
    outcome = verifier.verify_output(query="What is 2+2?", output="5")
    assert outcome.label == "fail"


@pytest.mark.parametrize(
    "raw_response, expected",
    [
        ("Pass", "pass"),
        ("  pass.  ", "pass"),
        ("PASS", "pass"),
        ("Fail.", "fail"),
        ("  FAIL\n", "fail"),
    ],
)
def test_verify_is_robust_to_case_and_punctuation(raw_response: str, expected: str):
    client = FakeChatCompletionClient(next_content=raw_response)
    verifier = Verifier(client, model="test-model")
    assert verifier.verify_output("q", "a").label == expected


def test_verify_resolves_response_containing_both_words_by_first_occurrence():
    client = FakeChatCompletionClient(next_content="This does not pass, it should fail.")
    verifier = Verifier(client, model="test-model")
    assert verifier.verify_output("q", "a").label == "pass"


def test_verify_raises_on_unparseable_response():
    client = FakeChatCompletionClient(next_content="banana")
    verifier = Verifier(client, model="test-model")
    with pytest.raises(VerificationError, match="banana"):
        verifier.verify_output("q", "a")


def test_verify_cache_hit_includes_new_and_cached_content_in_prompt():
    client = FakeChatCompletionClient(next_content="pass")
    verifier = Verifier(client, model="test-model")

    verifier.verify_cache_hit(
        query="new question here", cached_query="cached question here", cached_response="cached answer here"
    )

    sent = client.last_messages
    assert sent is not None
    assert sent[0].role == "system"
    user_content = sent[1].content
    assert "new question here" in user_content
    assert "cached question here" in user_content
    assert "cached answer here" in user_content


def test_verify_output_includes_question_and_answer_in_prompt():
    client = FakeChatCompletionClient(next_content="pass")
    verifier = Verifier(client, model="test-model")

    verifier.verify_output(query="my question", output="my proposed answer")

    user_content = client.last_messages[1].content
    assert "my question" in user_content
    assert "my proposed answer" in user_content


def test_verify_returns_token_usage():
    client = FakeChatCompletionClient(next_content="pass")
    verifier = Verifier(client, model="my-model")

    outcome = verifier.verify_output("q", "a")

    assert outcome == VerificationOutcome(label="pass", model="my-model", prompt_tokens=7, completion_tokens=1)


def test_verify_passes_model_and_temperature_to_client():
    client = FakeChatCompletionClient(next_content="pass")
    verifier = Verifier(client, model="my-model", temperature=0.1)

    verifier.verify_output("q", "a")

    assert client.last_model == "my-model"
    assert client.last_temperature == 0.1
