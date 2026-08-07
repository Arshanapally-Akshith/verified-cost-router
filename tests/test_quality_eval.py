"""Unit tests for verified_cost_router.eval.quality_eval."""

from __future__ import annotations

import random

import pytest
from fakes import FakeChatCompletionClient

from verified_cost_router.eval.quality_eval import JudgeError, judge_response, spot_check_quality


def test_judge_response_returns_comparable():
    client = FakeChatCompletionClient(next_content="comparable")
    verdict = judge_response(client, "judge-model", "q", "served", "reference")
    assert verdict == "comparable"


def test_judge_response_returns_worse():
    client = FakeChatCompletionClient(next_content="worse")
    verdict = judge_response(client, "judge-model", "q", "served", "reference")
    assert verdict == "worse"


@pytest.mark.parametrize(
    "raw_response, expected",
    [("Comparable", "comparable"), ("  comparable.  ", "comparable"), ("Worse.", "worse"), ("  WORSE\n", "worse")],
)
def test_judge_response_is_robust_to_case_and_punctuation(raw_response, expected):
    client = FakeChatCompletionClient(next_content=raw_response)
    assert judge_response(client, "judge-model", "q", "a", "b") == expected


def test_judge_response_raises_on_unparseable_response():
    client = FakeChatCompletionClient(next_content="banana")
    with pytest.raises(JudgeError, match="banana"):
        judge_response(client, "judge-model", "q", "a", "b")


def test_judge_response_includes_query_served_and_reference_in_prompt():
    client = FakeChatCompletionClient(next_content="comparable")
    judge_response(client, "judge-model", "my query", "my served answer", "my reference answer")

    user_content = client.last_messages[1].content
    assert "my query" in user_content
    assert "my served answer" in user_content
    assert "my reference answer" in user_content


def test_spot_check_quality_samples_up_to_requested_size():
    client = FakeChatCompletionClient(next_content="comparable")
    served = [(f"q{i}", f"served-{i}") for i in range(10)]

    result = spot_check_quality(served, client, "strong-model", sample_size=3, rng=random.Random(42))

    assert len(result.items) == 3
    assert result.comparable_rate == 1.0


def test_spot_check_quality_caps_sample_size_at_available_items():
    client = FakeChatCompletionClient(next_content="comparable")
    served = [("q1", "served-1"), ("q2", "served-2")]

    result = spot_check_quality(served, client, "strong-model", sample_size=10, rng=random.Random(0))

    assert len(result.items) == 2


def test_spot_check_quality_is_deterministic_given_the_same_rng_seed():
    client = FakeChatCompletionClient(next_content="comparable")
    served = [(f"q{i}", f"served-{i}") for i in range(20)]

    result_a = spot_check_quality(served, client, "strong-model", sample_size=5, rng=random.Random(7))
    result_b = spot_check_quality(served, client, "strong-model", sample_size=5, rng=random.Random(7))

    assert [item.query for item in result_a.items] == [item.query for item in result_b.items]


def test_spot_check_quality_generates_a_fresh_strong_model_reference_per_item():
    client = FakeChatCompletionClient(next_content="comparable")
    served = [("q1", "served-1")]

    spot_check_quality(served, client, "strong-model", sample_size=1, rng=random.Random(0))

    assert client.last_model == "strong-model"


def test_spot_check_quality_skips_only_the_failing_item_not_the_whole_batch():
    class _JudgeRefusesForQ2Client:
        def chat_completion(self, model, messages, *, temperature=0.0, max_tokens=None):
            from verified_cost_router.llm.groq_client import ChatCompletionResult

            user_content = messages[-1].content
            if "SERVED answer: served-2" in user_content:
                content = "I'm not able to make that comparison."
            else:
                content = "comparable"
            return ChatCompletionResult(content=content, model=model, prompt_tokens=1, completion_tokens=1)

    served = [("q1", "served-1"), ("q2", "served-2")]
    result = spot_check_quality(served, _JudgeRefusesForQ2Client(), "strong-model", sample_size=2, rng=random.Random(0))

    assert len(result.items) == 1
    assert result.items[0].query == "q1"


def test_spot_check_quality_mixed_verdicts_compute_correct_rate():
    class _AlternatingClient:
        def __init__(self):
            self.calls = 0

        def chat_completion(self, model, messages, *, temperature=0.0, max_tokens=None):
            from verified_cost_router.llm.groq_client import ChatCompletionResult

            self.calls += 1
            # Each item makes 2 calls (reference generation, then judge).
            # Alternate the judge's verdict per item.
            item_index = (self.calls - 1) // 2
            content = "comparable" if item_index % 2 == 0 else "worse"
            return ChatCompletionResult(content=content, model=model, prompt_tokens=1, completion_tokens=1)

    served = [(f"q{i}", f"served-{i}") for i in range(4)]
    result = spot_check_quality(served, _AlternatingClient(), "strong-model", sample_size=4, rng=random.Random(1))

    assert result.comparable_rate == 0.5
