"""Unit tests for verified_cost_router.pipeline.request_log."""

from __future__ import annotations

import logging

import pytest

from verified_cost_router.pipeline.request_log import (
    ModelPricing,
    RequestLog,
    RequestLogger,
    determine_path_taken,
    estimate_cost_usd,
    is_fresh_generation_path,
)
from verified_cost_router.state import LlmCallUsage


def test_determine_path_taken_high_confidence_hit():
    assert determine_path_taken({"cache_result": "high_confidence_hit"}) == "cache-hit"


def test_determine_path_taken_risky_hit_verified_pass():
    state = {"cache_result": "risky_hit", "verifier_cache_result": "pass"}
    assert determine_path_taken(state) == "cache-hit-verified"


def test_determine_path_taken_risky_hit_verified_fail_falls_to_router_simple():
    state = {
        "cache_result": "risky_hit",
        "verifier_cache_result": "fail",
        "route": "simple",
        "verifier_output_result": "pass",
    }
    assert determine_path_taken(state) == "router-8B"


def test_determine_path_taken_no_match_complex():
    state = {"cache_result": "no_match", "route": "complex"}
    assert determine_path_taken(state) == "router-70B"


def test_determine_path_taken_no_match_simple_verified_pass():
    state = {"cache_result": "no_match", "route": "simple", "verifier_output_result": "pass"}
    assert determine_path_taken(state) == "router-8B"


def test_determine_path_taken_no_match_simple_verified_fail_is_escalated():
    state = {"cache_result": "no_match", "route": "simple", "verifier_output_result": "fail"}
    assert determine_path_taken(state) == "router-8B-escalated"


def test_determine_path_taken_raises_on_incomplete_state():
    with pytest.raises(ValueError, match="could not determine path_taken"):
        determine_path_taken({"cache_result": "no_match"})


@pytest.mark.parametrize(
    "path_taken, expected",
    [
        ("cache-hit", False),
        ("cache-hit-verified", False),
        ("router-8B", True),
        ("router-8B-escalated", True),
        ("router-70B", True),
    ],
)
def test_is_fresh_generation_path(path_taken, expected):
    assert is_fresh_generation_path(path_taken) == expected


def test_estimate_cost_usd_sums_across_calls():
    pricing = {"model-a": ModelPricing(input_per_million=1.0, output_per_million=2.0)}
    calls = [
        LlmCallUsage("classify", "model-a", prompt_tokens=1_000_000, completion_tokens=0),
        LlmCallUsage("generate_cheap", "model-a", prompt_tokens=0, completion_tokens=500_000),
    ]
    cost = estimate_cost_usd(calls, pricing=pricing)
    assert cost == pytest.approx(1.0 + 1.0)  # $1 input + $1 output


def test_estimate_cost_usd_empty_calls_is_zero():
    assert estimate_cost_usd([]) == 0.0


def test_estimate_cost_usd_raises_for_unpriced_model():
    calls = [LlmCallUsage("classify", "unknown-model", prompt_tokens=10, completion_tokens=10)]
    with pytest.raises(KeyError, match="unknown-model"):
        estimate_cost_usd(calls, pricing={})


def test_estimate_cost_usd_uses_real_default_pricing_for_known_models():
    calls = [LlmCallUsage("generate_cheap", "llama-3.1-8b-instant", prompt_tokens=1_000_000, completion_tokens=0)]
    assert estimate_cost_usd(calls) == pytest.approx(0.05)


def test_request_logger_emits_one_info_line(caplog: pytest.LogCaptureFixture):
    logger = RequestLogger()
    entry = RequestLog(
        query="what is the capital of france?",
        path_taken="router-8B",
        latency_ms=123.4,
        cost_usd=0.000042,
        llm_calls=(),
    )
    with caplog.at_level(logging.INFO):
        logger.log(entry)

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "router-8B" in message
    assert "123.4" in message
