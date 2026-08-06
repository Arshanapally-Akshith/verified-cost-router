"""Unit tests for verified_cost_router.cache.thresholds."""

from __future__ import annotations

import pytest

from verified_cost_router.cache.thresholds import CacheThresholds, classify_similarity


def test_valid_thresholds_construct():
    thresholds = CacheThresholds(high_confidence=0.9, risky=0.75)
    assert thresholds.high_confidence == 0.9
    assert thresholds.risky == 0.75


@pytest.mark.parametrize(
    "high_confidence, risky",
    [
        (0.8, 0.9),  # risky above high_confidence
        (0.8, 0.8),  # equal, not strictly increasing
        (1.5, 0.5),  # out of [-1, 1] range
        (0.8, -1.5),
    ],
)
def test_invalid_thresholds_raise(high_confidence, risky):
    with pytest.raises(ValueError):
        CacheThresholds(high_confidence=high_confidence, risky=risky)


def test_classify_similarity_at_and_above_high_confidence():
    thresholds = CacheThresholds(high_confidence=0.9, risky=0.7)
    assert classify_similarity(0.9, thresholds) == "high_confidence_hit"
    assert classify_similarity(0.99, thresholds) == "high_confidence_hit"


def test_classify_similarity_in_risky_band():
    thresholds = CacheThresholds(high_confidence=0.9, risky=0.7)
    assert classify_similarity(0.7, thresholds) == "risky_hit"
    assert classify_similarity(0.89, thresholds) == "risky_hit"


def test_classify_similarity_below_risky_is_no_match():
    thresholds = CacheThresholds(high_confidence=0.9, risky=0.7)
    assert classify_similarity(0.69, thresholds) == "no_match"
    assert classify_similarity(-1.0, thresholds) == "no_match"
