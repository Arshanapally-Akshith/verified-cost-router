"""Tests for the adversarial eval set loader and its schema validation.

Loads the real committed data/adversarial_eval_set.json (the hand-built
set from BUILD.md section 2) to confirm it is well-formed, and exercises
the validator's failure modes against small synthetic fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verified_cost_router.data_prep.adversarial_eval import (
    MAX_TOTAL_ITEMS,
    MIN_TOTAL_ITEMS,
    load_adversarial_eval_set,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SET_PATH = REPO_ROOT / "data" / "adversarial_eval_set.json"


def test_real_eval_set_loads_within_expected_size_range():
    eval_set = load_adversarial_eval_set(EVAL_SET_PATH)
    assert MIN_TOTAL_ITEMS <= eval_set.total_items <= MAX_TOTAL_ITEMS


def test_real_eval_set_has_both_cache_pair_categories():
    eval_set = load_adversarial_eval_set(EVAL_SET_PATH)
    categories = {pair.category for pair in eval_set.cache_pairs}
    assert categories == {"true_duplicate", "near_miss"}


def test_real_eval_set_expect_cache_hit_matches_category():
    eval_set = load_adversarial_eval_set(EVAL_SET_PATH)
    for pair in eval_set.cache_pairs:
        expected = pair.category == "true_duplicate"
        assert pair.expect_cache_hit is expected


def test_real_eval_set_complexity_items_are_all_labeled_complex():
    eval_set = load_adversarial_eval_set(EVAL_SET_PATH)
    assert eval_set.complexity_items
    assert all(item.true_complexity == "complex" for item in eval_set.complexity_items)


def test_real_eval_set_ids_are_unique():
    eval_set = load_adversarial_eval_set(EVAL_SET_PATH)
    ids = [pair.id for pair in eval_set.cache_pairs] + [item.id for item in eval_set.complexity_items]
    assert len(ids) == len(set(ids))


def test_real_eval_set_pairs_have_non_trivial_rationale():
    eval_set = load_adversarial_eval_set(EVAL_SET_PATH)
    for pair in eval_set.cache_pairs:
        assert len(pair.rationale.strip()) > 10
    for item in eval_set.complexity_items:
        assert len(item.rationale.strip()) > 10


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "eval_set.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_rejects_true_duplicate_with_expect_cache_hit_false(tmp_path: Path):
    payload = {
        "cache_pairs": [
            {
                "id": "dup-001",
                "category": "true_duplicate",
                "query_a": "a",
                "query_b": "b",
                "expect_cache_hit": False,
                "rationale": "r",
            }
        ],
        "complexity_items": [],
    }
    with pytest.raises(ValueError, match="true_duplicate must expect a cache hit"):
        load_adversarial_eval_set(_write(tmp_path, payload))


def test_rejects_near_miss_with_expect_cache_hit_true(tmp_path: Path):
    payload = {
        "cache_pairs": [
            {
                "id": "nm-001",
                "category": "near_miss",
                "query_a": "a",
                "query_b": "b",
                "expect_cache_hit": True,
                "rationale": "r",
            }
        ],
        "complexity_items": [],
    }
    with pytest.raises(ValueError, match="near_miss must not expect a cache hit"):
        load_adversarial_eval_set(_write(tmp_path, payload))


def test_rejects_identical_query_a_and_query_b(tmp_path: Path):
    payload = {
        "cache_pairs": [
            {
                "id": "dup-001",
                "category": "true_duplicate",
                "query_a": "same text",
                "query_b": "same text",
                "expect_cache_hit": True,
                "rationale": "r",
            }
        ],
        "complexity_items": [],
    }
    with pytest.raises(ValueError, match="must differ in wording"):
        load_adversarial_eval_set(_write(tmp_path, payload))


def test_rejects_complexity_item_with_simple_true_complexity(tmp_path: Path):
    payload = {
        "cache_pairs": [],
        "complexity_items": [
            {
                "id": "cx-001",
                "category": "complexity_mislabeled",
                "query": "q",
                "true_complexity": "simple",
                "rationale": "r",
            }
        ],
    }
    with pytest.raises(ValueError, match="true_complexity='complex'"):
        load_adversarial_eval_set(_write(tmp_path, payload))


def test_rejects_duplicate_ids(tmp_path: Path):
    item = {
        "id": "cx-001",
        "category": "complexity_mislabeled",
        "query": "q",
        "true_complexity": "complex",
        "rationale": "r",
    }
    payload = {"cache_pairs": [], "complexity_items": [item, dict(item, query="q2")]}
    with pytest.raises(ValueError, match="duplicate eval item ids"):
        load_adversarial_eval_set(_write(tmp_path, payload))


def test_rejects_item_count_outside_expected_range(tmp_path: Path):
    items = [
        {
            "id": f"cx-{i:03d}",
            "category": "complexity_mislabeled",
            "query": "q",
            "true_complexity": "complex",
            "rationale": "r",
        }
        for i in range(3)
    ]
    payload = {"cache_pairs": [], "complexity_items": items}
    with pytest.raises(ValueError, match="expected 150-200 total eval items"):
        load_adversarial_eval_set(_write(tmp_path, payload))
