"""Loader and schema validation for the hand-built adversarial eval set.

The eval set (data/adversarial_eval_set.json) is authored by hand per
BUILD.md section 2: true-duplicate pairs, opposite-meaning near-miss
pairs, and complexity-mislabeled queries. This module only loads and
validates that fixed file -- generating candidates and manually
verifying each label is the point of Phase 1 and is not automated here.
Scoring against it is Phase 2 (cache thresholds) and Phase 3/5 (router
accuracy) work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CachePairCategory = Literal["true_duplicate", "near_miss"]
ComplexityLabel = Literal["simple", "complex"]

MIN_TOTAL_ITEMS = 150
MAX_TOTAL_ITEMS = 200


@dataclass(frozen=True)
class CachePair:
    """A labeled query pair used to evaluate the cache layer's match decision."""

    id: str
    category: CachePairCategory
    query_a: str
    query_b: str
    expect_cache_hit: bool
    rationale: str


@dataclass(frozen=True)
class ComplexityItem:
    """A simply-worded query whose true complexity requires the strong model."""

    id: str
    category: Literal["complexity_mislabeled"]
    query: str
    true_complexity: ComplexityLabel
    rationale: str


@dataclass(frozen=True)
class AdversarialEvalSet:
    cache_pairs: tuple[CachePair, ...]
    complexity_items: tuple[ComplexityItem, ...]

    @property
    def total_items(self) -> int:
        return len(self.cache_pairs) + len(self.complexity_items)


def load_adversarial_eval_set(path: Path) -> AdversarialEvalSet:
    """Load and validate the labeled adversarial eval set from `path`.

    Raises ValueError if the file has duplicate ids, labels inconsistent
    with their category, or a total item count outside the ~150-200
    range BUILD.md specifies.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))

    cache_pairs = tuple(_parse_cache_pair(item) for item in raw.get("cache_pairs", []))
    complexity_items = tuple(_parse_complexity_item(item) for item in raw.get("complexity_items", []))

    eval_set = AdversarialEvalSet(cache_pairs=cache_pairs, complexity_items=complexity_items)
    _validate(eval_set)
    return eval_set


def _parse_cache_pair(item: dict) -> CachePair:
    pair = CachePair(
        id=item["id"],
        category=item["category"],
        query_a=item["query_a"],
        query_b=item["query_b"],
        expect_cache_hit=item["expect_cache_hit"],
        rationale=item["rationale"],
    )
    if pair.category not in ("true_duplicate", "near_miss"):
        raise ValueError(f"{pair.id}: unknown cache pair category {pair.category!r}")
    if pair.category == "true_duplicate" and not pair.expect_cache_hit:
        raise ValueError(f"{pair.id}: true_duplicate must expect a cache hit")
    if pair.category == "near_miss" and pair.expect_cache_hit:
        raise ValueError(f"{pair.id}: near_miss must not expect a cache hit")
    if not pair.query_a.strip() or not pair.query_b.strip():
        raise ValueError(f"{pair.id}: empty query text")
    if pair.query_a.strip() == pair.query_b.strip():
        raise ValueError(f"{pair.id}: query_a and query_b must differ in wording")
    return pair


def _parse_complexity_item(item: dict) -> ComplexityItem:
    complexity_item = ComplexityItem(
        id=item["id"],
        category=item["category"],
        query=item["query"],
        true_complexity=item["true_complexity"],
        rationale=item["rationale"],
    )
    if complexity_item.category != "complexity_mislabeled":
        raise ValueError(
            f"{complexity_item.id}: unknown complexity item category {complexity_item.category!r}"
        )
    if complexity_item.true_complexity != "complex":
        raise ValueError(
            f"{complexity_item.id}: complexity_mislabeled items must have "
            "true_complexity='complex' (the query looks simple but truly "
            "needs the strong model)"
        )
    if not complexity_item.query.strip():
        raise ValueError(f"{complexity_item.id}: empty query text")
    return complexity_item


def _validate(eval_set: AdversarialEvalSet) -> None:
    ids = [pair.id for pair in eval_set.cache_pairs] + [item.id for item in eval_set.complexity_items]
    duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate eval item ids: {duplicates}")
    if not (MIN_TOTAL_ITEMS <= eval_set.total_items <= MAX_TOTAL_ITEMS):
        raise ValueError(
            f"expected {MIN_TOTAL_ITEMS}-{MAX_TOTAL_ITEMS} total eval items, "
            f"got {eval_set.total_items}"
        )
