"""Cache precision/recall against the labeled adversarial cache_pairs
(BUILD.md section 4: "does it correctly avoid the opposite-meaning-
similar-wording traps").

Evaluates the real SemanticCache class end-to-end (embed -> FAISS
search -> threshold classification) rather than raw cosine similarity,
so this measures actual system behavior, not just the embedding model.
Each pair gets its own fresh, isolated cache so pairs can never leak
into each other's results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from verified_cost_router.cache.embeddings import EmbeddingModel
from verified_cost_router.cache.semantic_cache import SemanticCache
from verified_cost_router.cache.thresholds import CacheThresholds, MatchCategory
from verified_cost_router.data_prep.adversarial_eval import CachePair


@dataclass(frozen=True)
class CachePairOutcome:
    """One pair's ground truth vs. what the real cache actually did."""

    pair_id: str
    category: str  # "true_duplicate" | "near_miss"
    expect_cache_hit: bool
    predicted_hit: bool
    match_category: MatchCategory


@dataclass(frozen=True)
class CacheEvalResult:
    outcomes: tuple[CachePairOutcome, ...]

    @property
    def precision(self) -> float:
        """Of pairs the cache flagged as any kind of match, how many actually should be."""
        predicted_positive = [o for o in self.outcomes if o.predicted_hit]
        if not predicted_positive:
            return 1.0
        true_positive = sum(1 for o in predicted_positive if o.expect_cache_hit)
        return true_positive / len(predicted_positive)

    @property
    def recall(self) -> float:
        """Of true_duplicate pairs, how many the cache actually flagged (any band)."""
        actual_positive = [o for o in self.outcomes if o.expect_cache_hit]
        if not actual_positive:
            return 1.0
        true_positive = sum(1 for o in actual_positive if o.predicted_hit)
        return true_positive / len(actual_positive)


def evaluate_cache_pairs(
    pairs: Sequence[CachePair], embedder: EmbeddingModel, thresholds: CacheThresholds
) -> CacheEvalResult:
    """Run every labeled pair through a fresh SemanticCache: put(query_a),
    lookup(query_b), and check whether the resulting match category is
    consistent with the pair's expect_cache_hit label."""
    outcomes = []
    for pair in pairs:
        cache = SemanticCache(embedder, thresholds)
        cache.put(pair.query_a, "dummy-answer")
        result = cache.lookup(pair.query_b)
        outcomes.append(
            CachePairOutcome(
                pair_id=pair.id,
                category=pair.category,
                expect_cache_hit=pair.expect_cache_hit,
                predicted_hit=result.category != "no_match",
                match_category=result.category,
            )
        )
    return CacheEvalResult(outcomes=tuple(outcomes))
