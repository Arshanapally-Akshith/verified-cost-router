"""Reproducible threshold selection for the semantic cache, swept against
the labeled adversarial eval set instead of hand-picked (ARCHITECTURE.md
4.1: "the selection method has to be reproducible").

Metric definitions (this project's choice, since ARCHITECTURE.md
specifies the goal -- "maximize precision at an acceptable recall" --
but not the exact formula):

- high_confidence_precision: of cache_pairs classified "high_confidence_hit"
  (which the real pipeline would auto-serve with no verification), what
  fraction are actual true_duplicate pairs. This is the number that
  matters most before a verifier exists (Phase 4) -- it measures how
  often an *unverified* auto-serve would be wrong.
- recall: of true_duplicate pairs, what fraction reach at least
  "risky_hit" (i.e. aren't silently dropped to "no_match", which never
  reaches the Verifier and is a full miss).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from verified_cost_router.cache.embeddings import EmbeddingModel
from verified_cost_router.cache.thresholds import CacheThresholds, classify_similarity
from verified_cost_router.data_prep.adversarial_eval import CachePair

# Cosine-similarity cutoff candidates to sweep, 0.50-0.99 in steps of 0.01.
DEFAULT_GRID: tuple[float, ...] = tuple(round(0.50 + 0.01 * i, 2) for i in range(50))


@dataclass(frozen=True)
class ThresholdEvaluation:
    """Metrics for one candidate (high_confidence, risky) threshold pair."""

    thresholds: CacheThresholds
    high_confidence_precision: float
    recall: float
    near_miss_leaks: int
    true_duplicate_high_confidence: int
    true_duplicate_missed: int


def compute_pair_similarities(pairs: Sequence[CachePair], embedder: EmbeddingModel) -> dict[str, float]:
    """Cosine similarity between query_a and query_b for each pair, keyed by pair id.

    Embeddings are L2-normalized (EmbeddingModel contract), so the dot
    product of query_a's and query_b's vectors is their cosine similarity.
    """
    if not pairs:
        return {}
    vectors_a = embedder.embed_batch([pair.query_a for pair in pairs])
    vectors_b = embedder.embed_batch([pair.query_b for pair in pairs])
    return {
        pair.id: float(vector_a @ vector_b)
        for pair, vector_a, vector_b in zip(pairs, vectors_a, vectors_b)
    }


def evaluate_candidate(
    pairs: Sequence[CachePair],
    similarities: Mapping[str, float],
    thresholds: CacheThresholds,
) -> ThresholdEvaluation:
    """Score one threshold pair against the labeled cache pairs."""
    true_duplicate_total = 0
    true_duplicate_detected = 0
    true_duplicate_high_confidence = 0
    true_duplicate_missed = 0
    high_confidence_true_positives = 0
    high_confidence_false_positives = 0

    for pair in pairs:
        category = classify_similarity(similarities[pair.id], thresholds)
        if pair.category == "true_duplicate":
            true_duplicate_total += 1
            if category != "no_match":
                true_duplicate_detected += 1
            else:
                true_duplicate_missed += 1
            if category == "high_confidence_hit":
                true_duplicate_high_confidence += 1
                high_confidence_true_positives += 1
        elif category == "high_confidence_hit":  # near_miss leaking into auto-serve
            high_confidence_false_positives += 1

    recall = true_duplicate_detected / true_duplicate_total if true_duplicate_total else 0.0
    high_confidence_total = high_confidence_true_positives + high_confidence_false_positives
    precision = (
        high_confidence_true_positives / high_confidence_total if high_confidence_total else 1.0
    )

    return ThresholdEvaluation(
        thresholds=thresholds,
        high_confidence_precision=precision,
        recall=recall,
        near_miss_leaks=high_confidence_false_positives,
        true_duplicate_high_confidence=true_duplicate_high_confidence,
        true_duplicate_missed=true_duplicate_missed,
    )


def sweep_thresholds(
    pairs: Sequence[CachePair],
    similarities: Mapping[str, float],
    min_recall: float = 0.90,
    grid: Sequence[float] = DEFAULT_GRID,
) -> list[ThresholdEvaluation]:
    """Evaluate every valid (high, risky) pair from `grid` that meets `min_recall`.

    Returns evaluations sorted best first: highest high_confidence_precision,
    ties broken by highest recall. Raises ValueError if no candidate on the
    grid reaches `min_recall`.
    """
    evaluations = [
        evaluate_candidate(pairs, similarities, CacheThresholds(high_confidence=high, risky=risky))
        for risky in grid
        for high in grid
        if high > risky
    ]
    evaluations = [e for e in evaluations if e.recall >= min_recall]
    if not evaluations:
        raise ValueError(f"no threshold candidate on the grid reached the minimum recall of {min_recall}")
    evaluations.sort(key=lambda e: (e.high_confidence_precision, e.recall), reverse=True)
    return evaluations
