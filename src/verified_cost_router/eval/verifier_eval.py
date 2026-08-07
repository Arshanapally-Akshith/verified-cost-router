"""Verifier catch rate (BUILD.md section 4: "how often it correctly
flags a bad cache hit or bad route").

Two independent sub-evaluations, since the verifier is used in two
places (ARCHITECTURE.md 4.2):

- Bad cache hit: a near_miss pair whose lookup lands in the risky band
  (the only band the verifier actually sees -- no_match never reaches
  it, and high_confidence_hit skips it by design) should be flagged
  "fail" by verify_cache_hit. Pairs that never reach the verifier are
  reported separately, not folded into the catch rate, since the
  verifier was never given a chance on them.
- Bad route: a complexity_mislabeled item the router mislabels "simple"
  produces a cheap-model answer that verify_output should flag "fail",
  triggering escalation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from verified_cost_router.cache.embeddings import EmbeddingModel
from verified_cost_router.cache.semantic_cache import SemanticCache
from verified_cost_router.cache.thresholds import CacheThresholds
from verified_cost_router.data_prep.adversarial_eval import CachePair, ComplexityItem
from verified_cost_router.eval.router_eval import ClassifierLike
from verified_cost_router.llm.generation import generate
from verified_cost_router.llm.groq_client import ChatCompletionClient
from verified_cost_router.verifier.verifier import VerificationOutcome


class VerifierLike(Protocol):
    def verify_cache_hit(self, query: str, cached_query: str, cached_response: str) -> VerificationOutcome: ...
    def verify_output(self, query: str, output: str) -> VerificationOutcome: ...


# --- Bad cache hit ----------------------------------------------------------


@dataclass(frozen=True)
class CacheVerifierOutcome:
    pair_id: str
    category: str  # "true_duplicate" | "near_miss"
    match_category: str  # "risky_hit" | "high_confidence_hit" (never "no_match" here)
    verifier_label: str  # "pass" | "fail"
    correctly_flagged: bool


@dataclass(frozen=True)
class SkippedCachePair:
    """A pair the verifier never got a chance to judge, and why."""

    pair_id: str
    category: str  # "true_duplicate" | "near_miss"
    reason: Literal["no_match", "high_confidence_hit"]


@dataclass(frozen=True)
class CacheVerifierEvalResult:
    reached_verifier: tuple[CacheVerifierOutcome, ...]
    skipped: tuple[SkippedCachePair, ...]

    @property
    def near_miss_catch_rate(self) -> float:
        """Of near_miss pairs the verifier actually saw, how many it correctly failed."""
        near_miss = [o for o in self.reached_verifier if o.category == "near_miss"]
        if not near_miss:
            return 1.0
        return sum(1 for o in near_miss if o.correctly_flagged) / len(near_miss)

    @property
    def true_duplicate_pass_rate(self) -> float:
        """Of true_duplicate pairs the verifier actually saw, how many it correctly passed."""
        duplicates = [o for o in self.reached_verifier if o.category == "true_duplicate"]
        if not duplicates:
            return 1.0
        return sum(1 for o in duplicates if o.correctly_flagged) / len(duplicates)

    @property
    def near_miss_high_confidence_leaks(self) -> int:
        """near_miss pairs that leaked straight past the verifier entirely
        (a cache-layer failure the verifier never had a chance to catch)."""
        return sum(1 for s in self.skipped if s.category == "near_miss" and s.reason == "high_confidence_hit")


def evaluate_cache_verifier(
    pairs: Sequence[CachePair], embedder: EmbeddingModel, thresholds: CacheThresholds, verifier: VerifierLike
) -> CacheVerifierEvalResult:
    reached: list[CacheVerifierOutcome] = []
    skipped: list[SkippedCachePair] = []

    for pair in pairs:
        cache = SemanticCache(embedder, thresholds)
        cache.put(pair.query_a, "dummy-answer")
        result = cache.lookup(pair.query_b)

        if result.category == "no_match":
            skipped.append(SkippedCachePair(pair.id, pair.category, "no_match"))
            continue
        if result.category == "high_confidence_hit":
            skipped.append(SkippedCachePair(pair.id, pair.category, "high_confidence_hit"))
            continue  # verifier is never invoked on a high-confidence hit

        outcome = verifier.verify_cache_hit(
            query=pair.query_b, cached_query=pair.query_a, cached_response="dummy-answer"
        )
        correctly_flagged = (outcome.label == "fail") if pair.category == "near_miss" else (outcome.label == "pass")
        reached.append(
            CacheVerifierOutcome(
                pair_id=pair.id,
                category=pair.category,
                match_category=result.category,
                verifier_label=outcome.label,
                correctly_flagged=correctly_flagged,
            )
        )

    return CacheVerifierEvalResult(reached_verifier=tuple(reached), skipped=tuple(skipped))


# --- Bad route ---------------------------------------------------------------


@dataclass(frozen=True)
class RouteVerifierOutcome:
    item_id: str
    verifier_label: str  # "pass" | "fail"
    correctly_flagged: bool  # verifier said "fail", catching the misroute


@dataclass(frozen=True)
class RouteVerifierEvalResult:
    misrouted: tuple[RouteVerifierOutcome, ...]
    correctly_routed_count: int

    @property
    def catch_rate(self) -> float:
        """Of items the router misrouted to the cheap model, how many the verifier caught."""
        if not self.misrouted:
            return 1.0
        return sum(1 for o in self.misrouted if o.correctly_flagged) / len(self.misrouted)


def evaluate_route_verifier(
    items: Sequence[ComplexityItem],
    classifier: ClassifierLike,
    groq_client: ChatCompletionClient,
    cheap_model: str,
    verifier: VerifierLike,
) -> RouteVerifierEvalResult:
    misrouted: list[RouteVerifierOutcome] = []
    correctly_routed_count = 0

    for item in items:
        classification = classifier.classify_with_usage(item.query)
        if classification.label != "simple":
            correctly_routed_count += 1
            continue

        generation = generate(groq_client, cheap_model, item.query)
        outcome = verifier.verify_output(query=item.query, output=generation.content)
        misrouted.append(
            RouteVerifierOutcome(
                item_id=item.id, verifier_label=outcome.label, correctly_flagged=(outcome.label == "fail")
            )
        )

    return RouteVerifierEvalResult(misrouted=tuple(misrouted), correctly_routed_count=correctly_routed_count)
