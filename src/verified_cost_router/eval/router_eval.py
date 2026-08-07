"""Router accuracy against the labeled adversarial complexity_items
(BUILD.md section 4: "router accuracy against labeled complexity").

Every item in this eval category is "complexity_mislabeled" by
construction (Phase 1): worded simply, but genuinely complex. There are
no labeled "actually simple" counterexamples in this set, so what's
measured here is specifically the router's recall on adversarial
complex queries -- how often it resists the misleadingly simple
wording -- not full precision/accuracy across both labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from verified_cost_router.data_prep.adversarial_eval import ComplexityItem
from verified_cost_router.router.classifier import ClassificationResult


class ClassifierLike(Protocol):
    def classify_with_usage(self, query: str) -> ClassificationResult: ...


@dataclass(frozen=True)
class ComplexityItemOutcome:
    item_id: str
    query: str
    predicted_label: str  # "simple" | "complex"
    correct: bool


@dataclass(frozen=True)
class RouterEvalResult:
    outcomes: tuple[ComplexityItemOutcome, ...]

    @property
    def complex_recall(self) -> float:
        """Fraction of complexity-mislabeled items correctly labeled "complex"."""
        if not self.outcomes:
            return 1.0
        correct = sum(1 for outcome in self.outcomes if outcome.correct)
        return correct / len(self.outcomes)


def evaluate_complexity_items(items: Sequence[ComplexityItem], classifier: ClassifierLike) -> RouterEvalResult:
    outcomes = []
    for item in items:
        result = classifier.classify_with_usage(item.query)
        outcomes.append(
            ComplexityItemOutcome(
                item_id=item.id,
                query=item.query,
                predicted_label=result.label,
                correct=(result.label == item.true_complexity),
            )
        )
    return RouterEvalResult(outcomes=tuple(outcomes))
