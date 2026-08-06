"""Similarity-cutoff classification shared by the live cache and by
threshold_tuning's offline sweep, so both use the exact same decision rule
(ARCHITECTURE.md 4.1: two cutoffs -- high-confidence auto-serve, mid-range
"risky" band routed to the Verifier, below that is no match).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MatchCategory = Literal["no_match", "risky_hit", "high_confidence_hit"]


@dataclass(frozen=True)
class CacheThresholds:
    """A pair of cosine-similarity cutoffs defining the three match bands.

    similarity >= high_confidence  -> "high_confidence_hit"
    risky <= similarity < high_confidence -> "risky_hit"
    similarity < risky             -> "no_match"
    """

    high_confidence: float
    risky: float

    def __post_init__(self) -> None:
        if not (-1.0 <= self.risky < self.high_confidence <= 1.0):
            raise ValueError(
                "thresholds must satisfy -1 <= risky < high_confidence <= 1, "
                f"got risky={self.risky}, high_confidence={self.high_confidence}"
            )


def classify_similarity(similarity: float, thresholds: CacheThresholds) -> MatchCategory:
    """Bucket a cosine similarity score into a match category."""
    if similarity >= thresholds.high_confidence:
        return "high_confidence_hit"
    if similarity >= thresholds.risky:
        return "risky_hit"
    return "no_match"
