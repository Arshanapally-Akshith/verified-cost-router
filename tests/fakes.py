"""Test-only fake embedders. Not part of the shipped package.

Two flavors:
- FakeEmbedder: deterministic, hash-derived vectors -- good for
  mechanics tests (put/lookup wiring, TTL) where the exact similarity
  value doesn't matter, only that it's a valid, reproducible embedder.
- ScriptedEmbedder: hand-registered vectors per exact text -- good for
  tests that need to pin an exact cosine similarity between two prompts.
"""

from __future__ import annotations

import numpy as np


class FakeEmbedder:
    """Deterministic, semantically-meaningless embedder for fast offline tests."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vector = rng.normal(size=self._dim).astype(np.float32)
        return vector / np.linalg.norm(vector)

    def embed_batch(self, texts):
        return np.stack([self.embed(text) for text in texts]) if texts else np.empty((0, self._dim), dtype=np.float32)


class ScriptedEmbedder:
    """Fake embedder returning pre-registered vectors for exact texts."""

    def __init__(self, vectors: dict[str, np.ndarray], dim: int) -> None:
        self._vectors = vectors
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        if text not in self._vectors:
            raise KeyError(f"ScriptedEmbedder has no vector registered for {text!r}")
        return self._vectors[text]

    def embed_batch(self, texts):
        return np.stack([self.embed(text) for text in texts])


def unit_vectors_with_similarity(similarity: float) -> tuple[np.ndarray, np.ndarray]:
    """Two 2D unit vectors whose dot product equals `similarity`."""
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([similarity, np.sqrt(max(0.0, 1.0 - similarity**2))], dtype=np.float32)
    return a, b
