"""Local FAISS-backed nearest-neighbor index (ARCHITECTURE.md 4.1: FAISS
or Chroma, local, no infra dependency).

Vectors must already be L2-normalized by the caller (SentenceTransformerEmbedder
does this) so that inner product, which IndexFlatIP computes, equals cosine
similarity.
"""

from __future__ import annotations

from typing import Sequence

import faiss
import numpy as np


class FaissVectorStore:
    """Cosine-similarity index over vectors keyed by an arbitrary int64 id.

    Wraps IndexIDMap2(IndexFlatIP) so entries can be added and removed by
    id -- needed for TTL-based eviction, which removes by id rather than
    rebuilding the whole index.
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def add(self, entry_id: int, vector: np.ndarray) -> None:
        """Add a single vector under `entry_id`."""
        self._index.add_with_ids(
            np.asarray(vector, dtype=np.float32).reshape(1, -1),
            np.array([entry_id], dtype=np.int64),
        )

    def remove(self, entry_ids: Sequence[int]) -> None:
        """Remove all vectors whose id is in `entry_ids` (no-op if empty)."""
        if entry_ids:
            self._index.remove_ids(np.array(list(entry_ids), dtype=np.int64))

    def search(self, vector: np.ndarray, k: int = 1) -> list[tuple[int, float]]:
        """Return up to `k` nearest (entry_id, cosine_similarity) pairs, best first."""
        if self._index.ntotal == 0:
            return []
        k = min(k, self._index.ntotal)
        scores, ids = self._index.search(
            np.asarray(vector, dtype=np.float32).reshape(1, -1), k
        )
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def __len__(self) -> int:
        return self._index.ntotal
