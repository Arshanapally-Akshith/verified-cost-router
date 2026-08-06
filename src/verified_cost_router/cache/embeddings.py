"""Local embedding pipeline (BUILD.md section 1: sentence-transformer, no API cost).

`EmbeddingModel` is the interface the rest of the cache layer depends on,
so tests can swap in a deterministic fake instead of loading real model
weights.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@runtime_checkable
class EmbeddingModel(Protocol):
    """Anything that can turn text into a fixed-size, L2-normalized vector."""

    @property
    def dim(self) -> int: ...

    def embed(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """`EmbeddingModel` backed by a local sentence-transformers model.

    Embeddings are L2-normalized so cosine similarity reduces to a plain
    inner product, which is what FaissVectorStore's IndexFlatIP computes.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        # Imported lazily so importing this module doesn't require torch
        # unless a real embedder is actually constructed.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self._dim = get_dim()

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        vector = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vector, dtype=np.float32)

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)
