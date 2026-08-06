"""Integration test for the real embedding model.

Unlike the rest of the cache test suite, this loads actual
sentence-transformers model weights (downloaded once, then cached under
~/.cache/huggingface -- see README for the one-time cost). It exists
because the whole cache design depends on real embedding behavior
(paraphrases scoring high, unrelated text scoring low) that a fake
embedder can't validate.
"""

from __future__ import annotations

import numpy as np
import pytest

from verified_cost_router.cache.embeddings import SentenceTransformerEmbedder


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


def test_embedding_dimension_is_positive(embedder: SentenceTransformerEmbedder):
    assert embedder.dim > 0


def test_embed_returns_l2_normalized_vector(embedder: SentenceTransformerEmbedder):
    vector = embedder.embed("What is the capital of France?")
    assert vector.shape == (embedder.dim,)
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-4)


def test_embed_batch_matches_embed_dimension(embedder: SentenceTransformerEmbedder):
    vectors = embedder.embed_batch(["hello", "world"])
    assert vectors.shape == (2, embedder.dim)


def test_paraphrase_scores_higher_than_unrelated_text(embedder: SentenceTransformerEmbedder):
    anchor = embedder.embed("What is the boiling point of water at sea level?")
    paraphrase = embedder.embed("At sea level, what temperature does water boil at?")
    unrelated = embedder.embed("What's a good recipe for banana bread?")

    similarity_paraphrase = float(anchor @ paraphrase)
    similarity_unrelated = float(anchor @ unrelated)

    assert similarity_paraphrase > similarity_unrelated
