"""Unit tests for verified_cost_router.cache.vector_store.FaissVectorStore."""

from __future__ import annotations

import numpy as np
import pytest

from verified_cost_router.cache.vector_store import FaissVectorStore


def test_empty_store_search_returns_no_results():
    store = FaissVectorStore(dim=4)
    assert store.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)) == []
    assert len(store) == 0


def test_add_and_search_returns_exact_match():
    store = FaissVectorStore(dim=4)
    store.add(1, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    results = store.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert len(results) == 1
    entry_id, similarity = results[0]
    assert entry_id == 1
    assert similarity == pytest.approx(1.0, abs=1e-5)


def test_search_orders_by_similarity_best_first():
    store = FaissVectorStore(dim=2)
    store.add(1, np.array([1.0, 0.0], dtype=np.float32))  # far from query
    store.add(2, np.array([0.0, 1.0], dtype=np.float32))  # exact match
    query = np.array([0.0, 1.0], dtype=np.float32)
    results = store.search(query, k=2)
    assert [entry_id for entry_id, _ in results] == [2, 1]


def test_search_k_is_capped_at_index_size():
    store = FaissVectorStore(dim=2)
    store.add(1, np.array([1.0, 0.0], dtype=np.float32))
    results = store.search(np.array([1.0, 0.0], dtype=np.float32), k=10)
    assert len(results) == 1


def test_remove_deletes_entry_from_index():
    store = FaissVectorStore(dim=2)
    store.add(1, np.array([1.0, 0.0], dtype=np.float32))
    store.add(2, np.array([0.0, 1.0], dtype=np.float32))
    store.remove([1])
    assert len(store) == 1
    results = store.search(np.array([1.0, 0.0], dtype=np.float32), k=2)
    assert [entry_id for entry_id, _ in results] == [2]


def test_remove_with_empty_list_is_a_no_op():
    store = FaissVectorStore(dim=2)
    store.add(1, np.array([1.0, 0.0], dtype=np.float32))
    store.remove([])
    assert len(store) == 1
