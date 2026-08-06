"""The semantic cache itself: embed, nearest-neighbor search, threshold
classification, and TTL eviction wired together (ARCHITECTURE.md 4.1).

Not wired into the LangGraph pipeline yet -- that happens in Phase 4,
which replaces the cache_check/verifier_cache stub nodes with calls into
this class. This module is independently usable and tested.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from verified_cost_router.cache.embeddings import EmbeddingModel
from verified_cost_router.cache.entry import CacheEntry
from verified_cost_router.cache.thresholds import CacheThresholds, MatchCategory, classify_similarity
from verified_cost_router.cache.vector_store import FaissVectorStore

DEFAULT_TTL_SECONDS = 48 * 3600  # 48h, within ARCHITECTURE.md's 24-72h guidance


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class CacheLookupResult:
    """Outcome of a cache lookup: which band it fell in, and the best match if any."""

    category: MatchCategory
    match: CacheEntry | None
    similarity: float | None


class SemanticCache:
    """Embedding + FAISS index + threshold classification + TTL eviction.

    `clock` is injectable so tests can control TTL expiry deterministically
    instead of sleeping.
    """

    def __init__(
        self,
        embedder: EmbeddingModel,
        thresholds: CacheThresholds,
        default_ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._embedder = embedder
        self._thresholds = thresholds
        self._default_ttl_seconds = default_ttl_seconds
        self._clock = clock
        self._store = FaissVectorStore(dim=embedder.dim)
        self._entries: dict[int, CacheEntry] = {}
        self._id_counter = itertools.count(1)

    def put(self, prompt: str, response: str, ttl_seconds: float | None = None) -> str:
        """Embed and store a (prompt, response) pair; returns the new entry's id."""
        internal_id = next(self._id_counter)
        entry = CacheEntry(
            id=str(internal_id),
            prompt=prompt,
            response=response,
            created_at=self._clock(),
            ttl_seconds=ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds,
        )
        self._store.add(internal_id, self._embedder.embed(prompt))
        self._entries[internal_id] = entry
        return entry.id

    def lookup(self, prompt: str) -> CacheLookupResult:
        """Find the closest non-expired entry to `prompt` and classify the match."""
        self._purge_expired()
        if len(self._store) == 0:
            return CacheLookupResult(category="no_match", match=None, similarity=None)

        vector = self._embedder.embed(prompt)
        results = self._store.search(vector, k=1)
        if not results:
            return CacheLookupResult(category="no_match", match=None, similarity=None)

        internal_id, similarity = results[0]
        entry = self._entries[internal_id]
        category = classify_similarity(similarity, self._thresholds)
        if category == "no_match":
            return CacheLookupResult(category="no_match", match=None, similarity=similarity)
        return CacheLookupResult(category=category, match=entry, similarity=similarity)

    def __len__(self) -> int:
        return len(self._entries)

    def _purge_expired(self) -> None:
        now = self._clock()
        expired = [internal_id for internal_id, entry in self._entries.items() if entry.is_expired(now)]
        if not expired:
            return
        self._store.remove(expired)
        for internal_id in expired:
            del self._entries[internal_id]
