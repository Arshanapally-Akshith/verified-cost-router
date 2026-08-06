"""Cache entry and TTL-based expiry (ARCHITECTURE.md 4.1: TTL alone is
sufficient eviction at this scale -- no LRU or size-based eviction).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class CacheEntry:
    """One cached (prompt, response) pair with its own TTL."""

    id: str
    prompt: str
    response: str
    created_at: datetime
    ttl_seconds: float

    def is_expired(self, now: datetime) -> bool:
        """Whether this entry has outlived its TTL as of `now`."""
        return now >= self.created_at + timedelta(seconds=self.ttl_seconds)
