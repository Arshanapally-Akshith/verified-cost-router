"""Unit tests for verified_cost_router.cache.entry."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from verified_cost_router.cache.entry import CacheEntry

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _entry(ttl_seconds: float) -> CacheEntry:
    return CacheEntry(id="1", prompt="q", response="a", created_at=_T0, ttl_seconds=ttl_seconds)


def test_not_expired_before_ttl_elapses():
    entry = _entry(ttl_seconds=3600)
    assert entry.is_expired(_T0 + timedelta(minutes=30)) is False


def test_expired_exactly_at_ttl_boundary():
    entry = _entry(ttl_seconds=3600)
    assert entry.is_expired(_T0 + timedelta(hours=1)) is True


def test_expired_after_ttl_elapses():
    entry = _entry(ttl_seconds=3600)
    assert entry.is_expired(_T0 + timedelta(hours=2)) is True


def test_not_expired_at_creation_time():
    entry = _entry(ttl_seconds=3600)
    assert entry.is_expired(_T0) is False
