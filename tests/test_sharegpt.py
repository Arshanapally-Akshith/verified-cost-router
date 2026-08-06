"""Unit tests for verified_cost_router.data_prep.sharegpt.

All tests operate on synthetic in-memory byte streams -- no real network
calls -- so the suite stays fast and reliable while still exercising the
extraction logic that will run against the real dataset.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from verified_cost_router.data_prep import sharegpt


def _stream(conversations: list[dict]) -> io.BytesIO:
    return io.BytesIO(json.dumps(conversations).encode("utf-8"))


def test_iter_conversations_respects_limit():
    conversations = [{"id": str(i)} for i in range(10)]
    result = list(sharegpt.iter_conversations(_stream(conversations), limit=3))
    assert [c["id"] for c in result] == ["0", "1", "2"]


def test_iter_conversations_limit_zero_yields_nothing():
    assert list(sharegpt.iter_conversations(_stream([{"id": "0"}]), limit=0)) == []


def test_extract_first_human_turn_returns_first_human_message():
    conversation = {
        "conversations": [
            {"from": "system", "value": "be nice"},
            {"from": "human", "value": " What is 2+2? "},
            {"from": "gpt", "value": "4"},
            {"from": "human", "value": "and 3+3?"},
        ]
    }
    assert sharegpt.extract_first_human_turn(conversation) == "What is 2+2?"


@pytest.mark.parametrize(
    "conversation",
    [
        {},
        {"conversations": []},
        {"conversations": "not-a-list"},
        {"conversations": [{"from": "gpt", "value": "hi"}]},
        {"conversations": [{"from": "human", "value": "   "}]},
        {"conversations": [{"from": "human", "value": None}]},
    ],
)
def test_extract_first_human_turn_returns_none_for_malformed_records(conversation):
    assert sharegpt.extract_first_human_turn(conversation) is None


def test_sample_replay_queries_skips_unusable_conversations():
    conversations = [
        {"id": "a", "conversations": [{"from": "human", "value": "hello"}]},
        {"id": "b", "conversations": []},
        {"conversations": [{"from": "human", "value": "no id"}]},
        {"id": "c", "conversations": [{"from": "human", "value": "world"}]},
    ]
    result = list(sharegpt.sample_replay_queries(_stream(conversations), limit=10))
    assert [r.conversation_id for r in result] == ["a", "c"]
    assert [r.query for r in result] == ["hello", "world"]


def test_sample_replay_queries_limit_caps_raw_conversations_read():
    # limit caps how many raw conversations are read, matching BUILD.md's
    # sampling code (cap first, extract second) -- not how many succeed
    # extraction.
    conversations = [{"id": "bad"}, {"id": "bad2"}, {
        "id": "good", "conversations": [{"from": "human", "value": "hi"}]
    }]
    result = list(sharegpt.sample_replay_queries(_stream(conversations), limit=2))
    assert result == []


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._raw = io.BytesIO(payload)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def raise_for_status(self):
        pass

    @property
    def raw(self):
        return self._raw


def test_download_and_sample_writes_jsonl(monkeypatch, tmp_path: Path):
    conversations = [
        {"id": "a", "conversations": [{"from": "human", "value": "hello"}]},
        {"id": "b", "conversations": [{"from": "human", "value": "world"}]},
    ]
    payload = json.dumps(conversations).encode("utf-8")

    def fake_get(url, stream, timeout):
        assert stream is True
        return _FakeResponse(payload)

    monkeypatch.setattr(sharegpt.requests, "get", fake_get)

    out_path = tmp_path / "replay_sample.jsonl"
    written = sharegpt.download_and_sample(out_path, limit=5)

    assert written == 2
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"id": "a", "query": "hello"}
    assert json.loads(lines[1]) == {"id": "b", "query": "world"}
