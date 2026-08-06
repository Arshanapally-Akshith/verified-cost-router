"""Unit tests for verified_cost_router.llm.groq_client.GroqClient.

Uses a scripted fake `requests.Session` and an injected fake `sleep`, so
retry/backoff behavior is tested deterministically with no real network
calls and no real waiting.
"""

from __future__ import annotations

import pytest

from verified_cost_router.llm.groq_client import (
    ChatMessage,
    GroqAPIError,
    GroqClient,
    GroqRateLimitError,
)


class _FakeResponse:
    def __init__(self, status_code: int, json_data: dict | None = None, headers: dict | None = None, text: str = ""):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._json_data = json_data or {}
        self.headers = headers or {}
        self.text = text

    def json(self) -> dict:
        return self._json_data


class _ScriptedSession:
    def __init__(self, responses: list[_FakeResponse]):
        self._responses = iter(responses)
        self.calls: list[dict] = []

    def post(self, url, json, headers, timeout):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return next(self._responses)


def _success_response(content: str = "simple") -> _FakeResponse:
    return _FakeResponse(
        200,
        json_data={
            "model": "llama-3.1-8b-instant",
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
    )


def _make_client(session: _ScriptedSession, sleeps: list[float] | None = None, **kwargs) -> GroqClient:
    sleep_log = sleeps if sleeps is not None else []
    return GroqClient(
        api_key="test-key",
        session=session,
        sleep=lambda seconds: sleep_log.append(seconds),
        **kwargs,
    )


def test_successful_completion_parses_content_and_usage():
    session = _ScriptedSession([_success_response("complex")])
    client = _make_client(session)

    result = client.chat_completion("llama-3.1-8b-instant", [ChatMessage(role="user", content="hi")])

    assert result.content == "complex"
    assert result.model == "llama-3.1-8b-instant"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 2


def test_request_payload_includes_messages_and_temperature():
    session = _ScriptedSession([_success_response()])
    client = _make_client(session)

    client.chat_completion(
        "model-x",
        [ChatMessage(role="system", content="sys"), ChatMessage(role="user", content="hi")],
        temperature=0.2,
        max_tokens=5,
    )

    call = session.calls[0]
    assert call["json"]["model"] == "model-x"
    assert call["json"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert call["json"]["temperature"] == 0.2
    assert call["json"]["max_tokens"] == 5
    assert call["headers"]["Authorization"] == "Bearer test-key"


def test_retries_on_429_then_succeeds():
    session = _ScriptedSession([_FakeResponse(429, text="slow down"), _success_response("simple")])
    sleeps: list[float] = []
    client = _make_client(session, sleeps=sleeps)

    result = client.chat_completion("m", [ChatMessage(role="user", content="hi")])

    assert result.content == "simple"
    assert len(session.calls) == 2
    assert len(sleeps) == 1


def test_raises_rate_limit_error_after_exhausting_retries():
    session = _ScriptedSession([_FakeResponse(429) for _ in range(4)])
    sleeps: list[float] = []
    client = _make_client(session, sleeps=sleeps, max_retries=3)

    with pytest.raises(GroqRateLimitError):
        client.chat_completion("m", [ChatMessage(role="user", content="hi")])

    assert len(session.calls) == 4  # 1 initial + 3 retries
    assert len(sleeps) == 3


def test_retry_after_header_is_respected_for_backoff_delay():
    session = _ScriptedSession(
        [_FakeResponse(429, headers={"Retry-After": "7"}), _success_response()]
    )
    sleeps: list[float] = []
    client = _make_client(session, sleeps=sleeps)

    client.chat_completion("m", [ChatMessage(role="user", content="hi")])

    assert sleeps == [7.0]


def test_exponential_backoff_without_retry_after_header():
    session = _ScriptedSession([_FakeResponse(429), _FakeResponse(429), _success_response()])
    sleeps: list[float] = []
    client = _make_client(session, sleeps=sleeps, backoff_seconds=1.0)

    client.chat_completion("m", [ChatMessage(role="user", content="hi")])

    assert sleeps == [1.0, 2.0]  # 1 * 2**0, 1 * 2**1


def test_non_429_error_status_raises_groq_api_error():
    session = _ScriptedSession([_FakeResponse(500, text="internal error")])
    client = _make_client(session)

    with pytest.raises(GroqAPIError, match="500"):
        client.chat_completion("m", [ChatMessage(role="user", content="hi")])


def test_malformed_success_response_raises_groq_api_error():
    session = _ScriptedSession([_FakeResponse(200, json_data={"unexpected": "shape"})])
    client = _make_client(session)

    with pytest.raises(GroqAPIError, match="unexpected Groq response shape"):
        client.chat_completion("m", [ChatMessage(role="user", content="hi")])
