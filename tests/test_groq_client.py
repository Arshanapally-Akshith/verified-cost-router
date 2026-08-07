"""Unit tests for verified_cost_router.llm.groq_client.GroqClient.

Uses a scripted fake `requests.Session` and an injected fake `sleep`, so
retry/backoff behavior is tested deterministically with no real network
calls and no real waiting.
"""

from __future__ import annotations

import time

import pytest
import requests

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


def test_large_retry_after_is_capped_not_slept_verbatim():
    """Regression test for the run_eval.py hang (Phase 6): near a daily
    token-quota cap, Groq's Retry-After scales with how many tokens a
    request is short by and can be minutes long. Sleeping that verbatim
    turned a routine 429 into a silent, multi-minute block -- this caps
    it so a single retry never blocks longer than max_retry_delay_seconds.
    """
    session = _ScriptedSession(
        [_FakeResponse(429, headers={"Retry-After": "600"}), _success_response()]
    )
    sleeps: list[float] = []
    client = _make_client(session, sleeps=sleeps, max_retry_delay_seconds=30.0)

    client.chat_completion("m", [ChatMessage(role="user", content="hi")])

    assert sleeps == [30.0]


def test_retry_logs_a_warning_instead_of_sleeping_silently(caplog):
    session = _ScriptedSession(
        [_FakeResponse(429, headers={"Retry-After": "7"}), _success_response()]
    )
    client = _make_client(session, sleeps=[])

    with caplog.at_level("WARNING"):
        client.chat_completion("m", [ChatMessage(role="user", content="hi")])

    assert any("rate limited" in record.message for record in caplog.records)


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


# --- Watchdog / request-level failure handling --------------------------------
#
# These cover the Phase 6 fix for a reproducible hang in scripts/run_eval.py:
# a request whose underlying socket call never returns (a known
# requests/urllib3 failure class on long-lived, heavily reused Sessions --
# see GroqClient's class docstring) used to hang chat_completion forever.


class _HangingSession:
    """Fake session whose post() blocks far longer than any test's configured timeout."""

    def post(self, url, json, headers, timeout):
        time.sleep(5)
        return _FakeResponse(200)  # never actually reached within the test


class _RaisingSession:
    def __init__(self, exc: BaseException):
        self._exc = exc

    def post(self, url, json, headers, timeout):
        raise self._exc


def test_watchdog_raises_groq_api_error_when_post_never_returns():
    client = GroqClient(
        api_key="test-key",
        session=_HangingSession(),
        sleep=lambda seconds: None,
        timeout=0.05,
        watchdog_grace_seconds=0.05,
    )

    start = time.monotonic()
    with pytest.raises(GroqAPIError, match="watchdog timeout"):
        client.chat_completion("m", [ChatMessage(role="user", content="hi")])
    elapsed = time.monotonic() - start

    assert elapsed < 2.0  # bounded by timeout+grace (0.1s), not the fake's 5s hang


def test_connection_error_is_wrapped_as_groq_api_error():
    session = _RaisingSession(requests.exceptions.ConnectionError("connection reset by peer"))
    client = _make_client(session)

    with pytest.raises(GroqAPIError, match="request failed"):
        client.chat_completion("m", [ChatMessage(role="user", content="hi")])


def test_read_timeout_is_wrapped_as_groq_api_error():
    session = _RaisingSession(requests.exceptions.ReadTimeout("read timed out"))
    client = _make_client(session)

    with pytest.raises(GroqAPIError, match="request failed"):
        client.chat_completion("m", [ChatMessage(role="user", content="hi")])


def test_fast_response_is_unaffected_by_a_tight_watchdog():
    session = _ScriptedSession([_success_response("ok")])
    client = GroqClient(
        api_key="test-key", session=session, sleep=lambda seconds: None, timeout=0.5, watchdog_grace_seconds=0.5
    )

    result = client.chat_completion("m", [ChatMessage(role="user", content="hi")])

    assert result.content == "ok"
