"""Minimal Groq chat-completions client: Groq exposes an OpenAI-compatible
REST endpoint (ARCHITECTURE.md 4.4), so this is a thin `requests` wrapper
rather than a full SDK dependency. Retries with exponential backoff on
429s (BUILD.md section 1: "don't let a rate-limit response silently kill
a replay run").

Shared infrastructure: the Router (Phase 3) uses it for classification.
Generation and verifier nodes (Phase 4) will reuse it unchanged for
completions -- nothing here is Router-specific.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Literal, Protocol, Sequence

import requests

logger = logging.getLogger(__name__)

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_SECONDS = 1.0
DEFAULT_TIMEOUT_SECONDS = 30.0
# Extra time given to a request beyond DEFAULT_TIMEOUT_SECONDS before the
# watchdog gives up on it -- see GroqClient's docstring for why this exists.
DEFAULT_WATCHDOG_GRACE_SECONDS = 15.0
# Ceiling on how long a single 429 retry will actually sleep for. Groq's
# Retry-After header is honored (see _retry_delay), but honored verbatim
# and uncapped it turns a daily-token-quota 429 into a silent, multi-
# minute block per retry (observed: near a 100K TPD cap, Retry-After
# scales with how many tokens a request is short by, easily minutes for
# a real generation) -- indistinguishable from a genuine hang, and,
# unlike the network-level watchdog in _post(), a plain time.sleep() in
# the retry loop below isn't bounded by that watchdog at all. Capping
# the sleep keeps worst-case blocking time bounded and (via the log
# line at each retry) visible instead of silent.
DEFAULT_MAX_RETRY_DELAY_SECONDS = 30.0

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class ChatCompletionResult:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class ChatCompletionClient(Protocol):
    """Anything that can run a Groq-style chat completion.

    Lets callers (e.g. ComplexityClassifier) depend on this instead of
    GroqClient directly, so tests can inject a fake with no network call.
    """

    def chat_completion(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatCompletionResult: ...


class GroqAPIError(RuntimeError):
    """Raised for any non-retryable (non-429) Groq API failure."""


class GroqRateLimitError(RuntimeError):
    """Raised when Groq keeps returning 429 after all retries are exhausted."""


class GroqClient:
    """Thin wrapper around Groq's OpenAI-compatible chat completions endpoint.

    Every HTTP call runs behind a watchdog thread in addition to
    `requests`' own connect/read timeout. This client is designed to be
    reused for hundreds of sequential calls over a long-running process
    (e.g. scripts/run_eval.py) -- exactly the shape that occasionally
    triggers a known requests/urllib3 failure class where a pooled
    keep-alive connection goes stale (the peer drops it without a clean
    FIN/RST) and a later request reusing it blocks on the socket read
    without `requests`' configured timeout ever firing. The watchdog is
    an independent backstop: chat_completion always returns or raises
    within timeout + watchdog_grace_seconds, even if the underlying
    socket call never does.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = GROQ_CHAT_COMPLETIONS_URL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        watchdog_grace_seconds: float = DEFAULT_WATCHDOG_GRACE_SECONDS,
        max_retry_delay_seconds: float = DEFAULT_MAX_RETRY_DELAY_SECONDS,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._timeout = timeout
        self._watchdog_grace_seconds = watchdog_grace_seconds
        self._max_retry_delay_seconds = max_retry_delay_seconds
        self._session = session or requests.Session()
        self._sleep = sleep

    def chat_completion(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatCompletionResult:
        """Run one chat completion, retrying with exponential backoff on 429.

        Raises GroqRateLimitError if every retry is exhausted, or
        GroqAPIError for any other non-2xx response, request-level error
        (connection reset, DNS failure, etc.), or a request that doesn't
        return within timeout + watchdog_grace_seconds.

        A 429's Retry-After is honored but capped at max_retry_delay_seconds
        (see DEFAULT_MAX_RETRY_DELAY_SECONDS) -- long enough to ride out a
        transient burst, short enough that a near-exhausted daily quota
        fails fast with a clear GroqRateLimitError instead of blocking for
        minutes with no log output.
        """
        payload: dict[str, object] = {
            "model": model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

        last_error: GroqRateLimitError | None = None
        for attempt in range(self._max_retries + 1):
            response = self._post(payload, headers)

            if response.status_code == 429:
                last_error = GroqRateLimitError(
                    f"rate limited after {attempt + 1} attempt(s): {response.text}"
                )
                delay = min(self._retry_delay(attempt, response), self._max_retry_delay_seconds)
                if attempt < self._max_retries:
                    logger.warning(
                        "Groq rate limited (attempt %d/%d) for model %s, retrying in %.1fs: %s",
                        attempt + 1,
                        self._max_retries + 1,
                        model,
                        delay,
                        response.text,
                    )
                    self._sleep(delay)
                    continue
                raise last_error

            if not response.ok:
                raise GroqAPIError(f"Groq API error {response.status_code}: {response.text}")

            return self._parse_response(response.json(), model)

        raise last_error or GroqRateLimitError("retries exhausted with no response")

    def _post(self, payload: dict[str, object], headers: dict[str, str]) -> requests.Response:
        """POST with `requests`' own timeout plus an independent watchdog
        backstop (see class docstring), and convert any request-level
        failure (timeout, connection reset, DNS failure, ...) into
        GroqAPIError so it's uniformly catchable alongside HTTP-status
        failures -- previously these propagated as raw, uncaught
        `requests` exceptions.

        The POST runs on a daemon thread so a call that never returns
        can't block process exit either: daemon threads are torn down
        by the interpreter at exit rather than joined.
        """
        outcome: dict[str, requests.Response | BaseException] = {}

        def _do_post() -> None:
            try:
                outcome["response"] = self._session.post(
                    self._base_url, json=payload, headers=headers, timeout=self._timeout
                )
            except BaseException as exc:  # noqa: BLE001 -- forwarded to the caller, not swallowed
                outcome["error"] = exc

        thread = threading.Thread(target=_do_post, daemon=True, name="groq-client-post")
        thread.start()
        thread.join(timeout=self._timeout + self._watchdog_grace_seconds)

        if thread.is_alive():
            raise GroqAPIError(
                f"request did not complete within {self._timeout + self._watchdog_grace_seconds:.0f}s "
                "(watchdog timeout -- the network call never returned)"
            )
        if "error" in outcome:
            error = outcome["error"]
            raise GroqAPIError(f"request failed: {error}") from error
        return outcome["response"]

    def _retry_delay(self, attempt: int, response: requests.Response) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return self._backoff_seconds * (2**attempt)

    @staticmethod
    def _parse_response(data: dict, model: str) -> ChatCompletionResult:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise GroqAPIError(f"unexpected Groq response shape: {data!r}") from exc
        usage = data.get("usage", {})
        return ChatCompletionResult(
            content=content,
            model=data.get("model", model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
