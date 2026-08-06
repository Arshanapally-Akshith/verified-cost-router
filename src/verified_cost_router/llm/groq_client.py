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

import time
from dataclasses import dataclass
from typing import Callable, Literal, Protocol, Sequence

import requests

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_SECONDS = 1.0
DEFAULT_TIMEOUT_SECONDS = 30.0

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
    """Thin wrapper around Groq's OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str = GROQ_CHAT_COMPLETIONS_URL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._timeout = timeout
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
        GroqAPIError for any other non-2xx response.
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
            response = self._session.post(self._base_url, json=payload, headers=headers, timeout=self._timeout)

            if response.status_code == 429:
                last_error = GroqRateLimitError(
                    f"rate limited after {attempt + 1} attempt(s): {response.text}"
                )
                if attempt < self._max_retries:
                    self._sleep(self._retry_delay(attempt, response))
                    continue
                raise last_error

            if not response.ok:
                raise GroqAPIError(f"Groq API error {response.status_code}: {response.text}")

            return self._parse_response(response.json(), model)

        raise last_error or GroqRateLimitError("retries exhausted with no response")

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
