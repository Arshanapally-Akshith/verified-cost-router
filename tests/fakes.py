"""Test-only fakes. Not part of the shipped package.

Embedders:
- FakeEmbedder: deterministic, hash-derived vectors -- good for
  mechanics tests (put/lookup wiring, TTL) where the exact similarity
  value doesn't matter, only that it's a valid, reproducible embedder.
- ScriptedEmbedder: hand-registered vectors per exact text -- good for
  tests that need to pin an exact cosine similarity between two prompts.

LLM client:
- FakeChatCompletionClient: scripted single-response ChatCompletionClient
  for testing prompt-building/response-parsing without a real model call.

Pipeline-node doubles (verified_cost_router.pipeline.nodes.PipelineNodes
takes these as separate objects, so each can be configured independently
without needing to script a shared client across multiple call sites):
- FakeClassifier, FakeVerifier: same verdict for every query -- fine
  when a test only exercises one query at a time.
- ScriptedClassifier, ScriptedVerifier: per-query-text verdicts via a
  lookup dict -- needed by eval-module tests, which classify/verify a
  whole batch of *different* queries in one run and need each to behave
  differently (e.g. some correctly routed, some not).
"""

from __future__ import annotations

import numpy as np

from verified_cost_router.llm.groq_client import ChatCompletionResult, ChatMessage
from verified_cost_router.router.classifier import ClassificationResult, ComplexityLabel
from verified_cost_router.verifier.verifier import VerificationOutcome, VerifierLabel


class FakeEmbedder:
    """Deterministic, semantically-meaningless embedder for fast offline tests."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vector = rng.normal(size=self._dim).astype(np.float32)
        return vector / np.linalg.norm(vector)

    def embed_batch(self, texts):
        return np.stack([self.embed(text) for text in texts]) if texts else np.empty((0, self._dim), dtype=np.float32)


class ScriptedEmbedder:
    """Fake embedder returning pre-registered vectors for exact texts."""

    def __init__(self, vectors: dict[str, np.ndarray], dim: int) -> None:
        self._vectors = vectors
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        if text not in self._vectors:
            raise KeyError(f"ScriptedEmbedder has no vector registered for {text!r}")
        return self._vectors[text]

    def embed_batch(self, texts):
        return np.stack([self.embed(text) for text in texts])


def unit_vectors_with_similarity(similarity: float) -> tuple[np.ndarray, np.ndarray]:
    """Two 2D unit vectors whose dot product equals `similarity`."""
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([similarity, np.sqrt(max(0.0, 1.0 - similarity**2))], dtype=np.float32)
    return a, b


class FakeChatCompletionClient:
    """Scripted ChatCompletionClient: always returns `next_content`, and
    records the arguments of the most recent call for assertions."""

    def __init__(self, next_content: str) -> None:
        self.next_content = next_content
        self.call_count = 0
        self.last_model: str | None = None
        self.last_messages: list[ChatMessage] | None = None
        self.last_temperature: float | None = None
        self.last_max_tokens: int | None = None

    def chat_completion(
        self,
        model: str,
        messages,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> ChatCompletionResult:
        self.call_count += 1
        self.last_model = model
        self.last_messages = list(messages)
        self.last_temperature = temperature
        self.last_max_tokens = max_tokens
        return ChatCompletionResult(
            content=self.next_content, model=model, prompt_tokens=7, completion_tokens=1
        )


class FakeClassifier:
    """Fake ComplexityClassifier: always classifies as `label`."""

    def __init__(self, label: ComplexityLabel, model: str = "fake-cheap-model") -> None:
        self.label = label
        self.model = model
        self.calls: list[str] = []

    def classify_with_usage(self, query: str) -> ClassificationResult:
        self.calls.append(query)
        return ClassificationResult(label=self.label, model=self.model, prompt_tokens=6, completion_tokens=1)


class FakeVerifier:
    """Fake Verifier with independently configurable verdicts per method,
    since a single request can call verify_cache_hit and verify_output
    at different points (e.g. a risky cache-hit fails verification, then
    the router path's own output verification also runs)."""

    def __init__(
        self,
        cache_hit_label: VerifierLabel = "pass",
        output_label: VerifierLabel = "pass",
        model: str = "fake-cheap-model",
    ) -> None:
        self.cache_hit_label = cache_hit_label
        self.output_label = output_label
        self.model = model
        self.cache_hit_calls: list[tuple[str, str, str]] = []
        self.output_calls: list[tuple[str, str]] = []

    def verify_cache_hit(self, query: str, cached_query: str, cached_response: str) -> VerificationOutcome:
        self.cache_hit_calls.append((query, cached_query, cached_response))
        return VerificationOutcome(
            label=self.cache_hit_label, model=self.model, prompt_tokens=8, completion_tokens=1
        )

    def verify_output(self, query: str, output: str) -> VerificationOutcome:
        self.output_calls.append((query, output))
        return VerificationOutcome(
            label=self.output_label, model=self.model, prompt_tokens=8, completion_tokens=1
        )


class ScriptedClassifier:
    """Fake ComplexityClassifier returning a per-query-text label from `labels`."""

    def __init__(self, labels: dict[str, ComplexityLabel], default: ComplexityLabel = "simple", model: str = "fake-cheap-model") -> None:
        self._labels = labels
        self._default = default
        self.model = model
        self.calls: list[str] = []

    def classify_with_usage(self, query: str) -> ClassificationResult:
        self.calls.append(query)
        label = self._labels.get(query, self._default)
        return ClassificationResult(label=label, model=self.model, prompt_tokens=6, completion_tokens=1)


class ScriptedVerifier:
    """Fake Verifier returning per-call verdicts from lookup dicts, keyed
    by the exact arguments the real Verifier would see (not by any
    ground-truth label the real verifier wouldn't have access to)."""

    def __init__(
        self,
        cache_hit_labels: dict[tuple[str, str], VerifierLabel] | None = None,
        output_labels: dict[str, VerifierLabel] | None = None,
        default: VerifierLabel = "pass",
        model: str = "fake-cheap-model",
    ) -> None:
        self._cache_hit_labels = cache_hit_labels or {}
        self._output_labels = output_labels or {}
        self._default = default
        self.model = model
        self.cache_hit_calls: list[tuple[str, str, str]] = []
        self.output_calls: list[tuple[str, str]] = []

    def verify_cache_hit(self, query: str, cached_query: str, cached_response: str) -> VerificationOutcome:
        self.cache_hit_calls.append((query, cached_query, cached_response))
        label = self._cache_hit_labels.get((query, cached_query), self._default)
        return VerificationOutcome(label=label, model=self.model, prompt_tokens=8, completion_tokens=1)

    def verify_output(self, query: str, output: str) -> VerificationOutcome:
        self.output_calls.append((query, output))
        label = self._output_labels.get(query, self._default)
        return VerificationOutcome(label=label, model=self.model, prompt_tokens=8, completion_tokens=1)
