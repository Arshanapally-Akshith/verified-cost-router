"""Prompt-based complexity classifier (ARCHITECTURE.md 4.3).

Not a trained model -- a few-shot prompt asks the cheap-tier model to
label a query "simple" or "complex" using multi-step reasoning, domain
knowledge, and ambiguity as criteria, not prompt length alone (BUILD.md
flags length as a weak proxy on its own -- a short question can hide
real complexity, e.g. "Is it safe to mix bleach and ammonia?").

Not wired into the LangGraph pipeline yet -- that's Phase 4, which
replaces the `router` stub node with a call into this class.
"""

from __future__ import annotations

from typing import Literal

from verified_cost_router.llm.groq_client import ChatCompletionClient, ChatMessage

ComplexityLabel = Literal["simple", "complex"]

_MAX_RESPONSE_TOKENS = 8

_SYSTEM_PROMPT = """You are a query complexity classifier for an LLM routing system.
Label each user query as exactly one word: "simple" or "complex".

Judge complexity by whether answering well requires:
- multi-step reasoning, not just fact recall
- specialized domain knowledge (medical, legal, financial, technical, safety) \
where a wrong or oversimplified answer has real consequences
- resolving ambiguity or weighing tradeoffs

Do NOT judge complexity by sentence length or vocabulary alone. A short, \
plainly worded question can still be complex (e.g. "Is it safe to mix \
bleach and ammonia?"), and a long, elaborately worded question can still \
be simple.

Respond with exactly one word: simple or complex. No punctuation, no \
explanation."""

_FEW_SHOT_EXAMPLES: tuple[tuple[str, ComplexityLabel], ...] = (
    ("What is the capital of France?", "simple"),
    ("How do I reverse a string in Python?", "simple"),
    ("What's a good substitute for buttermilk in a pancake recipe?", "simple"),
    ("Is it safe to mix bleach and ammonia?", "complex"),
    ("Why does my recursive function cause a stack overflow?", "complex"),
    ("Should I refinance my mortgage right now?", "complex"),
)


class ClassificationError(RuntimeError):
    """Raised when the model's response can't be parsed into a ComplexityLabel."""


class ComplexityClassifier:
    """Classifies a query as simple/complex via a few-shot prompt to the cheap-tier model."""

    def __init__(self, client: ChatCompletionClient, model: str, temperature: float = 0.0) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    def classify(self, query: str) -> ComplexityLabel:
        """Classify `query`. Raises ClassificationError if the model's
        response can't be resolved to "simple" or "complex"."""
        messages = self._build_messages(query)
        result = self._client.chat_completion(
            self._model, messages, temperature=self._temperature, max_tokens=_MAX_RESPONSE_TOKENS
        )
        return self._parse_label(result.content, query)

    def _build_messages(self, query: str) -> list[ChatMessage]:
        messages = [ChatMessage(role="system", content=_SYSTEM_PROMPT)]
        for example_query, label in _FEW_SHOT_EXAMPLES:
            messages.append(ChatMessage(role="user", content=example_query))
            messages.append(ChatMessage(role="assistant", content=label))
        messages.append(ChatMessage(role="user", content=query))
        return messages

    @staticmethod
    def _parse_label(raw_response: str, query: str) -> ComplexityLabel:
        normalized = raw_response.strip().lower()

        has_simple = "simple" in normalized
        has_complex = "complex" in normalized
        if has_simple and not has_complex:
            return "simple"
        if has_complex and not has_simple:
            return "complex"
        if has_simple and has_complex:
            # Both words present (e.g. "this is not simple, it's complex") --
            # trust whichever appears first.
            return "simple" if normalized.index("simple") < normalized.index("complex") else "complex"

        raise ClassificationError(
            f"could not parse a simple/complex label from model response "
            f"{raw_response!r} for query {query!r}"
        )
