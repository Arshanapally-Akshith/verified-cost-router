"""Verifier agent (ARCHITECTURE.md 4.2): a lightweight LLM check used in
two places in the pipeline --

(a) verify_cache_hit: does a risky cache hit's cached answer actually
    answer the *new* query, not just the one it was cached under
    (the GPTCache opposite-meaning-similar-wording failure mode)?
(b) verify_output: is a cheap-model (8B) answer good enough to serve
    before it goes out (the RouteLLM misroute failure mode, caught late)?

Runs on the cheap tier by design -- verification is a lighter judgment
task than generation, so it should stay cheap (ARCHITECTURE.md 4.2).

Not wired into the LangGraph pipeline yet -- that's pipeline/nodes.py,
which calls both methods from the verifier_cache and verifier_output
nodes respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from verified_cost_router.llm.groq_client import ChatCompletionClient, ChatMessage

VerifierLabel = Literal["pass", "fail"]

_MAX_RESPONSE_TOKENS = 8

_CACHE_HIT_SYSTEM_PROMPT = """You are a strict verifier for a semantic response cache.

You will see a NEW user question and a CACHED question/answer pair that \
an embedding-similarity search flagged as a possible match. Similar \
wording does not guarantee the same answer applies -- watch especially \
for negation, opposite direction (increase/decrease, buy/sell, before/\
after), or swapped entities (public/private, left/right) between the new \
question and the cached question.

Decide: does the cached answer correctly and completely answer the NEW \
question? Respond with exactly one word: pass or fail. No punctuation, \
no explanation."""

_OUTPUT_SYSTEM_PROMPT = """You are a strict verifier sanity-checking an answer from a smaller, \
cheaper model before it is served to a user.

Decide: does the answer correctly and adequately address the question, \
without being misleading, incomplete on a point that matters, or \
confidently wrong? Respond with exactly one word: pass or fail. No \
punctuation, no explanation."""


class VerificationError(RuntimeError):
    """Raised when the model's response can't be parsed into pass/fail."""


@dataclass(frozen=True)
class VerificationOutcome:
    """A pass/fail verdict plus the token usage it cost, for cost accounting."""

    label: VerifierLabel
    model: str
    prompt_tokens: int
    completion_tokens: int


class Verifier:
    """Runs the two verification checks described in ARCHITECTURE.md 4.2."""

    def __init__(self, client: ChatCompletionClient, model: str, temperature: float = 0.0) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    def verify_cache_hit(self, query: str, cached_query: str, cached_response: str) -> VerificationOutcome:
        """Does `cached_response` (originally for `cached_query`) still answer `query`?"""
        user_content = (
            f"New question: {query}\n\n"
            f"Cached question: {cached_query}\n"
            f"Cached answer: {cached_response}\n\n"
            "Does the cached answer correctly and completely answer the new question?"
        )
        return self._verify(_CACHE_HIT_SYSTEM_PROMPT, user_content)

    def verify_output(self, query: str, output: str) -> VerificationOutcome:
        """Is `output` a correct, adequate answer to `query`?"""
        user_content = f"Question: {query}\n\nProposed answer: {output}\n\nIs this answer correct and adequate?"
        return self._verify(_OUTPUT_SYSTEM_PROMPT, user_content)

    def _verify(self, system_prompt: str, user_content: str) -> VerificationOutcome:
        messages = [ChatMessage(role="system", content=system_prompt), ChatMessage(role="user", content=user_content)]
        result = self._client.chat_completion(
            self._model, messages, temperature=self._temperature, max_tokens=_MAX_RESPONSE_TOKENS
        )
        label = self._parse_label(result.content)
        return VerificationOutcome(
            label=label,
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )

    @staticmethod
    def _parse_label(raw_response: str) -> VerifierLabel:
        normalized = raw_response.strip().lower()

        has_pass = "pass" in normalized
        has_fail = "fail" in normalized
        if has_pass and not has_fail:
            return "pass"
        if has_fail and not has_pass:
            return "fail"
        if has_pass and has_fail:
            return "pass" if normalized.index("pass") < normalized.index("fail") else "fail"

        raise VerificationError(f"could not parse a pass/fail verdict from model response {raw_response!r}")
