"""Quality-regression spot check (ARCHITECTURE.md 4.6 / BUILD.md section
4): re-score a small random sample of served responses against what the
strong model would have produced for the same query, judged by the
strong model itself -- to show cost savings aren't coming at the
expense of silently worse answers.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Literal, Sequence

from verified_cost_router.llm.generation import generate
from verified_cost_router.llm.groq_client import ChatCompletionClient, ChatMessage, GroqAPIError, GroqRateLimitError

JudgeVerdict = Literal["comparable", "worse"]

_MAX_RESPONSE_TOKENS = 8

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = """You are comparing two candidate answers to the same question: a SERVED \
answer (what was actually given to the user) and a REFERENCE answer \
(from a stronger, more capable model). Judge whether the SERVED answer \
is comparable in correctness and completeness to the REFERENCE answer, \
even if worded differently -- not whether it's identical.

Respond with exactly one word: comparable or worse. No punctuation, no \
explanation."""


class JudgeError(RuntimeError):
    """Raised when the judge's response can't be parsed into a verdict."""


@dataclass(frozen=True)
class QualitySpotCheckItem:
    query: str
    served_response: str
    reference_response: str
    verdict: JudgeVerdict


@dataclass(frozen=True)
class QualitySpotCheckResult:
    items: tuple[QualitySpotCheckItem, ...]

    @property
    def comparable_rate(self) -> float:
        if not self.items:
            return 1.0
        return sum(1 for item in self.items if item.verdict == "comparable") / len(self.items)


def judge_response(
    client: ChatCompletionClient, judge_model: str, query: str, served_response: str, reference_response: str
) -> JudgeVerdict:
    """Ask `judge_model` whether `served_response` is comparable to `reference_response`."""
    user_content = (
        f"Question: {query}\n\n"
        f"SERVED answer: {served_response}\n\n"
        f"REFERENCE answer: {reference_response}\n\n"
        "Is the SERVED answer comparable to the REFERENCE answer?"
    )
    messages = [
        ChatMessage(role="system", content=_JUDGE_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_content),
    ]
    result = client.chat_completion(judge_model, messages, temperature=0.0, max_tokens=_MAX_RESPONSE_TOKENS)
    return _parse_verdict(result.content)


def spot_check_quality(
    served: Sequence[tuple[str, str]],
    client: ChatCompletionClient,
    strong_model: str,
    sample_size: int,
    rng: random.Random,
) -> QualitySpotCheckResult:
    """Sample up to `sample_size` (query, served_response) pairs, generate
    a strong-model reference for each, and judge the served answer against it.

    `rng` is injected so sampling is reproducible in tests; callers doing
    a real run should pass a seeded random.Random for a reproducible report.

    Real replay-derived queries can trigger a model refusal that doesn't
    parse as comparable/worse, or a rate limit surviving GroqClient's own
    retries -- one such item is skipped (and logged) rather than failing
    the whole spot check, so the result may have fewer than `sample_size`
    items.
    """
    sample = rng.sample(list(served), k=min(sample_size, len(served)))
    items = []
    for query, served_response in sample:
        try:
            reference = generate(client, strong_model, query)
            verdict = judge_response(client, strong_model, query, served_response, reference.content)
        except (JudgeError, GroqAPIError, GroqRateLimitError) as exc:
            logger.warning("quality spot check: skipping a query after %s: %s", type(exc).__name__, exc)
            continue
        items.append(
            QualitySpotCheckItem(
                query=query,
                served_response=served_response,
                reference_response=reference.content,
                verdict=verdict,
            )
        )
    return QualitySpotCheckResult(items=tuple(items))


def _parse_verdict(raw_response: str) -> JudgeVerdict:
    normalized = raw_response.strip().lower()

    has_comparable = "comparable" in normalized
    has_worse = "worse" in normalized
    if has_comparable and not has_worse:
        return "comparable"
    if has_worse and not has_comparable:
        return "worse"
    if has_comparable and has_worse:
        return "comparable" if normalized.index("comparable") < normalized.index("worse") else "worse"

    raise JudgeError(f"could not parse a comparable/worse verdict from judge response {raw_response!r}")
