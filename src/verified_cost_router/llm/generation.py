"""Plain generation calls to a Groq model tier (ARCHITECTURE.md 4.4).

Deliberately minimal: unlike the classifier and verifier, generation
doesn't need few-shot examples or output parsing -- just a system prompt
and the user's query, returning the full ChatCompletionResult (including
token usage, needed for cost accounting).
"""

from __future__ import annotations

from verified_cost_router.llm.groq_client import ChatCompletionClient, ChatCompletionResult, ChatMessage

_SYSTEM_PROMPT = "You are a helpful, direct assistant. Answer the user's question."


def generate(
    client: ChatCompletionClient, model: str, query: str, temperature: float = 0.2
) -> ChatCompletionResult:
    """Generate a response to `query` using `model`."""
    messages = [ChatMessage(role="system", content=_SYSTEM_PROMPT), ChatMessage(role="user", content=query)]
    return client.chat_completion(model, messages, temperature=temperature)
