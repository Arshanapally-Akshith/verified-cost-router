"""Streaming access to the ShareGPT replay-traffic dataset (BUILD.md section 2).

Streams ShareGPT_V3_unfiltered_cleaned_split.json from Hugging Face and
extracts the first human turn of each conversation as a replay query,
without downloading the full ~673MB file: the HTTP connection is closed
as soon as the requested number of conversations has been read.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterator

import ijson
import requests

logger = logging.getLogger(__name__)

SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/"
    "resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)
DEFAULT_SAMPLE_SIZE = 5000


@dataclass(frozen=True)
class ReplayQuery:
    """One replay-traffic record: a conversation id and its opening query."""

    conversation_id: str
    query: str


def iter_conversations(stream: IO[bytes], limit: int) -> Iterator[dict]:
    """Yield up to `limit` raw conversation records from a ShareGPT JSON stream.

    Returns as soon as `limit` records have been yielded, so callers can
    stop reading (and close the connection) instead of consuming the
    whole file.
    """
    if limit <= 0:
        return
    count = 0
    for conversation in ijson.items(stream, "item"):
        yield conversation
        count += 1
        if count >= limit:
            return


def extract_first_human_turn(conversation: dict) -> str | None:
    """Return the first human message in a ShareGPT conversation, if any.

    Returns None for conversations with no turns, no human turn, or an
    empty/whitespace-only human message. Malformed records are expected
    at this dataset's scale, so callers are expected to skip a None
    result rather than treat it as an error.
    """
    turns = conversation.get("conversations")
    if not isinstance(turns, list):
        return None
    for turn in turns:
        if isinstance(turn, dict) and turn.get("from") == "human":
            value = turn.get("value")
            return value.strip() if isinstance(value, str) and value.strip() else None
    return None


def sample_replay_queries(stream: IO[bytes], limit: int = DEFAULT_SAMPLE_SIZE) -> Iterator[ReplayQuery]:
    """Stream up to `limit` conversations and yield a ReplayQuery for each usable one.

    Conversations without an extractable first human turn (or without a
    string id) are dropped; the drop count is logged once streaming ends.
    """
    skipped = 0
    for conversation in iter_conversations(stream, limit):
        conversation_id = conversation.get("id")
        query = extract_first_human_turn(conversation)
        if query is None or not isinstance(conversation_id, str):
            skipped += 1
            continue
        yield ReplayQuery(conversation_id=conversation_id, query=query)
    if skipped:
        logger.info("skipped %d conversations with no usable human turn", skipped)


def download_and_sample(
    output_path: Path,
    limit: int = DEFAULT_SAMPLE_SIZE,
    url: str = SHAREGPT_URL,
    timeout: float = 30.0,
) -> int:
    """Stream `url`, extract up to `limit` replay queries, and write them as JSONL.

    Each output line is `{"id": ..., "query": ...}`. Returns the number of
    queries written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        with output_path.open("w", encoding="utf-8") as out:
            for record in sample_replay_queries(response.raw, limit=limit):
                out.write(json.dumps({"id": record.conversation_id, "query": record.query}))
                out.write("\n")
                written += 1
    logger.info("wrote %d replay queries to %s", written, output_path)
    return written
