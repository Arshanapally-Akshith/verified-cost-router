"""Lightweight, keyword-based traffic-composition analysis.

BUILD.md flags that ShareGPT dumps often skew heavily toward
code-generation requests and asks for the replay sample's composition to
be checked and reported rather than assumed representative. This module
gives a fast, reproducible (if approximate) breakdown for that check --
the categories are heuristics for spotting skew, not ground-truth labels.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

CATEGORY_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "code": (
        "code", "function", "python", "javascript", "sql", "bug", "compile",
        "regex", "algorithm", "class ", "def ", "html", "css", " api ",
        "json", "debug", "programming", "script", "typescript", "java ",
    ),
    "creative_writing": (
        "story", "poem", "write a song", "screenplay", "fictional",
        "creative writing", "write a scene", "novel", "lyrics",
    ),
    "summarization": (
        "summarize", "summary", "tl;dr", "key points", "bullet points", "main ideas",
    ),
    "translation": (
        "translate", "translation", "in spanish", "in french", "in german",
        "in japanese", "in chinese",
    ),
    "math_reasoning": (
        "solve", "calculate", "equation", "derivative", "integral",
        "probability", "how many", "math problem",
    ),
    "general_qa": (
        "what is", "who is", "when did", "where is", "why does", "explain", "how does",
    ),
}
OTHER_CATEGORY = "other"


def categorize(query: str) -> str:
    """Assign one heuristic category to a query via keyword matching.

    Categories are checked in CATEGORY_KEYWORDS order and the first match
    wins; a query matching no keyword set falls back to "other".
    """
    lowered = f" {query.lower()} "
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return OTHER_CATEGORY


def compute_composition(queries: Iterable[str]) -> Counter[str]:
    """Count queries per heuristic category."""
    return Counter(categorize(query) for query in queries)


def render_composition_report(counts: Counter[str], sample_size: int) -> str:
    """Render a markdown table of category share, most frequent first."""
    lines = ["| category | count | share |", "|---|---:|---:|"]
    for category, count in counts.most_common():
        share = count / sample_size if sample_size else 0.0
        lines.append(f"| {category} | {count} | {share:.1%} |")
    return "\n".join(lines)
