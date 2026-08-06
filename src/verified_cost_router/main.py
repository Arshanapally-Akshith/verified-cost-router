"""CLI entry point for running one query through the real pipeline.

Usage:
    python -m verified_cost_router.main "some query"

Requires GROQ_API_KEY (see .env.example) and data/cache_thresholds.json
(produced by scripts/tune_cache_thresholds.py, Phase 2).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from verified_cost_router.config import load_groq_settings
from verified_cost_router.graph import build_pipeline_graph
from verified_cost_router.pipeline.dependencies import build_pipeline_nodes, load_cache_thresholds

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THRESHOLDS_PATH = REPO_ROOT / "data" / "cache_thresholds.json"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    query = sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France?"

    groq_settings = load_groq_settings()
    cache_thresholds = load_cache_thresholds(DEFAULT_THRESHOLDS_PATH)
    nodes = build_pipeline_nodes(groq_settings, cache_thresholds)

    app = build_pipeline_graph(nodes)
    result = app.invoke({"query": query})

    print(f"path: {' -> '.join(result['visited'])}")
    print(f"response: {result['response']}")


if __name__ == "__main__":
    main()
