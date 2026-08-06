"""CLI entry point for exercising the walking skeleton end-to-end.

Usage:
    python -m verified_cost_router.main "some query"
"""

from __future__ import annotations

import logging
import sys

from verified_cost_router.graph import build_graph


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    query = sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France?"

    app = build_graph()
    result = app.invoke({"query": query})

    print(f"path: {' -> '.join(result['visited'])}")
    print(f"final state: {result}")


if __name__ == "__main__":
    main()
