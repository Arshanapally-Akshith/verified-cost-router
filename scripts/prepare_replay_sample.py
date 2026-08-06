"""CLI: download the ShareGPT replay sample and report its composition.

Usage:
    python scripts/prepare_replay_sample.py [--limit 5000] [--out data/replay_sample.jsonl]

Writes replay queries as JSONL and a markdown composition report next to
it (see verified_cost_router.data_prep.composition for what "composition"
means here).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from verified_cost_router.data_prep.composition import compute_composition, render_composition_report
from verified_cost_router.data_prep.sharegpt import DEFAULT_SAMPLE_SIZE, download_and_sample

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "replay_sample.jsonl"
DEFAULT_REPORT = REPO_ROOT / "data" / "replay_sample_composition.md"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    written = download_and_sample(args.out, limit=args.limit)

    with args.out.open(encoding="utf-8") as f:
        queries = [json.loads(line)["query"] for line in f]
    counts = compute_composition(queries)
    table = render_composition_report(counts, sample_size=len(queries))

    header = (
        "# Replay sample composition\n\n"
        f"Heuristic, keyword-based category breakdown of {written} replay "
        "queries sampled from ShareGPT (BUILD.md section 2). This is not a "
        "ground-truth label set -- it exists only to check the sample isn't "
        "overwhelmingly skewed toward one traffic type (e.g. code generation) "
        "before using it as replay traffic in later phases, and should be read "
        "as a rough skew check, not a precise taxonomy.\n\n"
    )
    args.report.write_text(header + table + "\n", encoding="utf-8")
    logger.info("wrote composition report to %s", args.report)


if __name__ == "__main__":
    main()
