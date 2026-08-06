"""CLI: sweep cache similarity thresholds against the labeled adversarial
eval set and persist the chosen (high_confidence, risky) pair.

Usage:
    python scripts/tune_cache_thresholds.py [--min-recall 0.9]

Writes data/cache_thresholds.json (the thresholds Phase 4 will load when
it wires the real cache into the LangGraph pipeline) and a markdown
report of the top candidates considered.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from verified_cost_router.cache.embeddings import SentenceTransformerEmbedder
from verified_cost_router.cache.threshold_tuning import compute_pair_similarities, sweep_thresholds
from verified_cost_router.data_prep.adversarial_eval import load_adversarial_eval_set

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_SET = REPO_ROOT / "data" / "adversarial_eval_set.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "cache_thresholds.json"
DEFAULT_REPORT = REPO_ROOT / "data" / "cache_threshold_sweep.md"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    eval_set = load_adversarial_eval_set(args.eval_set)
    embedder = SentenceTransformerEmbedder()
    similarities = compute_pair_similarities(eval_set.cache_pairs, embedder)
    evaluations = sweep_thresholds(eval_set.cache_pairs, similarities, min_recall=args.min_recall)
    best = evaluations[0]

    n_true_duplicate = sum(1 for p in eval_set.cache_pairs if p.category == "true_duplicate")
    n_near_miss = sum(1 for p in eval_set.cache_pairs if p.category == "near_miss")

    dup_sims = sorted(similarities[p.id] for p in eval_set.cache_pairs if p.category == "true_duplicate")
    nm_sims = sorted(similarities[p.id] for p in eval_set.cache_pairs if p.category == "near_miss")
    hardest_near_misses = sorted(
        ((similarities[p.id], p) for p in eval_set.cache_pairs if p.category == "near_miss"),
        reverse=True,
    )[:5]

    args.out.write_text(
        json.dumps(
            {
                "high_confidence": best.thresholds.high_confidence,
                "risky": best.thresholds.risky,
                "high_confidence_precision": best.high_confidence_precision,
                "recall": best.recall,
                "min_recall_constraint": args.min_recall,
                "embedding_model": embedder.__class__.__name__,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# Cache threshold sweep\n\n",
        f"Swept against {len(eval_set.cache_pairs)} labeled cache pairs "
        f"({n_true_duplicate} true_duplicate, {n_near_miss} near_miss), "
        f"requiring recall >= {args.min_recall:.0%}.\n\n",
        "## Chosen thresholds\n\n",
        f"- high_confidence = {best.thresholds.high_confidence:.2f}\n",
        f"- risky = {best.thresholds.risky:.2f}\n",
        f"- high-confidence precision = {best.high_confidence_precision:.1%}\n",
        f"- recall (true duplicates reaching risky_hit or higher) = {best.recall:.1%}\n",
        f"- near-miss pairs leaking into high_confidence_hit = {best.near_miss_leaks}\n\n",
        "## Why precision is capped\n\n",
        f"true_duplicate similarity: min={dup_sims[0]:.3f}, median={dup_sims[len(dup_sims)//2]:.3f}, "
        f"max={dup_sims[-1]:.3f}\n\n",
        f"near_miss similarity: min={nm_sims[0]:.3f}, median={nm_sims[len(nm_sims)//2]:.3f}, "
        f"max={nm_sims[-1]:.3f}\n\n",
        "The two distributions overlap heavily -- no single cutoff cleanly separates them. "
        "This is the exact GPTCache failure mode ARCHITECTURE.md section 1 describes (embedding "
        "similarity can match opposite-meaning, similar-wording text), confirmed empirically here "
        "rather than assumed. It's also why the pipeline routes the risky band to a Verifier "
        "(Phase 4) instead of trusting a single threshold. Hardest near-miss pairs (highest "
        "similarity despite differing in meaning):\n\n",
        "| similarity | query_a | query_b |\n",
        "|---:|---|---|\n",
    ]
    for similarity, pair in hardest_near_misses:
        report_lines.append(f"| {similarity:.3f} | {pair.query_a} | {pair.query_b} |\n")
    report_lines += [
        "\n",
        "## Top 10 threshold candidates\n\n",
        "| high | risky | precision | recall | near-miss leaks |\n",
        "|---:|---:|---:|---:|---:|\n",
    ]
    for evaluation in evaluations[:10]:
        report_lines.append(
            f"| {evaluation.thresholds.high_confidence:.2f} | {evaluation.thresholds.risky:.2f} | "
            f"{evaluation.high_confidence_precision:.1%} | {evaluation.recall:.1%} | "
            f"{evaluation.near_miss_leaks} |\n"
        )
    args.report.write_text("".join(report_lines), encoding="utf-8")

    logger.info(
        "chosen thresholds: high=%.2f risky=%.2f (precision=%.1f%%, recall=%.1f%%)",
        best.thresholds.high_confidence,
        best.thresholds.risky,
        best.high_confidence_precision * 100,
        best.recall * 100,
    )
    logger.info("wrote %s and %s", args.out, args.report)


if __name__ == "__main__":
    main()
