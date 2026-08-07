"""CLI: run the full Phase 5 eval harness in one script (BUILD.md
section 5: "Eval harness produces the precision/recall + cost numbers
... from a single script run").

Usage:
    python scripts/run_eval.py [--replay-sample-size 30] [--quality-sample-size 8] [--seed 42]

Requires GROQ_API_KEY and data/cache_thresholds.json (Phase 2). Reads
data/adversarial_eval_set.json (Phase 1) and data/replay_sample.jsonl
(Phase 1 -- regenerate with scripts/prepare_replay_sample.py if
missing). Writes data/eval_report.json and data/eval_report.md.

--replay-sample-size defaults to a modest 30 (not the full 5,000-query
replay set): the baseline comparison alone makes ~2-3 real Groq calls
per query per baseline across 3 baselines, and BUILD.md's own
"deliberately light -- not a research benchmark" testing philosophy
plus the free tier's 30 RPM cap make a full 5,000-query x 3-baseline
run impractical (~15+ hours). Pass a larger --replay-sample-size for a
more thorough run if you have the time and quota.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from functools import partial
from pathlib import Path
from typing import Callable, TypeVar

from verified_cost_router.cache.embeddings import SentenceTransformerEmbedder
from verified_cost_router.cache.semantic_cache import SemanticCache
from verified_cost_router.config import load_groq_settings
from verified_cost_router.data_prep.adversarial_eval import load_adversarial_eval_set
from verified_cost_router.eval.baselines import run_cache_router_no_verifier, run_full_system, run_no_system
from verified_cost_router.eval.cache_eval import evaluate_cache_pairs
from verified_cost_router.eval.quality_eval import JudgeError, spot_check_quality
from verified_cost_router.eval.report import (
    CACHE_ROUTER_NO_VERIFIER,
    FULL_SYSTEM,
    NO_SYSTEM,
    EvalReport,
    summarize_baseline,
)
from verified_cost_router.eval.router_eval import evaluate_complexity_items
from verified_cost_router.eval.verifier_eval import evaluate_cache_verifier, evaluate_route_verifier
from verified_cost_router.graph import build_pipeline_graph
from verified_cost_router.llm.groq_client import GroqAPIError, GroqClient, GroqRateLimitError
from verified_cost_router.pipeline.dependencies import load_cache_thresholds
from verified_cost_router.pipeline.nodes import PipelineNodes
from verified_cost_router.pipeline.request_log import DEFAULT_PRICING
from verified_cost_router.router.classifier import ClassificationError, ComplexityClassifier
from verified_cost_router.verifier.verifier import VerificationError, Verifier

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_SET = REPO_ROOT / "data" / "adversarial_eval_set.json"
DEFAULT_THRESHOLDS = REPO_ROOT / "data" / "cache_thresholds.json"
DEFAULT_REPLAY_SAMPLE = REPO_ROOT / "data" / "replay_sample.jsonl"
DEFAULT_REPORT_JSON = REPO_ROOT / "data" / "eval_report.json"
DEFAULT_REPORT_MD = REPO_ROOT / "data" / "eval_report.md"

logger = logging.getLogger(__name__)

# Real, uncurated replay traffic can trigger edge cases a curated eval set
# won't: a model refusal that doesn't parse as simple/complex or pass/fail,
# or a rate limit surviving GroqClient's own retries. One bad query must not
# crash the whole batch (BUILD.md's Definition of Done requires the pipeline
# to run "end-to-end... without manual intervention"), so these are caught
# and skipped per-query, not per-run.
_SKIPPABLE_ERRORS = (ClassificationError, VerificationError, JudgeError, GroqAPIError, GroqRateLimitError)

_T = TypeVar("_T")


def _run_over_queries(label: str, run_one: Callable[[str], _T], queries: list[str]) -> list[_T]:
    """Run `run_one(query)` for each query, skipping (and logging) any that
    raise one of `_SKIPPABLE_ERRORS` instead of aborting the whole batch.

    Logs progress every 5 queries -- each query is a real, potentially
    slow API call, and a long silent gap is otherwise indistinguishable
    from a genuine hang.
    """
    results: list[_T] = []
    skipped = 0
    for i, query in enumerate(queries, start=1):
        if i == 1 or i % 5 == 0:
            logger.info("%s: query %d/%d", label, i, len(queries))
        try:
            results.append(run_one(query))
        except _SKIPPABLE_ERRORS as exc:
            skipped += 1
            logger.warning("%s: skipping a query after %s: %s", label, type(exc).__name__, exc)
    if skipped:
        logger.info("%s: skipped %d/%d queries", label, skipped, len(queries))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--replay-sample", type=Path, default=DEFAULT_REPLAY_SAMPLE)
    parser.add_argument("--replay-sample-size", type=int, default=30)
    parser.add_argument("--quality-sample-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    return parser.parse_args()


def load_replay_queries(path: Path, sample_size: int, rng: random.Random) -> list[str]:
    with path.open(encoding="utf-8") as f:
        all_queries = [json.loads(line)["query"] for line in f]
    return rng.sample(all_queries, k=min(sample_size, len(all_queries)))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    rng = random.Random(args.seed)

    logger.info("loading settings, thresholds, and labeled eval data")
    groq_settings = load_groq_settings()
    thresholds = load_cache_thresholds(args.thresholds)
    eval_set = load_adversarial_eval_set(args.eval_set)

    embedder = SentenceTransformerEmbedder()
    groq_client = GroqClient(api_key=groq_settings.api_key)
    classifier = ComplexityClassifier(groq_client, model=groq_settings.cheap_model)
    verifier = Verifier(groq_client, model=groq_settings.cheap_model)

    logger.info("[1/6] cache precision/recall against %d labeled pairs", len(eval_set.cache_pairs))
    cache_eval = evaluate_cache_pairs(eval_set.cache_pairs, embedder, thresholds)
    logger.info("      precision=%.1f%% recall=%.1f%%", cache_eval.precision * 100, cache_eval.recall * 100)

    logger.info("[2/6] router accuracy against %d labeled complexity items", len(eval_set.complexity_items))
    router_eval = evaluate_complexity_items(eval_set.complexity_items, classifier)
    logger.info("      complex_recall=%.1f%%", router_eval.complex_recall * 100)

    logger.info("[3/6] verifier catch rate: cache-hit verification")
    cache_verifier_eval = evaluate_cache_verifier(eval_set.cache_pairs, embedder, thresholds, verifier)
    logger.info(
        "      near_miss_catch_rate=%.1f%% true_duplicate_pass_rate=%.1f%%",
        cache_verifier_eval.near_miss_catch_rate * 100,
        cache_verifier_eval.true_duplicate_pass_rate * 100,
    )

    logger.info("[3/6] verifier catch rate: route verification")
    route_verifier_eval = evaluate_route_verifier(
        eval_set.complexity_items, classifier, groq_client, groq_settings.cheap_model, verifier
    )
    logger.info("      route_catch_rate=%.1f%%", route_verifier_eval.catch_rate * 100)

    logger.info("[4/6] loading %d replay queries for baseline comparison", args.replay_sample_size)
    replay_queries = load_replay_queries(args.replay_sample, args.replay_sample_size, rng)

    logger.info("[4/6] running baseline 1/3: no_system (%d queries)", len(replay_queries))
    no_system_results = _run_over_queries(
        "no_system",
        partial(run_no_system, groq_client=groq_client, strong_model=groq_settings.strong_model, pricing=DEFAULT_PRICING),
        replay_queries,
    )

    logger.info("[4/6] running baseline 2/3: cache_router_no_verifier (%d queries)", len(replay_queries))
    cache_router_cache = SemanticCache(embedder, thresholds)
    cache_router_results = _run_over_queries(
        "cache_router_no_verifier",
        partial(
            run_cache_router_no_verifier,
            cache=cache_router_cache,
            classifier=classifier,
            groq_client=groq_client,
            cheap_model=groq_settings.cheap_model,
            strong_model=groq_settings.strong_model,
            pricing=DEFAULT_PRICING,
        ),
        replay_queries,
    )

    logger.info("[4/6] running baseline 3/3: full_system (%d queries)", len(replay_queries))
    full_system_cache = SemanticCache(embedder, thresholds)
    full_system_nodes = PipelineNodes(
        cache=full_system_cache,
        classifier=classifier,
        verifier=verifier,
        groq_client=groq_client,
        cheap_model=groq_settings.cheap_model,
        strong_model=groq_settings.strong_model,
    )
    full_system_app = build_pipeline_graph(full_system_nodes)
    full_system_results = _run_over_queries(
        "full_system", partial(run_full_system, app=full_system_app, pricing=DEFAULT_PRICING), replay_queries
    )

    baseline_raw_results = {
        NO_SYSTEM: tuple(no_system_results),
        CACHE_ROUTER_NO_VERIFIER: tuple(cache_router_results),
        FULL_SYSTEM: tuple(full_system_results),
    }
    for summary in (summarize_baseline(name, results) for name, results in baseline_raw_results.items()):
        logger.info(
            "      %s: mean_cost_usd=%.6f cache_hit_rate=%.1f%%",
            summary.name,
            summary.mean_cost_usd,
            summary.cache_hit_rate * 100,
        )

    logger.info("[5/6] quality-regression spot check (sample size %d)", args.quality_sample_size)
    non_strong_served = [
        (r.query, r.response) for r in full_system_results if not r.used_strong_model
    ]
    quality_spot_check = spot_check_quality(
        non_strong_served, groq_client, groq_settings.strong_model, args.quality_sample_size, rng
    )
    logger.info(
        "      comparable_rate=%.1f%% (%d spot-checked out of %d non-strong-model responses)",
        quality_spot_check.comparable_rate * 100,
        len(quality_spot_check.items),
        len(non_strong_served),
    )

    logger.info("[6/6] writing report")
    report = EvalReport(
        cache_eval=cache_eval,
        router_eval=router_eval,
        cache_verifier_eval=cache_verifier_eval,
        route_verifier_eval=route_verifier_eval,
        baseline_raw_results=baseline_raw_results,
        quality_spot_check=quality_spot_check,
    )
    report.write(args.report_json, args.report_md)
    logger.info("wrote %s and %s", args.report_json, args.report_md)


if __name__ == "__main__":
    main()
