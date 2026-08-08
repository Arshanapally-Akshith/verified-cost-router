"""CLI: run the full Phase 5 eval harness in one script (BUILD.md
section 5: "Eval harness produces the precision/recall + cost numbers
... from a single script run").

Usage:
    python scripts/run_eval.py [--replay-sample-size 30] [--quality-sample-size 8]
                                [--cache-reuse-sample-pct 0.2] [--cache-reuse-only] [--seed 42]

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

--cache-reuse-sample-pct (default 0.2, i.e. 20%; 0 disables) runs a
small, separate benchmark after the 3 baselines: a seeded random sample
of that percentage of the true_duplicate pairs in the labeled
adversarial eval set (data-driven -- tracks however many true_duplicate
pairs that set actually has, instead of a hardcoded pair count that
could silently over- or under-sample it), each pair asked as 2 queries
(original, then paraphrase) through both no_system and full_system --
full_system gets its own fresh, isolated cache per pair, so no pair's
cached answer can affect another pair's result -- giving a real
cost-savings number for a workload that actually has cache reuse in it.
20% of the current 50 true_duplicate pairs is 10 pairs (20 queries),
comfortably inside the free tier's 30 RPM cap. See
verified_cost_router.eval.cache_reuse_benchmark for why this exists. It
never touches baseline_raw_results, so the 3-baseline natural-replay
comparison above is unaffected by it.

--cache-reuse-only skips the 6-step evaluation entirely (no cache/
router/verifier evals, no 3-baseline replay -- none of the Groq calls
those make) and only runs the cache-reuse benchmark above, against an
*existing* report at --report-json: every other section of that report
is carried over unchanged, and only cache_reuse_raw_results (plus its
population/sample-pct/seed metadata) is replaced. Lets the benchmark be
iterated on (different --cache-reuse-sample-pct, a code fix, etc.)
without re-running the full evaluation each time. Fails fast if
--report-json doesn't exist yet -- run once without this flag first.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
import sys
from functools import partial
from pathlib import Path
from typing import Callable, TypeVar

from verified_cost_router.cache.embeddings import EmbeddingModel, SentenceTransformerEmbedder
from verified_cost_router.cache.semantic_cache import SemanticCache
from verified_cost_router.cache.thresholds import CacheThresholds
from verified_cost_router.config import GroqSettings, load_groq_settings
from verified_cost_router.data_prep.adversarial_eval import AdversarialEvalSet, load_adversarial_eval_set
from verified_cost_router.eval.baselines import run_cache_router_no_verifier, run_full_system, run_no_system
from verified_cost_router.eval.cache_eval import evaluate_cache_pairs
from verified_cost_router.eval.cache_reuse_benchmark import run_cache_reuse_benchmark, select_cache_reuse_pairs
from verified_cost_router.eval.quality_eval import JudgeError, spot_check_quality
from verified_cost_router.eval.report import (
    CACHE_ROUTER_NO_VERIFIER,
    FULL_SYSTEM,
    NO_SYSTEM,
    EvalReport,
    load_eval_report,
    summarize_baseline,
)
from verified_cost_router.eval.router_eval import evaluate_complexity_items
from verified_cost_router.eval.verifier_eval import evaluate_cache_verifier, evaluate_route_verifier
from verified_cost_router.graph import build_pipeline_graph
from verified_cost_router.llm.groq_client import ChatCompletionClient, GroqAPIError, GroqClient, GroqRateLimitError
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
    parser.add_argument(
        "--cache-reuse-sample-pct",
        type=float,
        default=0.2,
        help="Fraction (0-1) of the labeled adversarial eval set's true_duplicate pairs to "
        "seed-sample for the cache-reuse benchmark, each selected pair asked as 2 queries. "
        "Data-driven -- tracks the true_duplicate population instead of a hardcoded pair "
        "count. 0 disables the benchmark. Default 0.2 (20%%) keeps the sample comfortably "
        "inside the free tier's 30 RPM cap on the current eval set.",
    )
    parser.add_argument(
        "--cache-reuse-only",
        action="store_true",
        help="Skip the 6-step evaluation (no cache/router/verifier evals, no 3-baseline "
        "replay) and only run the cache-reuse benchmark, merging its results into the "
        "existing report at --report-json (every other section is carried over unchanged). "
        "Requires that report to already exist and --cache-reuse-sample-pct > 0.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    return parser.parse_args()


def load_replay_queries(path: Path, sample_size: int, rng: random.Random) -> list[str]:
    with path.open(encoding="utf-8") as f:
        all_queries = [json.loads(line)["query"] for line in f]
    return rng.sample(all_queries, k=min(sample_size, len(all_queries)))


def _true_duplicate_population(eval_set: AdversarialEvalSet) -> int:
    return sum(1 for pair in eval_set.cache_pairs if pair.category == "true_duplicate")


def run_cache_reuse_step(
    eval_set: AdversarialEvalSet,
    sample_pct: float,
    seed: int,
    embedder: EmbeddingModel,
    thresholds: CacheThresholds,
    classifier: ComplexityClassifier,
    verifier: Verifier,
    groq_client: ChatCompletionClient,
    groq_settings: GroqSettings,
) -> dict[str, tuple]:
    """Select a seeded `sample_pct` fraction of the eval set's
    true_duplicate pairs and run the cache-reuse benchmark (see
    eval.cache_reuse_benchmark), returning results keyed like
    baseline_raw_results (NO_SYSTEM / FULL_SYSTEM). Empty if
    `sample_pct <= 0`. Shared by both the normal 6-step run and
    --cache-reuse-only, so the benchmark behaves identically either way.
    """
    if sample_pct <= 0:
        return {}

    population_size = _true_duplicate_population(eval_set)
    cache_reuse_pairs = select_cache_reuse_pairs(eval_set.cache_pairs, sample_pct, seed)
    logger.info(
        "cache-reuse benchmark: sampled %d/%d true_duplicate pairs (%.0f%%, seed=%d) -- "
        "%d queries each through no_system and full_system; full_system gets its own fresh, "
        "isolated cache per pair. Separate from the 3 baselines above -- never merged into them.",
        len(cache_reuse_pairs),
        population_size,
        sample_pct * 100,
        seed,
        len(cache_reuse_pairs) * 2,
    )
    cache_reuse_result = run_cache_reuse_benchmark(
        cache_reuse_pairs,
        embedder=embedder,
        thresholds=thresholds,
        classifier=classifier,
        verifier=verifier,
        groq_client=groq_client,
        cheap_model=groq_settings.cheap_model,
        strong_model=groq_settings.strong_model,
        pricing=DEFAULT_PRICING,
    )
    cache_reuse_raw_results = {
        NO_SYSTEM: cache_reuse_result.no_system_raw_results,
        FULL_SYSTEM: cache_reuse_result.full_system_raw_results,
    }
    for summary in (summarize_baseline(name, results) for name, results in cache_reuse_raw_results.items()):
        logger.info(
            "      cache_reuse_benchmark/%s: mean_cost_usd=%.6f cache_hit_rate=%.1f%%",
            summary.name,
            summary.mean_cost_usd,
            summary.cache_hit_rate * 100,
        )
    return cache_reuse_raw_results


def run_cache_reuse_only(
    args: argparse.Namespace,
    eval_set: AdversarialEvalSet,
    embedder: EmbeddingModel,
    thresholds: CacheThresholds,
    classifier: ComplexityClassifier,
    verifier: Verifier,
    groq_client: ChatCompletionClient,
    groq_settings: GroqSettings,
) -> None:
    """--cache-reuse-only: run just the cache-reuse benchmark and merge its
    results into the existing report at args.report_json -- every other
    section (cache/router/verifier evals, the 3 natural-replay baselines)
    is carried over from that file exactly as it was, so the benchmark
    can be iterated on without re-running the full 6-step evaluation
    (and its own Groq calls) each time.
    """
    if args.cache_reuse_sample_pct <= 0:
        sys.exit("--cache-reuse-only requires --cache-reuse-sample-pct > 0 (got 0)")
    if not args.report_json.exists():
        sys.exit(
            f"--cache-reuse-only requires an existing report at {args.report_json} "
            "-- run without this flag first to produce one."
        )

    logger.info("--cache-reuse-only: loading existing report from %s", args.report_json)
    existing_report = load_eval_report(args.report_json)

    cache_reuse_raw_results = run_cache_reuse_step(
        eval_set, args.cache_reuse_sample_pct, args.seed, embedder, thresholds, classifier, verifier,
        groq_client, groq_settings,
    )

    updated_report = dataclasses.replace(
        existing_report,
        cache_reuse_raw_results=cache_reuse_raw_results,
        cache_reuse_population_size=_true_duplicate_population(eval_set),
        cache_reuse_sample_pct=args.cache_reuse_sample_pct,
        cache_reuse_seed=args.seed,
    )
    updated_report.write(args.report_json, args.report_md)
    logger.info(
        "wrote %s and %s (cache-reuse benchmark only -- every other section unchanged)",
        args.report_json,
        args.report_md,
    )


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

    if args.cache_reuse_only:
        run_cache_reuse_only(args, eval_set, embedder, thresholds, classifier, verifier, groq_client, groq_settings)
        return

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

    cache_reuse_raw_results = run_cache_reuse_step(
        eval_set, args.cache_reuse_sample_pct, args.seed, embedder, thresholds, classifier, verifier,
        groq_client, groq_settings,
    )

    logger.info("[6/6] writing report")
    report = EvalReport(
        cache_eval=cache_eval,
        router_eval=router_eval,
        cache_verifier_eval=cache_verifier_eval,
        route_verifier_eval=route_verifier_eval,
        baseline_raw_results=baseline_raw_results,
        quality_spot_check=quality_spot_check,
        cache_reuse_raw_results=cache_reuse_raw_results,
        cache_reuse_population_size=_true_duplicate_population(eval_set) if cache_reuse_raw_results else 0,
        cache_reuse_sample_pct=args.cache_reuse_sample_pct if cache_reuse_raw_results else 0.0,
        cache_reuse_seed=args.seed if cache_reuse_raw_results else 0,
    )
    report.write(args.report_json, args.report_md)
    logger.info("wrote %s and %s", args.report_json, args.report_md)


if __name__ == "__main__":
    main()
