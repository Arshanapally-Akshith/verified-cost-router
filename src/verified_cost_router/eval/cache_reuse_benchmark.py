"""Cache-reuse benchmark (Phase 6 addendum): measures cost savings on a
workload deliberately constructed to have semantic repeats, unlike the
natural ShareGPT replay sample used for the 3-baseline comparison
(ARCHITECTURE.md section 6), which saw a 0% cache hit rate because a
small, non-repetitive random sample rarely contains near-duplicates.

Compares no_system against full_system -- the same pair of systems, and
the same cost-savings computation, ARCHITECTURE.md section 6 already
uses for the natural replay -- on a controlled duplicate-query workload,
so a real "does caching save money" number can actually be computed.
cache_router_no_verifier is deliberately not included here: this
benchmark exists to isolate caching's effect specifically, and
no_system vs full_system is the whole story for that.

Not a new labeled dataset: reuses the same hand-verified `true_duplicate`
pairs from data/adversarial_eval_set.json that the cache precision/
recall eval (eval/cache_eval.py) already scores against -- selecting a
seeded random subset, not hand-picking favorable pairs, so the
methodology stays honest and auditable.

Each selected pair becomes two queries asked in order: query_a first (a
cache miss -- generates and caches a fresh answer), then query_b, its
paraphrase (if the cache and verifier correctly recognize it, this
should be served from cache). Every pair's full_system run gets its own
fresh SemanticCache -- the same "each pair gets its own fresh, isolated
cache so pairs can never leak into each other's results" design
eval.cache_eval.evaluate_cache_pairs already uses -- so one pair's
cached answer can never affect another pair's hit/miss outcome. Both
run_no_system and run_full_system are the same, unmodified functions
scripts/run_eval.py already uses for the natural-replay baselines.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from verified_cost_router.cache.embeddings import EmbeddingModel
from verified_cost_router.cache.semantic_cache import SemanticCache
from verified_cost_router.cache.thresholds import CacheThresholds
from verified_cost_router.data_prep.adversarial_eval import CachePair
from verified_cost_router.eval.baselines import BaselineResult, run_full_system, run_no_system
from verified_cost_router.graph import build_pipeline_graph
from verified_cost_router.llm.groq_client import ChatCompletionClient, GroqAPIError, GroqRateLimitError
from verified_cost_router.pipeline.nodes import PipelineNodes
from verified_cost_router.pipeline.request_log import DEFAULT_PRICING, ModelPricing
from verified_cost_router.router.classifier import ClassificationError, ComplexityClassifier
from verified_cost_router.verifier.verifier import VerificationError, Verifier

logger = logging.getLogger(__name__)

# Same convention as scripts/run_eval.py's _SKIPPABLE_ERRORS: a model
# refusal that doesn't parse, or a rate limit surviving GroqClient's own
# retries, must skip that one query, not abort the whole benchmark.
_SKIPPABLE_ERRORS = (ClassificationError, VerificationError, GroqAPIError, GroqRateLimitError)


def select_cache_reuse_pairs(
    cache_pairs: Sequence[CachePair], sample_pct: float, seed: int
) -> tuple[CachePair, ...]:
    """A seeded random sample of `sample_pct` of the available
    true_duplicate pairs -- not the "best" or most-similar ones, so the
    benchmark can't be accused of cherry-picking pairs that flatter the
    cache hit rate.

    Data-driven (a fraction of whatever the labeled eval set happens to
    contain) rather than a hardcoded pair count, so the sample size tracks
    the population instead of silently under- or over-sampling it as the
    labeled set grows or shrinks.
    """
    true_duplicates = [pair for pair in cache_pairs if pair.category == "true_duplicate"]
    n = round(len(true_duplicates) * sample_pct)
    rng = random.Random(seed)
    return tuple(rng.sample(true_duplicates, k=min(n, len(true_duplicates))))


def build_cache_reuse_queries(pairs: Sequence[CachePair]) -> list[str]:
    """query_a (miss, caches an answer) then query_b (paraphrase, should
    hit) for each pair, in pair order -- the queries a full_system run
    would see. Used for logging/counting; run_cache_reuse_benchmark below
    iterates pairs directly since it also needs each pair's identity to
    build that pair's isolated cache."""
    queries: list[str] = []
    for pair in pairs:
        queries.append(pair.query_a)
        queries.append(pair.query_b)
    return queries


@dataclass(frozen=True)
class CacheReuseBenchmarkResult:
    """no_system vs full_system on the same controlled duplicate-pair
    workload. Keyed like eval.report's baseline_raw_results (by system
    name) but returned separately -- this never merges into or overwrites
    the natural-replay baselines."""

    pair_count: int
    no_system_raw_results: tuple[BaselineResult, ...]
    full_system_raw_results: tuple[BaselineResult, ...]


def run_cache_reuse_benchmark(
    pairs: Sequence[CachePair],
    *,
    embedder: EmbeddingModel,
    thresholds: CacheThresholds,
    classifier: ComplexityClassifier,
    verifier: Verifier,
    groq_client: ChatCompletionClient,
    cheap_model: str,
    strong_model: str,
    pricing: Mapping[str, ModelPricing] = DEFAULT_PRICING,
) -> CacheReuseBenchmarkResult:
    """For each pair: run [query_a, query_b] through run_no_system (no
    cache involved, so no isolation concern), then through run_full_system
    against a brand-new SemanticCache built just for that pair -- pair 2's
    cache starts empty even though pair 1's full_system run just cached
    an answer, so pair 2's hit/miss outcome can only be explained by pair
    2's own queries.
    """
    no_system_results: list[BaselineResult] = []
    full_system_results: list[BaselineResult] = []

    for pair in pairs:
        queries = (pair.query_a, pair.query_b)

        for query in queries:
            try:
                no_system_results.append(
                    run_no_system(query, groq_client=groq_client, strong_model=strong_model, pricing=pricing)
                )
            except _SKIPPABLE_ERRORS as exc:
                logger.warning(
                    "cache_reuse_benchmark (no_system, pair %s): skipping after %s: %s",
                    pair.id,
                    type(exc).__name__,
                    exc,
                )

        pair_cache = SemanticCache(embedder, thresholds)
        pair_nodes = PipelineNodes(
            cache=pair_cache,
            classifier=classifier,
            verifier=verifier,
            groq_client=groq_client,
            cheap_model=cheap_model,
            strong_model=strong_model,
            pricing=pricing,
        )
        pair_app = build_pipeline_graph(pair_nodes)
        for query in queries:
            try:
                full_system_results.append(run_full_system(query, app=pair_app, pricing=pricing))
            except _SKIPPABLE_ERRORS as exc:
                logger.warning(
                    "cache_reuse_benchmark (full_system, pair %s): skipping after %s: %s",
                    pair.id,
                    type(exc).__name__,
                    exc,
                )

    return CacheReuseBenchmarkResult(
        pair_count=len(pairs),
        no_system_raw_results=tuple(no_system_results),
        full_system_raw_results=tuple(full_system_results),
    )
