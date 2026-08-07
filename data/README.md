# data/

Phase 1 data-prep outputs (BUILD.md section 2), Phase 2 cache-tuning
outputs (BUILD.md section 3 / ARCHITECTURE.md 4.1), and Phase 5 eval
outputs (BUILD.md section 5). This is dataset and artifact provenance
documentation, not the project README (that's Phase 7).

## `replay_sample.jsonl` (generated, gitignored)

First human turn of up to 5,000 ShareGPT conversations, used as replay
traffic in later phases. Regenerate with:

```bash
python scripts/prepare_replay_sample.py
```

Not committed: it's ~4MB of third-party data reproducible from a fixed,
public source, so the script is the source of truth, not the file.

## `replay_sample_composition.md` (generated, committed)

Heuristic keyword-based category breakdown of the sample above, generated
by the same script. From the most recent local run (4,923 usable queries
out of a 5,000-conversation cap): **code-related queries were ~27% of the
sample**, confirming BUILD.md's expectation that ShareGPT skews toward
code generation. Traffic-cost and cache/router numbers computed against
this sample in later phases should be read with that skew in mind --
it is not a representative production traffic mix.

## `adversarial_eval_set.json` (hand-built, committed)

~150-200 labeled pairs/items used only to evaluate the cache and router
(never as replay traffic), per BUILD.md section 2:

- `cache_pairs` (`true_duplicate` / `near_miss`) -- test whether the
  cache layer correctly hits on paraphrases and correctly avoids hitting
  on similarly-worded but different-meaning queries (the GPTCache
  failure mode).
- `complexity_items` (`complexity_mislabeled`) -- simply-worded queries
  that actually require the strong model, to test whether the router is
  fooled by wording alone (the RouteLLM failure mode).

Candidates were drafted then manually reviewed against their category
definition (each item's `rationale` field records why the label was
assigned) before being committed. Schema is enforced by
`verified_cost_router.data_prep.adversarial_eval.load_adversarial_eval_set`,
exercised in `tests/test_adversarial_eval.py`.

## `cache_thresholds.json` (generated, committed)

The similarity cutoffs the cache layer uses to classify a lookup as
`no_match` / `risky_hit` / `high_confidence_hit`, chosen by sweeping a
grid against `adversarial_eval_set.json`'s `cache_pairs` rather than
picked by eye (ARCHITECTURE.md 4.1). Regenerate with:

```bash
python scripts/tune_cache_thresholds.py
```

**Current result: high_confidence=0.86, risky=0.50, precision=59.7% at
recall=100%.** That precision ceiling is a real finding, not a bug: this
project's own true_duplicate and near_miss similarity distributions
overlap heavily under `all-MiniLM-L6-v2` (near_miss pairs range up to
0.996 cosine similarity, true_duplicate pairs only up to 0.973 -- see
`cache_threshold_sweep.md` for the full breakdown and the specific
pairs responsible, e.g. "rewrite passive to active voice" vs "rewrite
active to passive voice" at 0.996). No single cutoff can cleanly serve
that top ~40% of high-confidence matches unverified -- which is the
GPTCache failure mode ARCHITECTURE.md section 1 names, confirmed
empirically here, and precisely why the pipeline routes the risky band
through a Verifier (Phase 4) instead of trusting a threshold alone.

## `cache_threshold_sweep.md` (generated, committed)

The full sweep report behind the thresholds above: similarity
distribution summary, the hardest (highest-similarity) near-miss pairs,
and the top 10 threshold candidates considered.

## `eval_report.json` / `eval_report.md` (generated, committed)

The Phase 5 eval harness output (BUILD.md section 4-5): cache
precision/recall, router accuracy, verifier catch rate, the 3-baseline
cost comparison (ARCHITECTURE.md section 6) over a sample of replay
traffic, and the quality-regression spot check. Regenerate with:

```bash
python scripts/run_eval.py
```

**Current result highlights** (100 labeled cache pairs, 50 labeled
complexity items, 30 replay queries):

- Cache precision 50.0% / recall 100.0% via the real SemanticCache
  (consistent with Phase 2's own sweep finding of ~60% precision at
  this recall -- the two independent measurements agree).
- Verifier near-miss catch rate 100.0% (every near_miss pair that
  reached the risky band was correctly failed), true-duplicate pass
  rate 85.7% (occasionally over-cautious with genuine duplicates).
- Router complex-recall 98.0% (49/50 adversarial items correctly
  routed); the one misroute wasn't caught by output verification
  (0.0% route catch rate on n=1 -- too small a sample to generalize,
  reported honestly rather than hidden).
- Baseline comparison: on this 30-query sample of real (mostly
  code-heavy, per Phase 1's composition finding) ShareGPT traffic,
  full-system cost was only ~0.6% below the no-system baseline, and
  6/30 queries triggered a genuine model refusal (skipped and logged,
  not silently dropped -- see `run_eval.py`'s `_run_over_queries`).
  With zero semantic duplicates among 30 random diverse queries, no
  baseline saw a cache hit in this run -- caching's benefit shows up
  over larger or more repetitive traffic, not a small random sample.
- Quality spot check: 100% comparable rate, but only 3 non-strong-model
  responses existed to sample from in this run (most traffic was
  correctly routed to the strong model already).

`--replay-sample-size` defaults to 30, not the full 5,000-query replay
set -- see `run_eval.py`'s module docstring for why (free-tier rate
limits make a full 3-baseline replay impractical, ~15+ hours).
