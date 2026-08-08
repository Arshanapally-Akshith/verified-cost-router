# Eval report

## Cache precision/recall (labeled cache_pairs, BUILD.md section 4)

- precision: 50.0%
- recall: 100.0%
- pairs evaluated: 100

## Router accuracy (labeled complexity_items)

- complex-recall on adversarial complexity-mislabeled queries: 98.0%
- items evaluated: 50

## Verifier catch rate

- bad cache-hit catch rate (near_miss pairs correctly failed): 100.0%
- good cache-hit pass rate (true_duplicate pairs correctly passed): 85.7%
- pairs that reached the verifier: 28 (skipped, no_match: 0; skipped, leaked into high_confidence: 72, of which near_miss: 29)
- bad route catch rate (misrouted items correctly failed): 0.0%
- items router misrouted to cheap model: 1 (router correctly routed: 49)

## Baseline comparison (ARCHITECTURE.md section 6, replayed traffic)

| baseline | queries | mean cost/query (USD) | total cost (USD) | mean LLM calls | cache hit rate |
|---|---:|---:|---:|---:|---:|
| no_system | 30 | 0.000494 | 0.014812 | 1.00 | 0.0% |
| cache_router_no_verifier | 24 | 0.000533 | 0.012797 | 2.00 | 0.0% |
| full_system | 24 | 0.000491 | 0.011773 | 2.12 | 0.0% |

- full-system savings vs. no-system baseline: 0.6%
- verifier's added cost per query (full_system - cache_router_no_verifier): $-0.000043

## Quality-regression spot check

- comparable-to-strong-model rate: 100.0%
- responses spot-checked: 3

## Cache-reuse benchmark (synthetic, illustrative -- see eval.cache_reuse_benchmark)

Seeded true_duplicate pairs from the labeled adversarial eval set, each asked as two queries (original, then paraphrase); full_system uses a fresh, isolated cache per pair so no pair's cached answer can affect another pair's result.

- population (true_duplicate pairs available): 50
- sample percentage: 20%
- seed: 42
- sample size: 10 pairs (20 queries)

| system | queries | mean cost/query (USD) | cache hit rate |
|---|---:|---:|---:|
| no_system | 20 | 0.000272 | 0.0% |
| full_system | 20 | 0.000103 | 50.0% |

- cache-reuse savings vs. no-system: 62.2%

