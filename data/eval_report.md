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

