# data/

Phase 1 data-prep outputs (BUILD.md section 2). This is dataset provenance
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
