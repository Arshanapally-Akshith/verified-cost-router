# Cache threshold sweep

Swept against 100 labeled cache pairs (50 true_duplicate, 50 near_miss), requiring recall >= 90%.

## Chosen thresholds

- high_confidence = 0.86
- risky = 0.50
- high-confidence precision = 59.7%
- recall (true duplicates reaching risky_hit or higher) = 100.0%
- near-miss pairs leaking into high_confidence_hit = 29

## Why precision is capped

true_duplicate similarity: min=0.663, median=0.919, max=0.973

near_miss similarity: min=0.614, median=0.909, max=0.996

The two distributions overlap heavily -- no single cutoff cleanly separates them. This is the exact GPTCache failure mode ARCHITECTURE.md section 1 describes (embedding similarity can match opposite-meaning, similar-wording text), confirmed empirically here rather than assumed. It's also why the pipeline routes the risky band to a Verifier (Phase 4) instead of trusting a single threshold. Hardest near-miss pairs (highest similarity despite differing in meaning):

| similarity | query_a | query_b |
|---:|---|---|
| 0.996 | How do I rewrite a sentence from passive voice to active voice? | How do I rewrite a sentence from active voice to passive voice? |
| 0.990 | Is it safe to take ibuprofen with blood pressure medication? | Is it unsafe to take ibuprofen with blood pressure medication? |
| 0.987 | Is it legal to make a U-turn at a red light in this state? | Is it illegal to make a U-turn at a red light in this state? |
| 0.973 | Is it legal to turn right on red in this state? | Is it legal to turn left on red in this state? |
| 0.971 | Does exercise have a positive correlation with mood? | Does exercise have a negative correlation with mood? |

## Top 10 threshold candidates

| high | risky | precision | recall | near-miss leaks |
|---:|---:|---:|---:|---:|
| 0.86 | 0.50 | 59.7% | 100.0% | 29 |
| 0.86 | 0.51 | 59.7% | 100.0% | 29 |
| 0.86 | 0.52 | 59.7% | 100.0% | 29 |
| 0.86 | 0.53 | 59.7% | 100.0% | 29 |
| 0.86 | 0.54 | 59.7% | 100.0% | 29 |
| 0.86 | 0.55 | 59.7% | 100.0% | 29 |
| 0.86 | 0.56 | 59.7% | 100.0% | 29 |
| 0.86 | 0.57 | 59.7% | 100.0% | 29 |
| 0.86 | 0.58 | 59.7% | 100.0% | 29 |
| 0.86 | 0.59 | 59.7% | 100.0% | 29 |
