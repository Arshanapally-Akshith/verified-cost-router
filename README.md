<div align="center">

# Verified Cost Router

**A LangGraph pipeline that adds a verification gate to semantic caching and complexity routing — so a risky decision is checked before it's served, not trusted blindly.**

[![Version](https://img.shields.io/badge/version-v1.0.0-blue)](https://github.com/Arshanapally-Akshith/verified-cost-router/releases/tag/v1.0.0)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-264%20passing-brightgreen)](#testing)
[![Orchestration](https://img.shields.io/badge/orchestration-LangGraph-1C3C3C)](https://github.com/langchain-ai/langgraph)
[![Inference](https://img.shields.io/badge/inference-Groq-F55036?logo=groq&logoColor=white)](https://groq.com)
[![Dashboard](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](dashboard/app.py)
[![Live Demo](https://img.shields.io/badge/live%20demo-Streamlit%20Cloud-FF4B4B?logo=streamlit&logoColor=white)](https://verified-cost-router-1.streamlit.app/)

[Live Demo](https://verified-cost-router-1.streamlit.app/) · [Problem](#the-problem) · [Results](#key-results) · [Architecture](#architecture--pipeline) · [Methodology & limitations](#evaluation-methodology--limitations) · [Dashboard](#dashboard--demo) · [Setup](#setup--usage)

</div>

---

## The problem

Semantic caching (GPTCache) and complexity routing (RouteLLM) are both
established ways to cut LLM API costs — and both **trust their own
decision without checking it**. GPTCache's own documentation
acknowledges embedding similarity can match texts with opposite meaning
but similar wording. RouteLLM's classifier can misroute a query that's
worded simply but requires strong-model reasoning. Gateway products
(Portkey-style) expose both as tunable thresholds, with no published,
workload-specific evaluation methodology behind the knob.

**Verified Cost Router** treats both decisions as probabilistic and adds
an explicit verification gate after each one: a Verifier agent
sanity-checks every risky cache hit and every cheap-model output before
it reaches the user. It then *measures*, not just claims, how often that
gate catches a bad decision — the numbers below are from a real
evaluation harness, not aspirational.

| Tool | What it does | What it doesn't do |
|---|---|---|
| **GPTCache** | Semantic cache: embed, vector search, threshold-based serve | No routing at all. No verification of cache-hit correctness — trusts the similarity score. |
| **RouteLLM** | Trained classifier routes to weak/strong model based on complexity | No caching. No verification of routing correctness — trusts the classifier. |
| **Portkey / gateways** | Config-driven caching + routing + observability behind one gateway | Tunable knobs, not a workload-specific evaluation methodology. No verification gate. |
| **Verified Cost Router** | Cache + router, same as above | Adds an explicit verification gate on risky cache hits and cheap-model outputs, **and reports precision/recall of that gate catching bad decisions.** |

## Key results

Measured against a 100-pair labeled adversarial set, 50 labeled
complexity items, and a 30-query replay of real ShareGPT traffic
([full report](data/eval_report.md)). Every number below is read
directly from that committed report — nothing here is rounded up or
cherry-picked.

| | Metric | Result |
|---|---|---|
| 🎯 | Cache precision / recall | **50.0% / 100.0%** |
| 🧭 | Router complex-recall (adversarial) | **98.0%** (49/50) |
| 🛡️ | Verifier catch rate — bad cache hit | **100.0%** near-miss |
| 🛡️ | Verifier catch rate — bad route | 0.0% *(n=1, too small to generalize — reported as-is)* |
| ✅ | Quality-regression spot check | **100%** comparable to strong model *(n=3, small sample)* |
| 💰 | Full-system cost vs. no-system baseline | **-0.6%** on natural replay traffic |

The honest headline: cache precision tops out at **~60%** because
true-duplicate and near-miss queries genuinely overlap in embedding
space under `all-MiniLM-L6-v2` — that's the GPTCache failure mode this
project set out to measure, confirmed empirically, and exactly why the
Verifier exists instead of a single similarity threshold. See
[Evaluation methodology & limitations](#evaluation-methodology--limitations)
for what the -0.6% figure does and doesn't show.

## Architecture / Pipeline

```mermaid
flowchart TD
    Q([Query in]) --> C{Cache check}
    C -->|no match| R{Router}
    C -->|risky hit| VC[["Verifier<br/>(cache hit)"]]
    C -->|high-confidence hit| LOG[["Log +<br/>cache write"]]
    VC -->|pass| LOG
    VC -->|fail| R
    R -->|simple| CHEAP["Generate<br/>Groq 8B"]
    R -->|complex| STRONG["Generate<br/>Groq 70B"]
    CHEAP --> VO[["Verifier<br/>(output)"]]
    VO -->|pass| LOG
    VO -->|fail — escalate| STRONG
    STRONG --> LOG

    style VC fill:#7c3aed,color:#fff,stroke:none
    style VO fill:#7c3aed,color:#fff,stroke:none
    style LOG fill:#16a34a,color:#fff,stroke:none
    style STRONG fill:#dc2626,color:#fff,stroke:none
    style CHEAP fill:#2563eb,color:#fff,stroke:none
```

| Component | Path | What it does |
|---|---|---|
| **Cache layer** | `cache/` | Local sentence-transformer embeddings + FAISS. Two similarity cutoffs (`no_match` / `risky_hit` / `high_confidence_hit`), chosen by sweeping the labeled eval set, not picked by eye — `scripts/tune_cache_thresholds.py`. |
| **Router** | `router/` | Prompt-based complexity classifier (Groq 8B) labeling each query `simple` / `complex` — not a trained model, and not just a length heuristic. |
| **Verifier** | `verifier/` | A Groq-8B agent used in two places: sanity-checking a risky cache hit against the new query, and sanity-checking a cheap-model output before it's served. A fail escalates to the router or the strong model respectively. |
| **Generation** | `llm/` | Thin `requests` wrapper around Groq's OpenAI-compatible endpoint. Retry-with-backoff on 429s (capped, so a near-exhausted quota fails fast instead of blocking silently) plus a watchdog thread that guarantees every call returns or raises within a bounded time. |
| **Orchestration** | `graph.py`, `pipeline/` | The topology above wired as a LangGraph state machine. `build_graph()` wires stub nodes (topology-only proof); `build_pipeline_graph()` wires the real components via `pipeline.nodes.PipelineNodes`. |
| **Eval harness** | `eval/` | Precision/recall/catch-rate scoring against the labeled set, the 3-baseline cost comparison, and the quality-regression spot check — all reused by `scripts/run_eval.py`. |
| **Dashboard** | `dashboard/` | Streamlit UI over the eval report; `dashboard/data.py` is pure data-transform (no Streamlit import), independently unit-tested. |

## Tech stack

- **Language / packaging**: Python 3.11+, `setuptools`, `pytest`
- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) — the pipeline is a real `StateGraph` with conditional edges, not a linear script
- **Inference**: [Groq](https://groq.com) (`llama-3.1-8b-instant` cheap tier, `llama-3.3-70b-versatile` strong tier), via a thin `requests`-based client
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`), local, no API cost
- **Vector search**: FAISS, local
- **Dashboard**: Streamlit + pandas
- **Data**: ShareGPT (replay traffic) + a hand-built, hand-verified labeled adversarial set

## Evaluation methodology & limitations

**Methodology.** `scripts/run_eval.py` runs, in one pass: (1) cache
precision/recall against 100 hand-labeled pairs, (2) router accuracy
against 50 hand-labeled complexity-mislabeled items, (3) verifier catch
rate on both the cache and route paths, (4) a 3-baseline cost comparison
(`no_system` / `cache_router_no_verifier` / `full_system`) replayed over
the same sample of real ShareGPT traffic, and (5) an LLM-judged
quality-regression spot check. Full methodology and numbers:
[`data/eval_report.md`](data/eval_report.md), [`data/README.md`](data/README.md).

**Known limitations, stated plainly:**

- **The replay sample is small (30 queries) and not repetitive.** Groq's
  free-tier rate limits make a full 5,000-query × 3-baseline replay
  impractical (~15+ hours) — see `run_eval.py`'s module docstring. At
  n=30 with zero semantic duplicates by chance, **no baseline saw a
  single cache hit**, so the reported -0.6% cost delta reflects
  router/verifier efficiency only, not caching. It is not a caching
  benchmark, and shouldn't be read as one.
- **A dedicated cache-reuse benchmark, on a deliberately synthetic
  workload.** `eval/cache_reuse_benchmark.py` and `scripts/run_eval.py
  --cache-reuse-only` add a second evaluation: a seeded random sample of
  the labeled set's true-duplicate pairs, each asked twice (original,
  then paraphrase) through both `no_system` and `full_system`, with a
  cache freshly isolated per pair so no pair can leak into another's
  result. The sample is data-driven, not a hardcoded count — 20% of
  whatever the labeled set's true-duplicate pairs happen to number
  (currently 50), seeded for reproducibility (`--cache-reuse-sample-pct`,
  default 0.2; `--seed`, default 42; population/percentage/seed/sample
  size are all recorded in [`data/eval_report.md`](data/eval_report.md)
  and the dashboard). At 10/50 (seed 42): **62.2% cost savings**
  vs. no-system, 50% cache hit rate — a real, unmodified number from
  that report, on a workload constructed specifically to contain
  repetition (unlike the 30-query natural replay above, which saw none
  by chance). Small n and a synthetic workload — illustrative of the
  cache paying off when it actually gets hit, not a claim about typical
  traffic.
- **Some sub-metrics have very small n.** Route-verification catch rate
  is 0.0% on n=1; the quality spot check is n=3. Both are reported as-is
  rather than hidden, but neither generalizes.
- **No load testing, concurrency testing, or CI** — out of scope by
  design (see [Non-goals](#non-goals)).
- **No trained routing classifier** — the router is a prompt-based
  few-shot classifier, a deliberate scope choice for a solo build.

## Dashboard / Demo

**Live demo**: [verified-cost-router-1.streamlit.app](https://verified-cost-router-1.streamlit.app/) — deployed on Streamlit Community Cloud, reading the same committed `data/eval_report.json` as below.

Or run it locally:

```bash
streamlit run dashboard/app.py
```

Reads `data/eval_report.json` (produced by `scripts/run_eval.py`) and
shows, without any live API calls: a results-at-a-glance summary,
natural-replay cost comparison and cumulative-cost chart, cache-hit-rate
over time, path distribution, cache/router/verifier precision-recall-
catch-rate, and the quality-regression spot check. `dashboard/data.py`
holds all data transforms as plain, Streamlit-free functions — tested
independently of the UI layer.

## Setup & Usage

Requires Python ≥3.11 and a free [Groq](https://console.groq.com) API key (no card required).

```bash
python -m venv .venv
source .venv/Scripts/activate   # .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"

cp .env.example .env            # then fill in GROQ_API_KEY
```

`GROQ_CHEAP_MODEL` / `GROQ_STRONG_MODEL` in `.env` are optional overrides
of the defaults, in case Groq renames or retires a model.

```bash
# Run one query through the real pipeline
python -m verified_cost_router.main "What is the capital of France?"

# Regenerate the cache similarity thresholds (sweeps the labeled eval set)
python scripts/tune_cache_thresholds.py

# Run the eval harness -- precision/recall, catch rates, 3-baseline cost
# comparison over replayed traffic (writes data/eval_report.json / .md)
python scripts/run_eval.py

# Run only the cache-reuse benchmark against an existing report,
# without re-running the full evaluation (see Limitations above)
python scripts/run_eval.py --cache-reuse-only

# View the results
streamlit run dashboard/app.py

# Regenerate the ShareGPT replay sample (not committed -- see data/README.md)
python scripts/prepare_replay_sample.py
```

## Testing

```bash
pytest
```

254 offline tests (fakes only, no network) plus a handful of real-API
integration tests that auto-skip unless `GROQ_API_KEY` is set. Testing
is deliberately light per this project's own scope: the labeled
adversarial set drives the precision/recall/catch-rate numbers, and the
eval harness above is the correctness signal against real traffic —
see [Non-goals](#non-goals).

## Project structure

```
src/verified_cost_router/
  cache/        semantic cache: embeddings, FAISS store, threshold tuning
  router/       prompt-based complexity classifier
  verifier/     cache-hit and output verification agent
  llm/          Groq chat-completions client + generation helper
  pipeline/     real LangGraph node implementations, request logging
  eval/         precision/recall/catch-rate + baseline/cache-reuse harness
  dashboard/    Streamlit data-transform layer (no Streamlit import)
  data_prep/    ShareGPT replay sampling, adversarial eval set loading
  graph.py      LangGraph topology (shared by stub and real nodes)
  state.py      shared GraphState / LlmCallUsage types
  config.py     GROQ_API_KEY / model-tier settings
  main.py       CLI entry point for one query
scripts/        run_eval.py, tune_cache_thresholds.py, prepare_replay_sample.py
dashboard/      dashboard/app.py -- Streamlit rendering layer
data/           datasets, tuned thresholds, eval reports (see data/README.md)
tests/
```

## Non-goals

- No trained routing classifier (prompt-based is sufficient for this scope)
- No load testing / concurrency benchmarking
- No CI pipeline
- No more than two LLM-driven agent roles (Verifier, Router)

---

<div align="center">

Built by [Arshanapally Akshith](https://github.com/Arshanapally-Akshith) · [Repository](https://github.com/Arshanapally-Akshith/verified-cost-router)

</div>
