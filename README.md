# Verified Cost Router

A LangGraph pipeline that combines semantic caching and complexity-based
model routing with an explicit **verification gate** on both decisions,
so a risky cache hit or a misrouted query gets checked before it's
served, not trusted blindly.

## Why this exists

Semantic caching (e.g. GPTCache) and complexity routing (e.g. RouteLLM)
are both established ways to cut LLM API costs, and gateway products
like Portkey bundle them as configurable features. None of them verify
their own decision before serving it: GPTCache's own documentation
acknowledges embedding similarity can match texts with opposite meaning
but similar wording, and RouteLLM's classifier can misroute a query
that's worded simply but requires strong-model reasoning. Gateways
expose these as tunable thresholds without a published, workload-specific
evaluation methodology. This project adds a verification stage after
both the cache and the router, and reports precision/recall on how
often that stage actually catches a bad decision -- see
[Evaluation results](#evaluation-results) below for the measured numbers,
not just the claim.

## Pipeline

```
Query in
   |
Cache check (embed + similarity search)
   |-- no match ------------------------> Router
   |-- risky hit --> Verifier (cache)
   |                    |-- fail -------> Router
   |                    |-- pass ------------------------------> Log + cache write
   |-- high-confidence hit -------------------------------------> Log + cache write
                        |
                     Router (complexity classifier)
                        |-- simple --> Groq 8B --> Verifier (output) --.
                        |-- complex -> Groq 70B ------------------->|
                                                                     |
                                                    pass -> Log + cache write
                                                    fail -> escalate to Groq 70B
```

- **Cache layer** (`cache/`): local sentence-transformer embeddings +
  FAISS, with two similarity cutoffs -- selected by sweeping the labeled
  eval set rather than picked by eye (`scripts/tune_cache_thresholds.py`)
  -- separating `no_match` / `risky_hit` / `high_confidence_hit`.
- **Router** (`router/`): a prompt-based complexity classifier (Groq 8B)
  labeling each query `simple` or `complex` -- not a trained model, and
  not just a length heuristic.
- **Verifier** (`verifier/`): a Groq-8B agent used in two places --
  sanity-checking a risky cache hit against the new query, and
  sanity-checking a cheap-model output before it's served. A fail
  escalates to the router or the strong model respectively.
- **Generation** (`llm/`): a thin `requests` wrapper around Groq's
  OpenAI-compatible chat completions endpoint, with retry-with-backoff
  on 429s and a watchdog thread guaranteeing every call returns or
  raises within a bounded time, even if the underlying socket never
  does (see `llm/groq_client.py`'s docstring).
- **Orchestration** (`graph.py`, `pipeline/`): the topology above wired
  as a LangGraph state machine. `graph.build_graph()` wires stub nodes
  (topology-only proof); `graph.build_pipeline_graph()` wires the real
  components via `pipeline.nodes.PipelineNodes`.

## Setup

Requires Python >=3.11 and a free [Groq](https://console.groq.com) API
key (no card required).

```bash
python -m venv .venv
source .venv/Scripts/activate   # .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"

cp .env.example .env            # then fill in GROQ_API_KEY
```

`GROQ_CHEAP_MODEL` / `GROQ_STRONG_MODEL` in `.env` are optional overrides
of the defaults (`llama-3.1-8b-instant` / `llama-3.3-70b-versatile`),
in case Groq renames or retires a model.

## Usage

Run one query through the real pipeline:

```bash
python -m verified_cost_router.main "What is the capital of France?"
```

Regenerate the cache similarity thresholds (sweeps the labeled eval set):

```bash
python scripts/tune_cache_thresholds.py
```

Run the eval harness -- cache precision/recall, router accuracy,
verifier catch rate, and the 3-baseline cost comparison over replayed
traffic (writes `data/eval_report.json` / `.md`):

```bash
python scripts/run_eval.py
```

View the results in the dashboard:

```bash
streamlit run dashboard/app.py
```

Regenerate the ShareGPT replay sample (not committed -- see
`data/README.md`):

```bash
python scripts/prepare_replay_sample.py
```

## Evaluation results

Full numbers and methodology notes: [`data/eval_report.md`](data/eval_report.md)
and [`data/README.md`](data/README.md). Headline results from the most
recent committed run (100 labeled cache pairs, 50 labeled complexity
items, 30 replay queries):

- **Cache**: 50.0% precision / 100.0% recall via the real `SemanticCache`.
  Precision is capped by real overlap between this project's
  true-duplicate and near-miss similarity distributions under
  `all-MiniLM-L6-v2` -- an empirical confirmation of the GPTCache
  failure mode this project targets, and the reason the risky band
  exists at all instead of a single threshold.
- **Router**: 98.0% complex-recall (49/50 adversarial, simply-worded-but-complex
  items correctly routed to the strong model).
- **Verifier**: 100.0% near-miss catch rate (every risky-band near-miss
  correctly failed), 85.7% true-duplicate pass rate. Route-verification
  catch rate is 0.0%, but on n=1 (only one router misroute occurred in
  this sample) -- reported as-is rather than hidden; too small to
  generalize from.
- **Baselines** (ARCHITECTURE section 6): on this replay sample,
  full-system cost was ~0.6% below the no-system baseline, with the
  verifier adding negligible net cost. See `data/eval_report.md` for
  why a small, non-repetitive sample understates caching's benefit.
- **Quality spot check**: 100% comparable-to-strong-model rate on
  non-strong-model responses, judged by the strong model.

Re-run `python scripts/run_eval.py` for fresh numbers against a larger
`--replay-sample-size`; the default of 30 is deliberately small given
Groq's free-tier rate limits (see the script's module docstring).

## Testing

```bash
pytest
```

Deliberately light per this project's own scope: the labeled adversarial
set drives precision/recall/catch-rate numbers, and the eval harness
above is the correctness signal against real traffic. No load testing,
concurrency testing, or CI -- out of scope (see Non-goals).

## Project layout

```
src/verified_cost_router/
  cache/        semantic cache: embeddings, FAISS store, threshold tuning
  router/       prompt-based complexity classifier
  verifier/     cache-hit and output verification agent
  llm/          Groq chat-completions client + generation helper
  pipeline/     real LangGraph node implementations, request logging
  eval/         precision/recall/catch-rate + baseline-comparison harness
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
