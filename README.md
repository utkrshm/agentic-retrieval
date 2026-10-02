# PRISM GenAI Hackathon 2026 — Theme 01: Agentic Code Intelligence

Natural-language → code retrieval. Given a query and a library of code snippets, return the snippets ranked by relevance. Only retrieval is in scope; query-time inference runs on CPU.

Screening metric: NDCG@10 and MRR on the test split of CoIR **AppsRetrieval** (3,765 problem statements × 8,765 Python solutions), produced with [MTEB](https://docs.mteb.org/).

## Layout

```
src/coderet/
  config.py            model registry, prompts, run fingerprints
  data/apps.py         CoIR APPS loading and grouped dev/holdout splits
  models/encoders.py   prompted sentence-transformer encoders
  mteb_adapters/       PrePostPipelineEncoder (MTEB AbsEncoder)
  retrieval/dense.py   exact dense top-k search
  eval/                metrics, bootstrap intervals, latency harness
scripts/               run_mteb.py (submission run), bakeoff.py, demo.py
experiments/           EXPERIMENTS.md (results log)
tests/
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

## Produce the submission JSON

```bash
uv run python scripts/run_mteb.py
```

Writes `appsretrieval_results.json` (upload it to a GitHub release).
