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

## Quickstart demo (CPU)

Ranks a small built-in snippet library (it includes the guideline's three example snippets) and prints per-stage timings:

```bash
uv run python scripts/demo.py "How is the input preprocessed before going to the main function?"
```

Run it without arguments for four sample queries. Pass `--model qwen3-0.6b` to switch encoders.

## Results

| Model | AppsRetrieval test NDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|
| jina-code-embeddings-0.5b (current submission) | 83.87 | 80.83 | 98.51 |

Dev-split bakeoff, CPU latency and caveats are in [experiments/EXPERIMENTS.md](experiments/EXPERIMENTS.md).

## Reproduce the submission JSON

```bash
uv run python scripts/run_mteb.py --model jina-code-0.5b --out results/appsretrieval_results.json
```

Upload the JSON to a GitHub release. To rerun the dev bakeoff, use `uv run python scripts/bakeoff.py`. `embeddinggemma-300m` is gated: accept its license on Hugging Face and run `hf auth login` first.
