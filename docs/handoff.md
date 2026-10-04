# Handoff

State of the project for whoever picks it up next. See `architecture.md` for the modules and `experiments/` for every
experiment. Written 2026-10-04.

## Where things stand

- Branch `feat/v1` (local; nothing has been pushed). It starts from the initial scaffold.
- Built and tested (51 tests): the tree-sitter chunker (Python, JavaScript, TypeScript, TSX), the vector cache, an exact FAISS
  index per model, the gated BM25 lane with test-file scope, the `FusionSearcher` (Gemma top-50 re-scored by jina-code, Gemma
  and BM25), the demo backend that opens a GitHub repository, and the official-format benchmark run.
- The Reflex demo UI (`main.py`, `rxconfig.py`, `assets/`, the `reflex` dependency) was built last and is being tested by hand.
- Final pipeline, 50 labelled questions per repository (weights chosen on the other repository): Node-RED MRR@10 0.908,
  Recall@1 0.88, Recall@10 0.96; BrowserOS MRR@10 0.777, Recall@1 0.66, Recall@10 1.00. Scored on the AppsRetrieval data with
  `scripts/apps_two_stage.py`: NDCG@10 88.61, MRR@10 85.92, Recall@1 79.52, Recall@10 96.89. CPU query time 224 to 321 ms
  median end to end (fp32, both models).
- The official submission artifact is still the plain jina-code run through `scripts/run_mteb.py` (NDCG@10 83.83). Only a
  weighted sum of the two models' cosines (0.7 jina-code, 0.3 Gemma, 87.35 NDCG@10 in our own scoring) fits the official
  plain-encoder format; it has not been run through `mteb.evaluate`.

## Run it

```bash
uv sync --group dev
uv run pytest
uv run python scripts/index_repo.py eval/BrowserOS --out outputs/index/browseros-jina
uv run python scripts/index_repo.py eval/BrowserOS --out outputs/index/browseros-gemma --model embeddinggemma-300m
uv run reflex run --env prod --single-port           # demo at http://localhost:3000
CUDA_VISIBLE_DEVICES="" uv run python scripts/eval_fusion.py --repo browseros=eval/browseros/queries.json
uv run python scripts/run_mteb.py                    # official-format benchmark run
```

The README has every command as its own block. The labelled repositories are cloned to `eval/BrowserOS` (git-ignored) and
`../prism-test-repos/node-red`.

## Things that will bite you

- **A 4 GB GPU is tight.** Never run two GPU jobs at once. A CUDA out-of-memory error during indexing makes sentence-transformers
  fail; index on the CPU (`--device cpu`) if the GPU is busy.
- **CPU indexing is slow:** about 1.6 units/s for both models (ten minutes per thousand units). The vector cache makes a repeat
  free.
- **Gemma is gated** on Hugging Face (accept the licence, `hf auth login`) and must run in fp32; jina-code ships bfloat16, which
  is about 3.4x slower on a CPU without AVX-512 or AMX, so it is always loaded in fp32.
- **tree-sitter 0.26.0 returns wrong node positions and segfaults on real JavaScript.** `pyproject.toml` pins it below 0.26.
- **MTEB caches results** under `~/.cache/mteb` by (model name, revision); `run_mteb.py` forces a fresh run and uses
  `fingerprint()` as the revision.
- **The raw AppsRetrieval dataset leaks the answer** (corpus `meta_information.url` equals the query's url, and a `partition`
  field separates test from train). Code must never reload it inside an encoder or search function; `apps_two_stage.py` reads
  only text, ids and relevance judgements.
- **jina-code was trained on AppsRetrieval's training split:** APPS-train splits cannot validate it.
- **Both indexes of a repository must hold identical units in the same order:** rebuild both when chunking changes.
- **Vectors of different models or precisions must never share an index.**

## Outside the repository

- `../prism-genai-2k26-backup-2026-10-04/`: the earlier plans, code, results and cached vectors from before the reset.
- `origin/feat/phase-0-1-baseline` on the remote is a teammate's earlier work and was not touched.
- Local, git-ignored: `.cache/` (vector cache, cloned repositories, agent briefs), `outputs/` (indexes, results).

## Decisions still open

- Whether the plain-encoder ensemble (0.7 jina-code + 0.3 Gemma) should replace the plain jina-code file as the official
  submission; it is not yet run through MTEB.
- A fresh third set of labelled questions is needed before any final claim: both current sets are exposed.
- Whether to revisit an agent loop (planning, searching, reading, refining); the council advised against it without an LLM.
- Search across commits: per-commit manifests, grouping near-duplicate versions across commits, "as of version X" search.
