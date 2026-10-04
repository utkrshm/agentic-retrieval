# Architecture

Samsung PRISM GenAI Hackathon 2026, Theme 01 (code retrieval). What is judged: the retrieval
engine's ranking quality, and partly search across versions of code. Reranking is not tested; the
live demo is run by a person. This file separates **implemented**, **planned** and **measured**
so nothing planned is mistaken for something that exists.

## Design rules (decided)

- One embedding model: `jinaai/jina-code-embeddings-0.5b`, pinned to revision `4db23513`, code-trained,
  last-token pooling, L2-normalised, 896 dimensions. No licence considerations.
- Query-time inference on CPU. A GPU is only for offline work (document encoding, evaluation).
- Plain MTEB encoder path (`PrePostPipelineEncoder`) for the official benchmark. No `SearchProtocol`.
- Exact search only (FAISS flat inner product, planned). No SQLite, no graph, no reranker, no LLM at
  query time, no agent loop (decision pending, see `handoff.md`).
- PyTorch fp32 is the reference backend. OpenVINO int8 is an optional, query-side speed-up with
  automatic fallback. Documents are always fp32: int8 and fp32 vectors are never mixed in one index.
- One pipeline for every language. Tree-sitter covers Python and JavaScript now.

## Implemented

```text
source files --> coderet.chunking (tree-sitter) --> units (file:line, kind, qualname, text, ast_hash)
                                                        |
                                                        v   embed_text(unit) = path + "kind qualname" + code
queries / documents --> coderet.embed.ResilientEncoder --> L2-normalised float32 vectors (896-d)
                          chain for queries:  OpenVINO int8 -> OpenVINO fp32 -> PyTorch fp32
                          documents:          PyTorch fp32 (GPU if available, else CPU)
                                                        |
                          coderet.mteb_adapters.PrePostPipelineEncoder --> mteb.evaluate(AppsRetrieval)
```

| Module | What it does |
|---|---|
| `coderet/config.py` | `ModelSpec`, the `JINA_CODE` entry (prompts, revision, 1,024-token limit) and `fingerprint()`, a hash of every vector-affecting setting, used as the MTEB model revision so stale scores are never reused. |
| `coderet/chunking/treesitter.py` | Splits Python and JavaScript into units. Functions and methods are units, with leading comments or JSDoc inside the span; a class header is a unit only if it holds more than the declaration line; remaining top-level statements are grouped into `module` units; oversized code is split by descending the syntax tree (wrapper functions and IIFEs surface their inner functions), and only an indivisible leaf is cut by lines. Default limit 2,500 characters (jina was trained on 512-token sequences). `ast_hash` hashes the token stream without comments and whitespace and is for grouping versions only, never a cache key. `iter_source_files` uses `git ls-files` in a checkout (Node-RED keeps real source under `packages/node_modules/`), and skips dist/build/vendor and minified files. |
| `coderet/embed/backends.py` | `TorchBackend` (sentence-transformers, forced fp32 because the checkpoint ships bfloat16) and `OpenVinoBackend` (raw OpenVINO runtime, own tokenisation, last-token pooling, batch 1). |
| `coderet/embed/export.py` | Exports the model to OpenVINO with optimum-intel, compresses weights to int8 with NNCF, and stores fp32 PyTorch reference vectors for the health probe under `.cache/openvino/`. |
| `coderet/embed/probe.py` | Fixed startup health probe: finite, unit-norm, right dimension, semantic ordering on three triples, and cosine to the stored reference. |
| `coderet/embed/resilient.py` | `ResilientEncoder`: activates the first backend that builds and passes the probe; on any error or invalid output at runtime it demotes that backend permanently and retries the same batch on the next one; raises `EncoderUnavailable` if all fail; records every decision in `events`. |
| `coderet/mteb_adapters/encoder.py` | `PrePostPipelineEncoder` (the class named in the theme-1 guidelines): `preprocess -> embed -> postprocess`, role derived from MTEB's `PromptType`. It only sees what MTEB passes in. |

| Script | Purpose |
|---|---|
| `scripts/run_mteb.py` | Official-format run on AppsRetrieval; writes `outputs/appsretrieval_results.json`. Uses the test split: run only for a frozen configuration. |
| `scripts/export_openvino.py` | One-off export (about 40 s, about 4 GB RAM). |
| `scripts/bench_encoder.py` | Agreement with PyTorch and warm batch-1 latency per backend on real code units. |
| `scripts/failure_drill.py` | Breaks the real exported models seven ways and checks that the encoder recovers. |
| `scripts/chunk_repo.py` | Chunks a repository and reports units, coverage, size distribution and parse errors; checks the span invariant. |

Tests: `uv run pytest` (40 tests, no model downloads or GPU needed).

## Planned (not built)

- `repo` and `library` input modes. In `library` mode each document is one unit with no tree-sitter
  (AppsRetrieval documents are whole programs ranked by document id; chunking them would break the
  mapping). In `repo` mode, tree-sitter units.
- FAISS flat inner-product index, with the vector cache key `hash(model revision + prompt + exact embed text)`.
- Per-commit JSON manifests (path, span, kind, qualname, vector key, content key). "As of commit X"
  rebuilds a small flat index from the manifest, so results are filtered before the top-k, never after.
- Search across versions: search unique variants, group hits by `(path, kind, qualname)`, show the best
  variant with the others expandable, newest first.
- Identifier-aware BM25 fused with dense scores by plain RRF, gated to queries that contain code-like
  tokens, kept only if it wins on a held-out set.
- A hand-labelled JavaScript query set on Node-RED (about 30 dev, 20 holdout) and the evaluation that
  decides unit size, BM25 and the path header.
- A one-page Reflex demo (query box, version selector, timings, ranked snippets with file:line).

## Measured

All on one laptop: i5-12500H (AVX2 and AVX-VNNI, no AVX-512/AMX), RTX 3050 4 GB, 15 GB RAM.

**Encoder backends** (`scripts/bench_encoder.py`, 40 real Node-RED units, warm, batch 1):

| backend | short query | long code query | code document | cosine to PyTorch fp32 (min) |
|---|---|---|---|---|
| PyTorch fp32 | 80 ms | 377 ms | 324 ms | 1.0 |
| OpenVINO fp32 | 61 ms | 424 ms | 299 ms | 1.0 |
| OpenVINO int8 | 24 ms | 196 ms | 154 ms | 0.9987 |

int8 queries against the shipped fp32 document vectors, 500 APPS-train dev queries: NDCG@10 96.85 against
96.96 for fp32 (paired change -0.10 points, 95% interval -0.42 to +0.17), MRR@10 95.86 against 95.99,
top-10 overlap 97.1%, top-1 agreement 99.2%. These scores are inflated because jina-code was trained on
APPS train; the paired comparison is what matters.

**Chunker** on Node-RED (467 JS files): 5,787 units, median 1,200 characters, none above 2,500, 97.8% of
non-blank lines covered, no parse errors, no span-invariant violations, 2.9 s.

**AppsRetrieval, test split, official format** (`scripts/run_mteb.py`, PyTorch fp32, GPU):

| metric (test split, 3,765 queries x 8,765 documents) | this run (fp32, GPU) | earlier frozen run (bf16, GPU) |
|---|---|---|
| NDCG@10 | 83.83 | 83.87 |
| MRR@10 | 80.79 | 80.83 |
| Recall@1 | 73.81 | 73.84 |
| Recall@10 | 93.23 | 93.28 |
| Recall@20 | 95.64 | 95.62 |
| Recall@100 | 98.51 | 98.51 |

Run: `scripts/run_mteb.py`, PyTorch fp32 for documents and queries, RTX 3050, 16.5 minutes, run fingerprint
`642fc13092c7`, output `outputs/appsretrieval_results.json` (git-ignored; MTEB's own loader reads it back).
fp32 against bf16 changes nothing measurable (under 0.05 points). jina-code-0.5b was trained on AppsRetrieval
train, no tuning of any kind was done on this split, and published scores for this model are 81.6 to 84.2.

## Known limitations

- jina-code was trained on AppsRetrieval train, so APPS-train splits cannot validate it, and the test
  score is a property of this model more than of this pipeline.
- Chunk size, BM25 gating and the path header are untested on labelled queries (none exist yet).
- Tree-sitter call and definition facts are name-based. There is no call graph and no support for
  questions such as "which files call A before B".
- OpenVINO exports of this model must run through the raw runtime: sentence-transformers' OpenVINO
  wrapper returned NaN for every input.
