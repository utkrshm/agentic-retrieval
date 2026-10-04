# Architecture

Samsung PRISM GenAI Hackathon 2026, Theme 01 (code retrieval): rank the code snippets of a repository for a natural-language
question and return `file:line` ranges. Query time runs on CPU; a GPU is only used offline to embed repositories. This file
separates **implemented** from **measured**; measurements live in `experiments/`.

## Design rules (decided)

- Two embedding models, both open-weight and run as fp32 PyTorch: `google/embeddinggemma-300m` (revision `57c266a7`, 768-d)
  and `jinaai/jina-code-embeddings-0.5b` (revision `4db23513`, 896-d, trained on AppsRetrieval's training split). Gemma is
  mean-pooled with dense layers and must not run in fp16; jina-code is last-token pooled. Vectors are L2-normalised.
- Exact FAISS inner-product search, one index per model, over identical units. No SQLite, no graph, no LLM at query time, no
  learned reranker, no cross-encoder.
- A gated BM25 lane for queries that name an identifier or quote a message, merged by reciprocal rank fusion (it builds the
  candidate pool); the final order is a min-max blend of jina-code, Gemma and BM25 scores.
- Test files are excluded unless the query asks for tests; candidates are filtered before the top-k.
- The official AppsRetrieval path is a plain encoder with pre and post processing (`PrePostPipelineEncoder`) and stays free of
  the repository additions (the BM25 lane lowers the benchmark score by about 11 NDCG points).
- Both indexes of a repository hold identical units in the same order; vectors are cached by the exact embedded text.

## Implemented

```text
offline:  checkout --coderet.chunking--> units (file:line, kind, qualname, text)
          units --TorchBackend(Gemma)--> FAISS A        units --TorchBackend(jina-code)--> FAISS B
          (vector cache key = hash(model revision, role, prompt, length limit, precision, exact embedded text))

query:    Gemma encodes the query --> Searcher (test scope, gated BM25 + RRF) --> Gemma's top-50 units
          jina-code encodes the query --> blend: (1 - g) * (0.7 jina + 0.3 gemma) + g * bm25 --> top-10
```

| Module | What it does |
|---|---|
| `coderet/config.py` | `ModelSpec` registry (jina-code, embeddinggemma: prompts, pinned revision, sequence limit) and `fingerprint()`, a hash of every vector-affecting setting. |
| `coderet/chunking/treesitter.py` | Python, JavaScript, TypeScript and TSX into units: functions, methods, class headers (only if more than the declaration line) and statement groups; oversized code is split along the syntax tree; statement groups under 60 non-blank characters merge into a contiguous neighbour; units are at most 2,500 characters. `.d.ts`, `generated/`, dist, build, vendor and minified files are skipped; in a git checkout the tracked files are enumerated. |
| `coderet/embed/backends.py`, `vectors.py` | `TorchBackend` (sentence-transformers, fp32, role prompts as text prefixes) and `validate_vectors` (finite, unit norm, right dimension). |
| `coderet/index/cache.py` | `vector_key`, `VectorCache`: vectors keyed by the exact embedded text, never the syntax-tree hash. |
| `coderet/index/repo_index.py` | `RepoIndex`: chunk (or take pre-chunked units), embed, build an exact FAISS `IndexFlatIP`, save and load (`meta.json`, `units.jsonl`, `vectors.npy`, `index.faiss`). |
| `coderet/index/lexical.py` | Identifier-aware BM25 (whole identifiers plus snake_case and camelCase sub-tokens; a quoted phrase is one rare term), the anchor gate (a quoted literal or code-shaped token that exists in the code) and reciprocal rank fusion (k = 60). |
| `coderet/index/search.py` | `Searcher`: test-file scope applied through a FAISS ID selector before the top-k, plus the gated BM25 lane. |
| `coderet/index/fusion.py` | `FusionSearcher`: Gemma's top-50 from `Searcher`, re-scored by the min-max blend of jina-code, Gemma and BM25 (weights 0.7 and 0.25); returns per-model scores, the gate flag, the test-scope flag and per-stage timings. |
| `coderet/demo/` | `RepositoryIndexer` (open a GitHub link: shallow clone and CPU indexing with both models) and `Engine` (loads both models once, serves searches, builds result cards with commit-pinned GitHub links). |
| `coderet/eval/repo_queries.py` | Hit rule (same file, overlapping line range) and metrics (Recall@1/5/10, MRR@10, gold recall). |
| `coderet/mteb_adapters/encoder.py` | `PrePostPipelineEncoder` for the official benchmark: `preprocess -> embed -> postprocess`, role from MTEB's `PromptType`. |
| `main.py`, `rxconfig.py`, `assets/` | The Reflex demo UI. |

| Script | Purpose |
|---|---|
| `scripts/run_mteb.py` | Official-format AppsRetrieval run (`--model` jina-code or embeddinggemma); test split, frozen configurations only. |
| `scripts/index_repo.py` | Build one model's index of a checkout. |
| `scripts/run_queries.py`, `ablate_repo.py`, `two_stage_repo.py`, `fusion_repo.py` | Evaluate labelled questions: one index, search settings, the two-stage search, the fusion experiment. |
| `scripts/eval_fusion.py`, `bench_two_stage.py` | Metrics and CPU latency of the settled search; latency of the variants. |
| `scripts/apps_two_stage.py` | Scores the search variants on the benchmark data with MTEB's metrics (not the submission path). |
| `scripts/chunk_repo.py` | Inspect the chunker on a repository. |

Data: `eval/node-red/queries.json` and `eval/browseros/queries.json` hold 50 hand-labelled questions each (30 dev, 20
holdout); gold is resolved mechanically to chunker units. Tests: `uv run pytest` (51 tests, no models, dataset or GPU).

## Known limitations

- jina-code was trained on AppsRetrieval's training split, so APPS-train splits cannot validate it and its benchmark score is
  partly in-domain.
- Both labelled question sets are now exposed (Node-RED designed the BM25 gate, BrowserOS chose settings); 50 questions resolve
  only large differences.
- Two encoders at query time cost about the sum of both encodings and keep both models in memory.
- Tree-sitter call and definition facts are name-based: there is no call graph.
