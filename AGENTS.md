# AGENTS.md

## Project purpose and non-negotiables

Samsung PRISM GenAI Hackathon 2026, **Theme 01 - Agentic Code Intelligence**. Build natural-language-to-code
retrieval: given a query and a code library or repository, rank code snippets by relevance. Judged on retrieval
ranking quality and partly on search across code versions. The core goal is high retrieval quality with a fast,
practical CPU query path. Reranking is not tested by the organisers; the live demo is run by a person.

- **Online inference and the product demo must run on CPU.** A GPU may be used offline for corpus and repository
  encoding and for evaluation preparation. Keep those costs separate from serving latency.
- Evaluate every proposed approach for CPU feasibility before implementing it: query encoding cost, memory,
  indexing and rebuild cost, quality-versus-latency trade-offs.
- Use local open-weight models. Hosted embedding or LLM APIs are outside the retrieval pipeline.
- **Never push directly to `main` (this repository's default branch is `master`).** Repository authors raise and
  merge PRs. Do not commit, push, create releases or open PRs unless asked; a roadmap instruction to commit or
  publish is not authorization.
- Preserve unrelated user changes and existing experiment evidence.
- The user's global rules apply: no em-dashes (use hyphens), never add an agent as commit co-author, descriptive
  commit messages, uv for Python, Context7 for documentation lookups.

## The settled approach (decided with the user)

Two embedding models and a gated lexical lane, all fp32 PyTorch on CPU at query time, exact FAISS search:

1. **Chunk** a repository with tree-sitter (Python, JavaScript, TypeScript, TSX) into function, method, class
   header and statement-group units with exact `file:line` spans. Statement groups under 60 non-blank characters are
   merged into a contiguous neighbour; `.d.ts` files and `generated/` directories are not indexed.
2. **Index** every unit twice offline (GPU): with `embeddinggemma-300m` (768-d) and with `jina-code-0.5b` (896-d),
   in two exact FAISS inner-product indexes over identical units. Vectors are cached by the exact embedded text.
3. **Search** (per query): Gemma encodes the query; test files are excluded unless the query asks for tests; if the
   query holds a code anchor (quoted string, snake_case, camelCase or dotted name that exists in the code) a BM25
   lane runs and is fused by RRF; Gemma's top-50 candidates are then re-scored by a min-max blend
   `s = (1 - g) * (0.7 * jina + 0.3 * gemma) + g * bm25` with `g = 0.25` when the BM25 gate fired and `0`
   otherwise, jina's cosine coming from the stored document vectors after a second (jina) query encoding.
4. **Official benchmark path** (MTEB AppsRetrieval): a plain encoder with pre/post processing
   (`PrePostPipelineEncoder`). It must stay free of the repository additions above (no BM25, no re-scoring).

Decided against, with the evidence in `docs/experiments/` (the code for these was removed from `src/`): the signature
header on split pieces, merging fragments under 150 characters, a LightGBM or cross-encoder reranker, ungated BM25, query
rewriting, int8/OpenVINO serving and its fallback chain, Qwen3 as a model, an agent loop (undecided if ever revisited).
The reciprocal rank fusion inside the BM25 lane stays: it builds the candidate pool for the final blend.

The old "one encoder only" rule no longer applies to the repository pipeline (two encoders are used on purpose).
It still applies to the official screening path.

## Sources of truth

Read the relevant sources before changing implementation or proposing a plan:

| Source | Use |
| --- | --- |
| `README.md` | Setup, supported workflows, headline results |
| `pyproject.toml`, `uv.lock`, `src/`, `scripts/`, `tests/` | Actual dependencies, interfaces, commands, behaviour |
| `docs/theme1 guidelines.pdf`, `docs/samsung prism 2026 brochure.pdf` | Official hackathon requirements; consult before submission-related changes |
| `docs/architecture.md` | Modules and design rules |
| `docs/experiments/` | Every experiment: expectation, result, conclusion, caveats |
| `docs/handoff.md` | State of the project and open decisions |
| `eval/node-red/queries.json`, `eval/browseros/queries.json` | Labelled repository queries |

Distinguish **implemented behaviour**, **planned work** and **measured results**. Do not assume a module or command
exists because a document mentions it. If implementation and docs disagree, say so and keep claims grounded in the
code. The settled search is implemented as `FusionSearcher` (`src/coderet/index/fusion.py`) on top of `Searcher`
(test scope and the gated BM25 lane); the demo backend is `src/coderet/demo/` and the UI is `main.py` (Reflex). The
experiment scripts (`scripts/fusion_repo.py`, `two_stage_repo.py`, `apps_two_stage.py`) produced the numbers in
`docs/experiments/`. Search across commits (per-commit manifests, grouping near-duplicate versions) is planned, not built.

## Environment and commands

Python 3.11 (`>=3.11`) and **uv**; run commands from the repository root. Dependencies are in `pyproject.toml`
and locked in `uv.lock`; MTEB is pinned to `2.21.0`; tree-sitter is pinned below 0.26 (0.26.0 returns wrong node
positions and segfaults on real JavaScript). torch comes from the cu128 index so documents can be encoded on GPU.
There is no lint, type-check command or CI workflow; do not invent one as a required check.

```bash
uv sync --group dev
uv run pytest                                              # no downloads, GPU or dataset needed
uv run python scripts/run_mteb.py [--model embeddinggemma-300m]   # official-format AppsRetrieval run
uv run python scripts/index_repo.py <repo> --out outputs/index/<name> [--model embeddinggemma-300m]
uv run reflex run --env prod --single-port                 # demo UI (Reflex; use --env prod, log to reflex.log)
uv run python scripts/ablate_repo.py --queries eval/<repo>/queries.json --index a=<dir> ...
uv run python scripts/fusion_repo.py                       # learned fusion, cross-repo (needs both repos' indexes)
uv run python scripts/eval_fusion.py --repo <name>=<queries.json>   # settled search: metrics and latency (CUDA_VISIBLE_DEVICES="")
```

Environment gotchas: a small GPU (4 GB class) is tight, so never run two GPU jobs at once; use a document batch of 8 and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. A CUDA out-of-memory error during indexing fails the
build; index on the CPU (`--device cpu`) when the GPU is busy. Run heavy jobs one at a time (15 GB RAM) and do not
time anything while another heavy job runs.

## Implementation conventions

- Keep reusable logic under `src/coderet/` and orchestration under `scripts/`. Follow existing typed Python,
  dataclasses and small protocol-based interfaces; avoid speculative abstractions and unrelated refactors.
- Preserve `PrePostPipelineEncoder` as the shared pipeline for standalone and MTEB paths. Keep query and document
  roles and asymmetric prompts explicit. Models run as plain fp32 `TorchBackend`s.
- Embedding contract: `(n, dim)` float32 arrays, L2-normalised; preserve document-ID/vector alignment and descending
  score order. Models are `ModelSpec` entries in `config.py` with an exact checkpoint revision, documented prompts
  and a sequence-length limit. jina-code is 896-d with last-token pooling; Gemma is 768-d with mean pooling and dense
  layers and must not be run in fp16.
- Include every embedding-affecting setting in run and cache identity (model revision, prompts, truncation,
  preprocessing, chunking, corpus content and order). The vector cache key hashes the exact embedded text, never the
  syntax-tree hash. Never mix vectors of different models or precisions in one index.
- Both indexes of a repository must hold identical units in the same order (scripts assert this); rebuild both when
  chunking changes.
- Filter candidates (test scope, later version scope) before the top-k, never after.
- Change dependencies intentionally and update the lockfile with them. Do not upgrade the pinned MTEB version.

## Evaluation and experiment discipline

- Screening is CoIR **AppsRetrieval** (3,765 test queries, 8,765 Python programs, one relevant program per query):
  NDCG@10 and MRR, also Recall@k. Use the **test split only for frozen configurations**; never tune prompts,
  thresholds, weights or model choice against its labels. Never reload the raw dataset inside encoder or search code:
  its metadata contains the answer. `scripts/apps_two_stage.py` reads only text, ids and relevance judgements.
- jina-code was trained on AppsRetrieval train, so APPS-train splits cannot validate it and its test score is partly
  in-domain. Disclose this wherever its score is quoted; Gemma's training data is not itemised.
- Repository evaluation: 50 labelled queries each on Node-RED (JavaScript) and BrowserOS (TypeScript). Both sets are
  now exposed (Node-RED was used to design the BM25 gate; BrowserOS was used to choose settings), so a final claim
  needs a fresh third set. 50 queries resolve only differences of roughly 0.08 MRR or more; treat gaps under about
  0.05 as noise and report per-query better/worse counts.
- Pre-register experiments before running them (grid, protocol, acceptance rule) in `docs/experiments/`, tune on one
  repository and evaluate on the other, and record what was expected, what happened and the caveats.
- Select improvements by retrieval quality and CPU latency, never by embedding similarity or parameter count.
- `outputs/appsretrieval_results.json` is the submission artifact. Do not overwrite it with exploratory runs. Keep
  MTEB's `to_disk` serializer so the output stays readable by MTEB.

## CPU performance requirements

- Profile first. Measure warm **batch-1** p50/p95 on a documented CPU, caches disabled, end to end; never add stage
  percentiles. Report cold-first-query separately and distinguish it from model-loading time.
- Test representative long queries as well as short ones, and disclose truncation.
- Two encoders at query time cost about the sum of both encodings; measure the whole fusion path on the demo
  hardware (`scripts/eval_fusion.py`, GPU hidden).

## Verification and handoff

- Run relevant unit tests for code changes and `uv run pytest` before handoff. Prefer small deterministic fixtures
  and fake encoders; ordinary unit tests must not need model downloads, dataset access or a GPU.
- For changes to ranking, scopes, caches or exports, verify IDs, scores, normalisation, top-k behaviour and parity
  with the dense baseline for queries the new stage should not touch.
- Full benchmark and indexing runs are expensive experiments, not routine checks; run them when the task calls for it.
- Update the README when supported commands or behaviour change and `docs/experiments/` when an experiment finishes
  or a decision changes. Keep this file aligned with the code, not with the plan.
- Summarize changes, checks actually run, measured trade-offs and blockers. Never claim an unrun check passed or a
  planned feature is implemented.
