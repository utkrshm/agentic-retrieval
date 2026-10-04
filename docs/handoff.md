# Handoff

State of the project for whoever picks it up next. See `architecture.md` for what each module does and
what is still only planned. Written 2026-10-04.

## Where things stand

- Branch `feat/v1` (local; nothing has been pushed). It starts from `main`'s initial scaffold.
- Built: tree-sitter chunker (Python and JavaScript), embedding backends with OpenVINO int8 and automatic
  failure recovery, the MTEB `PrePostPipelineEncoder`, the official-format AppsRetrieval run script, the FAISS
  repository index (latest commit only) with an exact-text vector cache, 50 labelled Node-RED queries and a
  query runner that logs top-k matches to JSON.
- Not built: per-commit manifests and version search, BM25, the Reflex demo. See "Planned" in `architecture.md`.
- Node-RED, no tuning: holdout Recall@5 0.95, Recall@10 1.0, MRR@10 0.705; dev Recall@10 0.90, MRR@10 0.746.
  Behavioural questions work (holdout MRR 0.85); literal and error-code questions are the weak spot.
  Per-query results: `outputs/node-red-results.json` and the readable `outputs/node-red-digest.md` (git-ignored).
- AppsRetrieval test split, official format, PyTorch fp32 (details in `architecture.md`):

  NDCG@10 **83.83**, MRR@10 **80.79**, Recall@1 73.81, Recall@10 93.23,
  Recall@100 98.51 (output: `outputs/appsretrieval_results.json`, git-ignored). This matches the
  earlier bf16 result (83.87) to within 0.05 points.

## Run it

```bash
uv sync --group dev                       # torch comes from the cu128 index (see pyproject.toml)
uv run pytest                             # 45 tests, no downloads or GPU
uv run python scripts/run_mteb.py         # official-format AppsRetrieval run -> outputs/appsretrieval_results.json
uv run python scripts/index_repo.py <repo> --out outputs/index/<name>   # FAISS index of the checkout (about 8 min on the GPU)
uv run python scripts/run_queries.py --index outputs/index/<name> --queries eval/node-red/queries.json --out outputs/<name>-results.json
uv run python scripts/export_openvino.py  # optional: OpenVINO int8 (about 40 s, about 4 GB RAM)
uv run python scripts/bench_encoder.py --repo <repo>   # backend speed and agreement on real code units
uv run python scripts/failure_drill.py    # proves the encoder recovers from broken models
uv run python scripts/chunk_repo.py <repo>             # inspect the chunker on any repository
```

`run_mteb.py` uses the AppsRetrieval **test** split. Run it only for a frozen configuration; never tune
anything against it. `outputs/` is git-ignored.

## Environment

Developed on an i5-12500H (4 performance cores, AVX2 and AVX-VNNI, no AVX-512/AMX), RTX 3050 4 GB and
15 GB RAM. Python 3.11, uv, `mteb==2.21.0` pinned.

## Things that will bite you

- **bfloat16 is slow on this CPU.** The checkpoint ships bf16; the backends force fp32 (about 3.4x faster
  on CPU). Never load the default dtype for serving.
- **sentence-transformers' OpenVINO backend returns NaN for this model.** The exported graph is fine; use
  `OpenVinoBackend` (raw runtime). The `ResilientEncoder` health probe exists because of this.
- **tree-sitter 0.26.0 corrupts node positions and segfaults on real JavaScript** (found on Node-RED).
  `pyproject.toml` pins `<0.26`. Do not unpin without re-running `scripts/chunk_repo.py` on a real repo.
- **Out-of-memory kills.** Loading several model copies in one process (or running the benchmark next to
  other heavy jobs) can pass the 15 GB limit; the Linux OOM killer ended one of my runs at about 6 GB of
  one process. Run heavy scripts one at a time, and use a fresh subprocess per scenario when comparing models.
- **`uv sync` removes packages that are not in the lockfile.** torchvision is a direct dependency on
  purpose: `tool.uv.sources` only applies to direct dependencies, and a CPU/PyPI torchvision next to a
  CUDA torch breaks `sentence_transformers` imports.
- **MTEB caches results** under `~/.cache/mteb` by (model name, revision). `run_mteb.py` forces a fresh run
  and uses `fingerprint()` as the revision; keep it that way.
- **The raw AppsRetrieval dataset leaks the answer:** corpus `meta_information.url` equals the query's url,
  and a `partition` field separates test from train. MTEB passes only id, text and title. Code must never
  reload the raw dataset inside an encoder or search function.
- **jina-code was trained on AppsRetrieval train.** APPS-train splits cannot validate it. The old dev split
  (in the backup) is also the wrong distribution (50% Codewars, while the test split is 78% Codeforces and
  18% AtCoder), and its grouping silently ignored source URLs.
- **Do not mix int8 and fp32 vectors in one index.** int8 is for queries only.

## Outside the repository

- `../prism-genai-2k26-backup-2026-10-04/`: the earlier plans, code, results and cached fp32 vectors that were
  removed in the reset (including the old frozen result JSON: NDCG@10 83.87, bf16 on GPU).
- `../prism-test-repos/node-red/`: the JavaScript repository used for chunker tests (full history, 10,812
  commits). Chosen for being plain JavaScript, mid-sized, active and rich in registration patterns.
- `origin/feat/phase-0-1-baseline` on the remote is a teammate's earlier work and was not touched.
- Local, git-ignored: `.cache/openvino/` (exported models) and `.cache/agent-briefs/` (briefings and the
  written replies of the three reviewing agents: simplicity review and the evidence-backed quality ideas).

## Decisions still open

- **Agent loop (plan, search, read, refine).** All three reviewers advised skipping it: without an LLM it
  would be pseudo-relevance feedback (drift, double latency, unmeasurable without labelled queries).
  Options: A skip; B interactive refinement in the UI (the user is the loop); C one bounded second search,
  kept only if it wins on a held-out set.
- **`AGENTS.md` is stale.** It still names the deleted `docs/PLAN-v3.md`, a `PrePostPipelineEncoder` in the old
  layout, `SearchProtocol` and the old dev splits. It was left untouched on request.
- **`README.md` is still the empty scaffold.**
- **ONNX export** (instead of OpenVINO) for faster CPU inference was deliberately left out of scope.
- Organiser questions never sent: whether a non-commercial encoder is acceptable, which document governs the
  hands-on round (the guideline says AppsRetrieval-style queries, the brochure a JavaScript repository), and
  whether the sample JavaScript repository will be provided.

## Suggested next steps

1. Use the labelled queries to decide the open retrieval questions (dense only against gated BM25 fusion,
   with and without the path header, unit size and fragment folding, test files). The 20 holdout queries have been
   seen by reviewers: write a fresh holdout before claiming a final number.
2. Per-commit manifests with "as of commit X" (flat index rebuilt from the manifest) and lineage grouping
   across versions.
3. Check how many gold units exceed 512 tokens and whether repeating the function header on split pieces helps.
4. Reflex page, then update `README.md` and `AGENTS.md` to match the code.
