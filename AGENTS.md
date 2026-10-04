# AGENTS.md

## Project purpose and non-negotiables

Samsung PRISM GenAI Hackathon 2026, **Theme 01 — Agentic Code Intelligence**.
Build natural-language-to-code retrieval: given a query and a code library, rank
snippets by relevance. The core goal is high retrieval quality with minimal GPU
usage and a fast, practical CPU query path.

- **Online inference and the product demo must run on CPU.** GPU may be used
  offline for training, corpus encoding, and evaluation preparation. Keep those
  costs separate from serving latency.
- Evaluate every proposed approach for CPU feasibility during ideation, before
  implementation. Consider query encoding cost, memory, indexing/rebuild cost,
  and quality-versus-latency trade-offs.
- Use local open-weight models. Hosted embedding or LLM APIs are outside the
  planned retrieval pipeline.
- **Never push directly to `main`.** Repository authors raise and merge PRs.
  Do not commit, push, create releases, or open PRs unless explicitly requested;
  a roadmap instruction to commit or publish is not authorization.
- Preserve unrelated user changes and existing experiment evidence.

## Sources of truth

Read the relevant sources before changing implementation or proposing a plan:

| Source | Use |
| --- | --- |
| `README.md` | Current setup, supported workflows, and submission baseline |
| `pyproject.toml`, `uv.lock`, `src/`, `scripts/`, `tests/` | Actual dependencies, interfaces, commands, and implemented behavior |
| `docs/theme1 guidelines.pdf`, `docs/samsung prism 2026 brochure.pdf` | Official hackathon requirements; consult before submission-related changes |
| `docs/PLAN-v3.md` | Current roadmap and phase acceptance gates; supersedes `PLAN.md` and `PLAN-v2.md` |
| `docs/APPROACH.md` | Rationale and ADOPT / PILOT / PPT-ONLY / REJECT decisions |
| `docs/ARCHITECTURE.md` | Intended architecture, including planned and experimental components |
| `docs/POSSIBLE-FAILURES.md` | Known retrieval limitations and hypotheses to validate on dev |
| `experiments/EXPERIMENTS.md`, `results/` | Recorded measurements, configurations, and caveats |

Distinguish **implemented behavior**, **planned work**, and **measured results**.
Do not assume a module or command exists because it appears in the roadmap.
If implementation and docs disagree, identify the discrepancy and keep claims
grounded in the code. This file applies repository-wide; follow any more-specific
`AGENTS.md` introduced under a subdirectory for work in that scope.

## Environment and commands

Use **Python 3.11** (`>=3.11,<3.12`) and **uv**. Run commands from the repository
root. Dependencies are declared in `pyproject.toml` and locked in `uv.lock`;
MTEB is pinned to `2.21.0`. There is currently no configured lint/type-check
command or CI workflow; do not invent one as a required check.

## Implementation conventions

- Keep reusable logic under `src/coderet/` and orchestration under `scripts/`.
  Follow existing typed Python functions, dataclasses, and small protocol-based
  interfaces; avoid speculative abstractions and unrelated refactors.
- Preserve `PrePostPipelineEncoder` as the shared pipeline for standalone and
  MTEB paths. Keep query/document roles and asymmetric prompts explicit.
- Maintain the embedding contract: `(n, dim)` NumPy float32 arrays, with L2
  normalization in postprocessing. Preserve document-ID/vector alignment and
  descending top-k score ordering.
- Add models through `ModelSpec` in `config.py`, with an exact checkpoint
  revision, documented prompts, sequence-length limit, and relevant caveats.
- Include every embedding-affecting setting in run/cache identity. Changes to
  preprocessing, tokenizer/model revision, prompts, truncation, pooling,
  normalization, or corpus ordering/content must not silently reuse stale
  vectors or MTEB scores. Extend fingerprints/manifests where needed; the current
  fingerprint alone is not a complete corpus manifest.
- Preserve the pure-encoder submission fallback while implementing more complex
  retrieval. Require ranking/score parity checks before promoting a new backend,
  adapter, quantization path, or projection-folding implementation.
- Change dependencies intentionally and update the lockfile with them. Do not
  upgrade the pinned MTEB version incidentally.

## Evaluation and experiment discipline

- Screening uses CoIR **AppsRetrieval**: 3,765 test queries and an 8,765-document
  Python-solution corpus. Optimize NDCG@10 and MRR; also report Recall@k.
- Internal train/dev/holdout queries come from APPS **train**, grouped by source
  URL or normalized statement. Keep duplicate groups together and preserve
  deterministic splitting. Search the **full corpus** for every split.
- Use dev for iteration, internal holdout for promotion gates, and test only for
  frozen submission evaluations. Do not tune prompts, thresholds, routing, or
  model selection against test labels or use train/test corpus membership as a
  ranking signal.
- Disclose inherited benchmark exposure: jina-code was trained on AppsRetrieval
  train, so internal dev/holdout performance is optimistic. Distinguish locally
  measured scores from published scores and roadmap targets.
- Select improvements by retrieval quality and CPU latency, not embedding
  similarity or parameter count alone. Use controlled ablations and paired
  bootstrap intervals for promotion decisions where appropriate.
- Log the fingerprint, model/config, split, metrics, hardware, backend, thread
  count, input lengths/truncation, and CPU p50/p95. Save supporting artifacts and
  update `experiments/EXPERIMENTS.md` with caveats. `bakeoff.py` appends log rows
  and merges results into its output JSON; choose output paths deliberately.
- `results/appsretrieval_results.json` is the existing submission artifact.
  Do not overwrite it with exploratory runs. Keep MTEB's `to_disk` serializer so
  the output remains readable by MTEB, including datetime fields.

## CPU performance requirements

- Profile first. Baseline measurements identify query encoding as the bottleneck;
  an exact scan over this corpus is already cheap. Do not add an approximate
  index, extra encoder, or reranker without measuring an end-to-end benefit.
- Measure warm **batch-1** p50/p95 by stage and end-to-end on a documented CPU,
  with query-result caches disabled. Measure end-to-end percentiles directly;
  never add stage percentiles. Report cold-first-query and throughput separately,
  and distinguish cold-first-query timing from model-loading/startup cost.
- Test representative long APPS statements, not only short demo queries. Preserve
  constraints and I/O details when changing query views; disclose truncation.
- Treat `PLAN-v3.md` latency targets as acceptance goals, not achieved performance.
  Measure backend/thread/quantization choices on the actual demo hardware.
- Keep offline teacher-index generation distinct from CPU-rebuildable repository
  indexing. A query-only distilled student cannot regenerate teacher doc vectors.

## Verification and handoff

- Run relevant unit tests for code changes and `uv run pytest` before handoff.
  Prefer small, deterministic fixtures and fake embedders; ordinary unit tests
  should not require model downloads, dataset access, or a GPU.
- For changes to ranking, adapters, caches, or exports, verify IDs, scores,
  normalization, top-k behavior, and parity with the baseline as applicable.
- Full bakeoffs and MTEB runs are expensive experiments, not routine unit checks.
  Run them when the task calls for them; freeze the configuration before test
  evaluation. Documentation-only edits need path/command checks rather than a
  full model evaluation.
- Update the README when supported commands or behavior change, and the approach
  record/roadmap when an architectural decision changes. Keep this file aligned
  with the actual repository rather than copying the planned layout into it.
- Summarize changes, checks actually run, measured trade-offs, and any remaining
  blockers. Never claim an unrun check passed or a planned feature is implemented.
