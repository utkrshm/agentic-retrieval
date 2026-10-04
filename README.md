# Agentic Code Retrieval

Video Demo Link: https://www.loom.com/share/d2efb905b1b04ebb873c2b4f3b7672c1

Team Sunday Wasters' for Theme 01 of the Samsung PRISM GenAI Hackathon 2026: natural-language-to-code retrieval. Given a question such
as "where is the retry logic for failed tool calls?", the system ranks the functions, methods and statement blocks of a code
repository and returns the best matches as `file:line` ranges, with a link to each on GitHub.

Query-time inference runs on CPU only. A GPU is optional and only used offline, to embed a repository's code ahead of time.
Search combines two open-weight embedding models (embeddinggemma-300m and jina-code-embeddings-0.5b), an exact FAISS index per
model and a BM25 keyword lane that switches on when a question names an identifier or quotes a message. The experiments
behind these choices, including the ones that did not help, are in `docs/experiments/`.

## Setup and commands

Python 3.11 and [uv](https://docs.astral.sh/uv/). Each block does one job and can be copied and run on its own after the
install and model-download blocks. Indexing and the benchmark use a GPU when one is available; searching does not need one.

### Install

```bash
uv sync --group dev
```

### Run the unit tests

No models, datasets or GPU are needed.

```bash
uv run pytest
```

### Download the models

Both models are open-weight and are fetched once into the local Hugging Face cache; revisions are pinned in
`src/coderet/config.py`.

- [jina-code-embeddings-0.5b](https://huggingface.co/jinaai/jina-code-embeddings-0.5b) (code-trained, 896-d)
- [embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) (768-d; gated: open the model page, accept the
  licence, then log in)

```bash
uv run hf download jinaai/jina-code-embeddings-0.5b --revision 4db235132dafbe56a8b9c5f59b59795ecf58a4a7
```

```bash
uv run hf auth login
```

```bash
uv run hf download google/embeddinggemma-300m --revision 57c266a740f537b4dc058e1b0cda161fd15afa75
```

### Download the evaluation repositories

The labelled questions in `eval/` were written for these commits. BrowserOS is cloned inside the repository (the directory is
git-ignored); Node-RED next to it.

```bash
git clone https://github.com/browseros-ai/BrowserOS.git eval/BrowserOS
git -C eval/BrowserOS checkout 0152e0a829
```

```bash
git clone https://github.com/node-red/node-red.git ../prism-test-repos/node-red
git -C ../prism-test-repos/node-red checkout cd05a9a38b
```

### Build the search indexes of a repository

The search uses one index per model over identical units. Indexing is the slow, offline step (minutes on a GPU) and re-runs
only embed code that changed. These names are what the experiment scripts expect.

```bash
uv run python scripts/index_repo.py eval/BrowserOS --out outputs/index/browseros-jina
```

```bash
uv run python scripts/index_repo.py eval/BrowserOS --out outputs/index/browseros-gemma --model embeddinggemma-300m
```

```bash
uv run python scripts/index_repo.py ../prism-test-repos/node-red --out outputs/index/node-red-jina
```

```bash
uv run python scripts/index_repo.py ../prism-test-repos/node-red --out outputs/index/node-red-gemma --model embeddinggemma-300m
```

### Run the demo

A Reflex web app: paste a GitHub link, the repository is cloned and indexed on the CPU, then ask questions. Open http://localhost:3000 once the server
reports it is running.

```bash
uv run reflex run --env prod --single-port
```

### Inspect the chunker on a repository

Prints unit counts, size distribution, line coverage and parse errors, and checks that every unit's text equals the file lines
of its span.

```bash
uv run python scripts/chunk_repo.py eval/BrowserOS
```

### Evaluate the labelled questions

The settled search on the 50 questions of one repository, on CPU, with end-to-end latency. Needs the four indexes above.

```bash
CUDA_VISIBLE_DEVICES="" uv run python scripts/eval_fusion.py --repo browseros=eval/browseros/queries.json
```

### Evaluate one model's index

Dense search with jina-code on one index; writes the top matches per question and per-category metrics to JSON.

```bash
uv run python scripts/run_queries.py --index outputs/index/browseros-jina --queries eval/browseros/queries.json --out outputs/browseros-results.json
```

### Compare search settings on one index

Dense search, test-file scope and the keyword lane, side by side.

```bash
uv run python scripts/ablate_repo.py --queries eval/browseros/queries.json --index jina=outputs/index/browseros-jina
```

### Compare the two models and the plain two-stage search

Gemma retrieves, jina-code re-scores the top-N (or the sum of both cosines), and the reverse.

```bash
uv run python scripts/two_stage_repo.py --queries eval/browseros/queries.json --jina outputs/index/browseros-jina --gemma outputs/index/browseros-gemma
```

### Run the score fusion experiment

Grid over the jina-code share and the BM25 share, chosen on one repository and tested on the other (and back), for pool sizes
10, 20 and 50. Needs the indexes of both repositories.

```bash
uv run python scripts/fusion_repo.py
```

### Measure CPU query latency

Warm, one query at a time, with the GPU hidden; reports jina-code alone, Gemma alone, the two-stage re-sort and the fusion.

```bash
CUDA_VISIBLE_DEVICES="" uv run python scripts/bench_two_stage.py --queries eval/browseros/queries.json --jina outputs/index/browseros-jina --gemma outputs/index/browseros-gemma
```

### Run the official benchmark (CoIR AppsRetrieval, MTEB)

`PrePostPipelineEncoder` through `mteb.evaluate`; writes an MTEB-readable result file (default
`outputs/appsretrieval_results.json`, git-ignored). Test split: run only for a frozen configuration.

```bash
uv run python scripts/run_mteb.py
```

```bash
uv run python scripts/run_mteb.py --model embeddinggemma-300m --out outputs/compare/gemma.json
```

### Score the search variants on the benchmark data

Not the official submission path: embeds the 8,765 programs and 3,765 queries once per model (GPU, cached under
`outputs/apps-vectors/`) and scores each variant with MTEB's metrics.

```bash
uv run python scripts/apps_two_stage.py
```

## Pipeline

```text
Offline (GPU), once per repository checkout
  files --tree-sitter--> units: function | method | class header | statement group, exact file:line spans
         (Python, JavaScript, TypeScript, TSX; .d.ts and generated/ skipped; statement groups < 60 non-blank
          chars merged into a contiguous neighbour; units <= 2,500 chars, oversized code split along the syntax tree)
  units --> embeddinggemma-300m (768-d)  --> exact FAISS IndexFlatIP A
  units --> jina-code-0.5b      (896-d)  --> exact FAISS IndexFlatIP B        same units, same order in both

Per query (CPU, fp32 PyTorch)
  1. Gemma encodes the query (task: code retrieval prompt).
  2. Candidate scope: test files (test dirs, *.test.*, *.spec.*, *_spec.js, test_*.py) are excluded before the
     top-k unless the query mentions tests.
  3. BM25 lane: if the query holds a quoted literal or a snake_case / camelCase / dotted identifier that occurs
     in the indexed code, identifier-aware BM25 (whole identifiers plus sub-tokens) is run and fused with the dense
     top-100 by reciprocal rank fusion (k = 60). Other queries stay dense-only.
  4. Gemma's top-50 are re-scored. jina-code encodes the query too and each candidate gets its stored jina cosine:
         s = (1 - g) * (0.7 * mm(jina) + 0.3 * mm(gemma)) + g * mm(bm25)        mm = per-query min-max
         g = 0.25 if the BM25 gate fired for this query, else 0
  5. Top-10 units, each with path, kind, qualified name, line range and a GitHub link.
```

Gemma and jina-code have complementary failure modes on code repositories: Gemma's top-10 almost always contains an answer
but it places the first correct unit at rank 1 less often than jina-code, which orders the top better. The blend keeps
Gemma's candidate pool, jina-code's ordering and, on identifier-bearing queries, the lexical evidence. The weights (0.7 and
0.25) were chosen on one repository and evaluated on the other in both directions. The BM25 lane is a repository feature
only: on the benchmark's problem statements it hurts, so the official AppsRetrieval path stays a plain encoder with a pre and
post processing hook (`PrePostPipelineEncoder`).

## Evaluation repositories

For repository retrieval we wrote 50 labelled questions on each of two repositories (30 dev, 20 holdout). Gold is one or more
`file:start-end` ranges; a retrieved unit counts as a hit when it is in the same file and overlaps a gold range. Gold ranges
were resolved mechanically to chunker units.

| | Node-RED | BrowserOS |
| --- | --- | --- |
| What it is | Flow-based visual programming tool for wiring devices, APIs and services | Open-source browser with a built-in AI agent |
| Language of the indexed code | JavaScript | TypeScript and TSX (agent application), some Python (build tooling) |
| Indexed | 467 files, 5,787 units (median 1,200 chars; 31% in test files) | 1,592 files, 11,483 units (median 445 chars; 34% in test files) |
| Commit | `cd05a9a38b` | `0152e0a829` |
| Question style | Short phrases, e.g. "where is the install_not_allowed error code raised" | Full questions, e.g. "How does the agent decide when to retry a failed tool call?" |
| Question types | behavioural (28), exact name (9), setting (6), registration (7) | behavioural (29), exact (17), which-test-covers (4) |
| File | `eval/node-red/queries.json` | `eval/browseros/queries.json` |

The screening benchmark is CoIR AppsRetrieval: 3,765 test queries over 8,765 Python programs, one relevant program per
query.

## Results

Final pipeline (Gemma top-50 re-scored by jina-code, Gemma and BM25), measured; the experiments that led to it are in
`docs/experiments/`.

| | MRR@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- |
| Node-RED, 50 questions | 0.908 | 0.88 | 0.96 |
| BrowserOS, 50 questions | 0.777 | 0.66 | 1.00 |

The blend weights used for each repository were chosen on the other repository. Fifty questions resolve only large
differences, so these are evidence of direction, not a final score.

| AppsRetrieval test, 3,765 queries | NDCG@10 | MRR@10 | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- |
| Same blend, scored by `scripts/apps_two_stage.py` | 88.61 | 85.92 | 79.52 | 96.89 |

The blend weights come from the repositories, not from this split. jina-code was trained on AppsRetrieval's training split,
so part of this score is in-domain.

**CPU query time**, warm, one question at a time, GPU hidden, models already loaded: 224 to 321 ms median and 266 to 364 ms
at the 95th percentile end to end, depending on thread count and repository (Gemma and jina-code encoding dominate; search and
blend take a few milliseconds). Loading the models takes 7 to 15 seconds once at startup.

## Layout

```text
main.py               Reflex demo UI
src/coderet/
  config.py           model registry (jina-code, embeddinggemma) and run fingerprints
  chunking/           tree-sitter chunker and file discovery
  embed/              PyTorch fp32 backend and an output sanity check
  index/              vector cache, FAISS index, BM25 and rank fusion, Searcher, FusionSearcher
  demo/               clone and index a repository, search engine for the UI
  eval/               hit rule and metrics for labelled questions
  mteb_adapters/      PrePostPipelineEncoder for the official benchmark
scripts/              one script per command above
eval/                 labelled questions; the git-ignored BrowserOS checkout
docs/                 architecture, experiments, handoff, the two hackathon PDFs
```

## Caveats

- jina-code saw AppsRetrieval's training split: APPS-train splits cannot validate it, and its benchmark score is partly in-domain.
- The search runs two models on every query and keeps both in memory (about 3 GB of weights).
- A small GPU can run out of memory while indexing; index on the CPU instead with `--device cpu`.
- Indexing a new repository from scratch on the CPU takes about ten minutes per thousand units; vectors are cached by the exact
  embedded text, so repeating a repository is free.
