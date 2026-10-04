# Experiments

Every experiment run on `feat/v1`: what was expected, what happened, what it means. Numbers are copied from run output;
where a figure is a guess or a reviewing agent's estimate, the text says so. Reviewing agents are "the council" (Sol,
Astra, Opus).

Hardware for every measurement: one laptop with a 4-performance-core CPU (AVX2 and AVX-VNNI, no AVX-512 or AMX), a 4 GB GPU used only for offline encoding, and 15 GB of RAM; Python 3.11, fp32 throughout.

Reading the repository results: Node-RED (JavaScript, 50 queries) is where the keyword lane and test scope were designed,
so it is development data. BrowserOS (TypeScript, 50 questions written blind by a subagent that saw no results) was the
fresh check and was afterwards also used to choose settings. 50 queries resolve only differences of roughly 0.08 MRR or
more (council estimate); treat gaps under about 0.05 as noise.

## Map

| File | Experiments |
|---|---|
| [pipeline-changes.md](pipeline-changes.md) | The pipeline before and after, in plain words |
| [01-foundations.md](01-foundations.md) | E1 official run, E2 int8, E3 failure drill, E5 first baseline, E6 label audit |
| [02-lexical-lane-and-test-scope.md](02-lexical-lane-and-test-scope.md) | E8 keyword lane and test scope, E11 BrowserOS check |
| [03-chunking.md](03-chunking.md) | E4 chunker, E9 `.d.ts` regression, E10 fragment merging and signature header |
| [04-model-choice.md](04-model-choice.md) | E7 model comparison, E12 Gemma against jina-code on repositories |
| [05-two-stage-gemma-jina.md](05-two-stage-gemma-jina.md) | E12b two-stage (N=10 against N=50), E12d why it differs by repository |
| [06-learned-fusion.md](06-learned-fusion.md) | E12c learned score fusion against rank fusion |
| [07-benchmark-apps.md](07-benchmark-apps.md) | Official runs and E12e: every variant scored on AppsRetrieval |
| [08-latency-and-cost.md](08-latency-and-cost.md) | CPU query latency and CPU indexing cost |
| [09-decisions-and-incidents.md](09-decisions-and-incidents.md) | Incidents, council advice on rerankers, open decisions |

## The three systems side by side

Single jina-code, single Gemma, and the combined search (Gemma retrieves the top 50, jina-code, Gemma and BM25 scores are
blended). The two single-model rows use the keyword lane and test scope on the repositories and no lane on the benchmark.

| Metric | jina-code only | Gemma only | Gemma + jina-code + BM25 |
|---|---|---|---|
| **AppsRetrieval test** (own scoring) NDCG@10 | 83.86 | 84.27 | **88.61** |
| MRR@10 | 80.80 | 80.82 | **85.92** |
| Recall@1 | 73.84 | 72.99 | **79.52** |
| Recall@5 | 89.61 | 91.10 | **94.18** |
| Recall@10 | 93.33 | 94.90 | **96.89** |
| Recall@100 | 98.57 | 99.47 | 99.42 |
| **Node-RED** MRR@10 | 0.848 | 0.801 | **0.908** |
| Recall@1 | 0.78 | 0.72 | **0.88** |
| Recall@5 | 0.94 | 0.90 | **0.96** |
| Recall@10 | 0.96 | **1.00** | 0.96 |
| gold recall@10 | 0.943 | **0.983** | 0.943 |
| **BrowserOS** MRR@10 | 0.734 | 0.717 | **0.777** |
| Recall@1 | 0.62 | 0.56 | **0.66** |
| Recall@5 | 0.86 | **0.96** | **0.96** |
| Recall@10 | 0.96 | **1.00** | **1.00** |
| gold recall@10 | 0.930 | **0.980** | **0.980** |
| **CPU query latency**, end to end p50 / p95 (short questions) | 85 to 96 / 101 to 115 ms | 47 to 71 / 70 to 89 ms | 224 to 321 / 266 to 364 ms |
| Models in memory | one (about 2 GB) | one (about 1.2 GB) | both (about 3 GB) |

Notes: the combined row's repository numbers use weights chosen on the other repository; the benchmark row uses the same
weights, fixed from the repositories (jina-code was trained on that benchmark's training split, Gemma's training data
is not itemised). The official submission file is still the plain jina-code run (83.83 NDCG@10). Differences between rows
on a single repository are at the edge of what 50 queries resolve; the benchmark rows rest on 3,765 queries.
