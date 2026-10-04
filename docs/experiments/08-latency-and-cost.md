# CPU latency and indexing cost

All query-time numbers are warm, one query at a time, with the GPU hidden (`CUDA_VISIBLE_DEVICES=""`), models already
loaded, caches off, measured end to end per query (stage medians are for orientation only and are never added).
Hardware for every measurement: one laptop with a 4-performance-core CPU (AVX2 and AVX-VNNI, no AVX-512 or AMX), a 4 GB GPU used only for offline encoding, and 15 GB of RAM; Python 3.11, fp32 throughout.

## The settled search (Gemma top-50, jina-code and BM25 blend), fp32 on CPU

`scripts/eval_fusion.py` runs the production `FusionSearcher` on the labelled queries.

| Repository (units) | Threads | End to end p50 | p95 | Max | First query | Stage medians (Gemma encode, search, jina encode, blend) |
|---|---|---|---|---|---|---|
| Node-RED (5,787), phrases of about 10 words | 12 | 279 ms | 318 ms | 333 ms | 296 ms | 127, 2, 145, 1 ms |
| BrowserOS (11,483), questions of about 17 words | 12 | 321 ms | 364 ms | 447 ms | 301 ms | 129, 5, 164, 1 ms |
| BrowserOS | 4 | 224 ms | 266 ms | 328 ms | 276 ms | 74, 3, 139, 0 ms |

Model loading (7 to 15 s) is startup and not included. Retrieval metrics from the same runs reproduce the experiment
numbers (Node-RED MRR@10 0.908, BrowserOS 0.777). Timings varied between sessions (an earlier measurement of the
two-stage re-sort alone gave 126 to 169 ms), because other processes shared the CPU; treat the figures as a range, 224
to 321 ms p50, not a constant.

## Components measured earlier

| Item | Short query | Long query (about 400 tokens) |
|---|---|---|
| jina-code fp32, PyTorch | 82 ms (p95 84) | 880 ms (p95 886) |
| embeddinggemma fp32, PyTorch | 65 ms (p95 81) | 338 ms (p95 371) |
| jina-code int8, OpenVINO (4 threads, other text; optional path, not used) | 24 ms | 196 ms |

| Path (50 BrowserOS questions of about 20 tokens) | jina p50 / p95 | Gemma p50 / p95 | Gemma then jina re-sort, N=5, p50 / p95 |
|---|---|---|---|
| 12 threads, fp32 | 96 / 115 ms | 71 / 89 ms | 169 / 198 ms |
| 4 threads, fp32 | 85 / 101 ms | 47 / 70 ms | 126 / 144 ms |
| 4 threads, jina int8 | 39 / 49 ms | 43 / 53 ms | 82 / 104 ms |

The second encoder roughly doubles the cost of the first; FAISS search (1.7 to 5 ms) and the blend (under 2 ms) are
negligible. Both models and both indexes stay in memory (about 3 GB of weights).

## Indexing a repository on the CPU

Measured on 96 random BrowserOS units (median 516 characters, mean 768), fp32, batch 8, 12 threads:

| Model | Throughput | Time for 1,000 units |
|---|---|---|
| embeddinggemma-300m | 5.1 units/s (196 ms per unit) | about 3.3 min |
| jina-code-0.5b | 2.3 units/s (440 ms per unit) | about 7.2 min |
| Both (needed for the settled search) | about 1.6 units/s | about 10.5 min |

So indexing a new repository from scratch on the CPU takes about ten minutes per thousand units (Node-RED, 5.8k units,
about an hour; BrowserOS, 11.5k, about two hours). Vectors are cached on disk by the exact embedded text, so a repository
that was indexed before costs nothing to index again. On the GPU the same indexing took 144 s for Node-RED (jina-code) and 257 s (Gemma).
