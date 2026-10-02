# Experiments

Every run is identified by its config fingerprint (`coderet.config`). Dev = grouped split of APPS train; test is touched only by the frozen submission run.

| Date | Fingerprint | Model | Split | NDCG@10 | MRR@10 | R@100 | Query p50/p95 ms (CPU) | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-02 | 71d2b37877db | jina-code-0.5b | dev | 97.19 | 96.32 | 100.00 | 4437 / 13478 | bakeoff.py |
| 2026-10-02 | cbb9ccd44151 | qwen3-0.6b | dev | 82.86 | 80.18 | 97.40 | 4447 / 14206 | bakeoff.py |

## Notes: baseline bakeoff (2026-10-02)

Hardware: Intel i7-13650HX (20 logical cores), torch 2.11 CPU path, batch 1, warm. Docs and dev queries encoded on an RTX 4050 (offline); latency is CPU query encoding plus exact scan against the cached doc matrix.

- **embeddinggemma-300m** not run: gated on Hugging Face (accept the license, then `hf auth login`).
- **jina-code-0.5b** wins on dev, but its training data includes AppsRetrieval train, which is where dev comes from, so 97.2 overstates it. Published test-split scores are 81.6 to 84.2.
- **qwen3-0.6b** has no APPS exposure: 82.9 on dev, Recall@100 97.4.
- **CPU latency is the bottleneck.** APPS queries are long (jina tokens: p50 372, max 916). Thread sweep on 12 dev queries (jina, p50 / p95 ms): 2 threads 11945 / 25220, 4 threads 7566 / 15597, 6 threads 6231 / 13166, 8 threads 6379 / 13329, 12 threads 6569 / 13885, 20 threads 5495 / 11166. The torch default (14 threads) is kept. The exact scan takes about 1 ms. The next lever is PLAN-v3 phase 1 export work (ONNX, int8) and then distillation into a smaller query student.

## Fallback submission (2026-10-02)

| Fingerprint | Model | Split | NDCG@10 | MRR@10 | R@10 | R@100 | File |
|---|---|---|---|---|---|---|---|
| 71d2b37877db | jina-code-0.5b | **test** (MTEB 2.21.0) | 83.87 | 80.83 | 93.28 | 98.51 | `results/appsretrieval_results.json` |

Pure-encoder path through `scripts/run_mteb.py` (docs and queries encoded on GPU, 293 s). The score is consistent with jina's published 81.6 to 84.2.
