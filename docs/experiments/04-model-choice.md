# Model choice: Gemma against jina-code

Which embedding model to use, on the benchmark and on real repositories.

### E7. Model comparison on AppsRetrieval test (same script, PyTorch fp32)

- **Expected:** embeddinggemma-300m near its published 84.4; Qwen3-Embedding-0.6B near 75; jina-code 83.8.
- **Happened:**

  | Model | NDCG@10 | MRR@10 | Recall@1 | Recall@10 | Recall@100 |
  |---|---|---|---|---|---|
  | embeddinggemma-300m | 84.28 | 80.81 | 72.96 | 94.95 | 99.47 |
  | jina-code-0.5b | 83.83 | 80.79 | 73.81 | 93.23 | 98.51 |
  | Qwen3-Embedding-0.6B | 73.49 | 68.89 | 59.39 | 87.89 | 97.77 |

  Qwen3 needed micro-batch 1 after a CUDA out-of-memory error at batch 4.
- **Conclusion:** Gemma and jina are tied on this benchmark (Gemma +0.45 NDCG@10, jina +0.85 Recall@1;
  single runs, no interval). Qwen3-0.6B is clearly behind. An older bakeoff on an APPS-train dev split put jina at
  96.96, which is inflated because jina trained on it; it should not be used to compare models.

### E12. embeddinggemma-300m against jina-code on the repository pipeline

- **Setup:** the same chunker, queries, test scope and gated lexical lane; only the embedding model changes.
  Gemma runs on plain PyTorch fp32 with its own prompts (`task: code retrieval | query: `, `title: none | text: `),
  768 dimensions, 2,048-token limit; indexes built on the GPU. Node-RED index has no merge (5,787 units, same
  as the jina run it is compared with); BrowserOS has merge-60 (11,483 units, compared with the jina merge-60
  index).
- **Expected:** equal or better than jina, since the two tie on AppsRetrieval (E7).
- **Happened:**

  | Repo / config | Model | Recall@1 | Recall@5 | Recall@10 | MRR@10 | gold recall@10 |
  |---|---|---|---|---|---|---|
  | Node-RED dense | jina | 0.66 | 0.88 | 0.94 | 0.755 | 0.887 |
  | Node-RED dense | Gemma | 0.62 | 0.80 | 0.94 | 0.700 | 0.873 |
  | Node-RED lane + tests | jina | 0.78 | 0.94 | 0.96 | 0.848 | 0.943 |
  | Node-RED lane + tests | Gemma | 0.72 | 0.90 | 1.00 | 0.801 | 0.983 |
  | BrowserOS dense | jina | 0.60 | 0.86 | 0.88 | 0.707 | 0.840 |
  | BrowserOS dense | Gemma | 0.50 | 0.84 | 0.94 | 0.649 | 0.890 |
  | BrowserOS lane + tests | jina | 0.62 | 0.86 | 0.96 | 0.734 | 0.930 |
  | BrowserOS lane + tests | Gemma | 0.56 | 0.96 | 1.00 | 0.717 | 0.980 |

  With the lexical lane, Gemma finds an answer in the top 10 for every query on both repos (Recall@10 1.00)
  and has higher gold recall@10, but puts the first correct answer at rank 1 less often (Recall@1 lower by
  0.06 on both repos) and has lower MRR@10 (0.801 against 0.848, 0.717 against 0.734). On BrowserOS Recall@5
  is higher (0.96 against 0.86). Per query, Gemma dense was better on 9 and worse on 16 Node-RED queries, and
  better on 9 and worse on 13 BrowserOS queries, compared with jina dense.
- **Conclusion:** the AppsRetrieval tie does not carry over to the repositories on top-1 precision: jina orders
  the first answer better, Gemma retrieves a slightly better candidate pool. The gaps are at the edge of what 50
  queries resolve (a Recall@10 difference of 0.04 is two queries). Gemma is smaller (308M against 0.5B) and was
  not timed in the retrieval runs; a separate CPU timing (i5-12500H, PyTorch fp32, batch 1, warm, 12 threads, 15
  runs) gave Gemma 65 ms p50 (p95 81) on a short query and 338 ms (p95 371) on a 398-token query, against jina
  fp32 82 ms (p95 84) and 880 ms (p95 886) on a 399-token query. jina int8 OpenVINO, measured earlier with 4
  threads on different text, was 24 ms short and 196 ms long, so Gemma fp32 is faster than jina fp32 but not
  faster than jina int8 (no Gemma int8 export exists).
