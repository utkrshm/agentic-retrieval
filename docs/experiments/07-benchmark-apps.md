# Benchmark: CoIR AppsRetrieval (test split)

3,765 test queries over 8,765 Python programs, one relevant program per query, scored with NDCG@10, MRR@10 and Recall@k.
Hardware for every measurement: one laptop with a 4-performance-core CPU (AVX2 and AVX-VNNI, no AVX-512 or AMX), a 4 GB GPU used only for offline encoding, and 15 GB of RAM; Python 3.11, fp32 throughout.

## Official runs (E1, E7)

These went through `PrePostPipelineEncoder` and `mteb.evaluate`, the format the organisers asked for.

| Model | NDCG@10 | MRR@10 | Recall@1 | Recall@10 | Recall@100 |
|---|---|---|---|---|---|
| jina-code-0.5b (the current submission file, `outputs/appsretrieval_results.json`) | 83.83 | 80.79 | 73.81 | 93.23 | 98.51 |
| embeddinggemma-300m | 84.28 | 80.81 | 72.96 | 94.95 | 99.47 |
| Qwen3-Embedding-0.6B | 73.49 | 68.89 | 59.39 | 87.89 | 97.77 |

jina-code was trained on AppsRetrieval's training split, so its score is partly in-domain. Gemma's training data is not
itemised. Single runs, no confidence intervals.

## E12e. Scoring the two-model variants on the benchmark data

- **What:** `scripts/apps_two_stage.py` embeds the 8,765 programs and 3,765 queries once per model (GPU, offline, cached
  under `outputs/apps-vectors/`) and scores every system with the benchmark's own metrics (one relevant program per
  query, so NDCG@10 is 1/log2(rank+1) when the rank is 10 or better). It reads only text, ids and relevance judgements
  from the task, never the dataset's metadata fields.
- **Sanity check:** the script reproduces the official runs, jina-code 83.86 against 83.83 and Gemma 84.27 against 84.28
  NDCG@10 (differences under 0.04, from batching).
- **Expected:** a blend of two models helps a little; the BM25 lane does not matter here (BM25 alone scores 4.8 on
  this benchmark in the literature).
- **Fixed in advance:** the systems and the blend weights (jina share 0.7, BM25 share 0.25, taken from the repository
  experiments in `06-learned-fusion.md`), nothing tuned on this split.

| System | NDCG@10 | MRR@10 | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@100 | queries better / worse than jina |
|---|---|---|---|---|---|---|---|---|
| jina-code alone | 83.86 | 80.80 | 73.84 | 89.61 | 93.33 | 95.64 | 98.57 | reference |
| embeddinggemma alone | 84.27 | 80.82 | 72.99 | 91.10 | 94.90 | 97.53 | 99.47 | 718 / 676 |
| jina-code + gated BM25 lane | 72.82 | 66.86 | 55.80 | 83.00 | 91.66 | 95.09 | 98.41 | 16 / 869 |
| Gemma + gated BM25 lane | 73.26 | 66.86 | 55.09 | 84.25 | 93.47 | 97.05 | 99.42 | 646 / 1322 |
| Two-stage: Gemma top-10, jina-code re-sort | 86.10 | 83.23 | 76.44 | 92.38 | 94.90 | 97.53 | 99.47 | 611 / 150 |
| Two-stage with the BM25 lane in stage one | 85.42 | 82.77 | 76.23 | 91.53 | 93.47 | 97.05 | 99.42 | 616 / 198 |
| Fusion N=10 (jina, Gemma, BM25 blend) | 86.77 | 84.55 | 78.73 | 92.08 | 93.47 | 97.05 | 99.42 | 754 / 316 |
| **Fusion N=50 (jina, Gemma, BM25 blend)** | 88.61 | 85.92 | 79.52 | 94.18 | 96.89 | 98.41 | 99.42 | 767 / 223 |
| Plain-encoder ensemble: 0.7 jina cosine + 0.3 Gemma cosine over the whole corpus | 87.35 | 84.77 | 78.78 | 92.56 | 95.30 | 97.24 | 99.36 | 682 / 47 |

The gate for the BM25 lane fires on 1,090 of 3,765 queries (29.0%).

- **Happened:** every two-model system beats both single models. Fusion N=50 gains +4.75 NDCG@10 and +5.68 Recall@1 over
  jina-code and +4.34 NDCG@10 over Gemma. The plain-encoder ensemble (a weighted sum of the two cosines, computed over the
  whole corpus) gains +3.49 NDCG@10 over jina-code, with 682 queries better and 47 worse.
- **BM25 hurts on this benchmark:** adding the gated lane to a single model drops NDCG@10 by about 11 points (jina-code
  83.86 to 72.82, Gemma 84.27 to 73.26), because problem statements contain code-like words that mislead it. The lane is
  therefore a repository feature only and stays out of the official path.
- **What can be submitted officially:** the official format accepts a plain encoder with pre- and post-processing, so
  only the concatenation ensemble fits it: the dot product of `[0.7 * jina ; 0.3 * gemma]` vectors equals
  `0.7 * jina cosine + 0.3 * gemma cosine`. It has been scored only by the script above, not run through `mteb.evaluate`,
  and the official artifact is still the jina-code file.
- **Caveats:** jina-code saw the training split of this benchmark, so part of any blend with it inherits that
  advantage; the fusion's weights were chosen on two repositories, not here; single runs; the test split must not be
  used to tune these weights further.
