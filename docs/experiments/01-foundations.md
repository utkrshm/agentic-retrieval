# Foundations: benchmark, int8, failure drill, first baseline

The first working pipeline: the official benchmark run, the optional int8 query path, its failure recovery, the first repository baseline and the label audit.

Hardware for every measurement: one laptop with a 4-performance-core CPU (AVX2 and AVX-VNNI, no AVX-512 or AMX), a 4 GB GPU used only for offline encoding, and 15 GB of RAM; Python 3.11, fp32 throughout.

### E1. Official benchmark run (AppsRetrieval, test split)

- **Setup:** `PrePostPipelineEncoder` plus `mteb.evaluate`, jina-code-0.5b, PyTorch fp32, 1,024-token limit.
  3,765 queries against 8,765 Python programs (one relevant program per query).
- **Expected:** about 83.9 NDCG@10 (an earlier bf16 run gave 83.87; published range 81.6 to 84.2).
- **Happened:** NDCG@10 83.83, MRR@10 80.79, Recall@1 73.81, Recall@10 93.23, Recall@100 98.51.
  fp32 against bf16 changes nothing measurable (under 0.05).
- **Conclusion:** the pipeline reproduces the published level. jina-code was trained on AppsRetrieval train, so
  the score says as much about the model as the pipeline.

### E2. Query-side int8 with OpenVINO

- **Setup:** export the model to OpenVINO, compress weights to int8 (NNCF), compare with PyTorch fp32 on real
  code and on 500 APPS-train queries searched against fp32 document vectors.
- **Expected:** 2 to 3 times faster queries with no measurable quality loss.
- **Happened:** warm batch-1 latency 80 to 24 ms (short query) and 377 to 196 ms (long code query); minimum
  cosine to fp32 0.9987; NDCG@10 96.85 against 96.96 (paired change -0.10, 95% interval -0.42 to +0.17).
  The sentence-transformers OpenVINO wrapper returned NaN for every input, so a raw-runtime backend was written.
- **Conclusion:** int8 is a safe optional query speed-up. Documents stay fp32; the two precisions are never mixed
  in one index.

### E3. Failure drill for the encoder fallback chain

- **Setup:** break the real exported models seven ways (truncated weights, missing graph, wrong reference, no
  OpenVINO, crash mid-run, NaN mid-run) in separate subprocesses.
- **Expected:** every scenario ends on a working backend with a vector within cosine 0.99 of the healthy one.
- **Happened:** all seven recovered. The first version of the drill leaked memory and the OOM killer ended it;
  it was rewritten to use one subprocess per scenario. A dangling relative symlink meant the middle fallback was
  not actually being exercised until it was fixed.
- **Conclusion:** the int8, OpenVINO fp32, PyTorch fp32 chain recovers from each tested fault.

### E5. Dense-only retrieval on Node-RED (baseline)

- **Setup:** jina-code fp32 for documents and queries, exact FAISS search, 50 hand-labelled queries (30 dev, 20
  holdout; behavioural, exact name, setting, registration).
- **Expected:** behavioural questions work; exact and setting questions are weaker (the council predicted this
  from how embeddings spread score over a file).
- **Happened (first labels):** dev Recall@1 0.667, Recall@10 0.90, MRR@10 0.746; holdout Recall@1 0.55,
  Recall@10 1.0, MRR@10 0.705. Holdout behavioural MRR 0.85, exact-name 0.40. `install_not_allowed` first hit
  at rank 7. Spec files and one-line fragments filled the top 5.
- **Conclusion:** the weak spot is exact literals and error codes, and the top 5 is noisy.

### E6. Label audit

- **Expected:** a few "misses" might be correct answers we had not labelled.
- **Happened:** h05 ("read a nested property from a message using a dotted path") also matches
  `getMessageProperty`, and d03 (remove context for nodes no longer present) also matches the context manager's
  `clean()`. Both were added to gold, recorded in a `label_fixes` field because the change was made after
  results were seen. Baselines moved slightly (dense MRR 0.746 to 0.771 on dev).
- **Conclusion:** use the corrected labels for every comparison after this point; earlier numbers are not
  directly comparable.
