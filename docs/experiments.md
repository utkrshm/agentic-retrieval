# Experiments

Every experiment run so far on `feat/v1`: what we expected, what happened, and what we concluded. Section 2
explains in plain words how the pipeline changed because of them. Numbers are copied from run output; where a
number is a guess or an estimate by a reviewing agent, it says so.

Hardware for everything: i5-12500H (4 performance cores, AVX2 and AVX-VNNI, no AVX-512/AMX), RTX 3050 4 GB,
15 GB RAM, Python 3.11. Reviewing agents are "the council" (Sol, Astra, Opus); their effect sizes are opinions
unless stated.

Reading the repository results: Node-RED (JavaScript, 50 queries) is where the lexical lane and test scope were
designed, so it is development data. BrowserOS (TypeScript, 50 questions written blind by a subagent that saw no
results) is the fresh check. 50 queries only resolve large differences (about 0.08 MRR or more, council estimate);
treat gaps under about 0.05 as noise.

## 1. Experiment log

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

### E4. Tree-sitter chunker on a real repository

- **Setup:** chunk Node-RED (467 JavaScript files).
- **Expected:** units that cover the code, stay under 2,500 characters, and have exact line spans.
- **Happened:** 5,787 units, median 1,200 characters, none above the limit, 97.8% of non-blank lines covered,
  no parse errors, no span violations, 2.9 s. tree-sitter 0.26.0 returned wrong node positions and crashed on
  real JavaScript; 0.24 and 0.25 are fine, so the dependency is pinned below 0.26.
- **Conclusion:** the chunker is usable. It was later extended to TypeScript (E9).

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

### E8. Gated lexical lane (BM25 plus rank fusion) and test-file scope on Node-RED

- **Setup:** identifier-aware BM25 (whole identifiers plus snake_case and camelCase parts, a quoted phrase as one
  rare term). It only runs when the query holds a quoted literal or a snake_case, camelCase or dotted identifier
  that actually occurs in the code; dense and BM25 top-100 are fused with reciprocal rank fusion (k=60). Test
  files (test dirs, `*.test.*`, `*.spec.*`, `*_spec.js`, `test_*.py`) are filtered out before the top-k unless
  the query mentions tests.
- **Expected (council estimates):** lexical lane +0.02 to +0.06 overall MRR@10, mostly on exact and setting
  queries, no change on behavioural; test scope +0.02 to +0.05 (Opus guess), 0 to +0.03 (Sol).
- **Happened (corrected labels, `.d.ts` files excluded):**

  | Config | MRR@10 all | exact | setting | behavioural | registration |
  |---|---|---|---|---|---|
  | dense | 0.755 | 0.677 | 0.417 | 0.809 | 0.929 |
  | + test scope | 0.761 | 0.689 | 0.417 | 0.815 | 0.929 |
  | + lexical lane | 0.845 | 0.944 | 0.764 | 0.809 | 0.929 |
  | + both | 0.848 | 0.944 | 0.764 | 0.815 | 0.929 |

  The lane improved 8 queries and made none worse. h13, h14, h15 moved from ranks 7, 4, 5 to 1, 2, 1.
- **Conclusion:** a much larger effect than predicted on Node-RED, but the lane was designed after seeing these
  failures, so Node-RED cannot confirm it (see E13).

### E9. TypeScript support exposed a distractor problem

- **Expected:** adding the TypeScript grammar would only add TypeScript units (needed for BrowserOS).
- **Happened:** Node-RED gained 51 `.d.ts` files (vendored Node typings, 1,045 units). They took the top ranks
  for d07 ("kill a child process") and d08 ("add HTTP headers"), pushing the real code out of the top 10: dense
  Recall@10 fell from 0.90 to 0.833 on dev.
- **Fix:** `.d.ts`, `.d.mts` and `.d.cts` files are not indexed (they declare types and contain no
  implementation); directories named `generated` are skipped too. Node-RED returned to 5,787 units.
- **Conclusion:** vendored declaration files are pure noise for code search; index only files with
  implementations.

### E10. Chunk-repair variants on Node-RED

- **Expected (Sol, Astra, Opus):** merging tiny fragments and adding the parent signature to split pieces would
  help by 0 to +0.04 MRR, with a warning that merging must not swallow one-line registration units.
- **Setup:** `min_chars` merges statement groups under N non-blank characters into a contiguous neighbour;
  `signature_header` adds the enclosing signature to statement groups cut from a large definition.
  (These runs still included the `.d.ts` files, so absolute values are slightly lower than in E8.)
- **Happened (MRR@10, lexical lane plus test scope; registration category in brackets):**

  | Variant | Overall | Registration |
  |---|---|---|
  | no change | 0.838 | 0.929 |
  | merge under 60 chars | 0.842 | 0.929 |
  | merge under 150 chars | 0.807 | 0.607 |
  | signature header only | 0.796 | 0.655 |
  | merge 150 plus header | 0.773 | 0.488 |

  Without `.d.ts` files (final): 0.848 and 0.852 for none and merge-60.
- **Conclusion:** merge-60 is neutral to slightly positive (h17 moved from rank 3 to 2) and is now the default
  index setting. The 150-character merge swallows the one-line `registerType` units and hurts registration. The
  signature header lowered MRR in every combination (later pieces already carry path and qualified name) and
  stays off.

### E11. BrowserOS fresh-repository check

- **Setup:** BrowserOS commit `0152e0a829` (TypeScript and TSX agent code, 1,592 files, about 11,900 units),
  50 questions (30 dev, 20 holdout; behavioural, exact, test-seeking) written by a subagent from source only.
- **Expected:** the Node-RED gains transfer (lexical lane +0.02 to +0.06, test scope helps).
- **Happened (MRR@10):** dense 0.707; test scope 0.731 (7 queries better, none worse); lexical lane alone 0.720;
  both 0.734 (9 better, 1 worse). Recall@10 rises from 0.88 to 0.96, almost all from test scope (about 390
  test files). Test-seeking questions are unaffected (MRR 0.875). Exact MRR 0.804 to 0.873. Merge-60 changes
  nothing here. The one regression, h07, is a behavioural question where the gate fired on the dotted name
  `models.dev`.
- **Conclusion:** test scope is the bigger win on a repo with many tests; the lexical lane is a small positive
  here, not the +0.09 seen on Node-RED. The gate can misfire on prose that contains a dotted name.

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

### E12b. Two-stage: Gemma retrieves the top-N, jina re-scores them

- **Idea (from the user):** Gemma finds a better candidate pool and jina orders the top better, so let Gemma
  retrieve and jina choose. Both indexes hold the same units, so jina's document vectors for Gemma's candidates
  are already stored; the only extra query cost is a second query encoding.
- **Setup:** stage one is Gemma with test scope and the lexical lane; its top-N is re-ordered by jina cosine
  (or by the sum of both normalised cosines), then the rest of the top 10 follows. Also the reverse (jina
  retrieves, Gemma re-scores). `scripts/two_stage_repo.py`; N in 5, 10, 20 on Node-RED, 5 and 10 on BrowserOS.
  N was explored after seeing the data, so this is exploratory.
- **Expected:** at or above jina alone on MRR, with Gemma's Recall@10.
- **Happened (MRR@10 / Recall@1 / Recall@10):**

  | System | Node-RED | BrowserOS |
  |---|---|---|
  | jina alone | 0.848 / 0.78 / 0.96 | 0.734 / 0.62 / 0.96 |
  | Gemma alone | 0.801 / 0.72 / 1.00 | 0.717 / 0.56 / 1.00 |
  | Gemma top-5, jina re-scores | 0.831 / 0.78 / 1.00 | 0.772 / 0.68 / 1.00 |
  | Gemma top-10, jina re-scores | 0.815 / 0.72 / 1.00 | 0.783 / 0.68 / 1.00 |
  | Gemma top-20, jina re-scores | 0.783 / 0.68 / 0.96 | not run |
  | jina top-5/10, Gemma re-scores | 0.798 / 0.792 | 0.733 / 0.765 |

  Against jina alone, per query: Node-RED top-5 6 better, 5 worse; BrowserOS top-5 10 better, 4 worse, and
  top-10 11 better, 1 worse. Re-scoring by the sum of both cosines is never better than jina alone's MRR on
  Node-RED. Wider pools (N=20) get worse. Using Gemma as the second stage is worse than jina alone in most rows.
- **Conclusion:** it keeps Gemma's Recall@10 of 1.00 on both repos (jina alone 0.96) and recovers most of the
  top-1 gap, but the MRR effect is mixed: above jina alone on BrowserOS (+0.04 to +0.05) and below it on
  Node-RED (-0.02 to -0.03). The two repos disagree, so the gain is not established on 100 queries.
- **CPU-only cost** (`scripts/bench_two_stage.py`, GPU hidden, warm batch 1, 50 BrowserOS questions that are
  short, about 20 tokens each, end-to-end per query, search plus lexical lane included, models already loaded):

  | Threads / jina query encoder | jina alone p50 / p95 | Gemma alone p50 / p95 | two-stage p50 / p95 |
  |---|---|---|---|
  | 12, fp32 | 96 / 115 ms | 71 / 89 ms | 169 / 198 ms |
  | 4, fp32 | 85 / 101 ms | 47 / 70 ms | 126 / 144 ms |
  | 4, jina int8 OpenVINO | 39 / 49 ms | 43 / 53 ms | 82 / 104 ms |

  So the two-stage path costs about the sum of both encoders: roughly 2 to 3 times Gemma alone. It also needs
  both models in memory and both indexes (about 3 GB of weights). Long queries cost more (earlier fp32 timing:
  Gemma 338 ms, jina 880 ms at about 400 tokens). Startup (model loading) was 7 to 15 s and is not included.

### E13. Incidents worth remembering

- **Silent CPU fallback during indexing:** a first re-index hit a CUDA out-of-memory error on its first batch;
  the resilient encoder demoted itself to the CPU for good and the build crawled. Fixed with a GPU batch of 8
  and `expandable_segments`. Two GPU jobs must not run at once on this 4 GB card.
- **Empty cache falsy:** an empty `VectorCache` was falsy through `__len__`, so it never filled until the check
  became `is not None` (found by a test).
- **Council reviews:** rounds on the plan, on quality boosts, on improvement ideas and on a LightGBM reranker.
  Findings used: cut symbol boosts, call maps, agent loop and evidence bundles; add the gated lexical lane; no
  reranker yet (see section 3).

## 2. What changed in the pipeline, in plain words

**Before (the first working version):** cut each file into function-sized pieces with tree-sitter, turn each
piece into a vector with jina-code, store the vectors in an exact FAISS index, and answer a question by turning
it into a vector and returning the closest pieces.

**Now, indexing:**
1. The chunker also reads TypeScript and TSX, and it skips type-declaration files (`.d.ts`) and generated code,
   which only added noise.
2. Tiny statement fragments under 60 non-blank characters (`var x;`, `})();`) are glued onto the neighbouring
   piece when they sit right next to it, so they stop showing up as results on their own.
3. Vectors are cached by the exact text that was embedded, so a changed unit is re-embedded and an unchanged
   one never is. Documents are always embedded in fp32 on the GPU, offline.

**Now, searching:**
1. Test files are left out unless the question mentions tests ("which test covers ..."). The exclusion happens
   before the top-k so they cannot take a slot.
2. If the question contains something that looks like a code name or an exact message (a quoted string, a
   snake_case or camelCase word, a dotted name) and that text really exists in the code, a keyword search
   (BM25) also runs. The dense and keyword top-100 are merged by reciprocal rank fusion, so a unit that contains
   the exact text rises to the top. Plain-language questions skip this and behave exactly as before.
3. The query is still embedded on the CPU (fp32, or the optional int8 OpenVINO path with automatic fallback), so
   serving stays CPU-only.

**Unchanged on purpose:** the official benchmark path (a plain encoder with pre/post processing) has none of
the repository additions, and the model, prompts, and exact FAISS search are the same.

**Tried and left out:** signature header on split pieces (hurt), 150-character merge (hurt registration),
a different embedding model (see E7 and E12), a LightGBM reranker (council advice below).

## 3. Open decisions and advice on record

- **LightGBM reranker:** all three reviewers said not now. The runtime cost is small (about 2 to 10 ms, an
  estimate), but there is no trustworthy training data (jina saw the APPS-train labels, 50 repo queries are
  exposed, and 50 queries only resolve large effects). Suggested next steps in order: a one-parameter blend of
  normalised dense and BM25 scores in place of RRF, then a small linear model on a few scale-free features,
  and LightGBM only with 200 or more labelled queries across at least two repositories.
- **Fresh queries:** every Node-RED query has been seen; BrowserOS is the only unseen set and is now also used
  for choices, so a third set is needed for any final claim.
- **Embedding model:** see E7 and E12.
