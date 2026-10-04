# Two-stage search: Gemma retrieves, jina-code re-scores

Letting one model fetch the candidates and the other order them, how the pool size N matters, and why the effect differs between Node-RED and BrowserOS.

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

### E12d. Why the two-stage helps on BrowserOS and hurts on Node-RED (analysis of E12b and E12c)

- **Question (from the user):** is it the nature of the queries (BrowserOS: full questions; Node-RED: phrases) or
  of the repositories?
- **Method:** split the per-query results of jina alone, Gemma alone, the plain two-stage (jina re-sort, N=10)
  and the fusion (N=50, beta 0.7, gamma 0.25) by query category, by whether the BM25 gate fires, and by length.
- **Happened (MRR@10):**

  | Subset | n | jina | Gemma | two-stage N=10 | fusion N=50 |
  |---|---|---|---|---|---|
  | Node-RED, all | 50 | 0.848 | 0.801 | 0.815 | 0.908 |
  | Node-RED, gate fires | 14 | 0.911 | 0.795 | 0.726 | 0.946 |
  | Node-RED, gate silent | 36 | 0.824 | 0.803 | 0.850 | 0.894 |
  | Node-RED, exact | 9 | 0.944 | 0.861 | 0.769 | 1.000 |
  | Node-RED, config | 6 | 0.764 | 0.729 | 0.597 | 0.875 |
  | Node-RED, behavioural | 28 | 0.815 | 0.789 | 0.848 | 0.863 |
  | BrowserOS, all | 50 | 0.734 | 0.717 | 0.783 | 0.777 |
  | BrowserOS, gate fires | 20 | 0.867 | 0.820 | 0.889 | 0.865 |
  | BrowserOS, gate silent | 30 | 0.646 | 0.649 | 0.712 | 0.718 |
  | BrowserOS, exact | 17 | 0.873 | 0.818 | 0.869 | 0.841 |
  | BrowserOS, behavioural | 29 | 0.634 | 0.636 | 0.720 | 0.708 |

  Query and repository properties: Node-RED queries are phrases (mean 10.2 words, none end with "?"), BrowserOS
  queries are full questions (mean 17.3 words, all end with "?"). Median unit size is 1,200 characters in Node-RED
  (56% of units are module statement groups) and 445 in BrowserOS (28%). Test files are 31% and 34% of units.
- **Finding 1, where Node-RED loses:** on the 14 queries where the BM25 gate fires, the plain two-stage falls
  from 0.911 to 0.726, because re-sorting by jina cosine alone discards the lane's ordering. jina plus the lane was
  already near ceiling there (exact 0.944), so the two-stage could only lose. On BrowserOS the gated queries were
  less saturated (0.867) and the two-stage gains 0.022 there.
- **Finding 2, where both gain:** on plain-language queries (gate silent) the two-stage improves both repos:
  Node-RED +0.026, BrowserOS +0.066, with the larger gain on BrowserOS where jina is weakest on behavioural
  questions (0.634 against 0.815) and on long questions (0.701 to 0.766).
- **Finding 3, why fusion works on both:** it keeps the lane's ordering on gated queries (Node-RED 0.946, above
  jina's 0.911; BrowserOS 0.865, level with 0.867) and keeps the behavioural gain on gate-silent queries
  (+0.07 on both repos).
- **Conclusion:** the difference is mostly the query mix and how saturated each type already was. The effect of the
  second stage depends on query type in the same direction in both repositories. Query style (question or phrase)
  and repository are confounded and cannot be separated from this data: the subgroups hold 4 to 36 queries, so
  differences under about 0.1 are not reliable. A clean test would rewrite one repository's queries in the other
  style and re-run (not done).
