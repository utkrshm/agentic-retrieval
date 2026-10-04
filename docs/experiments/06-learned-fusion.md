# Reciprocal rank fusion against learned score fusion

Two fusion mechanisms appear in this project. The keyword lane merges the dense and BM25 rankings with reciprocal rank fusion (rank-based, no parameters tuned; E8). The second stage blends three scores (jina-code, Gemma, BM25) with two weights chosen on one repository and tested on the other (below). Opus's suggested experiment of replacing reciprocal rank fusion with a one-weight dense-plus-BM25 blend on a single model was not run: the three-signal blend inside the two-stage pipeline was run instead, because it answered the question that mattered (keep the keyword evidence while letting both models vote).

### E12c. Learned score fusion inside the two-stage pipeline (pre-registered before running)

- **Why:** E12b re-scores Gemma's top-N by jina cosine alone, which discards the BM25 lane's ordering inside the
  top-N (the lane is what fixed exact-literal queries). Fusing the signals should keep it.
- **Candidates:** Gemma's top-N from the full lane pipeline (test scope plus gated lexical lane), N in {10, 20, 50}.
  Exact scores are computed for every candidate: Gemma cosine, jina cosine (stored vectors), BM25.
- **Score:** per-query min-max over the candidates, `s = (1 - g) * (beta * jina + (1 - beta) * gemma) + g * bm25`,
  where g = gamma when the BM25 gate fires for the query and 0 otherwise.
- **Grid (fixed now):** beta in {0.5, 0.7, 1.0} and gamma in {0, 0.25, 0.5}: 9 settings, ties go to the first.
- **Protocol:** choose the setting by MRR@10 on one repo, evaluate once on the other, then the reverse, for each N.
- **Accept a setting for N** only if on each held-out repo MRR@10 is at least jina alone's (lane plus tests) and
  Recall@10 is not lower than jina alone's, and the chosen settings of the two directions differ by at most one
  grid step in each parameter. Otherwise reject. Report per-query better/worse counts against jina alone.
- **Expected:** at most a few queries change (Opus: -0.01 to +0.03 MRR, a guess).

- **Happened** (`scripts/fusion_repo.py`; held-out repo, setting chosen on the other repo; jina alone = lane plus
  tests, Node-RED MRR@10 0.848 / R@1 0.78 / R@10 0.96, BrowserOS 0.734 / 0.62 / 0.96):

  | Pool N | Fit on, tested on | beta, gamma | R@1 / R@10 / MRR@10 | better / worse than jina | Rule |
  |---|---|---|---|---|---|
  | 10 | Node-RED, BrowserOS | 0.7, 0.5 | 0.66 / 1.00 / 0.779 | 11 / 3 | pass |
  | 10 | BrowserOS, Node-RED | 0.7, 0.0 | 0.78 / 1.00 / 0.839 | 8 / 7 | fail (MRR below jina) |
  | 20 | Node-RED, BrowserOS | 0.7, 0.5 | 0.64 / 1.00 / 0.768 | 11 / 4 | pass |
  | 20 | BrowserOS, Node-RED | 0.7, 0.25 | 0.84 / 0.96 / 0.888 | 7 / 4 | pass |
  | 50 | Node-RED, BrowserOS | 0.7, 0.25 | 0.66 / 1.00 / 0.777 | 11 / 2 | pass |
  | 50 | BrowserOS, Node-RED | 0.7, 0.25 | 0.88 / 0.96 / 0.908 | 7 / 2 | pass |

  Stable settings across the two directions: N=20 and N=50 (N=10 not stable). The chosen setting at N=50 is the
  same in both directions (beta 0.7, gamma 0.25).
- **Conclusion:** the first configuration that beats jina alone on both repos with weights chosen elsewhere:
  at N=50, MRR@10 +0.060 on Node-RED (Recall@1 0.78 to 0.88) and +0.043 on BrowserOS (Recall@1 0.62 to 0.66,
  Recall@10 0.96 to 1.00). A wider pool helps only once the BM25 and Gemma scores stay in the final score; the plain
  jina re-sort of E12b made wider pools worse. Caveats: both repos are exposed, each setting was picked on 50
  queries from a 9-point grid, and a 0.04 to 0.06 MRR difference is at the edge of what 50 queries resolve. It
  needs both encoders at query time (E12b cost: about 82 to 169 ms for short queries on CPU) and breaks the
  one-encoder rule.
