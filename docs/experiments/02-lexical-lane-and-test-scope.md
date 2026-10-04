# Keyword lane and test-file scope

The gated BM25 lane (keyword search merged by reciprocal rank fusion when a query names an identifier or quotes a message) and the test-file scope, first on Node-RED where they were designed, then on BrowserOS as the fresh check.

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
