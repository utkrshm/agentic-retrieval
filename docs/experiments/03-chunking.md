# Chunking

The tree-sitter chunker, what went wrong when TypeScript files were added, and the variants tried to repair split pieces and tiny fragments.

### E4. Tree-sitter chunker on a real repository

- **Setup:** chunk Node-RED (467 JavaScript files).
- **Expected:** units that cover the code, stay under 2,500 characters, and have exact line spans.
- **Happened:** 5,787 units, median 1,200 characters, none above the limit, 97.8% of non-blank lines covered,
  no parse errors, no span violations, 2.9 s. tree-sitter 0.26.0 returned wrong node positions and crashed on
  real JavaScript; 0.24 and 0.25 are fine, so the dependency is pinned below 0.26.
- **Conclusion:** the chunker is usable. It was later extended to TypeScript (E9).

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
