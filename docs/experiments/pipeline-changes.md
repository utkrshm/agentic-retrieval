# What changed in the pipeline, in plain words

**First working version:** cut each file into function-sized pieces with tree-sitter, turn each piece into a vector with
jina-code, store the vectors in an exact FAISS index, and answer a question by turning it into a vector and returning
the closest pieces.

**Indexing now**
1. The chunker also reads TypeScript and TSX, and it skips type-declaration files (`.d.ts`) and generated code, which
   only added noise (`03-chunking.md`).
2. Tiny statement fragments under 60 non-blank characters (`var x;`, `})();`) are glued onto the neighbouring piece when
   they sit right next to it, so they stop appearing as results on their own.
3. Every piece is embedded twice, with embeddinggemma-300m and with jina-code, into two exact FAISS indexes over
   identical pieces. Vectors are cached by the exact text that was embedded, so unchanged code is never re-embedded.
   Documents are always embedded in fp32 (on the GPU offline, or on the CPU when a new repository is opened live).

**Searching now**
1. Gemma encodes the question.
2. Test files are left out unless the question mentions tests; the exclusion happens before the top-k so they cannot take
   a slot.
3. If the question names an identifier or quotes a message that really exists in the code (a snake_case or camelCase word,
   a dotted name, a quoted string), a keyword search (BM25) also runs and is merged with the vector results by reciprocal
   rank fusion. Plain-language questions skip this and behave as before.
4. Gemma's best 50 pieces are re-scored: jina-code also reads the question and every candidate gets
   `(1 - g) * (0.7 * jina + 0.3 * gemma) + g * bm25` (per-question min-max scaling; g = 0.25 when step 3 ran, else 0).
5. The best 10 pieces are returned with file and line range.

**Unchanged on purpose:** the official benchmark path (a plain encoder with pre- and post-processing) has none of the
repository additions; the models, prompts and exact FAISS search are the same.

**Tried and left out:** signature header on split pieces (hurt), merging fragments under 150 characters (hurt
registration queries), a LightGBM or cross-encoder reranker (council advice in `09-decisions-and-incidents.md`),
int8 serving (works, but not needed at these latencies), an agent loop.
