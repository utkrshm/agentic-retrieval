"""Search over a RepoIndex: test-file scope and gated lexical fusion on top of exact dense search.

``Searcher.search`` filters candidates before the top-k (never after), so excluded files cannot take a
slot. Test files are excluded unless the query asks for tests; lexical fusion only runs for queries
with a code anchor. Both are switchable so each can be ablated.
"""

from __future__ import annotations

import re

import faiss
import numpy as np

from coderet.index.lexical import BM25, rrf
from coderet.index.repo_index import Hit, RepoIndex

_TEST_DIRS = {"test", "tests", "__tests__", "spec", "specs", "e2e", "__mocks__", "__fixtures__", "testdata"}
_TEST_FILE = re.compile(r"(\.|_)(test|spec)\.[a-z]+$|^test_[^/]*\.py$|_test\.py$|^conftest\.py$", re.I)
_TEST_INTENT = re.compile(r"\b(tests?|specs?|testing|fixtures?|mocks?|stubs?|assert\w*|e2e|pytest|jest|mocha|vitest)\b", re.I)


def is_test_path(path: str) -> bool:
    parts = path.lower().split("/")
    return any(p in _TEST_DIRS for p in parts[:-1]) or bool(_TEST_FILE.search(parts[-1]))


def wants_tests(query: str) -> bool:
    return bool(_TEST_INTENT.search(query))


class Searcher:
    """tests: 'auto' (exclude test files unless the query asks), 'include' or 'exclude'."""

    def __init__(self, index: RepoIndex, lexical: bool = True, tests: str = "auto",
                 pool: int = 100, rrf_k: int = 60) -> None:
        if tests not in {"auto", "include", "exclude"}:
            raise ValueError(f"tests must be auto, include or exclude, got {tests!r}")
        self.index, self.use_lexical, self.tests, self.pool, self.rrf_k = index, lexical, tests, pool, rrf_k
        self.is_test = np.array([is_test_path(u.path) for u in index.units], dtype=bool)
        self.non_test_ids = np.flatnonzero(~self.is_test).astype("int64")
        self.bm25 = BM25([f"{u.path}\n{u.qualname}\n{u.text}" for u in index.units]) if lexical else None

    def _allowed(self, query: str) -> np.ndarray | None:
        exclude = self.tests == "exclude" or (self.tests == "auto" and not wants_tests(query))
        return self.non_test_ids if exclude and self.is_test.any() else None

    def _dense(self, qv: np.ndarray, n: int, allowed: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        params = None
        if allowed is not None:
            params = faiss.SearchParameters()
            params.sel = faiss.IDSelectorBatch(allowed)
            n = min(n, len(allowed))
        scores, ids = self.index.index.search(np.ascontiguousarray(qv[None], dtype="float32"),
                                              min(n, len(self.index.units)), params=params)
        keep = ids[0] >= 0
        return scores[0][keep], ids[0][keep]

    def search(self, query_vector: np.ndarray, query: str, k: int = 10) -> list[Hit]:
        allowed = self._allowed(query)
        fuse = self.bm25 is not None and self.bm25.has_anchor(query)
        d_scores, d_ids = self._dense(np.asarray(query_vector).reshape(-1), max(k, self.pool) if fuse else k, allowed)
        units = self.index.units
        if not fuse:
            return [Hit(r + 1, float(s), units[i]) for r, (s, i) in enumerate(zip(d_scores, d_ids, strict=True))]
        lex = self.bm25.scores(query)
        if allowed is not None:
            mask = np.zeros(len(lex), dtype=bool)
            mask[allowed] = True
            lex = np.where(mask, lex, 0.0)
        top = np.argsort(-lex)[: self.pool]
        lex_ids = [int(i) for i in top if lex[i] > 0]
        fused = rrf([[int(i) for i in d_ids], lex_ids], self.rrf_k)[:k]
        return [Hit(r + 1, float(s), units[i]) for r, (i, s) in enumerate(fused)]
