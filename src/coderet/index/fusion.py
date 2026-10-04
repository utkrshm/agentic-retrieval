"""The settled repository search: Gemma retrieves, jina and BM25 re-score (learned score fusion).

Per query: Gemma encodes the query; ``Searcher`` (test scope, gated BM25 lane) returns Gemma's top-``pool`` units;
jina encodes the query too and every candidate gets its stored jina cosine from the jina index; the candidates are
re-ranked by a per-query min-max blend

    s = (1 - g) * (beta * jina + (1 - beta) * gemma) + g * bm25        g = gamma if the BM25 gate fired, else 0

with beta = 0.7 and gamma = 0.25, the setting both directions of the cross-repository experiment chose
(docs/experiments, E12c). Both indexes must hold the same units. Query encoders are any object with
``embed(texts, role) -> (n, dim) float32 L2-normalised`` (fp32 PyTorch backends in the demo).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from coderet.chunking import Unit
from coderet.index.repo_index import RepoIndex
from coderet.index.search import Searcher, wants_tests


@dataclass(frozen=True)
class FusionHit:
    rank: int  # 1-based
    score: float  # fused score in [0, 1] (min-max over the candidates, so only meaningful within one query)
    jina: float  # cosine of the query and the unit in the jina-code space
    gemma: float  # cosine in the embeddinggemma space
    bm25: float | None  # raw BM25 score when the lexical gate fired, else None
    unit: Unit


@dataclass
class FusionResult:
    query: str
    hits: list[FusionHit]
    gate_fired: bool  # the query named an identifier or quoted literal that exists in the code
    tests_included: bool  # test files were searchable for this query
    n_candidates: int
    timings_ms: dict[str, float] = field(default_factory=dict)


def _unit_key(u: Unit) -> tuple:
    return (u.path, u.start_line, u.end_line, u.part, u.kind)


def _minmax(x: np.ndarray) -> np.ndarray:
    span = float(x.max() - x.min())
    return (x - x.min()) / span if span > 1e-9 else np.zeros_like(x)


class FusionSearcher:
    def __init__(self, gemma_index: RepoIndex, jina_index: RepoIndex, gemma_encoder, jina_encoder,
                 beta: float = 0.7, gamma: float = 0.25, pool: int = 50, tests: str = "auto") -> None:
        gkeys = [_unit_key(u) for u in gemma_index.units]
        jkeys = [_unit_key(u) for u in jina_index.units]
        if set(gkeys) != set(jkeys) or len(gkeys) != len(jkeys):
            raise ValueError("the Gemma and jina indexes must hold the same units")
        if not (0.0 <= beta <= 1.0 and 0.0 <= gamma < 1.0):
            raise ValueError("beta must be in [0, 1] and gamma in [0, 1)")
        self.gemma_index, self.jina_index = gemma_index, jina_index
        self.gemma_encoder, self.jina_encoder = gemma_encoder, jina_encoder
        self.beta, self.gamma, self.pool = beta, gamma, pool
        self._gpos = {k: i for i, k in enumerate(gkeys)}
        self._jpos = {k: i for i, k in enumerate(jkeys)}
        self._searcher = Searcher(gemma_index, lexical=True, tests=tests)

    def search(self, query: str, k: int = 10, tests: str | None = None) -> FusionResult:
        """Top-k units for the query. ``tests``: auto (exclude test files unless the query asks), include, exclude."""
        t0 = time.perf_counter()
        gv = np.asarray(self.gemma_encoder.embed([query], "query"))[0]
        t1 = time.perf_counter()
        gate = self._searcher.bm25.has_anchor(query)
        pool = self._searcher.search(gv, query, max(self.pool, k), tests=tests)
        t2 = time.perf_counter()
        jv = np.asarray(self.jina_encoder.embed([query], "query"))[0]
        t3 = time.perf_counter()
        mode = tests or self._searcher.tests
        included = mode == "include" or (mode == "auto" and wants_tests(query))
        if not pool:
            return FusionResult(query, [], gate, included, 0, {"total": (time.perf_counter() - t0) * 1000})
        gc = np.array([float(self.gemma_index.vectors[self._gpos[_unit_key(h.unit)]] @ gv) for h in pool])
        jc = np.array([float(self.jina_index.vectors[self._jpos[_unit_key(h.unit)]] @ jv) for h in pool])
        if gate:
            lex = self._searcher.bm25.scores(query)
            bm = np.array([float(lex[self._gpos[_unit_key(h.unit)]]) for h in pool])
        else:
            bm = np.zeros(len(pool))
        g = self.gamma if gate else 0.0
        s = (1 - g) * (self.beta * _minmax(jc) + (1 - self.beta) * _minmax(gc)) + g * _minmax(bm)
        order = np.argsort(-s, kind="stable")[:k]
        hits = [FusionHit(r + 1, float(s[i]), float(jc[i]), float(gc[i]), float(bm[i]) if gate else None, pool[i].unit)
                for r, i in enumerate(order)]
        t4 = time.perf_counter()
        timings = {"gemma_encode": (t1 - t0) * 1000, "gemma_search": (t2 - t1) * 1000,
                   "jina_encode": (t3 - t2) * 1000, "fuse": (t4 - t3) * 1000, "total": (t4 - t0) * 1000}
        return FusionResult(query, hits, gate, included, len(pool), timings)


__all__ = ["FusionHit", "FusionResult", "FusionSearcher"]
