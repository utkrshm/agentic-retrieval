"""Exact dense retrieval: a resident fp32 doc matrix scanned with one matmul.

At APPS scale (8,765 docs) an exact scan takes a few ms, so no ANN index.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from coderet.mteb_adapters.encoder import PrePostPipelineEncoder


@dataclass
class DenseIndex:
    doc_ids: list[str]
    matrix: np.ndarray  # (n_docs, dim), unit-norm rows

    def __post_init__(self) -> None:
        self.matrix = np.ascontiguousarray(self.matrix, dtype=np.float32)

    def search(self, queries: np.ndarray, k: int) -> list[list[tuple[str, float]]]:
        scores = np.atleast_2d(queries).astype(np.float32, copy=False) @ self.matrix.T
        k = min(k, scores.shape[1])
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        results = []
        for row, idx in zip(scores, top):
            idx = idx[np.argsort(-row[idx], kind="stable")]
            results.append([(self.doc_ids[i], float(row[i])) for i in idx])
        return results


@dataclass
class Retriever:
    """Query text -> ranked doc ids, timing each stage."""

    pipeline: PrePostPipelineEncoder
    index: DenseIndex
    last_timings_ms: dict[str, float] = field(default_factory=dict)

    @classmethod
    def build(cls, pipeline: PrePostPipelineEncoder, docs: dict[str, str], batch_size: int = 32) -> "Retriever":
        ids = list(docs)
        return cls(pipeline, DenseIndex(ids, pipeline.encode_texts([docs[i] for i in ids], "document", batch_size)))

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        t0 = time.perf_counter()
        q = self.pipeline.encode_texts([query], "query", batch_size=1)
        t1 = time.perf_counter()
        hits = self.index.search(q, k)[0]
        t2 = time.perf_counter()
        self.last_timings_ms = {"encode": (t1 - t0) * 1e3, "scan": (t2 - t1) * 1e3, "total": (t2 - t0) * 1e3}
        return hits
