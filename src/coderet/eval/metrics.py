"""Ranking metrics over `run` = {qid: {doc_id: score}} and `qrels` = {qid: {doc_id: rel}}.

Matches MTEB's definitions (pytrec_eval): NDCG@k with linear gains, MRR@k,
Recall@k. Ties are broken by doc id, as pytrec_eval does.
"""

from __future__ import annotations

import math

import numpy as np

Run = dict[str, dict[str, float]]
Qrels = dict[str, dict[str, int]]


def ranked(scores: dict[str, float], k: int) -> list[str]:
    # pytrec_eval orders by score desc, then doc id desc
    return [d for d, _ in sorted(scores.items(), key=lambda x: (x[1], x[0]), reverse=True)[:k]]


def per_query(run: Run, qrels: Qrels, k: int = 10, recall_ks: tuple[int, ...] = (10, 20, 50, 100)) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for qid, rels in qrels.items():
        relevant = {d: r for d, r in rels.items() if r > 0}
        top = ranked(run.get(qid, {}), max(k, *recall_ks))
        dcg = sum(relevant.get(d, 0) / math.log2(i + 2) for i, d in enumerate(top[:k]))
        ideal = sorted(relevant.values(), reverse=True)[:k]
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
        first = next((i for i, d in enumerate(top[:k]) if d in relevant), None)
        m = {f"ndcg@{k}": dcg / idcg if idcg else 0.0, f"mrr@{k}": 1.0 / (first + 1) if first is not None else 0.0}
        for rk in recall_ks:
            m[f"recall@{rk}"] = len(relevant.keys() & set(top[:rk])) / len(relevant) if relevant else 0.0
        out[qid] = m
    return out


def aggregate(pq: dict[str, dict[str, float]]) -> dict[str, float]:
    names = next(iter(pq.values())).keys()
    return {n: float(np.mean([m[n] for m in pq.values()])) for n in names}


def by_length_bucket(pq: dict[str, dict[str, float]], lengths: dict[str, int], edges=(256, 512, 1024, 2048), metric: str = "ndcg@10") -> dict[str, tuple[int, float]]:
    """Mean `metric` per query-length bucket -> {label: (count, mean)}."""
    buckets: dict[str, list[float]] = {}
    for qid, m in pq.items():
        n = lengths[qid]
        label = next((f"<={e}" for e in edges if n <= e), f">{edges[-1]}")
        buckets.setdefault(label, []).append(m[metric])
    return {b: (len(v), float(np.mean(v))) for b, v in sorted(buckets.items(), key=lambda x: (x[0][0] == ">", len(x[0]), x[0]))}
