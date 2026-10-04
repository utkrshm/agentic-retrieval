"""AppsRetrieval (MTEB task, test split) scored for jina, Gemma and the two-stage pipelines.

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/apps_two_stage.py

NOT the official submission path (that stays a plain encoder, scripts/run_mteb.py). This is an experiment on the
benchmark's data: both models embed the 8,765 programs and 3,765 queries once (GPU, offline; vectors cached under
outputs/apps-vectors/), then every system is scored with the metrics MTEB reports (one relevant program per query, so
NDCG@10 = 1/log2(rank+1) when rank <= 10). jina and Gemma alone must reproduce the official runs (83.83 / 84.28).

Systems (all fixed before running; nothing is tuned on this split):
  jina, gemma                        dense cosine over the whole corpus
  jina+bm25, gemma+bm25              gated BM25 lane: RRF of dense top-100 and BM25 top-100 when the query has a code
                                     anchor that occurs in the corpus; other queries dense only
  two-stage                          Gemma ranking, its top-10 re-ordered by jina cosine (rest keep Gemma order)
  two-stage+bm25                     the same, with the gated BM25 lane in Gemma's stage
  fusion N=10 / N=50                 Gemma's top-N (with the lane) re-scored by a min-max blend of jina cosine, Gemma
                                     cosine and BM25 (beta 0.7, gamma 0.25, the setting both repo directions chose in
                                     E12c; fixed from the repositories, not tuned on this split)
Only text, ids and relevance judgements are read from the task; the dataset's metadata fields are never used.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mteb
import numpy as np

from coderet.config import EMBEDDINGGEMMA, JINA_CODE
from coderet.embed.backends import TorchBackend
from coderet.index.lexical import BM25, rrf

OUT = Path("outputs/apps-vectors")
POOL, N = 100, 10


def encode_model(spec, q_texts: list[str], doc_texts: list[str], batch: int) -> tuple[np.ndarray, np.ndarray]:
    """Queries and documents for one model, with a single model copy on the GPU (cached on disk)."""
    paths = {r: OUT / f"{spec.key}-{r}.npy" for r in ("query", "document")}
    if all(p.exists() for p in paths.values()):
        return np.load(paths["query"]), np.load(paths["document"])
    import gc

    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    be = TorchBackend(spec, "cuda", batch_size=batch)
    for role, texts in (("query", q_texts), ("document", doc_texts)):
        if paths[role].exists():
            continue
        t = time.perf_counter()
        np.save(paths[role], be.embed(texts, role))
        print(f"  {spec.key} {role}: {len(texts)} texts in {time.perf_counter() - t:.0f}s", flush=True)
    del be
    gc.collect()
    torch.cuda.empty_cache()
    return np.load(paths["query"]), np.load(paths["document"])


def metrics(ranks: np.ndarray) -> dict:
    """ranks: 1-based rank of the relevant document per query (large when not retrieved)."""
    r = ranks.astype(float)
    in10 = r <= 10
    return {
        "ndcg@10": float(np.where(in10, 1 / np.log2(r + 1), 0).mean() * 100),
        "mrr@10": float(np.where(in10, 1 / r, 0).mean() * 100),
        **{f"recall@{k}": float((r <= k).mean() * 100) for k in (1, 5, 10, 20, 100)},
    }


def main() -> None:
    task = mteb.get_task("AppsRetrieval")
    task.load_data()
    d = task.dataset["default"]["test"]
    corpus, queries, rel = d["corpus"], d["queries"], d["relevant_docs"]
    doc_ids, doc_texts = list(corpus["id"]), list(corpus["text"])
    q_ids, q_texts = list(queries["id"]), list(queries["text"])
    pos = {i: k for k, i in enumerate(doc_ids)}
    gold = np.array([pos[next(iter(rel[q]))] for q in q_ids])
    print(f"{len(doc_texts)} documents, {len(q_texts)} queries", flush=True)

    vec = {}
    for spec, batch in ((JINA_CODE, 8), (EMBEDDINGGEMMA, 4)):
        vec[spec.key] = encode_model(spec, q_texts, doc_texts, batch)
    sims = {k: q @ docs.T for k, (q, docs) in vec.items()}  # (queries, docs) float32
    jina_s, gem_s = sims[JINA_CODE.key], sims[EMBEDDINGGEMMA.key]

    bm = BM25(doc_texts)
    gated = np.array([bm.has_anchor(q) for q in q_texts])
    print(f"BM25 gate fires on {gated.sum()} of {len(q_texts)} queries ({100 * gated.mean():.1f}%)", flush=True)

    def order(scores: np.ndarray, qi: int, lane: bool) -> np.ndarray:
        """Document indices best first for one query: dense, or gated RRF of dense and BM25 top-100."""
        dense = np.argsort(-scores[qi], kind="stable")
        if not lane or not gated[qi]:
            return dense
        lex = bm.scores(q_texts[qi])
        lex_ids = [int(i) for i in np.argsort(-lex)[:POOL] if lex[i] > 0]
        fused = [i for i, _ in rrf([[int(i) for i in dense[:POOL]], lex_ids], 60)]
        seen = set(fused)
        return np.array(fused + [int(i) for i in dense if int(i) not in seen])

    def rank_of(ordered: np.ndarray, g: int) -> int:
        hit = np.flatnonzero(ordered == g)
        return int(hit[0]) + 1 if hit.size else 10**6

    systems = {
        "jina": lambda qi: order(jina_s, qi, False),
        "gemma": lambda qi: order(gem_s, qi, False),
        "jina+bm25": lambda qi: order(jina_s, qi, True),
        "gemma+bm25": lambda qi: order(gem_s, qi, True),
    }

    def two_stage(qi: int, lane: bool) -> np.ndarray:
        first = order(gem_s, qi, lane)
        top = first[:N]
        top = top[np.argsort(-jina_s[qi, top], kind="stable")]
        return np.concatenate([top, first[N:]])

    def mm(x: np.ndarray) -> np.ndarray:
        span = x.max() - x.min()
        return (x - x.min()) / span if span > 1e-9 else np.zeros_like(x)

    def fusion(qi: int, n: int, beta: float = 0.7, gamma: float = 0.25) -> np.ndarray:
        """Gemma's top-n (with the gated lane) re-scored by a min-max blend of jina, Gemma and BM25 (E12c setting)."""
        first = order(gem_s, qi, True)
        top = first[:n]
        g = gamma if gated[qi] else 0.0
        lex = bm.scores(q_texts[qi])[top] if gated[qi] else np.zeros(len(top))
        s = (1 - g) * (beta * mm(jina_s[qi, top]) + (1 - beta) * mm(gem_s[qi, top])) + g * mm(lex)
        return np.concatenate([top[np.argsort(-s, kind="stable")], first[n:]])

    systems["two-stage"] = lambda qi: two_stage(qi, False)
    systems["two-stage+bm25"] = lambda qi: two_stage(qi, True)
    for n in (10, 50):
        systems[f"fusion N={n}"] = lambda qi, n=n: fusion(qi, n)

    results, all_ranks = {}, {}
    for name, fn in systems.items():
        ranks = np.array([rank_of(fn(qi)[:200], int(gold[qi])) for qi in range(len(q_texts))])
        all_ranks[name] = ranks
        results[name] = metrics(ranks)
        m = results[name]
        print(f"{name:16s} NDCG@10 {m['ndcg@10']:6.2f}  MRR@10 {m['mrr@10']:6.2f}  R@1 {m['recall@1']:6.2f}  "
              f"R@5 {m['recall@5']:6.2f}  R@10 {m['recall@10']:6.2f}  R@20 {m['recall@20']:6.2f}  R@100 {m['recall@100']:6.2f}",
              flush=True)
    base = all_ranks["jina"]
    for name, r in all_ranks.items():
        better, worse = int((r < base).sum()), int((r > base).sum())
        results[name]["vs_jina_better_worse"] = [better, worse]
    Path("outputs/apps-two-stage.json").write_text(json.dumps(
        {"gate_fires": int(gated.sum()), "n_queries": len(q_texts), "pool": POOL, "n_rescored": N, "results": results}, indent=1))
    print("wrote outputs/apps-two-stage.json")


if __name__ == "__main__":
    main()
