"""CPU-only warm batch-1 latency of the two-stage pipeline (Gemma retrieves, jina re-scores the top-N).

    CUDA_VISIBLE_DEVICES="" uv run python scripts/bench_two_stage.py --queries eval/browseros/queries.json \
        --jina outputs/index/browseros-m60 --gemma outputs/index/browseros-gemma [--threads 4] [--jina-int8]

Per query, end to end: Gemma query encode, Gemma search (test scope plus gated lexical lane), jina query encode,
re-score the top-N by jina cosine from the stored document vectors, sort. Percentiles are measured on the whole
path per query (never summed from stages). Also reports jina alone and Gemma alone for reference. Documents are
encoded offline; no query-result cache exists. The first query is reported separately as cold.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from coderet.config import EMBEDDINGGEMMA, JINA_CODE
from coderet.embed import make_query_encoder
from coderet.embed.backends import TorchBackend
from coderet.eval.repo_queries import load_queries
from coderet.index import RepoIndex, Searcher


def pct(xs: list[float]) -> str:
    s = sorted(xs)
    return f"p50 {s[len(s) // 2]:7.0f}  p95 {s[int(0.95 * (len(s) - 1))]:7.0f}  max {s[-1]:7.0f} ms"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--jina", type=Path, required=True)
    ap.add_argument("--gemma", type=Path, required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0: library default)")
    ap.add_argument("--jina-int8", action="store_true", help="jina query encoder: OpenVINO int8 chain instead of fp32")
    args = ap.parse_args()
    assert not torch.cuda.is_available(), "run with CUDA_VISIBLE_DEVICES='' so the GPU is not used"
    if args.threads:
        torch.set_num_threads(args.threads)

    texts = [q["query"] for q in load_queries(args.queries)["queries"]]
    jina_ix, gemma_ix = RepoIndex.load(args.jina), RepoIndex.load(args.gemma)
    key = lambda u: (u.path, u.start_line, u.end_line, u.part, u.kind)  # noqa: E731
    jpos = {key(u): i for i, u in enumerate(jina_ix.units)}
    assert set(jpos) == {key(u) for u in gemma_ix.units}, "indexes differ"
    gsearch, jsearch = Searcher(gemma_ix, tests="auto"), Searcher(jina_ix, tests="auto")

    t0 = time.perf_counter()
    gemma = TorchBackend(EMBEDDINGGEMMA, "cpu", batch_size=1)
    jina = make_query_encoder(JINA_CODE) if args.jina_int8 else TorchBackend(JINA_CODE, "cpu", batch_size=1)
    jname = "int8 OpenVINO chain" if args.jina_int8 else "fp32 PyTorch"
    print(f"threads {torch.get_num_threads()} | jina query encoder: {jname} | load {time.perf_counter() - t0:.1f}s "
          f"(startup, not query latency) | {len(texts)} queries, N={args.n}")

    def jina_vec(q: str) -> np.ndarray:
        return np.asarray(jina.embed_queries([q]) if args.jina_int8 else jina.embed([q], "query"))[0]

    def gemma_vec(q: str) -> np.ndarray:
        return gemma.embed([q], "query")[0]

    def two_stage(q: str) -> None:
        gv = gemma_vec(q)
        hits = gsearch.search(gv, q, max(args.n, 10))
        jv = jina_vec(q)
        pool = hits[: args.n]
        s = np.array([float(jina_ix.vectors[jpos[key(h.unit)]] @ jv) for h in pool])
        _ = [pool[i] for i in np.argsort(-s, kind="stable")] + hits[args.n :]

    gpos = {key(u): i for i, u in enumerate(gemma_ix.units)}

    def mm(x: np.ndarray) -> np.ndarray:
        span = x.max() - x.min()
        return (x - x.min()) / span if span > 1e-9 else np.zeros_like(x)

    def fusion(q: str) -> None:
        """Gemma top-50 (test scope, gated lane) re-scored by 0.7 jina + 0.3 Gemma, blended with BM25 when gated."""
        gv = gemma_vec(q)
        hits = gsearch.search(gv, q, 50)
        jv = jina_vec(q)
        gated = gsearch.bm25.has_anchor(q)
        gc = np.array([float(gemma_ix.vectors[gpos[key(h.unit)]] @ gv) for h in hits])
        jc = np.array([float(jina_ix.vectors[jpos[key(h.unit)]] @ jv) for h in hits])
        g = 0.25 if gated else 0.0
        lex = gsearch.bm25.scores(q) if gated else None
        bm = np.array([float(lex[gpos[key(h.unit)]]) for h in hits]) if gated else np.zeros(len(hits))
        s = (1 - g) * (0.7 * mm(jc) + 0.3 * mm(gc)) + g * mm(bm)
        _ = [hits[i] for i in np.argsort(-s, kind="stable")[:10]]

    def gemma_only(q: str) -> None:
        gsearch.search(gemma_vec(q), q, 10)

    def jina_only(q: str) -> None:
        jsearch.search(jina_vec(q), q, 10)

    for name, fn in (("jina alone", jina_only), ("gemma alone", gemma_only), (f"two-stage (gemma top-{args.n} then jina)", two_stage),
                     ("fusion N=50 (the settled pipeline)", fusion)):
        t = time.perf_counter()
        fn(texts[0])
        cold = (time.perf_counter() - t) * 1000
        for q in texts[:3]:
            fn(q)  # warm up
        lat = []
        for q in texts:
            t = time.perf_counter()
            fn(q)
            lat.append((time.perf_counter() - t) * 1000)
        print(f"{name:40s} {pct(lat)}   (first call {cold:.0f} ms)")


if __name__ == "__main__":
    main()
