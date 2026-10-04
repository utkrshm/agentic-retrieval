"""Evaluate the production FusionSearcher (Gemma, jina, BM25) on labelled queries and time it on CPU.

    CUDA_VISIBLE_DEVICES="" uv run python scripts/eval_fusion.py \
        --repo node-red=eval/node-red/queries.json --repo browseros=eval/browseros/queries.json

For each repository (index directories outputs/index/<name>-{jina,gemma}) every query is encoded and searched one at
a time with the settled pipeline (fp32 PyTorch, batch 1, N=50, beta 0.7, gamma 0.25). Reports retrieval metrics with the
hit rule of coderet.eval.repo_queries and end-to-end warm latency percentiles measured per query (stage medians are
reported for orientation only, never summed). Model loading and the first (cold) query are reported separately.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from coderet.config import EMBEDDINGGEMMA, JINA_CODE
from coderet.embed.backends import TorchBackend
from coderet.eval.repo_queries import first_hit_rank, gold_found_at, load_queries, summarise
from coderet.index import FusionSearcher, RepoIndex


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[int(q * (len(s) - 1))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", action="append", required=True, metavar="NAME=QUERIES.json")
    ap.add_argument("--index-dir", type=Path, default=Path("outputs/index"))
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("run with CUDA_VISIBLE_DEVICES='' so the measurement is CPU only")
    if args.threads:
        torch.set_num_threads(args.threads)

    t0 = time.perf_counter()
    gemma = TorchBackend(EMBEDDINGGEMMA, "cpu", batch_size=1)
    jina = TorchBackend(JINA_CODE, "cpu", batch_size=1)
    print(f"threads {torch.get_num_threads()} | model loading {time.perf_counter() - t0:.1f}s (startup, not query latency)")
    for spec in args.repo:
        name, _, qfile = spec.partition("=")
        t0 = time.perf_counter()
        fs = FusionSearcher(RepoIndex.load(args.index_dir / f"{name}-gemma"), RepoIndex.load(args.index_dir / f"{name}-jina"),
                            gemma, jina)
        doc = load_queries(Path(qfile))
        queries = doc["queries"]
        print(f"\n== {name}: {len(fs.gemma_index.units)} units, index and BM25 setup {time.perf_counter() - t0:.1f}s")
        t = time.perf_counter()
        fs.search(queries[0]["query"])
        cold = (time.perf_counter() - t) * 1000
        for q in queries[:3]:
            fs.search(q["query"])
        rows = {"dev": [], "holdout": []}
        totals, stages = [], {}
        for q in queries:
            t = time.perf_counter()
            res = fs.search(q["query"], k=10)
            totals.append((time.perf_counter() - t) * 1000)
            for k, v in res.timings_ms.items():
                stages.setdefault(k, []).append(v)
            from coderet.index import Hit

            hits = [Hit(h.rank, h.score, h.unit) for h in res.hits]
            rows[q["split"]].append({"rank": first_hit_rank(hits, q["gold"]), "gold_at_10": gold_found_at(hits, q["gold"], 10),
                                     "category": q["category"]})
        m = summarise(rows["dev"] + rows["holdout"])["all"]
        print(f"retrieval  n={m['n']}  R@1 {m['recall@1']:.2f}  R@5 {m['recall@5']:.2f}  R@10 {m['recall@10']:.2f}  "
              f"MRR@10 {m['mrr@10']:.3f}  goldR@10 {m['gold_recall@10']:.3f}")
        print(f"latency    end-to-end p50 {pct(totals, .5):.0f} ms  p95 {pct(totals, .95):.0f} ms  max {max(totals):.0f} ms   "
              f"(first query {cold:.0f} ms)")
        print("stage p50  " + "  ".join(f"{k} {pct(v, .5):.0f}" for k, v in stages.items() if k != "total") + "  ms (orientation only)")


if __name__ == "__main__":
    main()
