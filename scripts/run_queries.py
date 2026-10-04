"""Run the labelled repository queries against a built index and log the top-k matches as JSON.

    uv run python scripts/run_queries.py --index outputs/index/node-red \
        --queries eval/node-red/queries.json --out outputs/node-red-results.json [--split holdout]

For every query the log holds the gold locations and the top-k units (rank, score, file:line, name,
whether it hits the gold, and a snippet), plus timings. Metrics are computed per split and per
category with the hit rule in coderet.eval.repo_queries. PyTorch fp32 is the default query backend.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from coderet.config import JINA_CODE
from coderet.embed import make_query_encoder
from coderet.embed.backends import TorchBackend
from coderet.eval.repo_queries import first_hit_rank, gold_found_at, is_hit, load_queries, summarise
from coderet.index import RepoIndex


def snippet(text: str, lines: int = 8) -> str:
    return "\n".join(text.split("\n")[:lines])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="all", choices=["all", "dev", "holdout"])
    ap.add_argument("--query-backend", default="torch", choices=["torch", "auto"],
                    help="torch: PyTorch fp32 on CPU; auto: OpenVINO int8 -> fp32 -> PyTorch chain")
    ap.add_argument("-k", type=int, default=5, help="matches to log per query (metrics always use the top 10)")
    args = ap.parse_args()

    index = RepoIndex.load(args.index)
    doc = load_queries(args.queries)
    if doc["commit"] != index.meta["commit"]:
        print(f"WARNING: queries were labelled at {doc['commit'][:10]} but the index is {index.meta['commit'][:10]}")
    queries = [q for q in doc["queries"] if args.split in ("all", q["split"])]
    encoder = TorchBackend(JINA_CODE, "cpu") if args.query_backend == "torch" else make_query_encoder(JINA_CODE)
    name = getattr(encoder, "name", None) or "chain"

    encoder.embed([queries[0]["query"]], "query")  # warm up
    results, rows = [], {"dev": [], "holdout": []}
    for q in queries:
        t0 = time.perf_counter()
        qv = encoder.embed([q["query"]], "query")
        t1 = time.perf_counter()
        hits = index.search(np.asarray(qv), 10)[0]
        t2 = time.perf_counter()
        rank = first_hit_rank(hits, q["gold"])
        rows[q["split"]].append({"rank": rank, "gold_at_10": gold_found_at(hits, q["gold"], 10), "category": q["category"]})
        results.append({
            "id": q["id"], "split": q["split"], "category": q["category"], "query": q["query"],
            "gold": q["gold"], "first_hit_rank": rank,
            "encode_ms": round((t1 - t0) * 1000, 1), "search_ms": round((t2 - t1) * 1000, 2),
            "top": [{"rank": h.rank, "score": round(h.score, 4), "hit": is_hit(h, q["gold"]),
                     "path": h.unit.path, "start": h.unit.start_line, "end": h.unit.end_line,
                     "kind": h.unit.kind, "qualname": h.unit.qualname, "snippet": snippet(h.unit.text)}
                    for h in hits[: args.k]],
        })
    metrics = {s: summarise(r) for s, r in rows.items() if r}
    enc_ms = sorted(r["encode_ms"] for r in results)
    log = {"index": index.meta, "queries_file": str(args.queries), "query_backend": name,
           "latency_ms": {"encode_p50": enc_ms[len(enc_ms) // 2], "encode_p95": enc_ms[int(0.95 * (len(enc_ms) - 1))],
                          "search_max": max(r["search_ms"] for r in results)},
           "metrics": metrics, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(log, indent=1))

    print(f"index {index.meta['repo']} @ {index.meta['commit'][:10]} ({index.meta['n_units']} units) | queries: {name} | wrote {args.out}")
    for split, m in metrics.items():
        a = m["all"]
        print(f"\n{split}: n={a['n']}  recall@1 {a['recall@1']}  @5 {a['recall@5']}  @10 {a['recall@10']}  MRR@10 {a['mrr@10']}  gold-recall@10 {a['gold_recall@10']}")
        for cat, c in m["by_category"].items():
            print(f"   {cat:13s} n={c['n']:2d}  recall@1 {c['recall@1']}  @5 {c['recall@5']}  @10 {c['recall@10']}  MRR@10 {c['mrr@10']}")


if __name__ == "__main__":
    main()
