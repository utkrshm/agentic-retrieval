"""Compare search configurations (and index variants) on a labelled query file, with cached query vectors.

    uv run python scripts/ablate_repo.py --queries eval/node-red/queries.json \
        --index base=outputs/index/node-red --index m60=outputs/index/node-red-m60 \
        --out outputs/ablation-node-red.json

Each index is searched with four configurations: dense only or gated lexical fusion, each with test files
included or in "auto" scope (excluded unless the query asks for tests). Queries are encoded once with
PyTorch fp32 (query vectors are the same for every index). The first index/config is the baseline for the
paired flip counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from coderet.config import MODELS, ModelSpec
from coderet.embed.backends import TorchBackend
from coderet.eval.repo_queries import first_hit_rank, gold_found_at, load_queries, summarise
from coderet.index import RepoIndex, Searcher

CONFIGS = {
    "dense": dict(lexical=False, tests="include"),
    "dense+tests-auto": dict(lexical=False, tests="auto"),
    "lex": dict(lexical=True, tests="include"),
    "lex+tests-auto": dict(lexical=True, tests="auto"),
}


def query_vectors(texts: list[str], spec: ModelSpec | None = None) -> np.ndarray:
    spec = spec or MODELS["jina-code-0.5b"]
    key = hashlib.sha256(("\n".join(texts) + spec.hf_id + spec.revision + spec.query_prompt).encode()).hexdigest()[:16]
    path = Path(".cache/queryvecs") / f"{key}.npy"
    if path.exists():
        return np.load(path)
    vecs = TorchBackend(spec, "cpu").embed(texts, "query")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    return vecs


def evaluate(index: RepoIndex, cfg: dict, queries: list[dict], qv: np.ndarray) -> tuple[dict, list[int | None]]:
    searcher = Searcher(index, **cfg)
    rows = {"dev": [], "holdout": []}
    ranks = []
    for q, v in zip(queries, qv, strict=True):
        hits = searcher.search(v, q["query"], 10)
        r = first_hit_rank(hits, q["gold"])
        ranks.append(r)
        rows[q["split"]].append({"rank": r, "gold_at_10": gold_found_at(hits, q["gold"], 10), "category": q["category"]})
    rows["all"] = rows["dev"] + rows["holdout"]
    return {s: summarise(r) for s, r in rows.items() if r}, ranks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--index", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    doc = load_queries(args.queries)
    queries = doc["queries"]
    texts = [q["query"] for q in queries]
    qvs: dict[str, np.ndarray] = {}
    out: dict = {}
    base: list[int | None] | None = None
    print(f"{'index / config':34s} {'split':8s} {'n':>3s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s} {'MRR@10':>7s} {'goldR@10':>8s}")
    for spec in args.index:
        name, _, directory = spec.partition("=")
        index = RepoIndex.load(Path(directory))
        spec = next(m for m in MODELS.values() if m.hf_id == index.meta["model"])
        if spec.key not in qvs:
            qvs[spec.key] = query_vectors(texts, spec)
        qv = qvs[spec.key]
        for cname in args.configs:
            metrics, ranks = evaluate(index, CONFIGS[cname], queries, qv)
            out[f"{name}/{cname}"] = {"metrics": metrics, "ranks": dict(zip((q["id"] for q in queries), ranks, strict=True)),
                                       "n_units": len(index.units)}
            if base is None:
                base = ranks
            for split in ("dev", "holdout", "all"):
                a = metrics[split]["all"]
                print(f"{name + ' / ' + cname:34s} {split:8s} {a['n']:3d} {a['recall@1']:6.3f} {a['recall@5']:6.3f} "
                      f"{a['recall@10']:6.3f} {a['mrr@10']:7.3f} {a['gold_recall@10']:8.3f}")
            better = sum(1 for r, b in zip(ranks, base, strict=True) if (r or 99) < (b or 99))
            worse = sum(1 for r, b in zip(ranks, base, strict=True) if (r or 99) > (b or 99))
            cats = "  ".join(f"{c} {v['mrr@10']:.3f}" for c, v in metrics["all"]["by_category"].items())
            print(f"{'':34s} vs first row: {better} queries better, {worse} worse | MRR by category: {cats}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
