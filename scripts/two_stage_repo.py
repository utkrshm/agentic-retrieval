"""Two-stage test: one encoder retrieves the top-N, the other encoder re-scores those N.

    uv run python scripts/two_stage_repo.py --queries eval/node-red/queries.json \
        --jina outputs/index/node-red-ts --gemma outputs/index/node-red-gemma

Both indexes must hold the same units (same chunking). Stage one is the full pipeline of one model
(test scope plus the gated lexical lane). Stage two reorders its top-N by the other model's cosine, or by the sum
of both cosines (min-max normalised over the N candidates). Cost at query time: both encoders embed the query;
the second one's document vectors are already stored, so re-scoring is N dot products.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ablate_repo import query_vectors
from coderet.config import EMBEDDINGGEMMA, JINA_CODE
from coderet.eval.repo_queries import first_hit_rank, gold_found_at, load_queries, summarise
from coderet.index import Hit, RepoIndex, Searcher


def key(u) -> tuple:
    return (u.path, u.start_line, u.end_line, u.part, u.kind)


def minmax(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 1e-9 else np.zeros_like(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--jina", type=Path, required=True)
    ap.add_argument("--gemma", type=Path, required=True)
    ap.add_argument("--n", type=int, nargs="*", default=[5, 10, 20])
    args = ap.parse_args()

    doc = load_queries(args.queries)
    queries = doc["queries"]
    texts = [q["query"] for q in queries]
    idx = {"jina": RepoIndex.load(args.jina), "gemma": RepoIndex.load(args.gemma)}
    keys = {m: {key(u): i for i, u in enumerate(ix.units)} for m, ix in idx.items()}
    if set(keys["jina"]) != set(keys["gemma"]):
        raise SystemExit("the two indexes do not hold the same units")
    qv = {"jina": query_vectors(texts, JINA_CODE), "gemma": query_vectors(texts, EMBEDDINGGEMMA)}
    search = {m: Searcher(ix, lexical=True, tests="auto") for m, ix in idx.items()}

    def score_other(first: str, other: str, hits: list[Hit], qi: int) -> tuple[np.ndarray, np.ndarray]:
        ids = [keys[other][key(h.unit)] for h in hits]
        own = np.array([float(idx[first].vectors[keys[first][key(h.unit)]] @ qv[first][qi]) for h in hits])
        oth = np.array([float(idx[other].vectors[i] @ qv[other][qi]) for i in ids])
        return own, oth

    def run(first: str, n: int, mode: str) -> dict:
        other = "jina" if first == "gemma" else "gemma"
        rows = {"dev": [], "holdout": []}
        ranks = []
        for qi, q in enumerate(queries):
            hits = search[first].search(qv[first][qi], q["query"], max(n, 10))
            pool = hits[:n]
            if mode != "single" and len(pool) > 1:
                own, oth = score_other(first, other, pool, qi)
                s = oth if mode == "other" else minmax(own) + minmax(oth)
                order = np.argsort(-s, kind="stable")
                pool = [pool[i] for i in order]
            final = [Hit(r + 1, h.score, h.unit) for r, h in enumerate(pool + hits[n:])][:10]
            r = first_hit_rank(final, q["gold"])
            ranks.append(r)
            rows[q["split"]].append({"rank": r, "gold_at_10": gold_found_at(final, q["gold"], 10), "category": q["category"]})
        all_rows = rows["dev"] + rows["holdout"]
        return {"m": summarise(all_rows)["all"], "ranks": ranks}

    print(f"{'system':44s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s} {'MRR@10':>7s} {'goldR@10':>8s}  vs jina-alone")
    base = run("jina", 10, "single")
    for label, res in [("jina alone (lane + tests)", base), ("gemma alone (lane + tests)", run("gemma", 10, "single"))]:
        m = res["m"]
        print(f"{label:44s} {m['recall@1']:6.3f} {m['recall@5']:6.3f} {m['recall@10']:6.3f} {m['mrr@10']:7.3f} {m['gold_recall@10']:8.3f}")
    for first in ("gemma", "jina"):
        other = "jina" if first == "gemma" else "gemma"
        for n in args.n:
            for mode, what in (("other", f"re-ranked by {other} only"), ("sum", f"re-ranked by {first}+{other} sum")):
                res = run(first, n, mode)
                m = res["m"]
                better = sum(1 for r, b in zip(res["ranks"], base["ranks"], strict=True) if (r or 99) < (b or 99))
                worse = sum(1 for r, b in zip(res["ranks"], base["ranks"], strict=True) if (r or 99) > (b or 99))
                label = f"{first} top-{n} {what}"
                print(f"{label:44s} {m['recall@1']:6.3f} {m['recall@5']:6.3f} {m['recall@10']:6.3f} {m['mrr@10']:7.3f} {m['gold_recall@10']:8.3f}  {better} better / {worse} worse")


if __name__ == "__main__":
    main()
