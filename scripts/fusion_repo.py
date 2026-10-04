"""Learned score fusion inside the two-stage pipeline, tuned on one repository and tested on the other.

    uv run python scripts/fusion_repo.py

Candidates are Gemma's top-N from the full lane pipeline; each gets Gemma cosine, jina cosine and BM25 scores,
min-max normalised per query, and s = (1 - g) * (beta * jina + (1 - beta) * gemma) + g * bm25 with g = gamma only
when the query's BM25 gate fires. The grid and the protocol are pre-registered in docs/experiments.md (E12c).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from ablate_repo import query_vectors
from coderet.config import EMBEDDINGGEMMA, JINA_CODE
from coderet.eval.repo_queries import first_hit_rank, gold_found_at, load_queries, summarise
from coderet.index import Hit, RepoIndex, Searcher

REPOS = {
    "node-red": ("eval/node-red/queries.json", "outputs/index/node-red-ts", "outputs/index/node-red-gemma"),
    "browseros": ("eval/browseros/queries.json", "outputs/index/browseros-m60", "outputs/index/browseros-gemma"),
}
BETAS, GAMMAS, NS = (0.5, 0.7, 1.0), (0.0, 0.25, 0.5), (10, 20, 50)
key = lambda u: (u.path, u.start_line, u.end_line, u.part, u.kind)  # noqa: E731


def mm(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 1e-9 else np.zeros_like(x)


class Repo:
    def __init__(self, name: str) -> None:
        qfile, jdir, gdir = REPOS[name]
        self.queries = load_queries(Path(qfile))["queries"]
        texts = [q["query"] for q in self.queries]
        self.jina, self.gemma = RepoIndex.load(Path(jdir)), RepoIndex.load(Path(gdir))
        jpos = {key(u): i for i, u in enumerate(self.jina.units)}
        assert set(jpos) == {key(u) for u in self.gemma.units}
        self.jq, self.gq = query_vectors(texts, JINA_CODE), query_vectors(texts, EMBEDDINGGEMMA)
        self.search = Searcher(self.gemma, lexical=True, tests="auto")
        self.cands = {}  # (query index, N) -> (hits, gemma cos, jina cos, bm25, gate)
        for qi, q in enumerate(self.queries):
            gate = self.search.bm25.has_anchor(q["query"])
            lex = self.search.bm25.scores(q["query"]) if gate else None
            pool = self.search.search(self.gq[qi], q["query"], max(NS))
            gi = [i for i, u in enumerate(self.gemma.units)]  # noqa: F841
            gidx = {key(u): i for i, u in enumerate(self.gemma.units)}
            gc = np.array([float(self.gemma.vectors[gidx[key(h.unit)]] @ self.gq[qi]) for h in pool])
            jc = np.array([float(self.jina.vectors[jpos[key(h.unit)]] @ self.jq[qi]) for h in pool])
            bm = np.array([float(lex[gidx[key(h.unit)]]) for h in pool]) if gate else np.zeros(len(pool))
            self.cands[qi] = (pool, gc, jc, bm, gate)

    def evaluate(self, n: int, beta: float, gamma: float) -> tuple[dict, list]:
        rows = {"dev": [], "holdout": []}
        ranks = []
        for qi, q in enumerate(self.queries):
            pool, gc, jc, bm, gate = self.cands[qi]
            k = min(n, len(pool))
            g = gamma if gate else 0.0
            s = (1 - g) * (beta * mm(jc[:k]) + (1 - beta) * mm(gc[:k])) + g * mm(bm[:k])
            order = np.argsort(-s, kind="stable")
            final = [Hit(r + 1, 0.0, pool[i].unit) for r, i in enumerate(order[:10])]
            r = first_hit_rank(final, q["gold"])
            ranks.append(r)
            rows[q["split"]].append({"rank": r, "gold_at_10": gold_found_at(final, q["gold"], 10), "category": q["category"]})
        return summarise(rows["dev"] + rows["holdout"])["all"], ranks

    def baseline(self, mode: str) -> tuple[dict, list]:
        """jina alone (its own lane pipeline) or the current two-stage (jina cosine only, top-10)."""
        if mode == "two-stage":
            return self.evaluate(10, 1.0, 0.0)
        s = Searcher(self.jina, lexical=True, tests="auto")
        rows, ranks = [], []
        for qi, q in enumerate(self.queries):
            hits = s.search(self.jq[qi], q["query"], 10)
            r = first_hit_rank(hits, q["gold"])
            ranks.append(r)
            rows.append({"rank": r, "gold_at_10": gold_found_at(hits, q["gold"], 10), "category": q["category"]})
        return summarise(rows)["all"], ranks


def fmt(m: dict) -> str:
    return f"R@1 {m['recall@1']:.2f} R@5 {m['recall@5']:.2f} R@10 {m['recall@10']:.2f} MRR@10 {m['mrr@10']:.3f} goldR@10 {m['gold_recall@10']:.3f}"


def main() -> None:
    repos = {n: Repo(n) for n in REPOS}
    base = {n: r.baseline("jina") for n, r in repos.items()}
    two = {n: r.baseline("two-stage") for n, r in repos.items()}
    for n in repos:
        print(f"{n:10s} jina alone     {fmt(base[n][0])}")
        print(f"{n:10s} two-stage N=10 {fmt(two[n][0])}")
    for n in NS:
        print(f"\n=== pool N={n}")
        chosen = {}
        for fit, test in (("node-red", "browseros"), ("browseros", "node-red")):
            grid = {(b, g): repos[fit].evaluate(n, b, g)[0] for b, g in itertools.product(BETAS, GAMMAS)}
            best = max(grid, key=lambda k: (round(grid[k]["mrr@10"], 6), -list(grid).index(k)))
            chosen[fit] = best
            m, ranks = repos[test].evaluate(n, *best)
            b_m, b_ranks = base[test]
            better = sum(1 for r, b in zip(ranks, b_ranks, strict=True) if (r or 99) < (b or 99))
            worse = sum(1 for r, b in zip(ranks, b_ranks, strict=True) if (r or 99) > (b or 99))
            ok = m["mrr@10"] >= b_m["mrr@10"] and m["recall@10"] >= b_m["recall@10"]
            print(f"fit {fit:9s} best beta={best[0]} gamma={best[1]} (fit MRR {grid[best]['mrr@10']:.3f}) -> test {test:9s} {fmt(m)} | "
                  f"vs jina alone {better} better / {worse} worse | MRR>=jina & R@10>=jina: {ok}")
        a, b = chosen["node-red"], chosen["browseros"]
        stable = abs(BETAS.index(a[0]) - BETAS.index(b[0])) <= 1 and abs(GAMMAS.index(a[1]) - GAMMAS.index(b[1])) <= 1
        print(f"settings stable across directions: {stable}")


if __name__ == "__main__":
    main()
