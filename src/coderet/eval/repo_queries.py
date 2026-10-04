"""Hand-labelled repository queries: the hit rule and the metrics.

A retrieved unit hits a query if it is in the same file and its line range overlaps a gold range.
Metrics use "any gold unit" (several units can be valid answers, e.g. every definition and call site).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from coderet.index import Hit


def load_queries(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def overlaps(hit_path: str, hit_start: int, hit_end: int, gold: dict) -> bool:
    return hit_path == gold["path"] and hit_start <= gold["end"] and gold["start"] <= hit_end


def is_hit(hit: Hit, gold: list[dict]) -> bool:
    return any(overlaps(hit.unit.path, hit.unit.start_line, hit.unit.end_line, g) for g in gold)


def first_hit_rank(hits: list[Hit], gold: list[dict]) -> int | None:
    for h in hits:
        if is_hit(h, gold):
            return h.rank
    return None


def gold_found_at(hits: list[Hit], gold: list[dict], k: int) -> float:
    """Fraction of gold units that some top-k hit overlaps."""
    top = hits[:k]
    found = sum(any(overlaps(h.unit.path, h.unit.start_line, h.unit.end_line, g) for h in top) for g in gold)
    return found / len(gold)


def summarise(rows: list[dict]) -> dict:
    """rows: dicts with 'rank' (first hit rank or None), 'gold_at_10' and 'category'."""
    def block(rs: list[dict]) -> dict:
        n = len(rs)
        if n == 0:
            return {"n": 0}
        rr = lambda r: (1.0 / r["rank"]) if r["rank"] and r["rank"] <= 10 else 0.0
        at = lambda k: sum(1 for r in rs if r["rank"] and r["rank"] <= k) / n
        return {"n": n, "recall@1": round(at(1), 3), "recall@5": round(at(5), 3), "recall@10": round(at(10), 3),
                "mrr@10": round(sum(rr(r) for r in rs) / n, 3),
                "gold_recall@10": round(sum(r["gold_at_10"] for r in rs) / n, 3)}
    out = {"all": block(rows)}
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    out["by_category"] = {c: block(v) for c, v in sorted(by_cat.items())}
    return out
