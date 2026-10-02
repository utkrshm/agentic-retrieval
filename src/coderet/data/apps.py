"""CoIR APPS: corpus, queries, qrels, and grouped internal splits.

Internal dev/holdout splits are carved from the 5,000 APPS-*train* queries only;
the test split is reserved for the frozen MTEB submission run. Queries that are
the same problem (same source URL or identical normalised statement) land in the
same split. The full 8,765-document corpus stays searchable for every split.
"""

from __future__ import annotations

import ast
import hashlib
import random
import re
from dataclasses import dataclass
from functools import cache

from datasets import load_dataset

DATASET = "CoIR-Retrieval/apps"


@dataclass(frozen=True)
class Apps:
    corpus: dict[str, str]  # doc id -> code
    queries: dict[str, str]  # query id -> problem statement
    qrels: dict[str, dict[str, dict[str, int]]]  # split -> query id -> {doc id: relevance}
    query_group: dict[str, str]  # query id -> duplicate-group key


@dataclass(frozen=True)
class Splits:
    train: list[str]
    dev: list[str]
    holdout: list[str]


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _group_key(text: str, meta: str) -> str:
    try:
        url = ast.literal_eval(meta).get("url") or ""
    except (ValueError, SyntaxError):
        url = ""
    basis = url or _normalise(text)
    return hashlib.sha1(basis.encode()).hexdigest()


@cache
def load_apps() -> Apps:
    corpus_rows = load_dataset(DATASET, "corpus", split="corpus")
    query_rows = load_dataset(DATASET, "queries", split="queries")
    qrel_sets = load_dataset(DATASET, "default")

    corpus = {r["_id"]: r["text"] for r in corpus_rows}
    queries = {r["_id"]: r["text"] for r in query_rows}
    groups = {r["_id"]: _group_key(r["text"], r["meta_information"]) for r in query_rows}
    qrels: dict[str, dict[str, dict[str, int]]] = {}
    for split, rows in qrel_sets.items():
        table = qrels.setdefault(split, {})
        for r in rows:
            table.setdefault(r["query-id"], {})[r["corpus-id"]] = int(r["score"])
    return Apps(corpus, queries, qrels, groups)


def grouped_splits(apps: Apps, dev: int = 500, holdout: int = 500, seed: int = 13) -> Splits:
    """Split APPS-train queries by duplicate group into train / dev / holdout."""
    by_group: dict[str, list[str]] = {}
    for qid in sorted(apps.qrels["train"]):
        by_group.setdefault(apps.query_group[qid], []).append(qid)
    groups = sorted(by_group)
    random.Random(seed).shuffle(groups)

    buckets: dict[str, list[str]] = {"dev": [], "holdout": [], "train": []}
    for g in groups:
        target = "dev" if len(buckets["dev"]) < dev else "holdout" if len(buckets["holdout"]) < holdout else "train"
        buckets[target].extend(by_group[g])
    return Splits(**buckets)
