"""Baseline bakeoff on the internal dev split (APPS-train derived, full corpus searchable).

Documents and dev queries are encoded once on GPU (offline) and cached per run
fingerprint. Query latency is then measured on CPU, batch 1, against the cached
doc matrix: the serving setup.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub.errors import GatedRepoError

from coderet.config import RunConfig, get_model
from coderet.data.apps import grouped_splits, load_apps
from coderet.eval.latency import hardware, measure, thread_sweep
from coderet.eval.metrics import aggregate, by_length_bucket, per_query
from coderet.models.encoders import SentenceTransformerEmbedder
from coderet.mteb_adapters import PrePostPipelineEncoder
from coderet.retrieval.dense import DenseIndex, Retriever

DEFAULT_MODELS = ["embeddinggemma-300m", "jina-code-0.5b", "qwen3-0.6b"]


def cached_matrix(path: Path, compute) -> np.ndarray:
    if path.exists():
        return np.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = compute()
    np.save(path, m)
    return m


def run_model(key: str, split: str, latency_queries: int, batch_size: int, sweep: bool) -> dict | None:
    run = RunConfig(get_model(key))
    apps = load_apps()
    qids = getattr(grouped_splits(apps), split)
    doc_ids = list(apps.corpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        gpu = PrePostPipelineEncoder(SentenceTransformerEmbedder(run.model, device=device))
    except (GatedRepoError, OSError) as e:
        if run.model.gated:
            print(f"[skip] {key}: gated model not accessible ({type(e).__name__}). {run.model.notes}")
            return None
        raise

    cache = run.cache_dir()
    docs = cached_matrix(cache / "docs.npy", lambda: gpu.encode_texts([apps.corpus[d] for d in doc_ids], "document", batch_size))
    queries = cached_matrix(cache / f"queries-{split}.npy", lambda: gpu.encode_texts([apps.queries[q] for q in qids], "query", batch_size))
    del gpu
    torch.cuda.empty_cache()

    index = DenseIndex(doc_ids, docs)
    hits = index.search(queries, k=100)
    run_scores = {q: dict(h) for q, h in zip(qids, hits)}
    qrels = {q: apps.qrels["train"][q] for q in qids}
    pq = per_query(run_scores, qrels)
    lengths = {q: len(apps.queries[q].split()) for q in qids}

    cpu = Retriever(PrePostPipelineEncoder(SentenceTransformerEmbedder(run.model, device="cpu")), index)
    sample = [apps.queries[q] for q in qids[:latency_queries]]
    latency = measure(cpu, sample)
    result = {
        "model": key,
        "fingerprint": run.fingerprint(),
        "split": split,
        "n_queries": len(qids),
        "metrics": aggregate(pq),
        "ndcg_by_query_words": by_length_bucket(pq, lengths),
        "latency_ms_cpu": latency,
        "hardware": hardware(),
        "per_query": pq,
    }
    if sweep:
        result["thread_sweep_total_ms"] = thread_sweep(cpu, sample[:20], [1, 2, 4, 8])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--split", default="dev", choices=["dev", "holdout"])
    parser.add_argument("--latency-queries", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sweep", action="store_true", help="thread sweep (use on the winner)")
    parser.add_argument("--out", type=Path, default=Path("results/bakeoff.json"))
    args = parser.parse_args()

    results = {}
    if args.out.exists():
        results = json.loads(args.out.read_text())
    for key in args.models:
        r = run_model(key, args.split, args.latency_queries, args.batch_size, args.sweep)
        if r is None:
            continue
        results[f"{key}/{args.split}"] = r
        m, lat = r["metrics"], r["latency_ms_cpu"]["total"]
        print(f"{key:20s} ndcg@10={m['ndcg@10']:.4f} mrr@10={m['mrr@10']:.4f} r@100={m['recall@100']:.4f} "
              f"cpu p50={lat['p50']:.0f}ms p95={lat['p95']:.0f}ms")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))
        with open("experiments/EXPERIMENTS.md", "a") as f:
            f.write(f"| {dt.date.today()} | {r['fingerprint']} | {key} | {args.split} | {m['ndcg@10']*100:.2f} | "
                    f"{m['mrr@10']*100:.2f} | {m['recall@100']*100:.2f} | {lat['p50']:.0f} / {lat['p95']:.0f} | bakeoff.py |\n")


if __name__ == "__main__":
    main()
