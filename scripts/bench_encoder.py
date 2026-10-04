"""Compare embedding backends on real code: agreement with PyTorch fp32, and warm batch-1 latency.

    uv run python scripts/bench_encoder.py --repo /path/to/repo [--units 40] [--repeats 2] [--threads N]

Texts: short natural-language queries, plus real code units from the repository (as long
"queries" and as documents). Reports per backend and per bucket p50/p95 in ms and the cosine of
every vector to the PyTorch fp32 vector. Run on an otherwise idle machine; latency is
meaningless under load.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import random
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from coderet.chunking import chunk_file, embed_text, iter_source_files
from coderet.config import JINA_CODE
from coderet.embed import DIM, INT8_MIN_COSINE, FP32_MIN_COSINE
from coderet.embed.backends import OpenVinoBackend, TorchBackend
from coderet.embed.export import DEFAULT_ROOT, export_dir

SHORT_QUERIES = [
    "where is the bluetooth settings deeplink handled",
    "function that retries a failed http request with backoff",
    "how is user input normalised before it reaches the main handler",
    "register an event listener for device connected",
    "parse command line arguments and validate the port number",
    "which code reads the config file and merges defaults",
    "implementation of debounce for search input",
    "where do we close the database connection on shutdown",
    "convert a camelCase identifier to snake_case",
    "load plugin modules from a directory at startup",
    "send a message to the speech synthesis queue",
    "check whether a path is inside the workspace",
]


def lat_stats(ms: list[float]) -> dict[str, float]:
    ms = sorted(ms)
    return {"n": len(ms), "p50": round(statistics.median(ms), 1),
            "p95": round(ms[min(len(ms) - 1, int(0.95 * len(ms)))], 1)}


def time_backend(backend, role: str, texts: list[str], repeats: int) -> tuple[np.ndarray, dict[str, float]]:
    vecs = np.vstack([backend.embed([t], role) for t in texts])  # also warms up
    ms: list[float] = []
    for _ in range(repeats):
        for t in texts:
            s = time.perf_counter()
            backend.embed([t], role)
            ms.append((time.perf_counter() - s) * 1000)
    return vecs, lat_stats(ms)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--units", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--threads", type=int, default=None, help="OpenVINO threads (default: its own choice)")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    units = []
    for p in iter_source_files(args.repo.resolve()):
        units.extend(u for u in chunk_file(p, args.repo.resolve())[0] if u.kind in ("function", "method"))
    random.Random(3).shuffle(units)
    unit_texts = [embed_text(u) for u in units[: args.units]]
    buckets = {
        "short query": ("query", SHORT_QUERIES),
        "unit as long query": ("query", unit_texts[: args.units // 2]),
        "unit as document": ("document", unit_texts),
    }

    base = export_dir(JINA_CODE, args.root)
    # factories, so only one model is in memory at a time
    backends: dict[str, Callable[[], object]] = {"torch-fp32": lambda: TorchBackend(JINA_CODE, "cpu")}
    for prec in ("fp32", "int8"):
        d = base / prec
        if (d / "openvino_model.xml").exists():
            backends[f"openvino-{prec}"] = lambda d=d, prec=prec: OpenVinoBackend(JINA_CODE, d, prec, args.threads)

    print(f"cpu: {platform.processor() or platform.machine()} | units sampled: {len(unit_texts)} "
          f"(of {len(units)}) | repeats: {args.repeats} | openvino threads: {args.threads or 'default'}")
    results: dict[str, dict] = {}
    ref: dict[str, np.ndarray] = {}
    for name, make_backend in backends.items():
        backend = make_backend()
        results[name] = {}
        for bucket, (role, texts) in buckets.items():
            vecs, lat = time_backend(backend, role, texts, args.repeats)
            assert vecs.shape[1] == DIM
            if name == "torch-fp32":
                ref[bucket] = vecs
                results[name][bucket] = {**lat, "cos_min": 1.0, "cos_mean": 1.0}
            else:
                cos = np.sum(vecs * ref[bucket], axis=1)
                results[name][bucket] = {**lat, "cos_min": round(float(cos.min()), 5),
                                         "cos_mean": round(float(cos.mean()), 5)}
        del backend
        gc.collect()
    print(f"\n{'backend':14s} {'bucket':20s} {'p50 ms':>8s} {'p95 ms':>8s} {'speedup':>8s} {'cos min':>9s} {'cos mean':>9s}")
    for name, per in results.items():
        for bucket, r in per.items():
            sp = results["torch-fp32"][bucket]["p50"] / r["p50"]
            print(f"{name:14s} {bucket:20s} {r['p50']:8.1f} {r['p95']:8.1f} {sp:7.2f}x {r['cos_min']:9.5f} {r['cos_mean']:9.5f}")
    print(f"\nacceptance: int8 cos_min >= {INT8_MIN_COSINE}, fp32 cos_min >= {FP32_MIN_COSINE}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
