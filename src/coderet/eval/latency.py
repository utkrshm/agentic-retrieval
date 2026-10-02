"""Batch-1 warm latency per stage and end-to-end, plus the hardware it ran on.

Percentiles are measured on each stage directly, never summed across stages.
"""

from __future__ import annotations

import os
import platform
import time
from collections.abc import Sequence

import numpy as np
import torch

from coderet.retrieval.dense import Retriever


def hardware() -> dict[str, str | int]:
    return {
        "cpu": platform.processor() or platform.machine(),
        "logical_cores": os.cpu_count() or 0,
        "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def measure(retriever: Retriever, queries: Sequence[str], warmup: int = 3, k: int = 10) -> dict[str, dict[str, float]]:
    """Cold first call, then warm p50/p95/mean per stage over `queries`."""
    t0 = time.perf_counter()
    retriever.search(queries[0], k)
    cold_ms = (time.perf_counter() - t0) * 1e3
    for q in queries[:warmup]:
        retriever.search(q, k)

    samples: dict[str, list[float]] = {}
    for q in queries:
        retriever.search(q, k)
        for stage, ms in retriever.last_timings_ms.items():
            samples.setdefault(stage, []).append(ms)
    table = {stage: _summary(v) for stage, v in samples.items()}
    table["cold_first_query"] = {"p50": cold_ms, "p95": cold_ms, "mean": cold_ms}
    return table


def thread_sweep(retriever: Retriever, queries: Sequence[str], threads: Sequence[int]) -> dict[int, dict[str, float]]:
    """End-to-end latency per torch thread count; restores the original setting."""
    original, out = torch.get_num_threads(), {}
    try:
        for n in threads:
            torch.set_num_threads(n)
            out[n] = measure(retriever, queries)["total"]
    finally:
        torch.set_num_threads(original)
    return out


def _summary(values: list[float]) -> dict[str, float]:
    a = np.asarray(values)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "mean": float(a.mean())}


def format_table(table: dict[str, dict[str, float]]) -> str:
    rows = [f"{'stage':18s} {'p50 ms':>9s} {'p95 ms':>9s} {'mean ms':>9s}"]
    rows += [f"{s:18s} {m['p50']:9.1f} {m['p95']:9.1f} {m['mean']:9.1f}" for s, m in table.items()]
    return "\n".join(rows)
