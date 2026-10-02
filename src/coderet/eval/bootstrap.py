"""Paired bootstrap confidence interval for the mean per-query difference B - A."""

from __future__ import annotations

import numpy as np


def paired_bootstrap(a: dict[str, float], b: dict[str, float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> dict[str, float]:
    qids = sorted(a.keys() & b.keys())
    diff = np.array([b[q] - a[q] for q in qids])
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi), "p_b_not_better": float((means <= 0).mean())}
