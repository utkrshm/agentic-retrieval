"""Sanity check for embedding output."""

from __future__ import annotations

import numpy as np


def validate_vectors(vecs: np.ndarray, dim: int | None = None, tol: float = 1e-3) -> str | None:
    """Return a reason string if the vectors are unusable, else None."""
    if not isinstance(vecs, np.ndarray) or vecs.ndim != 2:
        return "not a 2-D array"
    if dim is not None and vecs.shape[1] != dim:
        return f"dimension {vecs.shape[1]} != {dim}"
    if not np.isfinite(vecs).all():
        return "contains NaN or inf"
    norms = np.linalg.norm(vecs, axis=1)
    if np.any(norms < 1e-6):
        return "zero-norm vector"
    if np.any(np.abs(norms - 1.0) > tol):
        return f"not unit norm (min {norms.min():.4f}, max {norms.max():.4f})"
    return None
