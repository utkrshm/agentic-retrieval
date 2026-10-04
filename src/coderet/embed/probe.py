"""A tiny fixed health probe for embedding backends.

A backend that loads fine can still return garbage (sentence-transformers' OpenVINO wrapper
returned NaN for every input of this model). The probe catches that at startup:

- every vector is finite with unit norm and the right dimension;
- semantics are plausible: each anchor is closer to its positive than to its negative;
- if a stored reference (made once from fp32 PyTorch at export time) exists, the backend's
  vectors must stay within a cosine threshold of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# (role, text). Indices are referenced by TRIPLES below.
PROBE_ITEMS: list[tuple[str, str]] = [
    ("query", "function that reverses a string"),                                   # 0
    ("document", "def reverse(s):\n    return s[::-1]"),                             # 1
    ("document", "def read_file(path):\n    with open(path) as f:\n        return f.read()"),  # 2
    ("query", "retry a failed http request with exponential backoff"),             # 3
    ("document", "async function retry(fn, tries) {\n  for (let i = 0; i < tries; i++) {\n"
                 "    try { return await fn(); } catch (e) { await sleep(2 ** i * 100); }\n  }\n}"),  # 4
    ("document", "function add(a, b) {\n  return a + b;\n}"),                       # 5
    ("query", "where is the config file loaded"),                                   # 6
    ("document", "const cfg = JSON.parse(fs.readFileSync(configPath, 'utf8'));"),   # 7
    ("document", "class Stack:\n    def __init__(self):\n        self.items = []\n"
                 "    def push(self, x):\n        self.items.append(x)"),            # 8
]

# (anchor, positive, negative): the anchor must score higher with the positive.
TRIPLES: list[tuple[int, int, int]] = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]


@dataclass(frozen=True)
class Probe:
    items: list[tuple[str, str]]
    triples: list[tuple[int, int, int]]


DEFAULT_PROBE = Probe(PROBE_ITEMS, TRIPLES)


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    reason: str = ""
    min_cosine: float | None = None


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


def check_probe(vecs: np.ndarray, probe: Probe = DEFAULT_PROBE, reference: np.ndarray | None = None,
                min_cosine: float | None = None) -> ProbeResult:
    """Validate probe vectors (in probe.items order) against sanity checks and a reference."""
    bad = validate_vectors(vecs)
    if bad:
        return ProbeResult(False, bad)
    if vecs.shape[0] != len(probe.items):
        return ProbeResult(False, f"expected {len(probe.items)} probe vectors, got {vecs.shape[0]}")
    for a, p, n in probe.triples:
        if not float(vecs[a] @ vecs[p]) > float(vecs[a] @ vecs[n]):
            return ProbeResult(False, f"semantic check failed for probe triple {(a, p, n)}")
    if reference is not None and min_cosine is not None:
        if reference.shape != vecs.shape:
            return ProbeResult(False, "reference shape mismatch")
        cos = np.sum(vecs * reference, axis=1)
        if float(cos.min()) < min_cosine:
            return ProbeResult(False, f"cosine to reference {cos.min():.4f} < {min_cosine}", float(cos.min()))
        return ProbeResult(True, "", float(cos.min()))
    return ProbeResult(True)


def save_reference(path: Path, vecs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs.astype("float32"))
    path.with_suffix(".json").write_text(json.dumps({"items": len(PROBE_ITEMS)}))


def load_reference(path: Path) -> np.ndarray | None:
    try:
        ref = np.load(path)
    except (OSError, ValueError):
        return None
    return ref if ref.shape[0] == len(PROBE_ITEMS) else None
