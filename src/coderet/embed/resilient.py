"""An encoder that survives a broken backend.

``ResilientEncoder`` holds an ordered list of candidate backends (fastest first, the boring
reference last). It activates the first one that builds AND passes the health probe, and on
any exception or invalid output at runtime it demotes the active backend for good and retries
the same batch on the next one. Nothing is returned to the caller unless it is finite,
unit-norm and the right shape. If every backend fails it raises ``EncoderUnavailable``.

Every decision is recorded in ``events`` so a demo can show what happened.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from coderet.embed.probe import DEFAULT_PROBE, Probe, check_probe, validate_vectors

log = logging.getLogger(__name__)


class Backend(Protocol):
    name: str

    def embed(self, texts: list[str], role: str) -> np.ndarray:
        """Return (len(texts), dim) float32, L2-normalised. role is 'query' or 'document'."""
        ...


@dataclass
class Candidate:
    name: str
    factory: Callable[[], Backend]
    min_cosine: float | None = None  # required cosine to the stored reference vectors, if any


class EncoderUnavailable(RuntimeError):
    """Every backend failed to build, failed the probe, or failed at runtime."""


class ResilientEncoder:
    def __init__(self, candidates: list[Candidate], *, probe: Probe = DEFAULT_PROBE,
                 reference: np.ndarray | None = None, dim: int | None = None) -> None:
        if not candidates:
            raise ValueError("at least one candidate backend is required")
        self._candidates = candidates
        self._probe = probe
        self._reference = reference
        self._dim = dim
        self._idx = 0
        self._backend: Backend | None = None
        self._errors: list[str] = []
        self.events: list[dict[str, str]] = []

    # ---- introspection ---------------------------------------------------
    @property
    def active_name(self) -> str | None:
        return self._candidates[self._idx].name if self._backend is not None else None

    def _event(self, kind: str, name: str, detail: str = "") -> None:
        self.events.append({"event": kind, "backend": name, "detail": detail})
        level = logging.INFO if kind == "activated" else logging.WARNING
        log.log(level, "encoder %s: %s %s", kind, name, detail)

    # ---- activation ------------------------------------------------------
    def _run_probe(self, backend: Backend) -> np.ndarray:
        by_role: dict[str, list[int]] = {}
        for i, (role, _) in enumerate(self._probe.items):
            by_role.setdefault(role, []).append(i)
        rows: dict[int, np.ndarray] = {}
        for role, idxs in by_role.items():
            vecs = backend.embed([self._probe.items[i][1] for i in idxs], role)
            if not isinstance(vecs, np.ndarray) or vecs.ndim != 2 or vecs.shape[0] != len(idxs):
                raise ValueError("probe returned a wrong-shaped result")
            for j, i in enumerate(idxs):
                rows[i] = vecs[j]
        return np.vstack([rows[i] for i in range(len(self._probe.items))]).astype("float32")

    def _ensure(self) -> Backend:
        while self._backend is None:
            if self._idx >= len(self._candidates):
                raise EncoderUnavailable("all embedding backends failed: " + "; ".join(self._errors))
            cand = self._candidates[self._idx]
            try:
                backend = cand.factory()
                vecs = self._run_probe(backend)
                result = check_probe(vecs, self._probe, self._reference, cand.min_cosine)
                if not result.ok:
                    raise ValueError(f"health probe failed: {result.reason}")
            except Exception as exc:  # noqa: BLE001 - any failure means "try the next backend"
                self._errors.append(f"{cand.name}: {exc!r}")
                self._event("rejected", cand.name, repr(exc))
                self._idx += 1
                continue
            self._backend = backend
            self._event("activated", cand.name)
        return self._backend

    def _demote(self, reason: str) -> None:
        name = self._candidates[self._idx].name
        self._errors.append(f"{name}: {reason}")
        self._event("demoted", name, reason)
        self._backend = None
        self._idx += 1

    # ---- use -------------------------------------------------------------
    def embed(self, texts: list[str], role: str) -> np.ndarray:
        """Embed texts; on a backend failure, retry the same batch on the next backend."""
        if not texts:
            return np.zeros((0, self._dim or 0), dtype="float32")
        while True:
            backend = self._ensure()
            try:
                vecs = backend.embed(texts, role)
                if not isinstance(vecs, np.ndarray) or vecs.ndim != 2 or vecs.shape[0] != len(texts):
                    raise ValueError("backend returned a wrong-shaped result")
                bad = validate_vectors(vecs, self._dim)
                if bad:
                    raise ValueError(f"invalid vectors: {bad}")
                return vecs.astype("float32", copy=False)
            except Exception as exc:  # noqa: BLE001
                self._demote(repr(exc))

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts, "query")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts, "document")
