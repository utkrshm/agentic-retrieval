"""Vector cache keyed by the exact text that was embedded.

The key hashes everything that changes a vector: model id and revision, the role prompt, the
length limit, the precision, and the exact embedded text (path, kind, name and code). It is
deliberately NOT the syntax-tree hash of a unit: that hash ignores comments and the file path,
so a comment edit or a file move would silently reuse a stale vector.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from coderet.config import ModelSpec


def vector_key(spec: ModelSpec, role: str, text: str, precision: str = "fp32") -> str:
    prompt = spec.document_prompt if role == "document" else spec.query_prompt
    h = hashlib.sha256()
    h.update(f"{spec.hf_id}@{spec.revision}|{role}|{spec.max_seq_length}|{precision}|".encode())
    h.update(prompt.encode())
    h.update(b"\x00")
    h.update(text.encode())
    return h.hexdigest()[:32]


class VectorCache:
    """key -> float32 vector, persisted as keys.json + vectors.npy in one directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._vectors: dict[str, np.ndarray] = {}
        keys_path, vec_path = self.directory / "keys.json", self.directory / "vectors.npy"
        if keys_path.exists() and vec_path.exists():
            keys = json.loads(keys_path.read_text())
            mat = np.load(vec_path)
            if len(keys) == mat.shape[0]:
                self._vectors = {k: mat[i] for i, k in enumerate(keys)}

    def __len__(self) -> int:
        return len(self._vectors)

    def get(self, key: str) -> np.ndarray | None:
        return self._vectors.get(key)

    def put(self, key: str, vec: np.ndarray) -> None:
        self._vectors[key] = np.asarray(vec, dtype="float32")

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        keys = list(self._vectors)
        mat = np.vstack([self._vectors[k] for k in keys]) if keys else np.zeros((0, 0), dtype="float32")
        np.save(self.directory / "vectors.npy", mat)
        (self.directory / "keys.json").write_text(json.dumps(keys))
