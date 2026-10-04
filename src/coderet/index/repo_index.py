"""An exact (flat inner-product) FAISS index over the tree-sitter units of one repository checkout.

Vectors are L2-normalised, so inner product is cosine similarity. One index describes one
commit; there is no per-commit manifest yet.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np

from coderet.chunking import MAX_CHARS, Unit, chunk_file, embed_text, iter_source_files
from coderet.config import JINA_CODE, ModelSpec, fingerprint
from coderet.embed.probe import validate_vectors
from coderet.index.cache import VectorCache, vector_key


class DocumentEncoder(Protocol):
    def embed(self, texts: list[str], role: str) -> np.ndarray: ...


@dataclass(frozen=True)
class Hit:
    rank: int  # 1-based
    score: float
    unit: Unit


def _git_head(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class RepoIndex:
    def __init__(self, meta: dict, units: list[Unit], vectors: np.ndarray) -> None:
        if len(units) != vectors.shape[0]:
            raise ValueError(f"{len(units)} units but {vectors.shape[0]} vectors")
        bad = validate_vectors(vectors)
        if bad:
            raise ValueError(f"invalid vectors: {bad}")
        self.meta, self.units, self.vectors = meta, units, np.ascontiguousarray(vectors, dtype="float32")
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)

    # ---- build -----------------------------------------------------------
    @classmethod
    def build(cls, repo: Path, encoder: DocumentEncoder, spec: ModelSpec = JINA_CODE,
              max_chars: int = MAX_CHARS, cache: VectorCache | None = None, batch: int = 64,
              log=print, min_chars: int = 0, signature_header: bool = False,
              units: list[Unit] | None = None) -> RepoIndex:
        """Chunk (unless ``units`` is given, e.g. shared between two models) and embed a checkout."""
        t0 = time.perf_counter()
        repo = repo.resolve()
        n_files = 0
        if units is None:
            units = []
            for path in iter_source_files(repo):
                file_units, _ = chunk_file(path, repo, max_chars, min_chars)
                units.extend(file_units)
                n_files += bool(file_units)
        else:
            n_files = len({u.path for u in units})
        texts = [embed_text(u, signature_header) for u in units]
        keys = [vector_key(spec, "document", t) for t in texts]
        have = {k: cache.get(k) for k in set(keys)} if cache is not None else {}
        have = {k: v for k, v in have.items() if v is not None}
        todo = sorted({k: i for i, k in enumerate(keys) if k not in have}.items(),
                      key=lambda kv: len(texts[kv[1]]))  # similar lengths together: less padding
        log(f"{len(units)} units from {n_files} files; {len(have)} cached vectors, {len(todo)} to embed")
        for s in range(0, len(todo), batch):
            chunk = todo[s : s + batch]
            vecs = encoder.embed([texts[i] for _, i in chunk], "document")
            for (k, _), v in zip(chunk, vecs, strict=True):
                have[k] = v
                if cache is not None:
                    cache.put(k, v)
            if (s // batch) % 10 == 0:
                log(f"  embedded {min(s + batch, len(todo))}/{len(todo)}  ({time.perf_counter() - t0:.0f}s)")
        if cache is not None:
            cache.save()
        vectors = np.vstack([have[k] for k in keys])
        meta = {
            "repo": repo.name, "path": str(repo), "commit": _git_head(repo), "model": spec.hf_id,
            "revision": spec.revision, "fingerprint": fingerprint(spec, kind="document", precision="fp32"),
            "max_chars": max_chars, "min_chars": min_chars,
            "signature_header": signature_header, "n_units": len(units), "n_files": n_files, "dim": int(vectors.shape[1]),
            "embedded_now": len(todo), "from_cache": len(keys) - len(todo),
            "build_seconds": round(time.perf_counter() - t0, 1), "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return cls(meta, units, vectors)

    # ---- persistence -----------------------------------------------------
    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "meta.json").write_text(json.dumps(self.meta, indent=1))
        with (directory / "units.jsonl").open("w") as f:
            for u in self.units:
                f.write(json.dumps(asdict(u)) + "\n")
        np.save(directory / "vectors.npy", self.vectors)
        faiss.write_index(self.index, str(directory / "index.faiss"))

    @classmethod
    def load(cls, directory: Path) -> RepoIndex:
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        units = [Unit(**json.loads(line)) for line in (directory / "units.jsonl").read_text().splitlines()]
        idx = cls(meta, units, np.load(directory / "vectors.npy"))
        stored = faiss.read_index(str(directory / "index.faiss"))
        if stored.ntotal != len(units):
            raise ValueError("index.faiss does not match units.jsonl")
        return idx

    # ---- search ----------------------------------------------------------
    def search(self, query_vectors: np.ndarray, k: int = 10) -> list[list[Hit]]:
        q = np.ascontiguousarray(query_vectors, dtype="float32")
        scores, ids = self.index.search(q, min(k, len(self.units)))
        return [[Hit(r + 1, float(s), self.units[i]) for r, (s, i) in enumerate(zip(row_s, row_i, strict=True)) if i >= 0]
                for row_s, row_i in zip(scores, ids, strict=True)]
