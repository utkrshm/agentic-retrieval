"""Open a GitHub repository for the demo: clone it and index it on the CPU with both models.

A public GitHub link is cloned (shallow) and embedded with Gemma and jina-code on the CPU; vectors already in the on-disk
cache (keyed by the exact embedded text) cost nothing, new ones take roughly 0.2 s (Gemma) plus 0.45 s (jina-code) per unit
on a laptop CPU. ``progress`` exposes only the current step for a status line.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from coderet.chunking import MAX_CHARS, Unit, chunk_file, iter_source_files
from coderet.config import EMBEDDINGGEMMA, JINA_CODE, fingerprint
from coderet.index import FusionSearcher, RepoIndex, VectorCache

MIN_CHARS = 60
BATCH = 8
_GITHUB = re.compile(r"^(?:https?://)?(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?(?:[#?].*)?$")


class IndexingCancelled(Exception):
    pass


@dataclass(frozen=True)
class Progress:
    stage: str = "idle"  # idle | cloning | chunking | embedding | finishing | done | error | cancelled
    message: str = ""
    units: int = 0
    error: str = ""


@dataclass(frozen=True)
class IndexedRepo:
    key: str
    label: str  # owner/name
    commit: str
    units: int
    files: int
    seconds: float
    searcher: FusionSearcher = field(repr=False, compare=False)

    @property
    def github(self) -> str:
        return f"https://github.com/{self.label}"

    def file_url(self, path: str, start: int, end: int) -> str:
        return f"{self.github}/blob/{self.commit}/{path}#L{start}-L{end}"


def parse_github_url(text: str) -> tuple[str, str]:
    """(owner, name) of a GitHub repository link, or ValueError."""
    m = _GITHUB.match(text.strip())
    if not m:
        raise ValueError("Paste a GitHub repository link, for example https://github.com/owner/name")
    return m.group(1), m.group(2)


class RepositoryIndexer:
    """One job at a time; ``progress`` can be polled from another thread."""

    def __init__(self, gemma_backend, jina_backend, cache_root: Path = Path(".cache/vectors"),
                 repos_dir: Path = Path(".cache/repos")) -> None:
        self.gemma_backend, self.jina_backend = gemma_backend, jina_backend
        self.cache_root, self.repos_dir = cache_root, repos_dir
        self._progress = Progress()
        self._lock = threading.Lock()
        self._cancel = threading.Event()

    @property
    def progress(self) -> Progress:
        with self._lock:
            return self._progress

    def _set(self, **kw) -> None:
        with self._lock:
            self._progress = replace(self._progress, **kw)

    def cancel(self) -> None:
        self._cancel.set()

    def _check(self) -> None:
        if self._cancel.is_set():
            raise IndexingCancelled()

    # ---- clone and index ---------------------------------------------------------------------------
    def clone(self, owner: str, name: str) -> tuple[Path, str]:
        dest = self.repos_dir / f"{owner}__{name}"
        url = f"https://github.com/{owner}/{name}.git"
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        env = {"GIT_TERMINAL_PROMPT": "0", "PATH": os.environ.get("PATH", "")}
        if (dest / ".git").exists():
            r = subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "--quiet", "origin"], capture_output=True,
                               text=True, env=env, timeout=600)
            if r.returncode == 0:
                subprocess.run(["git", "-C", str(dest), "reset", "--hard", "--quiet", "FETCH_HEAD"], capture_output=True,
                               env=env, timeout=120)
            else:
                shutil.rmtree(dest)
        if not (dest / ".git").exists():
            r = subprocess.run(["git", "clone", "--depth", "1", "--quiet", url, str(dest)], capture_output=True, text=True,
                               env=env, timeout=900)
            if r.returncode != 0:
                last = (r.stderr.strip().splitlines() or ["git failed"])[-1]
                raise RuntimeError(f"Could not clone the repository (is it public?): {last}")
        commit = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        return dest, commit

    def index_checkout(self, repo: Path, label: str, commit: str, t_start: float) -> IndexedRepo:
        """Chunk and embed an existing checkout with both models."""
        self._set(stage="chunking", message="Reading the code")
        units: list[Unit] = []
        for path in iter_source_files(repo):
            self._check()
            file_units, _ = chunk_file(path, repo, MAX_CHARS, MIN_CHARS)
            units.extend(file_units)
        if not units:
            raise RuntimeError("No Python, JavaScript or TypeScript code found in this repository.")
        indexes = []
        for step, (spec, backend, who) in enumerate(((EMBEDDINGGEMMA, self.gemma_backend, "Gemma"),
                                                     (JINA_CODE, self.jina_backend, "jina-code")), 1):
            self._check()
            self._set(stage="embedding", units=len(units), message=f"Embedding {len(units):,} units with {who} ({step} of 2)")
            cache = VectorCache(self.cache_root / fingerprint(spec, kind="document", precision="fp32"))
            indexes.append(RepoIndex.build(repo, backend, spec, MAX_CHARS, cache, BATCH, log=lambda *_: None,
                                           min_chars=MIN_CHARS, units=units))
        self._set(stage="finishing", message="Opening the index")
        gi, ji = indexes
        searcher = FusionSearcher(gi, ji, self.gemma_backend, self.jina_backend)
        return IndexedRepo(key=label.lower().replace("/", "__"), label=label, commit=commit, units=len(units),
                           files=gi.meta["n_files"], seconds=time.perf_counter() - t_start, searcher=searcher)

    def index(self, link: str) -> IndexedRepo:
        """Open a GitHub repository; raises IndexingCancelled, ValueError or RuntimeError."""
        self._cancel.clear()
        t_start = time.perf_counter()
        self._progress = Progress(stage="cloning", message="Opening the repository")
        try:
            owner, name = parse_github_url(link)
            result = None
            if result is None:
                self._set(message="Cloning the repository")
                repo, commit = self.clone(owner, name)
                self._check()
                result = self.index_checkout(repo, f"{owner}/{name}", commit, t_start)
        except IndexingCancelled:
            self._set(stage="cancelled", message="Cancelled")
            raise
        except Exception as exc:
            self._set(stage="error", message="Could not open the repository", error=str(exc))
            raise
        self._set(stage="done", message="Ready", units=result.units)
        return result
