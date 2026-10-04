"""Search engine behind the Reflex demo (main.py): the settled Gemma, jina and BM25 pipeline, all on CPU.

Both models load once (fp32 PyTorch, batch 1 for queries, no GPU). A user pastes a GitHub link, ``Engine.index`` clones and
indexes it on the CPU (``Engine.progress`` can be polled), and ``Engine.search`` answers questions over any repository
indexed in this session. Results are JSON-friendly dicts so the UI state can hold them directly.
"""

from __future__ import annotations

import threading
from pathlib import Path

from coderet.demo.indexer import IndexedRepo, Progress, RepositoryIndexer
from coderet.index import FusionHit, FusionResult

SNIPPET_LINES = 14
_LANG = {".py": "python", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "jsx",
         ".ts": "typescript", ".mts": "typescript", ".cts": "typescript", ".tsx": "tsx"}


def to_card(hit: FusionHit, repo: IndexedRepo) -> dict:
    """One result as plain strings and numbers for the UI."""
    u = hit.unit
    lines = u.text.split("\n")
    shown = lines[:SNIPPET_LINES]
    directory, _, name = u.path.rpartition("/")
    return {
        "rank": hit.rank, "path": u.path, "dir": directory + "/" if directory else "", "file": name,
        "start": u.start_line, "end": u.end_line, "kind": u.kind, "name": u.qualname or u.name or "(module level)",
        "lang": _LANG.get(Path(u.path).suffix.lower(), "text"),
        "code": "\n".join(shown), "hidden": max(0, len(lines) - len(shown)),
        "url": repo.file_url(u.path, u.start_line, u.end_line),
    }


class Engine:
    def __init__(self, cache_root: Path = Path(".cache/vectors"), repos_dir: Path = Path(".cache/repos")) -> None:
        self.cache_root, self.repos_dir = cache_root, repos_dir
        self._lock = threading.Lock()  # one encoder pair: serialise indexing and searching
        self._load_lock = threading.Lock()
        self.repos: dict[str, IndexedRepo] = {}
        self.indexer: RepositoryIndexer | None = None
        self.ready = False

    def load_models(self) -> None:
        with self._load_lock:
            if self.ready:
                return
            from coderet.config import EMBEDDINGGEMMA, JINA_CODE
            from coderet.embed.backends import TorchBackend

            gemma = TorchBackend(EMBEDDINGGEMMA, "cpu", batch_size=8)
            jina = TorchBackend(JINA_CODE, "cpu", batch_size=8)
            for be in (gemma, jina):  # warm up so the first real query is not the slow one
                be.embed(["warm up"], "query")
            self.indexer = RepositoryIndexer(gemma, jina, self.cache_root, self.repos_dir)
            self.ready = True

    @property
    def progress(self) -> Progress:
        return self.indexer.progress if self.indexer else Progress(stage="idle", message="Loading models")

    def cancel(self) -> None:
        if self.indexer:
            self.indexer.cancel()

    def index(self, link: str) -> IndexedRepo:
        if not self.ready or self.indexer is None:
            raise RuntimeError("the models are still loading")
        with self._lock:
            repo = self.indexer.index(link)
        self.repos[repo.key] = repo
        return repo

    def search(self, repo_key: str, query: str, tests: str = "auto", k: int = 10) -> dict:
        repo = self.repos[repo_key]
        with self._lock:
            res: FusionResult = repo.searcher.search(query, k=k, tests=tests)
        return {
            "cards": [to_card(h, repo) for h in res.hits], "gate": res.gate_fired, "tests_included": res.tests_included,
            "candidates": res.n_candidates, "timings": {k_: round(v) for k_, v in res.timings_ms.items()},
        }
