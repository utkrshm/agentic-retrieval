"""RepoIndex, vector cache and repository-query evaluation, with a fake hashing encoder (no model)."""

from __future__ import annotations

import re
import zlib
from pathlib import Path

import numpy as np

from coderet.config import JINA_CODE
from coderet.eval.repo_queries import first_hit_rank, gold_found_at, overlaps, summarise
from coderet.index import Hit, RepoIndex, VectorCache, vector_key

DIM = 64


class HashingEncoder:
    """Bag-of-words hashing vectors: texts sharing words are close. Counts calls and texts."""

    def __init__(self) -> None:
        self.texts_embedded: list[str] = []

    def embed(self, texts: list[str], role: str) -> np.ndarray:
        self.texts_embedded.extend(texts)
        out = np.zeros((len(texts), DIM), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                out[i, zlib.crc32(w.encode()) % DIM] += 1.0
        out += 1e-3
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def _repo(tmp: Path) -> Path:
    (tmp / "a.py").write_text('def read_config(path):\n    """load the configuration file"""\n    return open(path).read()\n\n'
                              'def send_mail(to):\n    return smtp_send(to)\n')
    (tmp / "b.js").write_text("function retryRequest(fn) {\n  return fn().catch(() => fn());\n}\n")
    return tmp


def test_build_search_save_load_roundtrip(tmp_path: Path):
    repo = tmp_path / "repo"; repo.mkdir(); _repo(repo)
    enc = HashingEncoder()
    idx = RepoIndex.build(repo, enc, JINA_CODE, log=lambda *_: None)
    assert idx.meta["n_units"] == len(idx.units) >= 3 and idx.meta["dim"] == DIM
    q = enc.embed(["load the configuration file"], "query")
    best = idx.search(q, 3)[0]
    assert best[0].unit.qualname == "read_config" and best[0].rank == 1
    assert best[0].score >= best[1].score
    idx.save(tmp_path / "out")
    again = RepoIndex.load(tmp_path / "out")
    assert [u.qualname for u in again.units] == [u.qualname for u in idx.units]
    np.testing.assert_allclose(again.vectors, idx.vectors)
    assert again.search(q, 1)[0][0].unit.qualname == "read_config"


def test_vector_cache_reuses_unchanged_text_and_reembeds_changed_text(tmp_path: Path):
    repo = tmp_path / "repo"; repo.mkdir(); _repo(repo)
    cache = VectorCache(tmp_path / "cache")
    first = HashingEncoder()
    RepoIndex.build(repo, first, JINA_CODE, cache=cache, log=lambda *_: None)
    assert first.texts_embedded and len(cache) > 0
    second = HashingEncoder()
    RepoIndex.build(repo, second, JINA_CODE, cache=VectorCache(tmp_path / "cache"), log=lambda *_: None)
    assert second.texts_embedded == [], "identical texts must come from the cache"
    (repo / "b.js").write_text("function retryRequest(fn) {\n  // changed comment only\n  return fn().catch(() => fn());\n}\n")
    third = HashingEncoder()
    RepoIndex.build(repo, third, JINA_CODE, cache=VectorCache(tmp_path / "cache"), log=lambda *_: None)
    assert len(third.texts_embedded) == 1 and "changed comment only" in third.texts_embedded[0], \
        "a comment edit changes the embedded text, so exactly that unit is re-embedded"


def test_vector_key_depends_on_text_role_prompt_and_revision():
    from dataclasses import replace
    k = vector_key(JINA_CODE, "document", "x")
    assert k == vector_key(JINA_CODE, "document", "x")
    assert k != vector_key(JINA_CODE, "document", "y")
    assert k != vector_key(JINA_CODE, "query", "x")
    assert k != vector_key(replace(JINA_CODE, document_prompt="other"), "document", "x")
    assert k != vector_key(replace(JINA_CODE, revision="deadbeef"), "document", "x")
    assert k != vector_key(JINA_CODE, "document", "x", precision="int8")


def test_identical_code_in_two_files_stays_two_results(tmp_path: Path):
    repo = tmp_path / "repo"; repo.mkdir()
    for name in ("one.js", "two.js"):
        (repo / name).write_text("function same(x) {\n  return x + 1;\n}\n")
    idx = RepoIndex.build(repo, HashingEncoder(), JINA_CODE, log=lambda *_: None)
    assert {u.path for u in idx.units if u.qualname == "same"} == {"one.js", "two.js"}


def test_overlap_rule_and_metrics(tmp_path: Path):
    gold = [{"path": "a.js", "start": 10, "end": 20}]
    assert overlaps("a.js", 20, 30, gold[0]) and overlaps("a.js", 1, 10, gold[0])
    assert not overlaps("a.js", 21, 30, gold[0]) and not overlaps("b.js", 10, 20, gold[0])

    def unit(path, s, e):
        from coderet.chunking import Unit
        return Unit("javascript", path, "function", "f", "f", s, e, "", "x", "h")

    hits = [Hit(1, .9, unit("b.js", 1, 5)), Hit(2, .8, unit("a.js", 12, 15)), Hit(3, .7, unit("a.js", 1, 2))]
    assert first_hit_rank(hits, gold) == 2
    assert gold_found_at(hits, gold, 1) == 0.0 and gold_found_at(hits, gold, 3) == 1.0
    assert first_hit_rank(hits[:1], gold) is None
    m = summarise([{"rank": 1, "gold_at_10": 1.0, "category": "a"}, {"rank": 4, "gold_at_10": 1.0, "category": "a"},
                   {"rank": None, "gold_at_10": 0.0, "category": "b"}])
    assert m["all"]["n"] == 3 and m["all"]["recall@1"] == round(1 / 3, 3) and m["all"]["recall@5"] == round(2 / 3, 3)
    assert m["all"]["mrr@10"] == round((1 + 0.25) / 3, 3)
    assert m["by_category"]["b"]["recall@10"] == 0.0
