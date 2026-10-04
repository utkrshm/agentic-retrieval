"""FusionSearcher: formula, candidate scope, gate and unit alignment (fake hashing encoders, no model)."""

from __future__ import annotations

import re
import zlib
from pathlib import Path

import numpy as np
import pytest

from coderet.index import FusionSearcher, RepoIndex, Searcher


class SaltedHashing:
    """Bag-of-words hashing vectors; the salt makes two instances behave like two different models."""

    def __init__(self, dim: int, salt: str) -> None:
        self.dim, self.salt = dim, salt

    def embed(self, texts: list[str], role: str) -> np.ndarray:
        out = np.full((len(texts), self.dim), 1e-3, dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]+", t.lower()):
                out[i, zlib.crc32((self.salt + w).encode()) % self.dim] += 1.0
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def _repo(tmp: Path) -> Path:
    (tmp / "lib").mkdir()
    (tmp / "test").mkdir()
    (tmp / "lib" / "app.js").write_text(
        "function installModule(name) {\n  if (blocked) { throw new Error('install_not_allowed'); }\n  return fetchModule(name);\n}\n\n"
        "function loadSettings() {\n  return readFile('settings');\n}\n\n"
        "function sendMail(to) {\n  return smtpSend(to);\n}\n")
    (tmp / "test" / "app_spec.js").write_text(
        "it('install module', function() {\n  installModule('x');\n  expect(blocked).toBe(false);\n});\n")
    return tmp


@pytest.fixture()
def parts(tmp_path):
    repo = _repo(tmp_path)
    gemma, jina = SaltedHashing(64, "g"), SaltedHashing(96, "j")
    gi = RepoIndex.build(repo, gemma, log=lambda *_: None)
    ji = RepoIndex.build(repo, jina, log=lambda *_: None)
    return gi, ji, gemma, jina


def test_blend_matches_the_formula_and_is_sorted(parts):
    gi, ji, gemma, jina = parts
    fs = FusionSearcher(gi, ji, gemma, jina, beta=0.7, gamma=0.25, pool=50, tests="include")
    res = fs.search("read the settings file", k=5)
    assert not res.gate_fired
    scores = [h.score for h in res.hits]
    assert scores == sorted(scores, reverse=True) and [h.rank for h in res.hits] == list(range(1, len(res.hits) + 1))
    gc = np.array([h.gemma for h in res.hits])
    jc = np.array([h.jina for h in res.hits])
    # recompute over all candidates the searcher saw (pool covers every unit here)
    allh = fs.search("read the settings file", k=50).hits
    g, j = np.array([h.gemma for h in allh]), np.array([h.jina for h in allh])
    mm = lambda x: (x - x.min()) / (x.max() - x.min())  # noqa: E731
    expected = 0.7 * mm(j) + 0.3 * mm(g)
    assert np.allclose(sorted(expected, reverse=True)[: len(scores)], scores, atol=1e-5)
    assert gc.shape == jc.shape and set(res.timings_ms) >= {"gemma_encode", "gemma_search", "jina_encode", "fuse", "total"}


def test_gate_adds_bm25_and_surfaces_the_exact_literal(parts):
    gi, ji, gemma, jina = parts
    fs = FusionSearcher(gi, ji, gemma, jina, tests="include")
    res = fs.search("where is install_not_allowed raised", k=3)
    assert res.gate_fired
    assert res.hits[0].unit.qualname == "installModule" and res.hits[0].unit.path == "lib/app.js"
    assert res.hits[0].bm25 is not None and res.hits[0].bm25 > 0
    plain = fs.search("send an email", k=3)
    assert not plain.gate_fired and all(h.bm25 is None for h in plain.hits)


def test_test_scope_default_override_and_flag(parts):
    gi, ji, gemma, jina = parts
    fs = FusionSearcher(gi, ji, gemma, jina)  # tests="auto"
    paths = lambda r: [h.unit.path for h in r.hits]  # noqa: E731
    r = fs.search("install module", k=10)
    assert all(not p.startswith("test/") for p in paths(r)) and not r.tests_included
    r = fs.search("which test covers install module", k=10)
    assert any(p.startswith("test/") for p in paths(r)) and r.tests_included
    r = fs.search("install module", k=10, tests="include")
    assert any(p.startswith("test/") for p in paths(r)) and r.tests_included
    r = fs.search("which test covers install module", k=10, tests="exclude")
    assert all(not p.startswith("test/") for p in paths(r)) and not r.tests_included
    with pytest.raises(ValueError):
        fs.search("x", tests="nope")


def test_mismatched_indexes_and_bad_weights_are_rejected(parts, tmp_path):
    gi, ji, gemma, jina = parts
    other = tmp_path / "other"
    other.mkdir()
    (other / "x.js").write_text("function only() {\n  return 1;\n}\n")
    oi = RepoIndex.build(other, jina, log=lambda *_: None)
    with pytest.raises(ValueError):
        FusionSearcher(gi, oi, gemma, jina)
    with pytest.raises(ValueError):
        FusionSearcher(gi, ji, gemma, jina, beta=1.5)
    with pytest.raises(ValueError):
        FusionSearcher(gi, ji, gemma, jina, gamma=1.0)


def test_searcher_per_call_override_matches_constructor_scope(parts):
    gi, _, gemma, _ = parts
    qv = gemma.embed(["install module"], "query")[0]
    a = Searcher(gi, lexical=False, tests="include").search(qv, "install module", 5)
    b = Searcher(gi, lexical=False, tests="exclude").search(qv, "install module", 5, tests="include")
    assert [(h.unit.path, h.unit.start_line) for h in a] == [(h.unit.path, h.unit.start_line) for h in b]
