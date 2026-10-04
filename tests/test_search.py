"""Lexical lane, test-file scope and Searcher (fake hashing encoder, no model)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from coderet.index import BM25, RepoIndex, Searcher, is_test_path, wants_tests
from coderet.index.lexical import identifier_anchors, quoted_literals, rrf, tokenize
from test_index import HashingEncoder


def test_tokenize_keeps_whole_identifier_and_subtokens():
    toks = tokenize("install_not_allowed getObjectProperty HTTPRequest")
    assert {"install_not_allowed", "install", "not", "allowed"} <= set(toks)
    assert {"getobjectproperty", "get", "object", "property"} <= set(toks)
    assert "httprequest" in toks


def test_anchors_only_for_code_shaped_tokens_and_quotes():
    assert identifier_anchors("where is install_not_allowed raised") == ["install_not_allowed"]
    assert identifier_anchors("how does editorTheme get read") == ["editortheme"]
    assert identifier_anchors("what does msg.payload hold") == ["msg.payload"]
    assert identifier_anchors("where is the error code raised for nodes") == []
    assert quoted_literals('where is the "Invalid SSH Key name" error thrown') == ["invalid ssh key name"]


def test_bm25_gate_needs_the_anchor_to_exist_in_the_corpus():
    bm = BM25(["def install_not_allowed(): pass", 'raise Error("Invalid SSH Key name")', "def other(): pass"])
    assert bm.has_anchor("where is install_not_allowed raised")
    assert bm.has_anchor('where is "invalid ssh key name" thrown')
    assert not bm.has_anchor("where is missing_identifier_here raised")
    assert not bm.has_anchor("how are connections retried")
    top = int(np.argmax(bm.scores("where is install_not_allowed raised")))
    assert top == 0
    assert int(np.argmax(bm.scores('where is the "Invalid SSH Key name" thrown'))) == 1


def test_rrf_prefers_ids_ranked_high_in_both_lists():
    fused = rrf([[1, 2, 3], [3, 1, 9]], k=60)
    assert [i for i, _ in fused][:2] == [1, 3]
    assert {i for i, _ in fused} == {1, 2, 3, 9}


def test_test_path_and_intent_detection():
    for p in ["test/unit/a.js", "pkg/tests/x.py", "src/a.test.ts", "src/a.spec.tsx", "editor/a_spec.js",
              "pkg/test_util.py", "pkg/util_test.py", "a/__tests__/b.js", "conftest.py"]:
        assert is_test_path(p), p
    for p in ["packages/node_modules/@node-red/util/lib/util.js", "src/latest.js", "src/testing/helper.ts",
              "src/contest.py"]:
        assert not is_test_path(p), p
    assert wants_tests("which test covers the retry logic") and wants_tests("is there a spec for X")
    assert not wants_tests("how are requests retried")


def _build(tmp: Path) -> RepoIndex:
    (tmp / "lib").mkdir()
    (tmp / "test").mkdir()
    (tmp / "lib" / "app.js").write_text(
        "function installModule(name) {\n  if (blocked) { throw new Error('install_not_allowed'); }\n  return fetchModule(name);\n}\n\n"
        "function loadSettings() {\n  return readFile('settings');\n}\n")
    (tmp / "test" / "app_spec.js").write_text(
        "it('install module', function() {\n  installModule('x');\n  expect(blocked).toBe(false);\n});\n")
    return RepoIndex.build(tmp, HashingEncoder(), log=lambda *_: None)


def test_searcher_filters_tests_before_top_k_and_auto_follows_the_query(tmp_path):
    index = _build(tmp_path)
    enc = HashingEncoder()

    def run(query: str, **kw) -> list[str]:
        s = Searcher(index, **kw)
        return [h.unit.path for h in s.search(enc.embed([query], "query")[0], query, k=5)]

    assert any(p.startswith("test/") for p in run("install module", lexical=False, tests="include"))
    assert all(not p.startswith("test/") for p in run("install module", lexical=False, tests="exclude"))
    assert all(not p.startswith("test/") for p in run("install module", lexical=False, tests="auto"))
    assert any(p.startswith("test/") for p in run("which test checks install module", lexical=False, tests="auto"))


def test_searcher_fuses_lexical_only_when_an_anchor_exists(tmp_path):
    index = _build(tmp_path)
    s = Searcher(index, tests="include")
    enc = HashingEncoder()
    q = "where is install_not_allowed raised"
    hits = s.search(enc.embed([q], "query")[0], q, k=3)
    assert hits[0].unit.qualname == "installModule" and hits[0].unit.path == "lib/app.js"
    assert [h.rank for h in hits] == [1, 2, 3]
    plain = "read the settings"
    dense = RepoIndex.search(index, enc.embed([plain], "query"), 3)[0]
    via = s.search(enc.embed([plain], "query")[0], plain, k=3)
    assert [h.unit.start_line for h in via] == [h.unit.start_line for h in dense]
    assert [round(h.score, 5) for h in via] == [round(h.score, 5) for h in dense]
