"""Demo backend: link parsing and CPU indexing of a checkout (fake hashing encoders, no model)."""

from __future__ import annotations

from pathlib import Path

import pytest

from coderet.demo import IndexingCancelled, RepositoryIndexer, parse_github_url
from test_fusion import SaltedHashing, _repo


def _src(tmp: Path) -> Path:
    (tmp / "src").mkdir()
    return _repo(tmp / "src")


def _indexer(tmp: Path) -> RepositoryIndexer:
    return RepositoryIndexer(SaltedHashing(64, "g"), SaltedHashing(96, "j"), tmp / "vectors", tmp / "repos")


def test_parse_github_url_accepts_common_forms_and_rejects_others():
    for text in ["https://github.com/browseros-ai/BrowserOS", "https://github.com/browseros-ai/BrowserOS/",
                 "github.com/browseros-ai/BrowserOS.git", "  https://www.github.com/browseros-ai/BrowserOS#readme "]:
        assert parse_github_url(text) == ("browseros-ai", "BrowserOS")
    for text in ["", "BrowserOS", "https://gitlab.com/a/b", "https://github.com/onlyowner", "git@github.com:a/b.git"]:
        with pytest.raises(ValueError):
            parse_github_url(text)


def test_a_checkout_is_chunked_and_embedded_by_both_models(tmp_path):
    repo = _src(tmp_path)
    ix = _indexer(tmp_path)
    got = ix.index_checkout(repo, "owner/name", "abc123", 0.0)
    assert got.units >= 3 and got.files == 2
    res = got.searcher.search("where is install_not_allowed raised", k=2, tests="include")
    assert res.hits[0].unit.qualname == "installModule"
    assert ix.progress.stage == "finishing"
    # a second run is served from the vector cache: nothing new to embed
    again = ix.index_checkout(repo, "owner/name", "abc123", 0.0)
    assert again.units == got.units


def test_empty_checkout_and_cancel_are_reported(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "README.md").write_text("no code here")
    ix = _indexer(tmp_path)
    with pytest.raises(RuntimeError, match="No Python, JavaScript or TypeScript"):
        ix.index_checkout(empty, "o/e", "x", 0.0)
    repo = _src(tmp_path)
    ix.cancel()
    with pytest.raises(IndexingCancelled):
        ix.index_checkout(repo, "o/n", "x", 0.0)


def test_result_cards_are_plain_values_and_link_to_the_indexed_commit(tmp_path):
    from coderet.demo import to_card

    repo = _src(tmp_path)
    got = _indexer(tmp_path).index_checkout(repo, "node-red/node-red", "cd05a9a38b", 0.0)
    res = got.searcher.search("where is install_not_allowed raised", k=2, tests="include")
    card = to_card(res.hits[0], got)
    assert card["rank"] == 1 and card["path"].endswith("app.js") and card["lang"] == "javascript"
    assert card["url"].startswith("https://github.com/node-red/node-red/blob/cd05a9a38b/") and "#L" in card["url"]
    assert isinstance(card["code"], str) and card["hidden"] >= 0
    assert all(isinstance(v, (str, int)) for v in card.values())
