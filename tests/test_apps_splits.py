from coderet.data.apps import Apps, _group_key, grouped_splits


def make_apps(n: int, dup_every: int) -> Apps:
    qids = [f"q{i}" for i in range(n)]
    # every `dup_every`-th query repeats the previous problem's URL
    urls = {q: f"u{i - 1 if i % dup_every == 0 and i else i}" for i, q in enumerate(qids)}
    groups = {q: _group_key("stmt", repr({"url": urls[q]})) for q in qids}
    qrels = {"train": {q: {f"d{q}": 1} for q in qids}}
    return Apps(corpus={}, queries={}, qrels=qrels, query_group=groups)


def test_splits_are_disjoint_complete_and_group_pure():
    apps = make_apps(200, dup_every=5)
    s = grouped_splits(apps, dev=30, holdout=30)
    assert set(s.train) | set(s.dev) | set(s.holdout) == set(apps.qrels["train"])
    assert not (set(s.train) & set(s.dev) or set(s.train) & set(s.holdout) or set(s.dev) & set(s.holdout))
    for a, b in [(s.dev, s.train), (s.holdout, s.train), (s.dev, s.holdout)]:
        assert not {apps.query_group[q] for q in a} & {apps.query_group[q] for q in b}
    assert 30 <= len(s.dev) <= 31


def test_group_key_falls_back_to_normalised_statement():
    assert _group_key("A  b\nC", "{'url': ''}") == _group_key("a b c", "not a dict")
