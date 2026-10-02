import numpy as np
import pytrec_eval

from coderet.eval.bootstrap import paired_bootstrap
from coderet.eval.metrics import aggregate, by_length_bucket, per_query


def random_case(seed=0, nq=60, nd=150):
    rng = np.random.default_rng(seed)
    qrels = {f"q{i}": {f"d{j}": int(rng.integers(1, 3)) for j in rng.choice(nd, size=rng.integers(1, 4), replace=False)} for i in range(nq)}
    run = {q: {f"d{j}": float(rng.normal()) for j in range(nd)} for q in qrels}
    return run, qrels


def test_matches_pytrec_eval():
    run, qrels = random_case()
    ours = per_query(run, qrels)
    ref = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.10,100", "recip_rank"}).evaluate(run)
    for q in qrels:
        assert abs(ours[q]["ndcg@10"] - ref[q]["ndcg_cut_10"]) < 1e-9
        assert abs(ours[q]["recall@10"] - ref[q]["recall_10"]) < 1e-9
        assert abs(ours[q]["recall@100"] - ref[q]["recall_100"]) < 1e-9


def test_mrr_at_10_and_aggregate():
    qrels = {"a": {"x": 1}, "b": {"x": 1}, "c": {"x": 1}}
    run = {"a": {"x": 3, "y": 2}, "b": {"y": 3, "x": 2}, "c": {f"z{i}": 10 - i * 0.1 for i in range(10)} | {"x": 0}}
    pq = per_query(run, qrels)
    assert [pq[q]["mrr@10"] for q in "abc"] == [1.0, 0.5, 0.0]
    assert abs(aggregate(pq)["mrr@10"] - 0.5) < 1e-12
    assert by_length_bucket(pq, {"a": 100, "b": 300, "c": 5000}, metric="mrr@10") == {"<=256": (1, 1.0), "<=512": (1, 0.5), ">2048": (1, 0.0)}


def test_paired_bootstrap_detects_shift():
    a = {f"q{i}": 0.5 for i in range(200)}
    b = {f"q{i}": 0.5 + (0.1 if i % 2 else 0.0) for i in range(200)}
    r = paired_bootstrap(a, b)
    assert abs(r["mean_diff"] - 0.05) < 1e-12 and r["ci_low"] > 0
