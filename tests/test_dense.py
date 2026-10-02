import numpy as np

from coderet.retrieval.dense import DenseIndex


def test_topk_matches_full_sort():
    rng = np.random.default_rng(0)
    docs = rng.normal(size=(500, 16)).astype(np.float32)
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    q = rng.normal(size=(3, 16)).astype(np.float32)
    index = DenseIndex([f"d{i}" for i in range(500)], docs)
    hits = index.search(q, k=10)
    for row, h in zip(q @ docs.T, hits):
        assert [d for d, _ in h] == [f"d{i}" for i in np.argsort(-row)[:10]]
    assert len(index.search(q[0], k=10_000)[0]) == 500
