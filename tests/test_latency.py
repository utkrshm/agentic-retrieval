import numpy as np

from coderet.eval.latency import format_table, measure
from coderet.mteb_adapters import PrePostPipelineEncoder
from coderet.retrieval.dense import Retriever


class HashEmbedder:
    name = "test/hash"

    def embed(self, texts, role, batch_size):
        return np.array([[hash(t) % 97 + 1, len(t) + 1] for t in texts], dtype=np.float32)


def test_measure_reports_every_stage():
    r = Retriever.build(PrePostPipelineEncoder(HashEmbedder()), {f"d{i}": "x" * i for i in range(1, 50)})
    table = measure(r, [f"query {i}" for i in range(20)])
    assert set(table) == {"encode", "scan", "total", "cold_first_query"}
    assert all(m["p95"] >= m["p50"] >= 0 for m in table.values())
    assert "total" in format_table(table)
