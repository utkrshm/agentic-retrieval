import numpy as np
from mteb.types import PromptType

from coderet.mteb_adapters import PrePostPipelineEncoder


class FakeEmbedder:
    name = "fake/embedder"

    def __init__(self):
        self.calls = []

    def embed(self, texts, role, batch_size):
        self.calls.append((list(texts), role))
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)


def test_pipeline_routes_roles_and_normalises():
    fake = FakeEmbedder()
    enc = PrePostPipelineEncoder(fake, preprocess=lambda ts, role: [f"{role}:{t}" for t in ts])
    batches = [{"text": ["ab", "abcd"]}]

    q = enc.encode(batches, task_metadata=None, hf_split="test", hf_subset="default", prompt_type=PromptType.query)
    d = enc.encode(batches, task_metadata=None, hf_split="test", hf_subset="default", prompt_type=PromptType.document)

    assert [role for _, role in fake.calls] == ["query", "document"]
    assert fake.calls[0][0] == ["query:ab", "query:abcd"]
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, rtol=1e-6)
    assert d.shape == (2, 2)
    assert enc.mteb_model_meta.name == "coderet/fake__embedder"
