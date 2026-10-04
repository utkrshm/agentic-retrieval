"""PrePostPipelineEncoder with a fake embedder (no model, no dataset, no GPU)."""

from __future__ import annotations

import numpy as np
from mteb.types import PromptType

from coderet.config import JINA_CODE, fingerprint
from coderet.mteb_adapters import PrePostPipelineEncoder, l2_normalize


class RecordingEmbedder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def embed(self, texts: list[str], role: str) -> np.ndarray:
        self.calls.append((list(texts), role))
        return np.array([[float(len(t)), 1.0, 0.0] for t in texts], dtype="float64")


def _encode(enc: PrePostPipelineEncoder, batches, prompt_type):
    return enc.encode(batches, task_metadata=None, hf_split="test", hf_subset="default", prompt_type=prompt_type)


def test_roles_follow_the_prompt_type_and_batches_are_flattened_in_order():
    emb = RecordingEmbedder()
    enc = PrePostPipelineEncoder(emb, "m")
    batches = [{"text": ["a", "bb"]}, {"text": ["ccc"]}]
    out_q = _encode(enc, batches, PromptType.query)
    out_d = _encode(enc, batches, PromptType.document)
    out_none = _encode(enc, batches, None)
    assert emb.calls[0] == (["a", "bb", "ccc"], "query")
    assert emb.calls[1] == (["a", "bb", "ccc"], "document")
    assert emb.calls[2][1] == "document", "no prompt type means document"
    assert out_q.shape == (3, 3) and out_q.dtype == np.float32
    np.testing.assert_allclose(out_q, out_d)
    assert out_none.shape == (3, 3)


def test_default_postprocess_normalises_to_unit_length():
    enc = PrePostPipelineEncoder(RecordingEmbedder(), "m")
    out = _encode(enc, [{"text": ["a", "bbbb"]}], PromptType.document)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_hooks_run_in_order_and_receive_the_role():
    seen: list[str] = []

    def pre(texts, role):
        seen.append(f"pre:{role}")
        return [t.upper() for t in texts]

    def post(vecs, role):
        seen.append(f"post:{role}")
        return vecs * 2

    emb = RecordingEmbedder()
    enc = PrePostPipelineEncoder(emb, "m", preprocess=pre, postprocess=post)
    out = _encode(enc, [{"text": ["ab"]}], PromptType.query)
    assert seen == ["pre:query", "post:query"]
    assert emb.calls[0][0] == ["AB"]
    assert out[0, 1] == 2.0


def test_model_meta_carries_name_and_revision():
    enc = PrePostPipelineEncoder(RecordingEmbedder(), "org/model", revision="abc123")
    assert enc.mteb_model_meta.name == "coderet/org__model"
    assert enc.mteb_model_meta.revision == "abc123"


def test_l2_normalize_handles_zero_vectors():
    out = l2_normalize(np.zeros((1, 3), dtype="float32"), "document")
    assert np.isfinite(out).all()


def test_fingerprint_changes_with_every_setting_and_is_stable():
    a = fingerprint(JINA_CODE, backend="torch", precision="fp32")
    assert a == fingerprint(JINA_CODE, backend="torch", precision="fp32")
    assert a != fingerprint(JINA_CODE, backend="cpu", precision="fp32")
    assert a != fingerprint(JINA_CODE, backend="torch", precision="bf16")
    from dataclasses import replace
    assert a != fingerprint(replace(JINA_CODE, query_prompt="x"), backend="torch", precision="fp32")
