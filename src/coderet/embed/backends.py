"""Concrete embedding backends for jina-code: PyTorch fp32 (reference) and OpenVINO (fast).

Both apply the model's role prompt as a text prefix, use last-token pooling and return
L2-normalised float32 vectors, so they are interchangeable. The OpenVINO backend runs the
exported graph through the raw OpenVINO runtime with our own tokenisation and pooling:
sentence-transformers' OpenVINO wrapper returned NaN for every input of this model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from coderet.config import ModelSpec


def l2_normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype="float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _prefix(spec: ModelSpec, role: str) -> str:
    if role == "query":
        return spec.query_prompt
    if role == "document":
        return spec.document_prompt
    raise ValueError(f"role must be 'query' or 'document', got {role!r}")


class TorchBackend:
    """Reference backend: sentence-transformers, forced fp32 (the checkpoint ships bf16)."""

    def __init__(self, spec: ModelSpec, device: str = "cpu", batch_size: int = 16) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.batch_size = batch_size
        self.name = f"torch-fp32-{device}"
        self._model = SentenceTransformer(
            spec.hf_id, revision=spec.revision, device=device, trust_remote_code=True,
            model_kwargs={"dtype": torch.float32},
        )
        self._model.max_seq_length = spec.max_seq_length

    def embed(self, texts: list[str], role: str) -> np.ndarray:
        prefix = _prefix(self.spec, role)
        vecs = self._model.encode(
            [prefix + t for t in texts], batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=False, show_progress_bar=False,
        )
        return l2_normalise(vecs)


class OpenVinoBackend:
    """Fast CPU backend: an exported (optionally int8-weight-compressed) OpenVINO graph.

    Texts are embedded one at a time (batch 1: the serving case, and no padding concerns).
    """

    def __init__(self, spec: ModelSpec, model_dir: Path, precision: str = "int8",
                 threads: int | None = None) -> None:
        import openvino as ov
        from transformers import AutoTokenizer

        xml = Path(model_dir) / "openvino_model.xml"
        if not xml.exists():
            raise FileNotFoundError(f"no exported OpenVINO model at {xml}")
        self.spec = spec
        self.name = f"openvino-{precision}"
        self._tok = AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision, trust_remote_code=True)
        props: dict[str, object] = {"PERFORMANCE_HINT": "LATENCY"}
        if threads:
            props["INFERENCE_NUM_THREADS"] = threads
        self._compiled = ov.Core().compile_model(str(xml), "CPU", props)

    def _embed_one(self, text: str) -> np.ndarray:
        enc = self._tok(text, truncation=True, max_length=self.spec.max_seq_length, return_tensors="np")
        ids = enc["input_ids"].astype("int64")
        mask = enc["attention_mask"].astype("int64")
        pos = np.arange(ids.shape[1], dtype="int64")[None, :]
        hidden = self._compiled({"input_ids": ids, "attention_mask": mask, "position_ids": pos})[0]
        return hidden[0, -1]  # last-token pooling (single sequence, no padding)

    def embed(self, texts: list[str], role: str) -> np.ndarray:
        prefix = _prefix(self.spec, role)
        return l2_normalise(np.vstack([self._embed_one(prefix + t) for t in texts]))
