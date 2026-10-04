"""Embedding backend: fp32 PyTorch through sentence-transformers.

Applies the model's role prompt as a text prefix and returns L2-normalised float32 vectors. The checkpoints ship as
bfloat16 (jina-code) or do not support fp16 (Gemma), so the model is always loaded in fp32.
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
