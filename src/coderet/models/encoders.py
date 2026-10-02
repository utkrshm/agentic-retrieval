"""Embedders: concrete models behind the `Embedder` protocol."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from coderet.config import ModelSpec
from coderet.mteb_adapters.encoder import Role


class SentenceTransformerEmbedder:
    """A pinned sentence-transformers checkpoint with role-specific prompts.

    Vectors are returned raw; normalisation is the pipeline's postprocess stage.
    """

    def __init__(self, spec: ModelSpec, device: str | None = None) -> None:
        self.spec = spec
        self.name = spec.hf_id
        self.model = SentenceTransformer(
            spec.hf_id, revision=spec.revision, device=device, trust_remote_code=True
        )
        self.model.max_seq_length = spec.max_seq_length

    def embed(self, texts: list[str], role: Role, batch_size: int) -> np.ndarray:
        prompt = self.spec.query_prompt if role == "query" else self.spec.document_prompt
        return self.model.encode(
            texts,
            prompt=prompt or None,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > batch_size,
        )
