"""Embedders: concrete models behind the `Embedder` protocol."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from coderet.mteb_adapters.encoder import Role


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers checkpoint; vectors are L2-normalised downstream."""

    def __init__(self, model_id: str, device: str | None = None) -> None:
        self.name = model_id
        self.model = SentenceTransformer(model_id, device=device, trust_remote_code=True)

    def embed(self, texts: list[str], role: Role, batch_size: int) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=len(texts) > batch_size)
