"""MTEB adapter: the `PrePostPipelineEncoder` named in the theme-1 guidelines.

Every encode call runs three stages, each swappable without touching MTEB glue:

    preprocess(texts, role) -> embedder.embed(texts, role) -> postprocess(vectors, role)

`role` is "query" or "document", derived from MTEB's `PromptType`, so query-side
and document-side processing (prompts, views, normalisation) stay asymmetric.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Protocol

import numpy as np
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

Role = Literal["query", "document"]


class Embedder(Protocol):
    """Anything that turns texts into a (n, dim) float32 matrix."""

    name: str

    def embed(self, texts: list[str], role: Role, batch_size: int) -> np.ndarray: ...


TextHook = Callable[[list[str], Role], list[str]]
VectorHook = Callable[[np.ndarray, Role], np.ndarray]


def identity_texts(texts: list[str], role: Role) -> list[str]:
    return texts


def l2_normalize(vectors: np.ndarray, role: Role) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class PrePostPipelineEncoder(AbsEncoder):
    def __init__(
        self,
        embedder: Embedder,
        preprocess: TextHook = identity_texts,
        postprocess: VectorHook = l2_normalize,
        revision: str | None = None,
    ) -> None:
        self.embedder = embedder
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.mteb_model_meta = ModelMeta.create_empty(
            {
                "name": f"coderet/{embedder.name.replace('/', '__')}",
                # MTEB caches results per (name, revision): pass the run fingerprint
                # so a changed config never reuses stale scores.
                "revision": revision,
                "similarity_fn_name": ScoringFunction.COSINE,
            }
        )

    def encode(
        self,
        inputs: Any,
        *,
        task_metadata: Any,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        role: Role = "query" if prompt_type == PromptType.query else "document"
        texts = [text for batch in inputs for text in batch["text"]]
        texts = self.preprocess(texts, role)
        vectors = self.embedder.embed(texts, role, batch_size=kwargs.get("batch_size", 64))
        return self.postprocess(np.asarray(vectors, dtype=np.float32), role)
