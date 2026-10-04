"""MTEB adapter: the `PrePostPipelineEncoder` named in the theme-1 guidelines.

Every encode call runs three stages, each swappable without touching the MTEB glue:

    preprocess(texts, role) -> embedder.embed(texts, role) -> postprocess(vectors, role)

`role` is "query" or "document", derived from MTEB's `PromptType`, so query-side and
document-side processing (prompts, views, normalisation) stay asymmetric. The embedder is any
object with ``embed(texts, role) -> (n, dim) array`` (a backend or a ResilientEncoder).

The encoder only ever sees what MTEB passes in (ids are not exposed, titles are empty); it must
never reload the raw dataset, whose metadata contains the answer.
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
    def embed(self, texts: list[str], role: str) -> np.ndarray: ...


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
        name: str,
        revision: str | None = None,
        preprocess: TextHook = identity_texts,
        postprocess: VectorHook = l2_normalize,
    ) -> None:
        self.embedder = embedder
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.mteb_model_meta = ModelMeta.create_empty(
            {
                "name": f"coderet/{name.replace('/', '__')}",
                # MTEB caches results per (name, revision): pass a run fingerprint so a
                # changed configuration never reuses stale scores.
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
        return self.encode_texts(texts, role)

    def encode_texts(self, texts: list[str], role: Role) -> np.ndarray:
        """The pipeline itself, shared by the MTEB path and any standalone use."""
        vectors = self.embedder.embed(self.preprocess(texts, role), role)
        return self.postprocess(np.asarray(vectors, dtype=np.float32), role)
