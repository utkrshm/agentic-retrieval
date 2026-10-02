"""Model registry and run fingerprints.

Every model is pinned to an exact Hugging Face revision with its documented
asymmetric prompts. A run's fingerprint hashes everything that changes its
vectors, so caches and results never mix across configurations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

CACHE_ROOT = Path(".cache")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    revision: str
    query_prompt: str
    document_prompt: str
    max_seq_length: int = 1024
    gated: bool = False
    notes: str = ""


MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in [
        ModelSpec(
            key="embeddinggemma-300m",
            hf_id="google/embeddinggemma-300m",
            revision="57c266a740f537b4dc058e1b0cda161fd15afa75",
            query_prompt="task: code retrieval | query: ",
            document_prompt="title: none | text: ",
            max_seq_length=2048,
            gated=True,
            notes="Gated: accept the license on Hugging Face and run `hf auth login`.",
        ),
        ModelSpec(
            key="jina-code-0.5b",
            hf_id="jinaai/jina-code-embeddings-0.5b",
            revision="4db235132dafbe56a8b9c5f59b59795ecf58a4a7",
            query_prompt="Find the most relevant code snippet given the following query:\n",
            document_prompt="Candidate code snippet:\n",
            notes="Trained on AppsRetrieval train: dev scores on APPS-train splits are optimistic.",
        ),
        ModelSpec(
            key="qwen3-0.6b",
            hf_id="Qwen/Qwen3-Embedding-0.6B",
            revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            query_prompt=(
                "Instruct: Given a code contest problem description, "
                "retrieve relevant code that can help solve the problem\nQuery:"
            ),
            document_prompt="",
        ),
        ModelSpec(
            key="minilm-l6",
            hf_id="sentence-transformers/all-MiniLM-L6-v2",
            revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            query_prompt="",
            document_prompt="",
            max_seq_length=256,
            notes="Tiny smoke-test model, not a candidate.",
        ),
    ]
}


@dataclass(frozen=True)
class RunConfig:
    model: ModelSpec
    pooling_normalize: bool = True
    extra: dict = field(default_factory=dict)

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]

    def cache_dir(self) -> Path:
        return CACHE_ROOT / f"{self.model.key}-{self.fingerprint()}"


def get_model(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; choose from {sorted(MODELS)}")
    return MODELS[key]
