"""Model registry. One entry per embedding model, pinned to an exact checkpoint revision."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    revision: str
    query_prompt: str
    document_prompt: str
    max_seq_length: int = 1024  # the frozen AppsRetrieval result used 1,024 tokens
    notes: str = ""


JINA_CODE = ModelSpec(
    key="jina-code-0.5b",
    hf_id="jinaai/jina-code-embeddings-0.5b",
    revision="4db235132dafbe56a8b9c5f59b59795ecf58a4a7",
    query_prompt="Find the most relevant code snippet given the following query:\n",
    document_prompt="Candidate code snippet:\n",
    notes=(
        "Last-token pooling, L2-normalised, 896 dims. Trained on sequences of 512 tokens "
        "(arXiv 2508.21290) and on AppsRetrieval train, so APPS-train splits cannot validate it. "
        "The checkpoint ships as bfloat16, which is about 3.4x slower on CPUs without AVX-512/AMX: "
        "always load fp32 for CPU serving."
    ),
)

# Comparison models for the AppsRetrieval head-to-head (scripts/run_mteb.py --model). Not used by the
# repository pipeline: its OpenVINO export, probe and unit text are specific to jina-code.
EMBEDDINGGEMMA = ModelSpec(
    key="embeddinggemma-300m",
    hf_id="google/embeddinggemma-300m",
    revision="57c266a740f537b4dc058e1b0cda161fd15afa75",
    query_prompt="task: code retrieval | query: ",
    document_prompt="title: none | text: ",
    max_seq_length=2048,
    notes="Gated on Hugging Face. Comparison model only; pooling comes from the checkpoint's own config.",
)

QWEN3_06B = ModelSpec(
    key="qwen3-0.6b",
    hf_id="Qwen/Qwen3-Embedding-0.6B",
    revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    query_prompt=(
        "Instruct: Given a code contest problem description, "
        "retrieve relevant code that can help solve the problem\nQuery:"
    ),
    document_prompt="",
    notes="Comparison model only; pooling comes from the checkpoint's own config.",
)

MODELS: dict[str, ModelSpec] = {m.key: m for m in (JINA_CODE, EMBEDDINGGEMMA, QWEN3_06B)}


def get_model(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; choose from {sorted(MODELS)}")
    return MODELS[key]


def fingerprint(spec: ModelSpec, **settings: object) -> str:
    """Short hash of everything that changes the vectors: model spec plus run settings.

    Used as the MTEB model revision and in cache keys, so a changed prompt, precision, backend or
    truncation can never reuse stale scores or vectors.
    """
    payload = json.dumps({"spec": asdict(spec), "settings": settings}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
