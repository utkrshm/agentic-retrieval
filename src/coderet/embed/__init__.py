"""Embedding backends with failure recovery.

    enc = make_query_encoder()        # OpenVINO int8 if exported and healthy, else PyTorch fp32
    vectors = enc.embed_queries(["where is the config file loaded"])

Documents always use fp32 (``make_document_encoder``): int8 and fp32 vectors must never be
mixed in one index. The int8 path was checked for queries only (int8 queries against fp32
documents, see EXPERIMENTS notes in the repository history).
"""

from __future__ import annotations

from pathlib import Path

from coderet.config import JINA_CODE, ModelSpec
from coderet.embed.backends import OpenVinoBackend, TorchBackend
from coderet.embed.export import DEFAULT_ROOT, export_dir
from coderet.embed.probe import load_reference
from coderet.embed.resilient import Candidate, EncoderUnavailable, ResilientEncoder

__all__ = [
    "Candidate", "EncoderUnavailable", "OpenVinoBackend", "ResilientEncoder", "TorchBackend",
    "make_document_encoder", "make_query_encoder",
]

DIM = 896
INT8_MIN_COSINE = 0.99  # observed minimum 0.9985 on 72 texts, 0.9969 on 500 queries
FP32_MIN_COSINE = 0.999


def make_query_encoder(spec: ModelSpec = JINA_CODE, root: Path = DEFAULT_ROOT,
                       threads: int | None = None) -> ResilientEncoder:
    """Fastest healthy backend for queries: OpenVINO int8, then OpenVINO fp32, then PyTorch fp32."""
    base = export_dir(spec, root)
    reference = load_reference(base / "probe_reference.npy")
    candidates: list[Candidate] = []
    for precision, min_cos in (("int8", INT8_MIN_COSINE), ("fp32", FP32_MIN_COSINE)):
        model_dir = base / precision
        if (model_dir / "openvino_model.xml").exists():
            candidates.append(Candidate(
                f"openvino-{precision}",
                lambda d=model_dir, p=precision: OpenVinoBackend(spec, d, p, threads),
                min_cosine=min_cos,
            ))
    candidates.append(Candidate("torch-fp32-cpu", lambda: TorchBackend(spec, "cpu")))
    return ResilientEncoder(candidates, reference=reference, dim=DIM)


def make_document_encoder(spec: ModelSpec = JINA_CODE, device: str = "cpu") -> ResilientEncoder:
    """Documents: fp32 PyTorch only (on GPU when asked and available, falling back to CPU)."""
    candidates = []
    if device != "cpu":
        candidates.append(Candidate(f"torch-fp32-{device}", lambda: TorchBackend(spec, device)))
    candidates.append(Candidate("torch-fp32-cpu", lambda: TorchBackend(spec, "cpu")))
    return ResilientEncoder(candidates, dim=DIM)
