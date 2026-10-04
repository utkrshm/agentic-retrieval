"""Official-format run from the theme-1 guidelines: MTEB AppsRetrieval -> appsretrieval_results.json.

    uv run python scripts/run_mteb.py                      # PyTorch fp32 on the GPU if there is one
    uv run python scripts/run_mteb.py --device cpu         # PyTorch fp32 on the CPU (slow: hours)
    uv run python scripts/run_mteb.py --query-backend auto # int8 OpenVINO queries on CPU, fp32 documents

Documents are always encoded in fp32. The AppsRetrieval TEST split is used by this script: run it
only for a frozen configuration, never to tune anything. Output goes to outputs/ (git-ignored).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mteb
import numpy as np

from coderet.config import MODELS, JINA_CODE, fingerprint, get_model
from coderet.embed import make_query_encoder
from coderet.embed.backends import TorchBackend
from coderet.mteb_adapters import PrePostPipelineEncoder


class SplitEmbedder:
    """Queries and documents may use different (but each fp32-faithful) backends."""

    def __init__(self, query_embedder, document_embedder) -> None:
        self.query_embedder, self.document_embedder = query_embedder, document_embedder

    def embed(self, texts: list[str], role: str) -> np.ndarray:
        return (self.query_embedder if role == "query" else self.document_embedder).embed(texts, role)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=JINA_CODE.key, choices=sorted(MODELS),
                    help="comparison models run with PyTorch fp32 only (--query-backend torch)")
    ap.add_argument("--out", type=Path, default=Path("outputs/appsretrieval_results.json"))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--query-backend", default="torch", choices=["torch", "auto"],
                    help="torch: PyTorch fp32 for queries too; auto: OpenVINO int8 -> fp32 -> PyTorch chain")
    ap.add_argument("--batch-size", type=int, default=64, help="MTEB batch size (encoder re-batches internally)")
    ap.add_argument("--torch-batch", type=int, default=8, help="PyTorch micro-batch (memory bound on small GPUs)")
    args = ap.parse_args()

    import torch

    spec = get_model(args.model)
    if spec is not JINA_CODE and args.query_backend != "torch":
        raise SystemExit("the OpenVINO chain is exported for jina-code only; use --query-backend torch")
    device = "cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    doc_backend = TorchBackend(spec, device, batch_size=args.torch_batch)
    if args.query_backend == "torch":
        embedder, qname = doc_backend, doc_backend.name
    else:
        embedder, qname = SplitEmbedder(make_query_encoder(JINA_CODE), doc_backend), "chain"
    rev = fingerprint(spec, doc=doc_backend.name, queries=qname, precision="fp32 docs")
    encoder = PrePostPipelineEncoder(embedder, spec.key, revision=rev)

    print(f"model {spec.hf_id} @ {spec.revision[:8]} | docs: {doc_backend.name} | queries: {qname} | run {rev}")
    task = mteb.get_task("AppsRetrieval")  # Make sure you choose this task
    t0 = time.perf_counter()
    result = mteb.evaluate(encoder, [task], encode_kwargs={"batch_size": args.batch_size},
                           overwrite_strategy="always")
    elapsed = time.perf_counter() - t0

    # The guideline's json.dump(task_result.to_dict()) fails on the datetime field in mteb 2.21,
    # so use MTEB's own serializer, which mteb's from_disk can read back.
    task_result = list(result.task_results)[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    task_result.to_disk(args.out)
    scores = task_result.scores["test"][0]
    print(f"\nwrote {args.out}   ({elapsed / 60:.1f} min)")
    for k in ("ndcg_at_10", "mrr_at_10", "recall_at_1", "recall_at_10", "recall_at_20", "recall_at_100", "main_score"):
        if k in scores:
            print(f"  {k:14s} {100 * scores[k]:.2f}")
    if getattr(encoder.embedder, "events", None):
        print("  encoder events:", encoder.embedder.events)


if __name__ == "__main__":
    main()
