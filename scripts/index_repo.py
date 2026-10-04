"""Build the FAISS index of a repository's current checkout (latest commit only, no manifests).

    uv run python scripts/index_repo.py /path/to/repo --out outputs/index/node-red

Units come from the tree-sitter chunker, documents are embedded with jina-code in fp32 (on the GPU
when there is one: this is offline work), and vectors are cached by the exact embedded text under
.cache/vectors/, so a re-run only embeds units whose text changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coderet.chunking import MAX_CHARS
from coderet.config import JINA_CODE, MODELS, fingerprint, get_model
from coderet.embed.backends import TorchBackend
from coderet.index import RepoIndex, VectorCache


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=JINA_CODE.key, choices=sorted(MODELS),
                    help="embedding model; non-jina models are comparison runs on plain PyTorch fp32")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    ap.add_argument("--min-chars", type=int, default=60,
                    help="fold statement groups under this many non-blank characters into a neighbour (0: off; 60 was neutral to slightly positive on two repos)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    import torch

    device = "cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    spec = get_model(args.model)
    encoder = TorchBackend(spec, device, batch_size=8)
    cache = None if args.no_cache else VectorCache(
        Path(".cache/vectors") / fingerprint(spec, kind="document", precision="fp32"))
    index = RepoIndex.build(args.repo, encoder, spec, args.max_chars, cache, args.batch,
                            min_chars=args.min_chars)
    index.save(args.out)
    m = index.meta
    print(f"\nindexed {m['repo']} @ {m['commit'][:10]}: {m['n_units']} units from {m['n_files']} files, "
          f"{m['dim']}-d, {m['embedded_now']} embedded now, {m['from_cache']} from cache, "
          f"{m['build_seconds']}s, backend {encoder.name}")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
