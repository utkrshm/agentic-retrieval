"""Submission run from the theme-1 guidelines: MTEB AppsRetrieval -> appsretrieval_results.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mteb

from coderet.config import MODELS, get_model
from coderet.models.encoders import SentenceTransformerEmbedder
from coderet.mteb_adapters import PrePostPipelineEncoder

DEFAULT_MODEL = "jina-code-0.5b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS), help="registry key from coderet.config")
    parser.add_argument("--out", type=Path, default=Path("appsretrieval_results.json"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    model = PrePostPipelineEncoder(SentenceTransformerEmbedder(get_model(args.model)))
    task = mteb.get_task("AppsRetrieval")  # Make sure you choose this task
    result = mteb.evaluate(model, [task], encode_kwargs={"batch_size": args.batch_size})

    # Write the evaluation JSON you asked for.
    task_result = list(result.task_results)[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:  # Upload this file
        json.dump(task_result.to_dict(), f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
