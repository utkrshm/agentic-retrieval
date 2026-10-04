"""Export jina-code to OpenVINO (fp32 graph + int8-compressed copy + probe reference vectors).

    uv run python scripts/export_openvino.py [--no-int8] [--force]

Output goes to .cache/openvino/<model>-<revision8>/. Takes about a minute and about 4 GB of RAM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coderet.config import JINA_CODE
from coderet.embed.export import DEFAULT_ROOT, export_openvino


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--no-int8", action="store_true", help="skip the int8 weight-compressed copy")
    ap.add_argument("--force", action="store_true", help="re-export even if files exist")
    args = ap.parse_args()
    out = export_openvino(JINA_CODE, args.root, int8=not args.no_int8, force=args.force)
    print(f"exported to {out}")


if __name__ == "__main__":
    main()
