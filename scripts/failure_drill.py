"""Break the real exported models on purpose and check the encoder still returns correct vectors.

    uv run python scripts/failure_drill.py

Each scenario runs in its own subprocess (so memory is returned between scenarios) on a
temporary copy of the export; .cache/openvino itself is never modified. The vector the encoder
returns is compared with a healthy int8 vector (cosine must be >= 0.99).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from coderet.config import JINA_CODE
from coderet.embed import make_query_encoder
from coderet.embed.export import DEFAULT_ROOT, export_dir

QUERY = ["where is the bluetooth settings deeplink handled"]
Sabotage = Callable[[object], None] | None


def fresh_root(src: Path, tmp: Path, *, link_fp32: bool = True) -> tuple[Path, Path]:
    """A throwaway export tree: fp32 is symlinked (read-only), int8 and the reference are copied."""
    src = src.resolve()  # the symlink target must be absolute, or the link dangles
    root = tmp / "root"
    base = root / src.name
    base.mkdir(parents=True)
    if link_fp32:
        (base / "fp32").symlink_to(src / "fp32", target_is_directory=True)
    shutil.copytree(src / "int8", base / "int8")
    shutil.copy(src / "probe_reference.npy", base / "probe_reference.npy")
    return root, base


def _truncate_int8(base: Path) -> None:
    p = base / "int8" / "openvino_model.bin"
    p.write_bytes(p.read_bytes()[: p.stat().st_size // 2])


def _missing_int8_graph(base: Path) -> None:
    (base / "int8" / "openvino_model.xml").unlink()


def _mismatched_reference(base: Path) -> None:
    ref = np.load(base / "probe_reference.npy")
    np.save(base / "probe_reference.npy", np.roll(ref, 1, axis=0))


def _no_openvino(base: Path) -> None:
    (base / "int8" / "openvino_model.xml").unlink()
    if (base / "fp32").is_symlink():
        (base / "fp32").unlink()


def _crash(enc) -> None:
    def boom(*_a, **_k):
        raise RuntimeError("simulated OpenVINO runtime crash")

    enc._backend._compiled = boom


def _nan(enc) -> None:
    real = enc._backend._compiled

    def bad(feeds):
        out = real(feeds)[0].copy()
        out[...] = np.nan
        return [out]

    enc._backend._compiled = bad


# key -> (description, how to damage the tree before building, how to damage the running encoder)
SCENARIOS: dict[str, tuple[str, Callable[[Path], None] | None, Sabotage]] = {
    "A": ("healthy", None, None),
    "B": ("int8 weights file truncated", _truncate_int8, None),
    "C": ("int8 graph file missing", _missing_int8_graph, None),
    "D": ("probe reference does not match (all OpenVINO rejected)", _mismatched_reference, None),
    "E": ("no OpenVINO models at all", _no_openvino, None),
    "F": ("int8 crashes mid-run", None, _crash),
    "G": ("int8 starts returning NaN mid-run", None, _nan),
}


def run_one(key: str, healthy_path: Path) -> None:
    desc, damage, sabotage = SCENARIOS[key]
    src = export_dir(JINA_CODE, DEFAULT_ROOT)
    with tempfile.TemporaryDirectory() as t:
        root, base = fresh_root(src, Path(t))
        if damage is not None:
            damage(base)
        enc = make_query_encoder(JINA_CODE, root)
        if sabotage is not None:
            enc.embed_queries(QUERY)  # activate it first, then break it while it is running
            sabotage(enc)
        vec = enc.embed_queries(QUERY)
        healthy = np.load(healthy_path)
        cos = float(vec[0] @ healthy[0])
        events = "; ".join(f"{e['event']}:{e['backend']}" for e in enc.events)
        print("RESULT " + json.dumps({"key": key, "desc": desc, "active": enc.active_name or "none",
                                      "cos": cos, "events": events}))


def make_healthy(path: Path) -> None:
    np.save(path, make_query_encoder(JINA_CODE, DEFAULT_ROOT).embed_queries(QUERY))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--one", choices=sorted(SCENARIOS), help="(internal) run a single scenario")
    ap.add_argument("--healthy", type=Path, help="(internal) healthy reference vector")
    ap.add_argument("--make-healthy", type=Path, help="(internal) write the healthy vector and exit")
    args = ap.parse_args()
    if args.make_healthy:
        make_healthy(args.make_healthy)
        return 0
    if args.one:
        run_one(args.one, args.healthy)
        return 0

    if not (export_dir(JINA_CODE, DEFAULT_ROOT) / "int8" / "openvino_model.xml").exists():
        raise SystemExit("run scripts/export_openvino.py first")
    results = []
    with tempfile.TemporaryDirectory() as t:
        healthy = Path(t) / "healthy.npy"
        subprocess.run([sys.executable, __file__, "--make-healthy", str(healthy)], check=True,
                       stderr=subprocess.DEVNULL)
        for key in SCENARIOS:
            proc = subprocess.run([sys.executable, __file__, "--one", key, "--healthy", str(healthy)],
                                  capture_output=True, text=True)
            line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
            if line is None:
                results.append({"key": key, "desc": SCENARIOS[key][0], "active": "CRASHED", "cos": 0.0,
                                "events": f"exit {proc.returncode}: {proc.stderr.strip()[-200:]}"})
            else:
                results.append(json.loads(line[7:]))
    print(f"\n{'':2s}{'scenario':56s} {'ended on':16s} {'cos':>8s}  events")
    ok = True
    for r in results:
        good = r["cos"] >= 0.99
        ok &= good
        print(f"{r['key']:2s}{r['desc']:56s} {r['active']:16s} {r['cos']:8.5f}  {r['events']}  {'OK' if good else 'BAD'}")
    print("\nALL RECOVERED" if ok else "\nFAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
