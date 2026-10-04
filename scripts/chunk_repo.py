"""Chunk a repository with tree-sitter and report what came out.

    uv run python scripts/chunk_repo.py /path/to/repo [--max-chars 4000] [--dump units.jsonl]

Checks invariants (unit text equals the file lines of its span) and prints coverage,
size distribution, parse errors and a few samples, so the chunker can be inspected
on a real repository before anything is embedded.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from coderet.chunking import MAX_CHARS, Unit, chunk_file, iter_source_files


def pct(xs: list[int], q: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    ap.add_argument("--dump", type=Path, help="write units as JSONL")
    ap.add_argument("--samples", type=int, default=5)
    args = ap.parse_args()
    root = args.root.resolve()

    t0 = time.perf_counter()
    units: list[Unit] = []
    error_files: list[str] = []
    files_by_lang: Counter[str] = Counter()
    lines_total = lines_covered = 0
    low_cov: list[tuple[float, str, int]] = []
    violations = 0

    for path in iter_source_files(root):
        file_lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
        file_units, had_error = chunk_file(path, root, args.max_chars)
        rel = path.relative_to(root).as_posix()
        if had_error:
            error_files.append(rel)
        if file_units:
            files_by_lang[file_units[0].language] += 1
        covered: set[int] = set()
        for u in file_units:
            if u.text != "\n".join(file_lines[u.start_line - 1 : u.end_line]):
                violations += 1
            covered.update(range(u.start_line, u.end_line + 1))
        non_blank = {i + 1 for i, ln in enumerate(file_lines) if ln.strip()}
        lines_total += len(non_blank)
        lines_covered += len(non_blank & covered)
        if non_blank:
            low_cov.append((len(non_blank & covered) / len(non_blank), rel, len(non_blank)))
        units.extend(file_units)
    elapsed = time.perf_counter() - t0

    sizes = [len(u.text) for u in units]
    kinds = Counter(u.kind for u in units)
    parts = sum(1 for u in units if u.part > 0)
    by_text: dict[str, set[str]] = defaultdict(set)
    for u in units:
        by_text[u.text].add(u.path)
    dup_texts = sum(1 for v in by_text.values() if len(v) > 1)

    print(f"root: {root}")
    print(f"files chunked: {sum(files_by_lang.values())}  ({dict(files_by_lang)})   time: {elapsed:.1f}s")
    print(f"units: {len(units)}  kinds: {dict(kinds)}  pieces of oversized defs: {parts}")
    print(f"unit chars: p50 {pct(sizes, .5)}  p95 {pct(sizes, .95)}  max {max(sizes, default=0)}  "
          f"mean {statistics.fmean(sizes) if sizes else 0:.0f}   (about {sum(sizes) / 4 / 1000:.0f}k tokens in total)")
    print(f"non-blank line coverage: {100 * lines_covered / max(lines_total, 1):.1f}%  ({lines_covered}/{lines_total})")
    print(f"files with parse errors: {len(error_files)}  {error_files[:5]}")
    print(f"identical unit text appearing in several files: {dup_texts}")
    print(f"INVARIANT violations (unit text != file lines): {violations}")

    print("\nlowest-coverage files (>= 20 lines):")
    for cov, rel, n in sorted(x for x in low_cov if x[2] >= 20)[:5]:
        print(f"  {100 * cov:5.1f}%  {n:5d} lines  {rel}")
    print("\nlargest units:")
    for u in sorted(units, key=lambda x: -len(x.text))[:5]:
        print(f"  {len(u.text):6d} chars  {u.kind:8s} {u.qualname[:50]:50s} {u.path}:{u.start_line}-{u.end_line}")

    rng = random.Random(7)
    print(f"\n{args.samples} random units:")
    for u in rng.sample(units, min(args.samples, len(units))):
        head = "\n      ".join(u.text.split("\n")[:4])
        print(f"  [{u.kind}] {u.qualname}  {u.path}:{u.start_line}-{u.end_line}  ({len(u.text)} chars)\n      {head}")

    if args.dump:
        with args.dump.open("w") as f:
            for u in units:
                f.write(json.dumps(u.__dict__) + "\n")
        print(f"\nwrote {len(units)} units to {args.dump}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
