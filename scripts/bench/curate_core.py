#!/usr/bin/env python
"""Curate a generated benchmark: keep the items that are usable questions.

    python scripts/bench/curate_core.py --benchmark data/benchmarks/<kg>/<kg>_v0.jsonl --out data/benchmarks/<kg>/<kg>_core.jsonl

The input is the generator's output (scripts/bench/build_benchmark.py); neither
that file nor this script's output is tracked. The five filters are
`graph_reasoner.bench.curate`. Every dropped item is written, with its reason,
to `<out stem>.dropped.jsonl`.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from graph_reasoner.bench.curate import curate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, help="the generator's output, <kg>_v0.jsonl")
    ap.add_argument("--out", required=True, help="the curated file, <kg>_core.jsonl")
    args = ap.parse_args()

    with open(args.benchmark, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    keep, dropped, why = curate(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for path, data in ((out, keep), (out.with_suffix(".dropped.jsonl"), dropped)):
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{len(rows)} -> KEEP {len(keep)} ({len(keep)/len(rows):.0%})")
    for reason, n in why.most_common():
        print(f"   dropped {n:4d}  {reason}")
    print(f"\ntier:     {dict(sorted(collections.Counter(r['tier'] for r in keep).items()))}")
    print(f"path_len: {dict(sorted(collections.Counter(r['meta']['path_len'] for r in keep).items()))}")
    print(f"\nwrote {out}")
    print(f"wrote {out.with_suffix('.dropped.jsonl')}  (auditable, with reasons)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
