#!/usr/bin/env python
"""Merge train-benchmark batches with enforced disjointness, by gold node-set.

    python scripts/bench/merge_train.py \\
        --base data/benchmarks/tsmc/<batch-a>.jsonl \\
        --add data/benchmarks/tsmc/<batch-b>.jsonl:c \\
        --exclude data/benchmarks/tsmc/tsmc_core_agentic.jsonl \\
        --out data/benchmarks/tsmc/<merged>.jsonl \\
        --new-only data/benchmarks/tsmc/<batch-b-kept>.jsonl

Items are compared by `frozenset(gold_path.nodes)` and not by id, because ids
restart in every generation batch. An item whose gold node set is in the
evaluation file would leak an evaluation question into training, and one that
repeats across batches would count twice. `BenchmarkItem.gold_pathset`, which
the trainers' `exclude` filter reads, is the same key.

The merge aborts if a `--base` row collides with an `--exclude` file. Base rows
are written unchanged. An added row (`--add FILE:SUFFIX`) is dropped if its key
is excluded or already present; a kept one gets the suffix appended to its id.
`--new-only` also writes the kept added rows alone, so rollouts can be sampled
on the new items only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def key(row: dict) -> frozenset:
    return frozenset(row["gold_path"]["nodes"])



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--add", action="append", required=True,
                    metavar="FILE:SUFFIX")
    ap.add_argument("--exclude", action="append", type=Path, default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--new-only", type=Path, default=None)
    a = ap.parse_args()

    banned: set[frozenset] = set()
    for path in a.exclude:
        rows = load(path)
        banned |= {key(r) for r in rows}

    base = load(a.base)
    seen: set[frozenset] = {key(r) for r in base}
    if seen & banned:
        raise SystemExit(f"ABORT: base already collides with exclude on "
                         f"{len(seen & banned)} gold node-sets")

    out_rows = list(base)
    new_rows = []
    audit: Counter = Counter()
    for spec in a.add:
        path, _, suffix = spec.rpartition(":")
        if not suffix or "/" in suffix:
            raise SystemExit(f"--add wants FILE:SUFFIX, got {spec!r}")
        for r in load(Path(path)):
            k = key(r)
            if k in banned:
                audit[f"{suffix}: eval collision"] += 1
                continue
            if k in seen:
                audit[f"{suffix}: train duplicate"] += 1
                continue
            seen.add(k)
            r = dict(r, id=f"{r['id']}{suffix}")
            out_rows.append(r)
            new_rows.append(r)
            audit[f"{suffix}: kept"] += 1

    ids = [r["id"] for r in out_rows]
    assert len(ids) == len(set(ids)), "id collision after suffixing"

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.new_only:
        with a.new_only.open("w") as fh:
            for r in new_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    tiers = Counter(r["tier"] for r in out_rows)
    print(f"{len(base)} base + {len(new_rows)} new -> {len(out_rows)} items "
          f"({dict(sorted(tiers.items()))})")
    for reason, n in sorted(audit.items()):
        print(f"   {reason}: {n}")
    print(f"wrote {a.out}" + (f" and {a.new_only}" if a.new_only else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
