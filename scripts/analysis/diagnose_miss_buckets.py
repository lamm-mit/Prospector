#!/usr/bin/env python
"""Classify a run's headline misses into the three miss buckets.

    python scripts/analysis/diagnose_miss_buckets.py data/runs/flash20cap1024-s0--tsmc_core_agentic
    python scripts/analysis/diagnose_miss_buckets.py data/runs/flash20cap1024-s{0..7}--tsmc_core_agentic --ids

A miss is `not gated_acc`, recomputed against the benchmark's current answer
sets; the buckets (search, rival, gate) are defined in
`graph_reasoner.eval.misses`. Each run directory needs both results.jsonl and
trajectories.jsonl: a partial join or a repeated id raises.

Per run it prints the bucket counts, the rival misses that held every gold
edge (`gold_complete_at_call` set; marked `*` under `--ids`) and how many
misses and hits ended at the call budget (`status == 'budget'`). With several
runs it also prints the items missed in a majority of them, with their
majority bucket.

`--benchmark` overrides the answer sets the run directory's `--<benchmark>`
suffix resolves to; `--tier` restricts to one tier's items. No model is run.

The paper's figures/extract_data.py loads this file by path and calls
`diagnose(run_dir, None, None)`, so `diagnose` keeps that signature.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from graph_reasoner import benchmarks
from graph_reasoner.benchmarks import split_run_name
from graph_reasoner.eval.misses import classify, stable_misses
from graph_reasoner.select.pool import load_pair


def gold_map(bench_name: str) -> dict[str, list[str]]:
    return benchmarks.gold_answers(bench_name)


def diagnose(run: Path, benchmark: str | None, tier: str | None) -> dict[str, dict]:
    """item id -> `classify` row, over one run's joined results and
    trajectories, against the answer sets of `benchmark` (default: the run
    dir's own suffix)."""
    bench = benchmark or split_run_name(run.name)[1]
    golds = gold_map(bench)
    res_rows, traj_rows = load_pair(run)
    out: dict[str, dict] = {}
    for iid, res in res_rows.items():
        answers = golds.get(iid.split("#")[0]) or golds.get(iid)
        if answers is None:      # the item is not in this benchmark
            continue
        if tier and res.get("tier") != tier:
            continue
        out[iid] = classify(res, traj_rows[iid], answers)
    return out


def report(name: str, rows: dict[str, dict], show_ids: bool) -> None:
    misses = {i: r for i, r in rows.items() if not r["hit"]}
    b = Counter(r["bucket"] for r in misses.values())
    rivals = [i for i, r in misses.items() if r["bucket"] == "rival"]
    ev = sum(misses[i]["ev_complete"] for i in rivals)
    cap_m = sum(r["at_cap"] for r in misses.values())
    hits = [r for r in rows.values() if r["hit"]]
    cap_h = sum(r["at_cap"] for r in hits)
    print(f"{name:28s} n={len(rows)}  gated_acc={len(hits)/len(rows):.3f}  "
          f"miss={len(misses)}  search={b['search']}  rival={b['rival']} "
          f"(ev-complete {ev})  gate={b['gate']}  "
          f"at-cap {cap_m}/{len(misses)} miss vs {cap_h}/{len(hits)} hit")
    if show_ids:
        for bucket in ("search", "rival", "gate"):
            ids = sorted(i for i, r in misses.items() if r["bucket"] == bucket)
            mark = lambda i: i + ("*" if bucket == "rival" and misses[i]["ev_complete"] else "")
            if ids:
                print(f"  {bucket:6s} {' '.join(mark(i) for i in ids)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--tier", default=None)
    ap.add_argument("--ids", action="store_true",
                    help="print per-bucket item ids (rival* = evidence-complete)")
    a = ap.parse_args()
    per_run = {}
    for run in a.runs:
        rows = diagnose(run, a.benchmark, a.tier)
        tag = split_run_name(run.name)[0]
        per_run[tag] = rows
        report(tag, rows, a.ids)
    if len(per_run) > 1:
        maj = stable_misses(per_run)
        b = Counter(maj.values())
        ev = sum(1 for i, bk in maj.items()
                 if bk == "rival" and any(r[i]["ev_complete"] for r in per_run.values()))
        print(f"\nSTABLE (missed in majority of {len(per_run)} reads): {len(maj)} items — "
              f"search={b['search']}  rival={b['rival']} (ev-complete {ev})  gate={b['gate']}")
        if a.ids:
            for bucket in ("search", "rival", "gate"):
                sel = sorted(i for i, bk in maj.items() if bk == bucket)
                if sel:
                    print(f"  {bucket:6s} {' '.join(sel)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
