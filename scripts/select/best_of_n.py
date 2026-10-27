#!/usr/bin/env python
"""The gold-free vote over a pool of stored runs, beside its oracle.

    python scripts/select/best_of_n.py data/runs/<greedy> data/runs/<sample-1> ... --recompute

Run directories are positional and their order matters: the first is the
tie-break reference. Prints each run's own rates, then for every k the vote
over the first k runs (`graph_reasoner.select.select`) beside the oracle, the
share of items that any of the k runs gets right; the gap between the two is
what a better selector could still gain. A McNemar test compares the vote over
all runs with the first run. `--by-tier` breaks gated_acc and reach out by tier.

`--recompute` scores against the current benchmark instead of each run's
stored labels, and drops items the benchmark does not hold. No model is run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from graph_reasoner.benchmarks import split_run_name
from graph_reasoner.select import load, select


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--by-tier", action="store_true")
    # A run's stored labels are those of the benchmark at evaluation time.
    ap.add_argument("--recompute", action="store_true",
                    help="rescore against the current benchmark instead of the "
                         "run's stored labels, and drop items it does not hold")
    ap.add_argument("--scores", type=Path, default=None, metavar="DIR",
                    help="a verifier scoring run over the same pool (verifier_select.py): "
                         "the accuracy of its picks is printed beside the vote's")
    a = ap.parse_args()

    runs = [load(r, recompute=a.recompute) for r in a.runs]
    ids = sorted(set(runs[0]) & set.intersection(*(set(r) for r in runs)))
    dropped = len(runs[0]) - len(ids)
    if dropped:
        print(f"note: {dropped} item(s) missing from some run, excluded\n")

    def rate(rows, key):
        return sum(bool(r[key]) for r in rows) / len(rows) if rows else 0.0

    # gated_acc is the headline; em_claim is the same claim rule without the
    # gate, so the gap between them is what the gate refuses.
    cols = ("gated_acc", "em_claim", "reached")
    print(f"{len(ids)} items x {len(runs)} rollouts\n")
    print(f"{'':34s} {'gated_acc':>8s} {'em_claim':>8s} {'reach':>6s}")
    for i, run in enumerate(runs):
        rows = [run[i_] for i_ in ids]
        tag = f"run {i}  {split_run_name(a.runs[i].name)[0]}"
        print(f"{tag:34s} {rate(rows,'gated_acc'):8.3f} {rate(rows,'em_claim'):8.3f} "
              f"{rate(rows,'reached'):6.3f}")

    print()
    print(f"{'':34s} {'gated_acc':>8s} {'em_claim':>8s} {'reach':>6s}   "
          f"{'oracle gated_acc':>15s} {'oracle em_claim':>15s} {'oracle reach':>12s}")
    per_k = {}
    for k in range(1, len(runs) + 1):
        picked = [select([runs[j][i_] for j in range(k)]) for i_ in ids]
        union = [{key: any(runs[j][i_][key] for j in range(k)) for key in cols}
                 for i_ in ids]
        per_k[k] = (picked, union)
        print(f"{'best-of-' + str(k) + ' (gold-free)':34s} "
              f"{rate(picked,'gated_acc'):8.3f} {rate(picked,'em_claim'):8.3f} "
              f"{rate(picked,'reached'):6.3f}   {rate(union,'gated_acc'):15.3f} "
              f"{rate(union,'em_claim'):15.3f} {rate(union,'reached'):12.3f}")

    if a.scores:
        # The verifier's picks, one row per item, from the scoring run's
        # scores.jsonl: the stored labels of the candidate it chose.
        chosen = {}
        with (a.scores / "scores.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    if row.get("picked"):
                        chosen[row["id"]] = row
        picked = [chosen[i_] for i_ in ids if i_ in chosen]
        tag = f"verifier pick  {split_run_name(a.scores.name)[0]}"
        print(f"{tag:34s} {rate(picked,'gated_acc'):8.3f} {rate(picked,'em_claim'):8.3f}"
              + (f"   ({len(ids) - len(picked)} item(s) without a pick)" if len(picked) < len(ids) else ""))

    if a.by_tier:
        print()
        picked, union = per_k[len(runs)]
        base = [runs[0][i_] for i_ in ids]
        print(f"{'tier':8s} {'n':>4s} {'run 0 acc':>9s} {'bo-n acc':>8s} "
              f"{'oracle acc':>10s} {'run 0 reach':>12s} {'oracle reach':>13s}")
        for tier in ("easy", "medium", "hard", "ALL"):
            sel = [i for i, i_ in enumerate(ids)
                   if tier == "ALL" or runs[0][i_]["tier"] == tier]
            b = [base[i] for i in sel]
            p = [picked[i] for i in sel]
            u = [union[i] for i in sel]
            print(f"{tier:8s} {len(sel):4d} {rate(b,'gated_acc'):9.3f} "
                  f"{rate(p,'gated_acc'):8.3f} {rate(u,'gated_acc'):10.3f} "
                  f"{rate(b,'reached'):12.3f} {rate(u,'reached'):13.3f}")

    # McNemar against the reference run, on the full k.
    picked, _ = per_k[len(runs)]
    base = [runs[0][i_] for i_ in ids]
    for key in ("gated_acc", "em_claim"):
        b = sum(1 for x, y in zip(picked, base) if x[key] and not y[key])
        c = sum(1 for x, y in zip(picked, base) if y[key] and not x[key])
        if b + c:
            from math import comb
            p = sum(comb(b + c, i) for i in range(min(b, c) + 1)) / 2 ** (b + c) * 2
            print(f"\nMcNemar {key:8s} vs run 0: +{b} / -{c}, p = {min(p, 1.0):.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
