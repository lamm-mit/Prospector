#!/usr/bin/env python
"""Pick a GRPO round's checkpoint: per-checkpoint means over its stored
evaluation runs, and the recipe's pick rule applied to them
(`graph_reasoner.recipe.pick`, which the runner calls in-process).

    python scripts/analysis/ladder_peak.py q2bd2eval --benchmark tsmc_core_agentic            # the table and the pick
    python scripts/analysis/ladder_peak.py q2bd2eval --benchmark tsmc_core_agentic --pick     # the pick alone, on stdout
    python scripts/analysis/ladder_peak.py q2bd2eval --benchmark tsmc_core_agentic --top 2    # the two best means, on stdout
    python scripts/analysis/ladder_peak.py q2bd2eval --benchmark tsmc_core_agentic --cap 1024 # only seeds read at that turn budget

It reads `<runs>/<prefix>-ck<N>-s<K>--<benchmark stem>/`, one run per seed,
and skips a run whose summary.json is missing or unreadable; the stem is the
graph's evaluation file's (`prospector names --graph <graph> EVAL_STEM`).

The pick rule is a decision round's: checkpoints whose mean gated_acc is
within `--band` (default 0.03) of the best mean are tied, and a tie goes to
the checkpoint with more seeds, then to the later one. `--best` applies the
full round's rule instead: the best mean, the later checkpoint on a tie.

With `--pick` or `--top` the table goes to stderr and stdout carries only the
checkpoint numbers, or nothing when no run is found; the exit status is then
1. `--cap` keeps only seeds whose summary.json records that
`policy.turn_max_tokens`; without it every seed counts.
"""
from __future__ import annotations

import argparse
import statistics as st
import sys

from graph_reasoner.recipe.pick import BAND, leaders, means, pick, pick_best, read_grid, tied


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix", help="eval-run prefix, e.g. q2bd2eval")
    ap.add_argument("--runs", default="data/runs")
    ap.add_argument("--benchmark", required=True,
                    help="the evaluation benchmark's stem, the suffix of every read's directory")
    ap.add_argument("--band", type=float, default=BAND, help=f"the tie band (default {BAND})")
    ap.add_argument("--best", action="store_true", help="the full round's rule: the best mean, the later checkpoint on a tie")
    ap.add_argument("--cap", type=int, default=None, help="count only seeds read at this turn budget")
    out = ap.add_mutually_exclusive_group()
    out.add_argument("--pick", action="store_true", help="print only the picked checkpoint number")
    out.add_argument("--top", type=int, default=None, metavar="N",
                     help="print only the N checkpoint numbers with the best means")
    args = ap.parse_args()

    acc, caps = read_grid(args.prefix, args.benchmark, args.runs, args.cap)
    quiet = args.pick or args.top is not None
    table = sys.stderr if quiet else sys.stdout
    if not acc:
        print(f"{args.prefix}: no reads under {args.runs}", file=table)
        return 1
    m, near = means(acc), tied(acc, args.band)
    chosen = pick_best(acc) if args.best else pick(acc, args.band)
    for k in sorted(acc):
        v = sorted(acc[k])
        g = [x for _, x in v]
        se = st.stdev(g) / len(g) ** 0.5 if len(g) > 1 else float("nan")
        print(f"ck{k:>4}  {m[k]:.3f} +/- {se:.3f} SE (n={len(g)})  seeds={[s for s, _ in v]}  "
              f"caps={dict(caps[k])}" + ("  <- tied" if k in near else ""), file=table)
    print(f"pick: ck{chosen} ({m[chosen]:.3f} on {len(acc[chosen])} seeds; "
          f"{'best mean' if args.best else f'band {args.band}'}, cap filter {args.cap})", file=table)
    if args.pick:
        print(chosen)
    elif args.top is not None:
        print(" ".join(str(k) for k in leaders(acc, args.top)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
