#!/usr/bin/env python
"""Assemble a supervised training mix. Two forms.

The demonstrations (stage 1, `prospector teach mix`): the teacher's verified
episodes (keeps) plus synthetic detours (`build_error_pairs.py`), the detours
downsampled to at most `--ec-share` of the rows (default 0.08):

    python scripts/traj/build_substrate_mix.py \\
        --keeps data/traj/tsmc_flash20_keeps.jsonl \\
        --detours data/traj/tsmc_error_pairs_flash20.jsonl \\
        --out data/traj/tsmc_sft_mix_flash20.jsonl

The stage-3 blend (`prospector run`, stage rftbuild): the stage-1
demonstrations and the policy's own verified episodes after stage 2, both
whole, shuffled together:

    python scripts/traj/build_substrate_mix.py \\
        --demos data/traj/tsmc_sft_mix_flash20.jsonl \\
        --keeps data/traj/tsmc_2b_flash20g2_keeps.jsonl --keeps-name 2b_g2_keeps \\
        --out data/traj/tsmc_sft_mix_2b_flash20g2.jsonl

Every row keeps its `source` tag. `<out>.provenance.json` records the git
state, the argv, each component's path and row count, and the shares. One
seed (`--seed`, default 0) draws the detours and shuffles the mix, so a
rebuild from the same inputs writes the same rows in the same order.
"""
import argparse
import collections
import json
import pathlib
import random
import sys

from graph_reasoner import provenance
from graph_reasoner.files import write_jsonl_atomic


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def two_part(keeps: list, detours: list, ec_share: float, rng: random.Random) -> tuple[list, list]:
    """Keeps whole, detours downsampled to `ec_share` of the result; `(mix, detours used)`."""
    budget = int(ec_share / (1 - ec_share) * len(keeps))
    if budget < len(detours):
        detours = rng.sample(detours, budget)
    mix = keeps + detours
    rng.shuffle(mix)
    return mix, detours


def blend(demos: list, keeps: list, rng: random.Random) -> list:
    """Both parts whole, shuffled together."""
    mix = demos + keeps
    rng.shuffle(mix)
    return mix


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keeps", required=True, help="verified episodes (build_sft_rows.py)")
    form = ap.add_mutually_exclusive_group(required=True)
    form.add_argument("--detours", help="the demonstrations: synthetic detours (build_error_pairs.py)")
    form.add_argument("--demos", help="the stage-3 blend: the stage-1 demonstrations")
    ap.add_argument("--ec-share", type=float, default=0.08,
                    help="the detours' share of the demonstrations (default 0.08)")
    ap.add_argument("--keeps-name", default=None,
                    help="the keeps' name in the sidecar (default: the file's stem)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    keeps = _read(args.keeps)
    keeps_name = args.keeps_name or pathlib.Path(args.keeps).stem
    if args.detours:
        mix, used = two_part(keeps, _read(args.detours), args.ec_share, rng)
        components = {"keeps": {"path": args.keeps, "rows": len(keeps)},
                      "detours": {"path": args.detours, "rows_used": len(used)}}
        tally = collections.Counter(r.get("source", "?") for r in mix)
        extra = {"ec_share_target": args.ec_share}
    else:
        demos = _read(args.demos)
        mix = blend(demos, keeps, rng)
        stem = pathlib.Path(args.demos).stem
        demos_name = "mix_" + stem.split("sft_mix_", 1)[1] if "sft_mix_" in stem else stem
        components = {demos_name: {"path": args.demos, "rows": len(demos)},
                      keeps_name: {"path": args.keeps, "rows": len(keeps)}}
        tally = collections.Counter({demos_name: len(demos), keeps_name: len(keeps)})
        extra = {}

    out = pathlib.Path(args.out)
    write_jsonl_atomic(out, mix)
    prov = provenance.block("scripts/traj/build_substrate_mix.py", sys.argv[1:], extra={
        "components": components,
        "shares": {k: round(v / len(mix), 4) for k, v in tally.items()},
        **extra, "seed": args.seed, "rows_total": len(mix)})
    sidecar = provenance.write_sidecar(out, prov)
    print(f"wrote {len(mix)} rows -> {out} (+ {sidecar.name})")
    for k, v in sorted(tally.items()):
        print(f"  {k:24s} {v:7d}  ({100 * v / len(mix):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
