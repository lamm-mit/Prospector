#!/usr/bin/env python
"""The steering directions at one or more layers, from the fitting graph's
reads: the follow and answer axes, random draws and the projected-out axes,
with their dose units.

    python scripts/latent/directions.py data/runs/_probe/pilot/policy_all/<home run> data/runs/<home run> \\
        --layers 20 --n-random 3 --out-dir data/runs/_probe/pilot

Writes <out-dir>/directions_L<layer>.npz (unit vectors: follow, answer,
random, random2, ..., follow_perp, answer_perp) and
<out-dir>/directions_L<layer>.json (each axis's gap between its class means,
the class sizes, the AUC along it and the gap as a share of the median
residual norm; `random` carries the answer axis's gap). The construction is
`graph_reasoner.latent.directions`; `scripts/latent/steer.py` reads the files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from graph_reasoner.latent.directions import build, per_layer_files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("extract", type=Path, help="the fitting graph's reads directory")
    ap.add_argument("run", type=Path, help="the stored run the reads were taken from (data/runs/<run>)")
    ap.add_argument("--layers", default="20", help="comma-separated layers")
    ap.add_argument("--n-random", type=int, default=3, help="random unit vectors per layer: random, random2, ...")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    vec, meta = build(a.extract, a.run, layers, a.n_random, log=lambda s: print(s, flush=True))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        v, m = per_layer_files(vec, meta, layer)
        p = a.out_dir / f"directions_L{layer}"
        np.savez(p.with_suffix(".npz"), **v)
        p.with_suffix(".json").write_text(json.dumps(m, indent=1), encoding="utf-8")
        print("wrote", p.with_suffix(".npz"), "and", p.with_suffix(".json").name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
