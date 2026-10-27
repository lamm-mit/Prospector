#!/usr/bin/env python
"""The AUC of the two next-move readings at every read point: fitted on one
graph's reads, held out there by item-grouped five-fold cross-validation, and
zero-shot on a second graph.

    python scripts/latent/depth_profile.py \\
        --fit data/runs/_probe/pilot/policy_all/<home run> data/runs/_probe/pilot/labels_home.npz \\
        --test data/runs/_probe/pilot/policy_all/<unseen run> data/runs/_probe/pilot/labels_bio.npz \\
        --layers 0-32 --tag policy_all --out data/runs/_probe/pilot/depth_profile.json

Each of --fit and --test names a reads directory (index.jsonl, L<n>.npy) and
its labels file. The readings and the fit are `graph_reasoner.latent.readings`.
The file is {tag: {reading: {home: [AUC per read point], bio: [...]}}}, where
home is the fitting graph held out and bio the second graph zero-shot; a
layer list other than 0-32 is recorded under "layers".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from graph_reasoner.latent.readings import profile


def parse_layers(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", nargs=2, type=Path, required=True, metavar=("READS", "LABELS"),
                    help="the fitting graph's reads directory and labels file")
    ap.add_argument("--test", nargs=2, type=Path, required=True, metavar=("READS", "LABELS"),
                    help="the unseen graph's reads directory and labels file")
    ap.add_argument("--layers", default="0-32", help="read points, as ranges and lists: 0-32, or 1,8,20")
    ap.add_argument("--tag", default="policy_all", help="the key the profile is stored under")
    ap.add_argument("--out", type=Path, required=True, help="the profile to write (.json)")
    a = ap.parse_args()
    layers = parse_layers(a.layers)
    res = profile(a.fit[0], a.fit[1], a.test[0], a.test[1], layers, log=lambda s: print(s, flush=True))
    doc = {a.tag: res}
    if layers != list(range(33)):
        doc = {"layers": layers, **doc}
    a.out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
