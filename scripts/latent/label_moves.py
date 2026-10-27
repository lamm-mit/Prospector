#!/usr/bin/env python
"""Label every observation row of a reads extract with the move the policy
made next, read off the stored run, and save the labels beside the reads.

    python scripts/latent/label_moves.py data/runs/_probe/pilot/policy_all/<run> data/runs/<run> \\
        --out data/runs/_probe/pilot/labels_home.npz

The classes are `graph_reasoner.latent.moves`'s: answer, follow, backtrack,
fresh, find, connect, other. The file carries keep (the rows' indices in the
extract's index.jsonl), lab, G, hop, call, tool, tier, item and run, which the
readings, the directions and the figure join to the reads by (item, call).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from graph_reasoner.latent.moves import label_extract, write_labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("extract", type=Path, help="the reads of one run (scripts/latent/extract_states.py's output directory)")
    ap.add_argument("run", type=Path, help="the stored run the reads were taken from (data/runs/<run>)")
    ap.add_argument("--out", type=Path, required=True, help="the labels file to write (.npz)")
    a = ap.parse_args()
    rows, keep, lab = label_extract(a.extract, a.run)
    write_labels(a.out, rows, keep, lab, a.run.name)
    counts = {k: int((lab == k).sum()) for k in sorted(set(lab.tolist()))}
    print(f"{a.out}: {len(rows)} observation rows, {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
