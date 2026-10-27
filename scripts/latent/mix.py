#!/usr/bin/env python
"""Tabulate every finished steering arm under a knob root into <root>/mix.json:
the move shares, calls, early answers, budget exits and accuracy of each arm,
from its trajectories.jsonl and results.jsonl.

    python scripts/latent/mix.py data/runs/_probe/knob_vllm

An arm is a `knob-<arm>--<benchmark>/` directory with a summary.json (the
eval finished) and a knob.json (`scripts/latent/steer.py` wrote it); its kind,
setting, layer and direction come from the latter. The measures are
`graph_reasoner.latent.moves.analyse`; `--out` writes elsewhere than the root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from graph_reasoner.latent.moves import mix


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="a knob root: scripts/latent/steer.py's --out-root")
    ap.add_argument("--out", type=Path, default=None, help="default <root>/mix.json")
    a = ap.parse_args()
    out = a.out or a.root / "mix.json"
    doc = mix(a.root, log=lambda s: print(s, flush=True))
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
