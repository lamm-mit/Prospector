#!/usr/bin/env python
"""GRPO over live multi-turn rollouts with the gated reward, for one round of
the recipe. A GPU job.

    python scripts/train_traj_grpo.py --graph tsmc --size 4b --stage d1
    python scripts/train_traj_grpo.py --graph tsmc --size 4b --stage full --smoke   # three steps into <output>_smoke
    python scripts/train_traj_grpo.py --graph tsmc --size 4b --stage d2 --dry-run

`--stage` is `full` (whole episodes on the training questions, continuing
from the stage-1 adapter), `d1` or `d2` (the decision rounds, resuming the
policy's stored decision states on top of the stage-3 adapter and, for d2,
round 1's picked checkpoint). The adapters, the questions, the band, the
graph and the output directory are derived from --graph, --size and --tag
(`graph_reasoner.recipe`); the round's hyperparameters are
`configs/grpo_full.yaml` or `configs/grpo_decision.yaml`, or `--config FILE`.
The run records what it was trained from in `<output>/run.json`.

A rollout earns 1 when its answer is correct and its cited chain passes the
gate, and otherwise a quarter of the fraction of the gold path's edges its
calls returned. No judge is involved, and a fabricated citation fails the gate,
so it cannot earn the success reward. Observation tokens are masked out of the
loss. The reward is `graph_reasoner.train.reward` and the trainer
`graph_reasoner.train.grpo`. The recipe does not take a round's last
checkpoint: it evaluates every checkpoint and picks one (the grid of
`prospector run`, `graph_reasoner.recipe.pick`).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

from graph_reasoner.recipe import (TRAINING, Layout, add_spec_arguments, spec_from_args,
                                   training_config, write_manifest)
from graph_reasoner.train.grpo import run_grpo


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_spec_arguments(ap)
    ap.add_argument("--stage", required=True, choices=[s for s, st in TRAINING.items() if st.trainer == "grpo"],
                    help="which round's inputs and hyperparameters")
    ap.add_argument("--config", default=None,
                    help="a hyperparameter file in place of the round's configs/grpo_*.yaml")
    ap.add_argument("--kb", default=None,
                    help="the graph file, when not the substrate's own")
    ap.add_argument("--smoke", action="store_true",
                    help="train three steps into <output>_smoke, removing an earlier smoke first")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved config and the manifest, write and train nothing")
    args = ap.parse_args(argv)
    # The graph is checked and its file resolved together (recipe.Graph.of),
    # so an unregistered --graph is refused before anything else.
    layout = Layout.of(spec_from_args(args), kb=args.kb, root=Path.cwd())
    config, manifest = training_config(layout, args.stage, config_path=args.config,
                                       smoke=args.smoke, argv=sys.argv[1:] if argv is None else argv)
    missing = layout.check_inputs(args.stage)
    if missing:
        raise SystemExit(f"{args.stage} at --graph {args.graph} --size {args.size} --tag {args.tag} "
                         f"is missing its inputs:\n  " + "\n  ".join(missing))
    if args.dry_run:
        print(yaml.safe_dump(config, sort_keys=False).rstrip())
        print(json.dumps(manifest, indent=2))
        return 0
    out_dir = Path(config["output_dir"])
    if args.smoke and out_dir.exists():
        shutil.rmtree(out_dir)
    write_manifest(out_dir, manifest)
    run_grpo(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
