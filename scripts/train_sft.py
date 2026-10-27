#!/usr/bin/env python
"""Run LoRA SFT for one stage of the recipe: transformers' Trainer on a CUDA GPU.

    python scripts/train_sft.py --graph tsmc --size 4b --stage sft1
    python scripts/train_sft.py --graph tsmc --size 4b --stage sft1 --smoke      # three steps into <output>_smoke
    python scripts/train_sft.py --graph tsmc --size 4b --stage verifier --dry-run

`--stage` is `sft1` (from the base model on the teacher's demonstrations),
`sft2` (from the base model on the demonstrations plus the stage-2 policy's
verified episodes) or `verifier` (the yes/no verifier on the served policy's
own passes). The rows, the base model and the output directory are derived
from --graph, --size and --tag (`graph_reasoner.recipe`); the stage's
hyperparameters are `configs/<stage>.yaml`, or `--config FILE` for an
experiment. The run records what it was trained from in `<output>/run.json`,
copied beside the adapter in `final/`. Needs the `[train]` extra, which
installs torch; `--dry-run` and `--help` need nothing.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

from graph_reasoner.recipe import (TRAINING, Layout, add_spec_arguments, spec_from_args,
                                   training_config, write_manifest)
from graph_reasoner.train.sft import run_sft


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_spec_arguments(ap)
    ap.add_argument("--stage", required=True, choices=[s for s, st in TRAINING.items() if st.trainer == "sft"],
                    help="which stage's rows and hyperparameters")
    ap.add_argument("--config", default=None,
                    help="a hyperparameter file in place of the stage's configs/<stage>.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="train three steps into <output>_smoke, removing an earlier smoke first")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved config and the manifest, write and train nothing")
    args = ap.parse_args(argv)
    layout = Layout.of(spec_from_args(args), root=Path.cwd())
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
    run_sft(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
