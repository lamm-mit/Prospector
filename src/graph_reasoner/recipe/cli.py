"""`prospector`, the recipe's command line, for one graph and one size:
the teacher pass, the plan and the run of the four stages and the selection,
a stage's smoke, the machine check, the latent-state probe, and the names
everything is derived from. `python -m graph_reasoner.recipe` is the same
program.

    prospector teach  --graph tsmc 0
    prospector plan   --graph tsmc --size 2b --transfer-graph biograph
    prospector run    --graph tsmc --size 2b --transfer-graph biograph
    prospector smoke  --graph tsmc --size 2b --stage d1
    prospector verify --graph tsmc --size 4b
    prospector latent --graph tsmc --size 4b --transfer-graph biograph --home-run flash20cap1024-s0 --bio-run flash20stackbio-s0
    prospector names  --graph tsmc --size 2b --env
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

from ..benchmarks import ROOT
from . import names as _names
from .names import Layout, add_spec_arguments, spec_from_args
from .latent import STAGES as PROBE_STAGES
from .latent import Probe
from .names import DEFAULT_TAG, TRAINING, Spec, refuse
from .plan import STAGES, Context, plan, render
from .run import Failed, Runner, execute, sft_smoke, smoke
from .teach import DEFAULT_TEACHER, Teach
from .verify import verify

COMMANDS = ("names", "plan", "run", "smoke", "teach", "verify", "latent")


def add_run_arguments(ap: argparse.ArgumentParser) -> None:
    """The flags `plan` and `run` share."""
    add_spec_arguments(ap)
    ap.add_argument("--transfer-graph", default=None,
                    help="a second registered graph read at every gate and coda; none unless set")
    ap.add_argument("--kb", default=None, help="the graph file, when not the substrate's own")
    ap.add_argument("--start-at", default=None, metavar="STAGE", help=f"the first stage to run: {' '.join(STAGES)}")
    ap.add_argument("--stop-after", default=None, metavar="STAGE", help="the last stage to run")
    ap.add_argument("--floor", type=float, default=0.40, help="the gates' gated_acc floor (default 0.40)")
    ap.add_argument("--force-gate", action="store_true", help="continue past a failed gate")


def context_from_args(args, root: Path = ROOT) -> Context:
    layout = Layout.of(spec_from_args(args), kb=args.kb, transfer=args.transfer_graph, root=root)
    return Context(layout, floor=args.floor, force_gate=args.force_gate)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="prospector", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("names", help="print the names the recipe derives", add_help=False)
    p = sub.add_parser("plan", help="what a run would do, stage by stage, against the disk",
                       description="Print every derived name and what each stage of a run would do: which adapters "
                                   "are on disk, which reads are scored, which rows exist. Nothing runs.")
    add_run_arguments(p)
    p.add_argument("--no-names", action="store_true", help="the stage lines alone")
    r = sub.add_parser("run", help="the recipe for one graph and one size, stage by stage",
                       description="Stages 1 to 4 and the selection for one graph and one size, on the GPU machine: "
                                   f"{' '.join(STAGES)}. A stage whose outputs exist is skipped, so a relaunch resumes; "
                                   "the log is this command's stdout, the leaves write logs/<name>.log. Exit 1 when a "
                                   "stage fails, 2 when a prerequisite is missing, 3 when a gate is under the floor.")
    add_run_arguments(r)
    r.add_argument("--dry-run", action="store_true", help="print what each stage would do, then stop (as `plan`)")
    m = sub.add_parser("smoke", help="one training run's three-step twin, for the fit and the health of a stage",
                       description="Train three steps of one stage into <its directory>_smoke: a GRPO stage under the "
                                   "step timers (killed after three steps, the peak memory printed), an SFT stage under "
                                   "the memory watermark. An earlier smoke directory is removed first.")
    add_spec_arguments(m)
    m.add_argument("--stage", required=True, choices=list(TRAINING), help="which training run to smoke")
    m.add_argument("--kb", default=None, help="the graph file, when not the substrate's own")
    t = sub.add_parser("teach", help="stage 0, the teacher pass, on your own machine",
                       description="The teacher pass over the graph's training questions, in shards, then the second "
                                   "draw and the rows: <shard>... | ids | p2 | mix (graph_reasoner.recipe.teach).")
    t.add_argument("--graph", required=True, help="a registered substrate; its training questions are the pass's items")
    t.add_argument("--tag", default=DEFAULT_TAG, help=f"the name of this teacher pass, which every file carries (default {DEFAULT_TAG})")
    t.add_argument("--teacher", default=None, help=f"the teacher model id: the one the tag's stored pass records, "
                                                   f"or {DEFAULT_TEACHER} for a new tag; another teacher needs its own --tag")
    t.add_argument("--shards", type=int, default=8, help="tier-balanced shards of the pass (default 8)")
    t.add_argument("--dry-run", action="store_true", help="print which of the asked-for pieces exist, then stop")
    t.add_argument("pieces", nargs="+", metavar="PIECE", help="a shard number, ids, p2 or mix, in order")
    v = sub.add_parser("verify", help="verify the GPU machine before spending: imports, the vLLM probes, an SFT smoke",
                       description="Five checks of one size, stopping at the first failure (graph_reasoner.recipe.verify).")
    add_spec_arguments(v)
    v.add_argument("--gpu-frac", type=float, default=0.35, help="the engine's share of the card in the sleep probe (default 0.35)")
    la = sub.add_parser("latent", help="the latent-state probe of the served policy",
                        description="The hidden states of two stored runs, the next-move readings, the directions and "
                                    "the steering arms (graph_reasoner.recipe.latent).")
    add_spec_arguments(la)
    la.add_argument("--transfer-graph", required=True, help="the transfer graph whose stored run is read")
    la.add_argument("--home-run", required=True, metavar="TAG", help="the stored run of the served policy on the graph's evaluation questions (its run tag)")
    la.add_argument("--bio-run", required=True, metavar="TAG", help="the stored run on the transfer graph (its run tag)")
    la.add_argument("--out", default=None,
                    help="the data root (default data/runs/_probe, or data/runs/_probe_smoke with --limit)")
    la.add_argument("--reads-tag", default="policy_all", help="the reads' tag under <out>/pilot (default policy_all)")
    la.add_argument("--layer", type=int, default=20, help="the layer the directions and the arms live at (default 20)")
    la.add_argument("--layers", default="0-32", help="the layers read and profiled (default 0-32)")
    la.add_argument("--seeds", type=int, nargs="+", default=[0, 2, 3], help="the episode seeds of the arms (default 0 2 3)")
    la.add_argument("--limit", type=int, default=None, help="a smoke: episodes per run for the reads, items per tier for the arms")
    la.add_argument("--arms-home", nargs="+", default=None, help="the home graph's arms, in place of the paper's")
    la.add_argument("--arms-bio", nargs="+", default=None, help="the transfer graph's arms, in place of the paper's")
    la.add_argument("--start-at", default=None, metavar="STAGE", help=f"the first stage: {' '.join(PROBE_STAGES)}")
    la.add_argument("--stop-after", default=None, metavar="STAGE", help="the last stage")
    if argv and argv[0] == "names":
        return _names.main(argv[1:])
    args = ap.parse_args(argv)
    if not (ROOT / "pyproject.toml").exists():
        # The root is derived from this file's place in the source tree, so a
        # wheel or a copied package has none to run from.
        refuse(f"prospector runs from an editable install of the repository (uv pip install -e .); "
               f"{ROOT} holds no pyproject.toml")
    os.chdir(ROOT)

    def on_term(signum, frame):
        # A SIGTERM (`kill <pid>`, a scheduler's stop) unwinds like a SIGINT,
        # so a supervisor kills the trainer it launched (run.wait_watching)
        # instead of leaving it running under a directory the next launch removes.
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, on_term)
    if args.command == "plan" or (args.command == "run" and args.dry_run):
        ctx = context_from_args(args)
        print(render(ctx, plan(ctx, args.start_at, args.stop_after), names=getattr(args, "no_names", False) is False))
        return 0
    if args.command == "run":
        ctx = context_from_args(args)
        return execute(ctx, Runner(ROOT), args.start_at, args.stop_after)
    if args.command == "smoke":
        layout = Layout.of(spec_from_args(args), kb=args.kb, root=ROOT)
        ctx = Context(layout)
        out_dir = layout.resolved(args.stage)["output_dir"]
        try:
            if TRAINING[args.stage].trainer == "grpo":
                smoke(args.stage, out_dir, ctx, Runner(ROOT))
            else:
                sft_smoke(args.stage, out_dir, ctx, Runner(ROOT))
        except Failed as e:
            print(f"!!! SMOKE {e.message}")
            return e.code
        return 0
    if args.command == "teach":
        layout = Layout.of(Spec(args.graph, tag=args.tag), root=ROOT)
        teach = Teach(layout, Runner(ROOT), teacher=args.teacher, shards=args.shards)
        if args.dry_run:
            try:
                teach.dry_run(args.pieces)
            except Failed as e:
                print(f"!!! {e.message}")
                return e.code
            return 0
        return teach.execute(args.pieces)
    if args.command == "verify":
        return verify(Context(Layout.of(spec_from_args(args), root=ROOT)), Runner(ROOT), args.gpu_frac)
    if args.command == "latent":
        ctx = Context(Layout.of(spec_from_args(args), transfer=args.transfer_graph, root=ROOT))
        try:
            probe = Probe(ctx, Runner(ROOT), args.home_run, args.bio_run, out=args.out, tag=args.reads_tag,
                          layer=args.layer, layers=args.layers, seeds=tuple(args.seeds), limit=args.limit,
                          arms_home=args.arms_home, arms_bio=args.arms_bio, start=args.start_at, stop=args.stop_after)
        except Failed as e:
            print(f"!!! LATENT {e.message}")
            return e.code
        return probe.execute()
    return 2
