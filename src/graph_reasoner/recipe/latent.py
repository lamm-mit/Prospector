"""The latent-state probe of a served policy (the paper's Section 3.6), on the
GPU machine: the hidden states of two stored runs (the training graph's and a
transfer graph's), the next move after every observation, the AUC-by-layer
profile of the two next-move readings, the directions at one layer, the
steering arms at several episode seeds, and each root's mix.

    prospector latent --graph tsmc --size 4b --transfer-graph biograph \\
        --home-run flash20cap1024-s0 --bio-run flash20stackbio-s0

The stack is the recipe's served stack for `--graph` and `--size`; the two
runs are named by their run tag, and sit under `data/runs/<tag>--<stem>` for
the home graph's and the transfer graph's evaluation questions. Stages:
`states moves readings directions steer mix`; a stage whose output exists is
skipped, the arms skip per run directory. `--limit` makes a smoke of it, under
`data/runs/_probe_smoke` unless `--out` names another root, so a smoke's outputs
are never taken for the probe's.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..benchmarks import run_dir_name
from .plan import Context
from .run import Failed, Refused, Runner, check_chain, say, shown
from .plan import CheckChain

STAGES = ("states", "moves", "readings", "directions", "steer", "mix")


def layer_list(spec: str) -> list[int]:
    """The read points a `--layers` value names: `0-32`, `20`, or a comma list of either."""
    out: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out
#: The arms of the paper's figure. On the home graph the first seed reads
#: everything a repeat seed does plus the follow half doses (fol:+-0.5), the
#: quarter answer doses (ans:+-0.25) and the second and third random draws
#: (rnd2, rnd3, rndf2, rndf3). A repeat seed reads fol:+-1 and +-2, ans:+-0.5
#: and +-1 and one draw of each null (rnd, rndf). The transfer graph reads the
#: repeat arms without rnd, at every seed.
ARMS_HOME_FIRST = ("dose0 fol:+0.5 fol:-0.5 fol:+1 fol:-1 fol:+2 fol:-2 ans:+0.25 ans:-0.25 ans:+0.5 ans:-0.5 "
                   "ans:+1 ans:-1 rnd:+0.5 rnd:-0.5 rnd:+1 rnd:-1 rnd2:+0.5 rnd2:-0.5 rnd2:+1 rnd2:-1 rnd3:+0.5 "
                   "rnd3:-0.5 rnd3:+1 rnd3:-1 rndf:+1 rndf:-1 rndf:+2 rndf:-2 rndf2:+1 rndf2:-1 rndf2:+2 rndf2:-2 "
                   "rndf3:+1 rndf3:-1 rndf3:+2 rndf3:-2").split()
ARMS_HOME_REPEAT = ("dose0 fol:+1 fol:-1 fol:+2 fol:-2 ans:+0.5 ans:-0.5 ans:+1 ans:-1 rndf:+1 rndf:-1 rndf:+2 "
                    "rndf:-2 rnd:+0.5 rnd:-0.5 rnd:+1 rnd:-1").split()
ARMS_TRANSFER = "dose0 fol:+1 fol:-1 fol:+2 fol:-2 ans:+0.5 ans:-0.5 ans:+1 ans:-1 rndf:+1 rndf:-1 rndf:+2 rndf:-2".split()


@dataclass
class Probe:
    ctx: Context
    runner: Runner
    home_run: str                      # the run tag of the stored home-graph run
    bio_run: str                       # the run tag of the stored transfer-graph run
    out: str | None = None             # data/runs/_probe, or data/runs/_probe_smoke with a limit
    tag: str = "policy_all"
    layer: int = 20
    layers: str = "0-32"
    seeds: tuple[int, ...] = (0, 2, 3)
    limit: int | None = None
    arms_home: list[str] | None = None
    arms_bio: list[str] | None = None
    start: str | None = None
    stop: str | None = None
    stack: list[str] = field(init=False)

    def __post_init__(self):
        L = self.ctx.layout
        if self.out is None:
            self.out = "data/runs/_probe_smoke" if self.limit else "data/runs/_probe"
        if self.ctx.transfer is None:
            raise Refused("the probe reads a transfer graph's run too: pass --transfer-graph")
        self.stack = L.served_stack()
        self.home_dir = f"data/runs/{run_dir_name(self.home_run, L.graph.eval_benchmark)}"
        self.bio_dir = f"data/runs/{run_dir_name(self.bio_run, self.ctx.transfer.eval_benchmark)}"
        self.pilot = f"{self.out}/pilot"
        self.reads = f"{self.pilot}/{self.tag}"
        for s in (self.start, self.stop):
            if s is not None and s not in STAGES:
                raise Refused(f"{s!r} is not a stage of the probe ({' '.join(STAGES)})")
        if self.start and self.stop and STAGES.index(self.start) > STAGES.index(self.stop):
            raise Refused(f"--start-at {self.start} comes after --stop-after {self.stop} ({' '.join(STAGES)})")

    def have(self, rel: str) -> bool:
        return (self.ctx.root / rel).exists()

    def active(self) -> list[str]:
        i = STAGES.index(self.start or STAGES[0])
        j = STAGES.index(self.stop or STAGES[-1])
        return list(STAGES[i:j + 1])

    def cmd(self, gpu: bool, *parts: str) -> None:
        cmd = [self.runner.python(), *parts]
        say(f"    {shown(cmd)}")
        if gpu:
            self.runner.reap()
        rc = self.runner.run(cmd)
        if rc != 0:
            raise Failed(f"{Path(parts[0]).name} exited {rc}")

    def limit_args(self) -> list[str]:
        return ["--limit", str(self.limit)] if self.limit else []

    def reads_present(self, run_dir: str) -> bool:
        """Whether a run's reads are whole: the index and one array per read
        point. The arrays are not released and a copy may leave them out, so
        the index alone does not make the stage done."""
        d = f"{self.reads}/{run_dir.split('/')[-1]}"
        return self.have(f"{d}/index.jsonl") and all(self.have(f"{d}/L{n}.npy") for n in layer_list(self.layers))

    def states(self) -> None:
        if self.reads_present(self.home_dir) and self.reads_present(self.bio_dir):
            say(f"=== SKIP STATES ({self.reads} holds both runs' reads)")
            return
        self.cmd(True, "scripts/latent/extract_states.py", self.home_dir, self.bio_dir,
                 "--model", self.ctx.layout.spec.base_model, "--adapter", ",".join(self.stack),
                 "--layers", self.layers, "--out", self.reads, *self.limit_args())

    def moves(self) -> None:
        for name, run_dir in (("home", self.home_dir), ("bio", self.bio_dir)):
            if self.have(f"{self.pilot}/labels_{name}.npz"):
                say(f"=== SKIP MOVES {name}")
                continue
            self.cmd(False, "scripts/latent/label_moves.py", f"{self.reads}/{run_dir.split('/')[-1]}", run_dir,
                     "--out", f"{self.pilot}/labels_{name}.npz")

    def readings(self) -> None:
        if self.have(f"{self.pilot}/depth_profile.json"):
            say("=== SKIP READINGS")
            return
        self.cmd(False, "scripts/latent/depth_profile.py",
                 "--fit", f"{self.reads}/{self.home_dir.split('/')[-1]}", f"{self.pilot}/labels_home.npz",
                 "--test", f"{self.reads}/{self.bio_dir.split('/')[-1]}", f"{self.pilot}/labels_bio.npz",
                 "--layers", self.layers, "--tag", self.tag, "--out", f"{self.pilot}/depth_profile.json")

    def directions(self) -> None:
        if self.have(f"{self.pilot}/directions_L{self.layer}.json"):
            say("=== SKIP DIRECTIONS")
            return
        self.cmd(False, "scripts/latent/directions.py", f"{self.reads}/{self.home_dir.split('/')[-1]}", self.home_dir,
                 "--layers", str(self.layer), "--n-random", "3", "--out-dir", self.pilot)

    def arms(self, which: str, seed: int) -> list[str]:
        if which == "home":
            return self.arms_home or (ARMS_HOME_FIRST if seed == 0 else ARMS_HOME_REPEAT)
        return self.arms_bio or ARMS_TRANSFER

    def steer(self) -> None:
        L = self.ctx.layout
        for seed in self.seeds:
            root = f"{self.out}/knob_vllm" if seed == 0 else f"{self.out}/knob_vllm_r{seed}"
            for bench, which in ((L.graph.eval_benchmark, "home"), (self.ctx.transfer.eval_benchmark, "bio")):
                arms = self.arms(which, seed)
                self.runner.sleep(5)
                say(f"=== arms seed {seed} {Path(bench).stem}: {' '.join(arms)}")
                self.cmd(True, "scripts/latent/steer.py", "--benchmark", bench, "--model", L.spec.base_model,
                         "--adapter", ",".join(self.stack),
                         "--directions", f"{self.pilot}/directions_L{self.layer}.npz", "--layer", str(self.layer),
                         "--arms", *arms, "--out-root", root, "--seed", str(seed), *self.limit_args())

    def mix(self) -> None:
        roots = sorted(p for p in (self.ctx.root / self.out).glob("knob_vllm*") if p.is_dir())
        for root in roots:
            cmd = [self.runner.python(), "scripts/latent/mix.py", str(root.relative_to(self.ctx.root))]
            say(f"    {shown(cmd)}")
            rc, out = self.runner.capture(cmd)
            if out.strip():
                say(out.strip().splitlines()[-1])
            if rc != 0:
                raise Failed(f"mix.py {root.name} exited {rc}")

    def execute(self) -> int:
        say(f"=== LATENT start: stack {','.join(self.stack)} out {self.out} layer {self.layer} "
            f"seeds {' '.join(map(str, self.seeds))} limit {self.limit or ''} {self.runner.now()}")
        try:
            check_chain(CheckChain(tuple(self.stack)), self.ctx)
            for run_dir in (self.home_dir, self.bio_dir):
                if not self.have(f"{run_dir}/trajectories.jsonl"):
                    raise Refused(f"stored run missing: {run_dir}")
            for stage in self.active():
                say(f"=== LATENT {stage.upper()} start {self.runner.now()}")
                getattr(self, stage)()
                say(f"=== LATENT {stage.upper()} done {self.runner.now()}")
        except Failed as e:
            say(f"!!! LATENT {e.message} {self.runner.now()}")
            return e.code
        say(f"=== LATENT COMPLETE {self.runner.now()}")
        return 0
