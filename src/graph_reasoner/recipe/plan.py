"""What a run of the recipe does for one (graph, size), computed from the disk.

A run is a fixed sequence of stages (`STAGES`). Each stage is a list of steps
with its own done test on the files it writes: a read is done when its run
directory holds `summary.json`, a training run when its adapter is saved, a
round when its target checkpoint or its recorded pick is on disk. The steps
are data; `run.py` executes the ones that are not done, and `plan()` with
`render()` prints them, which is the dry run. The two cannot disagree, since
they are the same list.

A stage recomputes its steps from the disk on every call, so what follows a
pick or a round not yet made appears as `Later` until the pick or the round
exists, and a relaunch resumes wherever the files stop. Nothing here runs a
process or imports torch.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..benchmarks import run_dir_name
from ..train.adapters import ADAPTER_CONFIG, is_adapter_dir
from .names import BAND_SEEDS, G2_ROLL_SEEDS, GATE_SEEDS, POOL_SEEDS, Graph, Layout, refuse
from .pick import leaders as grid_leaders_rule
from .pick import pick as pick_rule
from .pick import pick_best, read_grid

#: The stages of `prospector run`, in order. `--start-at` and `--stop-after`
#: take a slice of this list.
STAGES = ("sft1", "gate1", "band", "round1", "rftroll", "rftbuild", "sft2", "gate2",
          "d1", "d2", "final", "rows", "verifier", "score")
#: The seeds a decision round's two leading checkpoints are read on before the pick.
LEADER_SEEDS = tuple(range(8))
#: The extra seeds read on the full round's pick after the grid.
FULL_EXTRA_SEEDS = (3, 4, 5)
#: The sampled seeds of a round's pass@5 coda, at temperature 0.9.
PASS5_SEEDS = (1, 2, 3, 4)
PASS5_TEMPERATURE = 0.9
ROLL_TEMPERATURE = 0.7


@dataclass(frozen=True)
class Context:
    """One run's fixed inputs: the layout and the knobs the user set."""
    layout: Layout
    floor: float = 0.40
    force_gate: bool = False
    target: int = int(os.environ.get("TARGET", "120"))       # the round's last step
    turn_cap: int = int(os.environ.get("TURN_CAP", "1024"))   # tokens a turn in every read

    @property
    def root(self) -> Path:
        return self.layout.root

    @property
    def graph(self) -> Graph:
        return self.layout.graph

    @property
    def transfer(self) -> Graph | None:
        return self.layout.transfer

    def exists(self, rel: str) -> bool:
        return (self.root / rel).exists()


# ---------------------------------------------------------------- the steps

@dataclass(frozen=True)
class Read:
    """One evaluation read: `run_agent_eval.py` with the recipe's protocol."""
    name: str
    chain: tuple[str, ...]
    benchmark: str
    kb: str
    seed: int
    temperature: float | None = None     # None is greedy
    group: str = "read"                  # the dry run counts reads by group

    @property
    def run_dir(self) -> str:
        return f"data/runs/{run_dir_name(self.name, self.benchmark)}"

    @property
    def key(self) -> str:
        return f"read:{self.name}"

    def done(self, ctx: Context) -> bool:
        return scored(ctx, self.run_dir)


@dataclass(frozen=True)
class Train:
    """An SFT run (sft1, sft2 or the verifier) into `out_dir`, saved at `final`."""
    stage: str
    out_dir: str
    final: str
    log: str

    @property
    def key(self) -> str:
        return f"train:{self.stage}"

    def done(self, ctx: Context) -> bool:
        return is_adapter_dir(ctx.root / self.final)


@dataclass(frozen=True)
class Smoke:
    """A GRPO round's three-step twin, run before the round itself."""
    stage: str
    out_dir: str

    @property
    def key(self) -> str:
        return f"smoke:{self.stage}"

    def done(self, ctx: Context) -> bool:
        return False


@dataclass(frozen=True)
class Round:
    """A supervised GRPO round. Running it first removes the round directory
    and the reads named after its checkpoints (a retrain gets fresh reads)."""
    stage: str
    out_dir: str
    prefix: str
    chain: tuple[str, ...]
    target: int

    @property
    def key(self) -> str:
        return f"round:{self.stage}"

    def done(self, ctx: Context) -> bool:
        return round_done(ctx, self.out_dir, self.target)


@dataclass(frozen=True)
class Build:
    """A builder script writing one file; `command` is the script and its
    arguments, the interpreter is the runner's. `always` rebuilds on every
    pass (the decision states)."""
    label: str
    out: str
    command: tuple[str, ...]
    log: str | None = None
    always: bool = False

    @property
    def key(self) -> str:
        return f"build:{self.label}"

    def done(self, ctx: Context) -> bool:
        return not self.always and ctx.exists(self.out)


@dataclass(frozen=True)
class Band:
    """The band: `sample_prefixes.py` over a decision round's states."""
    prefixes: str
    name: str
    run_dir: str
    chain: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"band:{self.name}"

    def done(self, ctx: Context) -> bool:
        return scored(ctx, self.run_dir)


@dataclass(frozen=True)
class Gate:
    """The floor on a gate's three greedy seeds: the means are printed, and
    the run stops (exit 3) unless gated_acc >= floor and calls >= 11, or
    `--force-gate`."""
    prefix: str
    floor: float

    @property
    def key(self) -> str:
        return f"gate:{self.prefix}"

    def done(self, ctx: Context) -> bool:
        return False


@dataclass(frozen=True)
class CheckChain:
    """Every adapter of a chain is on disk and none was trained on another graph."""
    chain: tuple[str, ...]

    @property
    def key(self) -> str:
        return "chain:" + ",".join(self.chain)

    def done(self, ctx: Context) -> bool:
        return False


@dataclass(frozen=True)
class Require:
    """Files a stage needs and does not make; a refusal (exit 2) when absent."""
    paths: tuple[str, ...]
    what: str

    @property
    def key(self) -> str:
        return f"require:{self.what}"

    def done(self, ctx: Context) -> bool:
        return all(ctx.exists(p) for p in self.paths)

    def missing(self, ctx: Context) -> list[str]:
        return [p for p in self.paths if not ctx.exists(p)]


@dataclass(frozen=True)
class ValidPick:
    """A round's recorded pick names a checkpoint on disk."""
    out_dir: str

    @property
    def key(self) -> str:
        return f"pick:{self.out_dir}"

    def done(self, ctx: Context) -> bool:
        return ctx.layout.has_pick(self.out_dir)


@dataclass(frozen=True)
class WritePeak:
    """Record a round's pick in its `peak.txt`."""
    out_dir: str
    checkpoint: int

    @property
    def key(self) -> str:
        return f"peak:{self.out_dir}"

    def done(self, ctx: Context) -> bool:
        return ctx.layout.has_pick(self.out_dir)


@dataclass(frozen=True)
class Report:
    """A printed summary: `means` of a gate's reads, `best_of_n` over the
    pool (with the verifier's pick beside the vote when `scores` names its
    scoring run), the stage-by-stage `summary`."""
    kind: str
    args: tuple[str, ...] = ()
    scores: str = ""

    @property
    def key(self) -> str:
        return f"report:{self.kind}:{','.join(self.args)}"

    def done(self, ctx: Context) -> bool:
        return False


@dataclass(frozen=True)
class Later:
    """What cannot be enumerated yet: it follows a round or a pick not yet made."""
    text: str

    @property
    def key(self) -> str:
        return f"later:{self.text}"

    def done(self, ctx: Context) -> bool:
        return False


Step = Read | Train | Smoke | Round | Build | Band | Gate | CheckChain | Require | ValidPick | WritePeak | Report | Later

#: Steps the dry run's status counts as work; the others print or check.
WORK = (Read, Train, Smoke, Round, Build, Band, Require, ValidPick, WritePeak, Later)


@dataclass(frozen=True)
class Stage:
    name: str
    steps: Callable[[], list[Step]]              # recomputed from the disk on each call
    skip: Callable[[], str | None] = field(default=lambda: None)   # a reason when the stage does not apply


# ---------------------------------------------------------------- the disk

def scored(ctx: Context, run_dir: str) -> bool:
    """A read or a band is done when its `summary.json` is on disk and whole:
    a file cut short by a kill is not a scored run, and the read runs again
    (its own writer overwrites the directory)."""
    try:
        return "aggregate" in json.loads((ctx.root / run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False


def round_done(ctx: Context, out_dir: str, target: int | None = None) -> bool:
    """The round need not train again: `checkpoint-<target>` is on disk, or
    its `peak.txt` names a checkpoint that is (a released round that holds
    only its pick)."""
    target = ctx.target if target is None else target
    return is_adapter_dir(ctx.root / out_dir / f"checkpoint-{target}") or ctx.layout.has_pick(out_dir)


def checkpoints(ctx: Context, out_dir: str) -> list[int]:
    """The round's saved checkpoints: a `checkpoint-<N>` directory counts once
    its adapter config is written, not while the trainer is still saving it."""
    d = ctx.root / out_dir
    if not d.is_dir():
        return []
    return sorted(int(p.name.split("-", 1)[1]) for p in d.iterdir()
                  if p.name.startswith("checkpoint-") and p.name.split("-", 1)[1].isdigit() and is_adapter_dir(p))


def grid(ctx: Context, prefix: str, out_dir: str, seeds=None):
    """The round's grid reads on disk, restricted to its checkpoints on disk."""
    acc, _ = read_grid(f"{prefix}eval", ctx.graph.eval_stem, ctx.root / "data/runs",
                       seeds=seeds, checkpoints=set(checkpoints(ctx, out_dir)))
    return acc


def grid_pick(ctx: Context, prefix: str, out_dir: str, seeds=None) -> int | None:
    """A decision round's pick, by the band rule over every seed read."""
    acc = grid(ctx, prefix, out_dir, seeds)
    return pick_rule(acc) if acc else None


def grid_pick_best(ctx: Context, prefix: str, out_dir: str, seeds=None) -> int | None:
    """The full round's pick: the best mean over the grid's seeds."""
    acc = grid(ctx, prefix, out_dir, seeds)
    return pick_best(acc) if acc else None


def grid_leaders(ctx: Context, prefix: str, out_dir: str, n: int = 2) -> list[int]:
    acc = grid(ctx, prefix, out_dir, GATE_SEEDS)
    return grid_leaders_rule(acc, n) if acc else []


# ---------------------------------------------------------------- the reads

def reads(ctx: Context, prefix: str, chain, seeds, *, graph: Graph | None = None,
          training: bool = False, temperature: float | None = None, group: str = "read") -> list[Read]:
    """`<prefix>-s<seed>` on a graph's evaluation questions (its training
    questions with `training`), with that graph's file."""
    g = graph or ctx.graph
    bench = g.train_benchmark if training else g.eval_benchmark
    return [Read(f"{prefix}-s{s}", tuple(chain), bench, g.kb, s, temperature, group) for s in seeds]


def gate_reads(ctx: Context, prefix: str, chain) -> list[Read]:
    """Three greedy seeds on the evaluation questions and, with a transfer
    graph, three on it (`<prefix><token>-s<k>`)."""
    out = reads(ctx, f"{prefix}eval", chain, GATE_SEEDS, group="gate")
    if ctx.transfer:
        out += reads(ctx, f"{prefix}{ctx.transfer.transfer_token}", chain, GATE_SEEDS,
                     graph=ctx.transfer, group="transfer")
    return out


def codas(ctx: Context, prefix: str, chain, checkpoint: int | None = None) -> list[Read]:
    """The closing reads on a round's pick: the transfer graph at three greedy
    seeds and four sampled seeds for pass@5; named by checkpoint when given."""
    ck = f"-ck{checkpoint}" if checkpoint is not None else ""
    out = []
    if ctx.transfer:
        out += reads(ctx, f"{prefix}{ctx.transfer.transfer_token}{ck}", chain, GATE_SEEDS,
                     graph=ctx.transfer, group="transfer")
    out += reads(ctx, f"{prefix}pass5{ck}", chain, PASS5_SEEDS, temperature=PASS5_TEMPERATURE, group="pass5")
    return out


def grid_reads(ctx: Context, prefix: str, substrate, out_dir: str) -> list[Read]:
    """Three greedy seeds on every checkpoint the round has saved."""
    out = []
    for ck in checkpoints(ctx, out_dir):
        out += reads(ctx, f"{prefix}eval-ck{ck}", (*substrate, f"{out_dir}/checkpoint-{ck}"), GATE_SEEDS, group="grid")
    return out


def results_of(runs) -> tuple[str, ...]:
    return tuple(f"{r}/results.jsonl" for r in runs)


# ---------------------------------------------------------------- the stages

def _train(ctx: Context, stage: str, out_dir: str, final: str, log: str) -> Stage:
    return Stage(stage, lambda: [Train(stage, out_dir, final, log)])


def _gate(ctx: Context, name: str, prefix: str, chain) -> Stage:
    return Stage(name, lambda: [CheckChain(tuple(chain)), *gate_reads(ctx, prefix, chain), Gate(prefix, ctx.floor)])


def _band(ctx: Context) -> Stage:
    L = ctx.layout
    return Stage("band",
                 lambda: [CheckChain((L.sft1,)),
                          *reads(ctx, L.full_roll_prefix, (L.sft1,), BAND_SEEDS, training=True,
                                 temperature=ROLL_TEMPERATURE, group="roll")],
                 skip=lambda: f"{L.full_dir} is on disk" if round_done(ctx, L.full_dir) else None)


def _round1(ctx: Context) -> Stage:
    L = ctx.layout
    chain = (L.sft1,)

    def steps():
        if not round_done(ctx, L.full_dir):
            return [Require(results_of(L.full_band_runs), "the band's four passes"), CheckChain(chain),
                    Smoke("full", L.full_dir), Round("full", L.full_dir, L.full_prefix, chain, ctx.target),
                    Later("the grid, the pick and the codas, once the round is on disk")]
        s = [CheckChain(chain), *grid_reads(ctx, L.full_prefix, chain, L.full_dir)]
        if not all(r.done(ctx) for r in s[1:]):
            return s + [Later("the pick, three more seeds on it and the codas, once the grid is read")]
        # The pick is made once, the best mean over the grid's three seeds; an existing one is final.
        pk = L.pick(L.full_dir) if L.has_pick(L.full_dir) else grid_pick_best(ctx, L.full_prefix, L.full_dir, GATE_SEEDS)
        if pk is None:
            return s + [Later("the pick: the round has no checkpoint")]
        picked = (*chain, f"{L.full_dir}/checkpoint-{pk}")
        s += reads(ctx, f"{L.full_prefix}eval-ck{pk}", picked, FULL_EXTRA_SEEDS, group="post")
        s += codas(ctx, L.full_prefix, picked)
        s.append(WritePeak(L.full_dir, pk))
        return s

    return Stage("round1", steps)


def _stage3_rows_skip(ctx: Context) -> str | None:
    """Why stage 3's roll and build do not run: its adapter is saved, so its
    rows are not needed, or its mix exists already."""
    L = ctx.layout
    if is_adapter_dir(ctx.root / L.sft2):
        return f"{L.sft2} is saved"
    return f"{L.sft2_mix} exists" if ctx.exists(L.sft2_mix) else None


def _rftroll(ctx: Context) -> Stage:
    L = ctx.layout

    def steps():
        if not L.has_pick(L.full_dir):
            return [ValidPick(L.full_dir), Later("the rejection roll on round 1's pick")]
        chain = tuple(L.full_chain())
        return [CheckChain(chain), *reads(ctx, L.g2_roll_prefix, chain, G2_ROLL_SEEDS, training=True,
                                          temperature=ROLL_TEMPERATURE, group="roll")]

    return Stage("rftroll", steps, skip=lambda: _stage3_rows_skip(ctx))


def _rftbuild(ctx: Context) -> Stage:
    L = ctx.layout
    rolls = [a for r in L.g2_roll_runs for a in ("--rollouts", r)]
    return Stage("rftbuild",
                 lambda: [Require(results_of(L.g2_roll_runs), "the rejection roll's four passes"),
                          Build("keeps", L.g2_keeps,
                                ("scripts/traj/build_sft_rows.py", *rolls, "--exclude", ctx.graph.eval_benchmark,
                                 "--cap", "2", "--abstain-ratio", "0.1", "--out", L.g2_keeps),
                                log=f"{L.P}_rftbuild"),
                          Build("mix", L.sft2_mix,
                                ("scripts/traj/build_substrate_mix.py", "--demos", L.demos_mix, "--keeps", L.g2_keeps,
                                 "--keeps-name", L.g2_keeps_name, "--out", L.sft2_mix))],
                 skip=lambda: _stage3_rows_skip(ctx))


def _decision(ctx: Context, d: str) -> Stage:
    L = ctx.layout
    n = L.round(d)
    prefix, out_dir = n["prefix"], n["dir"]

    def steps():
        if d == "d2" and not L.has_pick(L.d1_dir):
            return [ValidPick(L.d1_dir), Later("d2, on d1's pick")]
        chain = (L.sft2,) if d == "d1" else tuple(L.d2_substrate())
        s = [CheckChain(chain)]
        if not round_done(ctx, out_dir):
            s += reads(ctx, n["roll_prefix"], chain, BAND_SEEDS, training=True, temperature=ROLL_TEMPERATURE, group="roll")
            rolls = [a for r in n["roll_runs"] for a in ("--rollouts", r)]
            s.append(Build(f"prefixes:{d}", n["prefixes"],
                           ("scripts/traj/build_prefixes.py", *rolls, "--exclude", ctx.graph.eval_benchmark,
                            "--kb", ctx.graph.kb, "--success-share", "0.25", "--seed", "7", "--out", n["prefixes"]),
                           always=True))
            s.append(Band(n["prefixes"], n["band_name"], n["band_run"], chain))
            s += [Smoke(d, out_dir), Round(d, out_dir, prefix, chain, ctx.target),
                  Later("the grid, eight seeds on the two best checkpoints, the pick and the codas, once the round is on disk")]
            return s
        s += grid_reads(ctx, prefix, chain, out_dir)
        if L.has_pick(out_dir):            # an existing pick is final: no leader reads, the codas on it
            pk = L.pick(out_dir)
            return s + codas(ctx, prefix, (*chain, f"{out_dir}/checkpoint-{pk}"), checkpoint=pk)
        if not all(r.done(ctx) for r in s[1:]):
            return s + [Later("eight seeds on the two best checkpoints, the pick and the codas, once the grid is read")]
        top = grid_leaders(ctx, prefix, out_dir, 2)
        for ck in top:
            s += reads(ctx, f"{prefix}eval-ck{ck}", (*chain, f"{out_dir}/checkpoint-{ck}"), LEADER_SEEDS, group="post")
        if not all(r.done(ctx) for r in s[1:]):
            return s + [Later("the pick and the codas, once the two leaders have eight seeds")]
        pk = grid_pick(ctx, prefix, out_dir)
        if pk is None:
            return s + [Later("the pick: the round has no checkpoint")]
        s += codas(ctx, prefix, (*chain, f"{out_dir}/checkpoint-{pk}"), checkpoint=pk)
        s.append(WritePeak(out_dir, pk))
        return s

    return Stage(d, steps)


def _final(ctx: Context) -> Stage:
    L = ctx.layout

    def steps():
        picks = [ValidPick(L.d1_dir), ValidPick(L.d2_dir)]
        if not all(p.done(ctx) for p in picks):
            return picks + [Later("the verifier roll and the pool on the served stack")]
        stack = tuple(L.served_stack())
        s = [CheckChain(stack)]
        if not ctx.exists(L.verifier_rows):
            s += reads(ctx, L.v_roll_prefix, stack, BAND_SEEDS, training=True, temperature=ROLL_TEMPERATURE, group="roll")
        s += reads(ctx, L.pool_prefix, stack, POOL_SEEDS, temperature=ROLL_TEMPERATURE, group="pool")
        s += [Report("best_of_n", tuple(L.pool_runs)), Report("summary")]
        return s

    return Stage("final", steps)


def _rows(ctx: Context) -> Stage:
    L = ctx.layout
    rolls = [a for r in L.v_roll_runs for a in ("--rollouts", r)]
    return Stage("rows",
                 lambda: [Require(results_of(L.v_roll_runs), "the verifier roll's four passes"),
                          Build("rows", L.verifier_rows,
                                ("scripts/select/build_verifier_data.py", *rolls, "--exclude", ctx.graph.eval_benchmark,
                                 "--out", L.verifier_rows),
                                log=f"{L.P}_verifier_rows")],
                 skip=lambda: f"{L.verifier_rows} exists" if ctx.exists(L.verifier_rows) else None)


def _score(ctx: Context) -> Stage:
    L = ctx.layout
    out = f"data/runs/{run_dir_name(L.verifier_select_name, ctx.graph.eval_benchmark)}"
    return Stage("score",
                 lambda: [Require(results_of(L.pool_runs), "the pool's 32 passes"),
                          Require((f"{L.verifier}/{ADAPTER_CONFIG}",), "the verifier"),
                          Build("score", f"{out}/summary.json",
                                ("scripts/select/verifier_select.py", "--verifier", f"hf:{L.verifier}",
                                 "--runs", *L.pool_runs,
                                 "--name", L.verifier_select_name),
                                log=f"{L.P}_verifier_select"),
                          Report("best_of_n", tuple(L.pool_runs), scores=out)])


def stages(ctx: Context) -> dict[str, Stage]:
    """The run's stages, in `STAGES` order."""
    L = ctx.layout
    out = [
        # Stage 1 is the one stage whose input no stage makes: without the
        # demonstrations it refuses (exit 2) with the command, instead of the
        # trainer failing on a missing file.
        Stage("sft1", lambda: ([] if is_adapter_dir(ctx.root / L.sft1) else
                               [Require((L.demos_mix,), f"stage 1's demonstrations "
                                        f"(prospector teach --graph {L.graph.name} --tag {L.spec.tag})")])
              + [Train("sft1", L.sft1_dir, L.sft1, f"{L.P}sft1_train")]),
        _gate(ctx, "gate1", L.gate1_prefix, (L.sft1,)),
        _band(ctx),
        _round1(ctx),
        _rftroll(ctx),
        _rftbuild(ctx),
        _train(ctx, "sft2", L.sft2_dir, L.sft2, f"{L.P}g2_train"),
        _gate(ctx, "gate2", L.g2_prefix, (L.sft2,)),
        _decision(ctx, "d1"),
        _decision(ctx, "d2"),
        _final(ctx),
        _rows(ctx),
        _train(ctx, "verifier", L.verifier_dir, L.verifier, f"{L.P}_verifier_train"),
        _score(ctx),
    ]
    assert tuple(s.name for s in out) == STAGES
    return {s.name: s for s in out}


def slice_of(start: str | None, stop: str | None) -> tuple[str, ...]:
    """The stages from `start` to `stop` inclusive; a name that is not a stage is refused."""
    start = start or STAGES[0]
    stop = stop or STAGES[-1]
    for s in (start, stop):
        if s not in STAGES:
            refuse(f"{s!r} is not a stage (the stages: {' '.join(STAGES)})")
    i, j = STAGES.index(start), STAGES.index(stop)
    if i > j:
        refuse(f"--start-at {start} comes after --stop-after {stop}")
    return STAGES[i:j + 1]


# ---------------------------------------------------------------- the dry run

@dataclass(frozen=True)
class Planned:
    name: str
    status: str                 # done, skipped, todo
    reason: str | None          # why skipped
    steps: list


def target_problem(ctx: Context) -> str | None:
    """Why `TARGET` is not a checkpoint every GRPO round saves, or None. A
    round that never writes its target runs on to `final/`, counts as having
    left no output, and is removed and retrained on the next launch."""
    import yaml

    from ..benchmarks import ROOT
    from .names import TRAINING
    for stage, st in TRAINING.items():
        if st.trainer != "grpo":
            continue
        hp = yaml.safe_load((ROOT / st.config).read_text(encoding="utf-8"))
        last, every = int(hp["max_steps"]), int(hp["save_steps"])
        if not (0 < ctx.target <= last and ctx.target % every == 0):
            return (f"TARGET={ctx.target} is not a checkpoint the {stage} round saves: {st.config} "
                    f"saves every {every} steps up to {last}")
    return None


def budget_problem(ctx: Context, step) -> str | None:
    """Why a stored read cannot stand in for this run's: it records another
    turn budget than `TURN_CAP` (a different budget is a different policy).
    A read that records none is taken as it is."""
    try:
        summary = json.loads((ctx.root / step.run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    read_at = (summary.get("policy") or {}).get("turn_max_tokens")
    if read_at is None or int(read_at) == ctx.turn_cap:
        return None
    return (f"{step.run_dir} was read at {read_at} tokens a turn and TURN_CAP is {ctx.turn_cap}: "
            f"move the stored reads aside or use another --tag")


def plan(ctx: Context, start: str | None = None, stop: str | None = None) -> list[Planned]:
    """Each stage of the slice with its status against the disk now. A
    `TARGET` no round saves, a recorded pick that names no checkpoint on disk,
    and a stored read made at another turn budget than `TURN_CAP`, are refused
    first."""
    problem = target_problem(ctx)
    if problem:
        refuse(problem)
    L = ctx.layout
    for out_dir in (L.full_dir, L.d1_dir, L.d2_dir):
        if ctx.exists(f"{out_dir}/peak.txt") and not L.has_pick(out_dir):
            refuse(f"{L.pick_problem(out_dir)}: a recorded pick is final, so remove the file "
                   f"to have the round picked again from its grid")
    table = stages(ctx)
    out = []
    for name in slice_of(start, stop):
        st = table[name]
        why = st.skip()
        if why:
            out.append(Planned(name, "skipped", why, []))
            continue
        steps = st.steps()
        for s in steps:
            if isinstance(s, (Read, Band)) and s.done(ctx):
                problem = budget_problem(ctx, s)
                if problem:
                    refuse(problem)
        work = [s for s in steps if isinstance(s, WORK) and not s.done(ctx)]
        out.append(Planned(name, "todo" if work else "done", None, steps))
    return out


def _describe(ctx: Context, steps: list) -> str:
    """One line for a stage's steps: reads counted by group, the rest named."""
    parts = []
    by_group: dict[str, list[Read]] = {}
    for s in steps:
        if isinstance(s, Read):
            by_group.setdefault(s.group, []).append(s)
    for s in steps:
        if isinstance(s, Read):
            if s.group in by_group:
                rs = by_group.pop(s.group)
                todo = sum(not r.done(ctx) for r in rs)
                parts.append(f"{s.group} {todo} of {len(rs)} to read")
        elif isinstance(s, Train):
            parts.append(("have " if s.done(ctx) else "train ") + s.final)
        elif isinstance(s, Round):
            parts.append(("have " if s.done(ctx) else "train ") + s.out_dir)
        elif isinstance(s, Smoke):
            parts.append(f"smoke {s.stage}")
        elif isinstance(s, Build):
            parts.append(("have " if s.done(ctx) else "build ") + s.out)
        elif isinstance(s, Band):
            parts.append(("have " if s.done(ctx) else "sample ") + s.run_dir)
        elif isinstance(s, Require):
            missing = s.missing(ctx)
            if missing:
                parts.append(f"needs {s.what} ({len(missing)} missing: {missing[0]}" + (", ..." if len(missing) > 1 else "") + ")")
        elif isinstance(s, ValidPick):
            parts.append(("pick checkpoint-" + str(ctx.layout.pick(s.out_dir)) if s.done(ctx)
                          else f"no valid pick in {s.out_dir}/peak.txt"))
        elif isinstance(s, WritePeak):
            parts.append(f"pick checkpoint-{s.checkpoint}" + ("" if s.done(ctx) else " to record"))
        elif isinstance(s, Gate):
            parts.append(f"gate at {s.floor}")
        elif isinstance(s, Later):
            parts.append(f"then {s.text}")
    return "; ".join(parts)


def render(ctx: Context, planned: list[Planned], names: bool = True) -> str:
    """The dry run: the derived names, then one line per stage."""
    L = ctx.layout
    lines = [f"=== plan {L.P}: GRAPH={L.graph.name} SIZE={L.spec.size} MODEL={L.spec.base_model} TAG={L.spec.tag} "
             f"TRANSFER={L.transfer.name if L.transfer else 'none'} TARGET={ctx.target} TURN_CAP={ctx.turn_cap} "
             f"stages {planned[0].name}..{planned[-1].name}"]
    if names:
        lines += [f"  {k}={v}" for k, v in L.env().items()]
    lines.append(f"demos:    {'have ' if ctx.exists(L.demos_mix) else 'need '}{L.demos_mix} (prospector teach --graph {L.graph.name} --tag {L.spec.tag})")
    for g in (ctx.graph, ctx.transfer):
        if g is not None and not ctx.exists(g.kb):
            lines.append(f"graph:    need {g.kb} (data/raw/README.md says how to fetch it); a run refuses at its first stage")
    for p in planned:
        if p.status == "skipped":
            lines.append(f"{p.name + ':':9s} skipped ({p.reason})")
        else:
            lines.append(f"{p.name + ':':9s} {_describe(ctx, p.steps)}")
    return "\n".join(lines)
