"""Execute a plan: the leaves (a read, a training run, a builder, the band),
the supervision (a round's retry loop and watcher, a smoke's three steps) and
the loop over a stage's steps.

Everything heavy is a subprocess, launched with the environment the GPU
machine needs; nothing here imports torch or vLLM. The `Runner` owns the
processes, the clock, the engine reaper and `nvidia-smi`, and is injected, so
the tests run the whole ladder with a stub (tests/test_recipe_run.py). Every
command is printed before it runs, so a stage can be rerun by hand from the
log, and each leaf writes its own log under `logs/` (the recipe's README
lists the names).
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import signal
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

from ..train.adapters import is_adapter_dir
from .names import TRAINING, read_manifest
from .plan import (Band, Build, CheckChain, Context, Gate, Later, Read, Report, Require, Round, Smoke, Stage,
                   Train, ValidPick, WritePeak, grid, plan, stages)

#: Steps that must leave their output behind once run.
PRODUCES = (Read, Train, Build, Band, Round, WritePeak)
#: What ends a round's attempts early: the trainer refused the run, and would again.
REFUSALS = ("missing its inputs", "holds a run trained from other inputs")
#: What fails a smoke, read in its log.
SMOKE_FAILURE = re.compile(r"Traceback|CUDA out of memory|SystemExit|missing its inputs")
SMOKE_STEPS = 3
ROUND_ATTEMPTS = 8
WATCH_EVERY = 60          # seconds between the watcher's looks for the target checkpoint
WATERMARK_EVERY = 5       # seconds between nvidia-smi samples
READ_PAUSE = 10           # seconds before a read's engine starts, after the reaper
#: The run names of a round whose reads a retrain discards, by suffix after the round's prefix.
RETRAIN_GLOBS = ("eval-ck*--*", "pass5-*--*")


class Failed(SystemExit):
    """A stage failed (exit 1). `code` is the exit status, `message` the reason."""
    def __init__(self, message: str, code: int = 1):
        super().__init__(code)
        self.message = message

    def __str__(self) -> str:
        return self.message


class Refused(Failed):
    """A prerequisite is missing or an argument is wrong (exit 2), before anything ran."""
    def __init__(self, message: str):
        super().__init__(message, 2)


class GateFailed(Failed):
    """A gate's reads fell under the floor (exit 3)."""
    def __init__(self, message: str):
        super().__init__(message, 3)


# ---------------------------------------------------------------- the runner

def environment(base: dict | None = None) -> dict:
    """The environment the GPU machine's stack needs: the model cache,
    unbuffered logs, torch-native sampling in vLLM, expandable CUDA segments,
    and the environment's own `bin` first on PATH, for flashinfer's `ninja`
    (launch/README.md, section 4)."""
    env = dict(os.environ if base is None else base)
    env.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    env["PYTHONUNBUFFERED"] = "1"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    return env


class Proc:
    """A launched process in its own session, so its engine children die with it."""
    def __init__(self, popen: subprocess.Popen):
        self.popen = popen

    def poll(self):
        return self.popen.poll()

    def kill_group(self) -> None:
        try:
            os.killpg(self.popen.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.popen.wait()


class Runner:
    """Processes, the clock and the card, for the real machine."""
    def __init__(self, root: Path):
        self.root = Path(root)
        self.env = environment()
        (self.root / "logs").mkdir(exist_ok=True)

    def python(self) -> str:
        """The interpreter that runs `prospector`, for every leaf: on a CUDA
        machine `.venv` holds the trainers and vLLM (launch/setup.sh)."""
        return sys.executable

    def run(self, cmd: list[str], log: Path | None = None, env: dict | None = None) -> int:
        """Run a command to completion; stdout and stderr to `log`, or to ours;
        `env` replaces the runner's environment (the teacher pass drops a key)."""
        env = self.env if env is None else env
        if log is None:
            return subprocess.run(cmd, cwd=self.root, env=env).returncode
        with open(log, "w") as fh:
            return subprocess.run(cmd, cwd=self.root, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode

    def start(self, cmd: list[str], log: Path) -> Proc:
        """Launch a command in its own session, to be watched and, if need be, killed with its children."""
        fh = open(log, "w")
        return Proc(subprocess.Popen(cmd, cwd=self.root, env=self.env, stdout=fh, stderr=subprocess.STDOUT,
                                     start_new_session=True))

    def capture(self, cmd: list[str]) -> tuple[int, str]:
        p = subprocess.run(cmd, cwd=self.root, env=self.env, capture_output=True, text=True)
        return p.returncode, p.stdout + p.stderr

    def reap(self, everything: bool = False) -> None:
        """Kill the vLLM engines orphaned to PID 1, which hold most of the card
        and make every later engine start fail; `everything` kills every engine,
        for the grid after a round, when nothing else may hold the card."""
        out = subprocess.run(["pgrep", "-f", "[E]ngineCore"], capture_output=True, text=True).stdout.split()
        for pid in out:
            ppid = subprocess.run(["ps", "-o", "ppid=", "-p", pid], capture_output=True, text=True).stdout.strip()
            if everything or ppid == "1":
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    print(f"reaped EngineCore {pid}")
                except (ProcessLookupError, PermissionError):
                    pass

    def gpu_line(self) -> str | None:
        """`<memory used MiB>, <utilization %>` from nvidia-smi, or None without a card."""
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None

    def gpu_total(self) -> int | None:
        """The card's memory in MiB from nvidia-smi, or None without a card."""
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return int(out.stdout.strip().splitlines()[0]) if out.returncode == 0 and out.stdout.strip() else None

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def now(self) -> str:
        return time.strftime("%m-%d %H:%M:%S")


def say(text: str) -> None:
    print(text, flush=True)


def shown(cmd: list[str]) -> str:
    return " ".join(cmd)


# ---------------------------------------------------------------- the leaves

def trainer_args(ctx: Context, stage: str) -> list[str]:
    """The flags that name a training run to the two trainer scripts."""
    L = ctx.layout
    args = ["--graph", L.graph.name, "--size", L.spec.size, "--tag", L.spec.tag,
            "--model", L.spec.base_model, "--stage", stage]
    if TRAINING[stage].trainer == "grpo":
        args += ["--kb", L.graph.kb]
    return args


def log_lines(path: Path, pattern: str, last: int | None = None) -> list[str]:
    if not path.exists():
        return []
    lines = [l.rstrip("\n") for l in path.read_text(errors="replace").splitlines() if re.search(pattern, l)]
    return lines[-last:] if last else lines


def eval_read(step: Read, ctx: Context, runner: Runner) -> None:
    """One read at the recipe's protocol, retried once: the engine-start race
    ("Free memory on device ... less than desired") is transient."""
    L = ctx.layout
    cmd = [runner.python(), "scripts/run_agent_eval.py", "--model", f"vllm:{L.spec.base_model}",
           "--adapter", ",".join(step.chain), "--max-turns", "20",
           "--turn-max-tokens", str(ctx.turn_cap), "--name", step.name, "--benchmark", step.benchmark,
           "--kb", step.kb, "--seed", str(step.seed)]
    if step.temperature is not None:
        cmd += ["--temperature", str(step.temperature)]
    log = ctx.root / "logs" / f"{step.name}.log"
    for attempt in (1, 2):
        runner.reap()
        runner.sleep(READ_PAUSE)
        say(f"=== RUN {step.name} start {runner.now()}\n    {shown(cmd)}")
        rc = runner.run(cmd, log)
        if rc == 0 and step.done(ctx):
            for line in log_lines(log, r"^(ALL|hard)"):
                say(line)
            say(f"=== RUN {step.name} end {runner.now()}")
            return
        say(f"!!! RUN {step.name} FAILED (attempt {attempt}, exit {rc}; logs/{step.name}.log)")
    raise Failed(f"read {step.name} failed twice (logs/{step.name}.log)")


def train(step: Train, ctx: Context, runner: Runner) -> None:
    """A fresh SFT run: the output directory is removed first, since a stale
    one would silently resume."""
    out = ctx.root / step.out_dir
    if out.exists():
        shutil.rmtree(out)
    runner.reap()
    cmd = [runner.python(), "scripts/train_sft.py", *trainer_args(ctx, step.stage)]
    log = ctx.root / "logs" / f"{step.log}.log"
    say(f"    {shown(cmd)}")
    rc = runner.run(cmd, log)
    if rc != 0:
        why = log_lines(log, r"Error|Traceback|missing its inputs|holds a run", last=1)
        raise Failed(f"{step.stage} training exited {rc} ({why[0] if why else 'see'} logs/{step.log}.log)")
    if not step.done(ctx):
        raise Failed(f"{step.stage} training left no adapter at {step.final}")
    for line in log_lines(log, r"transcripts|dropped|Adapter saved|train_runtime|train_loss", last=4):
        say(line)
    if step.stage == "verifier" and log_lines(log, r"dropped [1-9][0-9]* transcripts"):
        say(f"!!! the verifier SFT dropped rows at its max_length; read logs/{step.log}.log")


def build(step: Build, ctx: Context, runner: Runner) -> None:
    cmd = [runner.python(), *step.command]
    say(f"    {shown(cmd)}")
    if step.log:
        log = ctx.root / "logs" / f"{step.log}.log"
        rc = runner.run(cmd, log)
        for line in log_lines(log, r".", last=12):
            say(line)
    else:
        rc = runner.run(cmd)
    if rc != 0:
        raise Failed(f"{step.label} exited {rc}" + (f" (logs/{step.log}.log)" if step.log else ""))
    if not ctx.exists(step.out):
        raise Failed(f"{step.label} wrote nothing at {step.out}")


def band(step: Band, ctx: Context, runner: Runner) -> None:
    """The band: eight continuations per decision state, into the band run."""
    L = ctx.layout
    run_dir = ctx.root / step.run_dir
    if run_dir.exists():
        shutil.rmtree(run_dir)
    runner.reap()
    cmd = [runner.python(), "scripts/traj/sample_prefixes.py", "--prefixes", step.prefixes,
           "--benchmark", L.graph.train_benchmark, "--kb", L.graph.kb, "--model", f"vllm:{L.spec.base_model}",
           "--adapter", ",".join(step.chain), "--temperature", "0.7",
           "--n", "8", "--seed", "911", "--chunk", "512", "--name", step.name,
           "--turn-max-tokens", str(ctx.turn_cap)]
    prefix = step.name.rsplit("-s", 1)[0]
    log = ctx.root / "logs" / f"{prefix}_stage0.log"
    say(f"    {shown(cmd)}")
    rc = runner.run(cmd, log)
    if rc != 0 or not step.done(ctx):
        raise Failed(f"the band {step.name} failed (exit {rc}; logs/{prefix}_stage0.log)")
    for line in log_lines(log, r"in band|class ", last=4):
        say(line)


def sd(xs) -> float:
    """The sample standard deviation the README and the paper report beside a
    mean over seeds; 0 for a single seed."""
    xs = list(xs)
    return st.stdev(xs) if len(xs) > 1 else 0.0


def seed_mean(ctx: Context, prefix: str, stem: str, floor: float | None = None) -> tuple[str, bool]:
    """One line of means over the scored seeds `<prefix>-s*--<stem>`; with a
    floor it is a gate: passed when mean gated_acc >= floor and mean calls >= 11."""
    runs = []
    for path in sorted(glob.glob(str(ctx.root / "data/runs" / f"{prefix}-s*--{stem}" / "summary.json"))):
        with open(path) as fh:
            runs.append(json.load(fh)["aggregate"]["ALL"])
    if not runs:
        return f"{prefix}: no scored seeds", False
    m = {k: st.mean(float(r.get(k) or 0) for r in runs)
         for k in ("gated_acc", "n_calls", "reached_answer", "gate_pass", "halluc_strict")}
    line = (f"{prefix}: gated_acc {m['gated_acc']:.3f} +/- {sd(float(r['gated_acc']) for r in runs):.3f} "
            f"(n={len(runs)}) calls {m['n_calls']:.1f} reach {m['reached_answer']:.3f} "
            f"gate {m['gate_pass']:.3f} halluc {m['halluc_strict']:.3f}")
    if floor is None:
        return line, True
    ok = m["gated_acc"] >= floor and m["n_calls"] >= 11.0
    return f"GATE {line} | floor {floor} and calls >= 11: {'PASSED' if ok else 'FAILED'}", ok


def gate(step: Gate, ctx: Context) -> None:
    """The gate's means, then the floor on the evaluation questions."""
    say(seed_mean(ctx, f"{step.prefix}eval", ctx.graph.eval_stem)[0])
    if ctx.transfer:
        say(seed_mean(ctx, f"{step.prefix}{ctx.transfer.transfer_token}", ctx.transfer.eval_stem)[0])
    line, ok = seed_mean(ctx, f"{step.prefix}eval", ctx.graph.eval_stem, step.floor)
    say(line)
    if not ok:
        if ctx.force_gate:
            say("--force-gate: continuing past a failed gate")
            return
        raise GateFailed(f"{step.prefix} is under the floor {step.floor}")


def check_chain(step: CheckChain, ctx: Context) -> None:
    """Every adapter of the chain is on disk, none whose `run.json` names a
    graph was trained on another graph than this run's, and the graph files
    the stage's reads open exist (a leaf would otherwise fail on the first
    read, after the gates before it ran)."""
    for g in (ctx.graph, ctx.transfer):
        if g is not None and not ctx.exists(g.kb):
            raise Refused(f"no graph file at {g.kb}: data/raw/README.md says how to fetch each graph")
    for a in step.chain:
        if not is_adapter_dir(ctx.root / a):
            raise Refused(f"no adapter at {a}")
        m = read_manifest(ctx.root / a)
        if m and m.get("graph", {}).get("name") not in (None, ctx.graph.name):
            raise Refused(f"{a} was trained on the graph {m['graph']['name']}; this run is on {ctx.graph.name}")


def require(step: Require, ctx: Context) -> None:
    missing = step.missing(ctx)
    if missing:
        raise Refused(f"{step.what}: {len(missing)} missing, the first {missing[0]}")


def valid_pick(step: ValidPick, ctx: Context) -> None:
    problem = ctx.layout.pick_problem(step.out_dir)
    if problem:
        raise Refused(problem)


def write_peak(step: WritePeak, ctx: Context) -> None:
    """Record the pick, after printing each checkpoint's mean gated_acc over
    every seed read on it. A decision round picks on that table; round 1 picks
    over the three grid seeds before its pick gets three more, so its pick
    need not top the printed table."""
    L = ctx.layout
    prefix = {L.full_dir: L.full_prefix, L.d1_dir: L.d1_prefix, L.d2_dir: L.d2_prefix}.get(step.out_dir)
    acc = grid(ctx, prefix, step.out_dir) if prefix else {}
    for ck in sorted(acc):
        xs = [g for _, g in acc[ck]]
        say(f"grid {prefix}eval-ck{ck}: gated_acc {st.mean(xs):.3f} +/- {sd(xs):.3f} (n={len(xs)})"
            + ("  <- pick" if ck == step.checkpoint else ""))
    (ctx.root / step.out_dir / "peak.txt").write_text(f"{step.checkpoint}\n")
    say(f"pick recorded: {step.out_dir}/peak.txt = checkpoint-{step.checkpoint}")


def report(step: Report, ctx: Context, runner: Runner) -> None:
    L = ctx.layout
    if step.kind == "best_of_n":
        cmd = [runner.python(), "scripts/select/best_of_n.py", *step.args, "--recompute"]
        if step.scores:
            cmd += ["--scores", step.scores]
        say(f"--- gold-free vote and ceiling over the {len(step.args)}"
            + (" and the verifier's pick" if step.scores else "") + f":\n    {shown(cmd)}")
        rc, out = runner.capture(cmd)
        for line in out.splitlines():
            if re.match(r"^best-of-(1|2|5|10|32) |^run 0|^verifier pick", line):
                say(line)
        if rc != 0:
            say(f"!!! best_of_n exited {rc}")
    elif step.kind == "summary":
        x = ctx.transfer
        say("--- this size's results, stage by stage (mean over the seeds on disk; the evaluation questions"
            + (f", then {x.name}" if x else "") + "):")
        full = str(L.pick(L.full_dir)) if L.has_pick(L.full_dir) else "?"
        d1, d2 = L.pick(L.d1_dir), L.pick(L.d2_dir)
        tok = x.transfer_token if x else ""
        rows = [(f"{L.gate1_prefix}eval", f"{L.gate1_prefix}{tok}"),
                (f"{L.full_prefix}eval-ck{full}", f"{L.full_prefix}{tok}"),
                (f"{L.g2_prefix}eval", f"{L.g2_prefix}{tok}"),
                (f"{L.d1_prefix}eval-ck{d1}", f"{L.d1_prefix}{tok}-ck{d1}"),
                (f"{L.d2_prefix}eval-ck{d2}", f"{L.d2_prefix}{tok}-ck{d2}")]
        for home, transfer in rows:
            say(seed_mean(ctx, home, ctx.graph.eval_stem)[0])
            if x:
                say(seed_mean(ctx, transfer, x.eval_stem)[0])
    else:
        raise Refused(f"unknown report {step.kind}")


# ---------------------------------------------------------------- the supervision

class Watermark:
    """`nvidia-smi` memory and utilization every WATERMARK_EVERY seconds, into
    a log, sampled from the supervising loop; `peak()` is the highest memory seen."""
    def __init__(self, runner: Runner, path: Path):
        self.runner, self.path, self.samples = runner, path, []
        self.fh = open(path, "w")

    def sample(self) -> None:
        line = self.runner.gpu_line()
        if line:
            self.fh.write(f"{self.runner.now()} {line}\n")
            self.fh.flush()
            try:
                self.samples.append(int(line.split(",")[0].strip()))
            except ValueError:
                pass

    def close(self) -> int | None:
        self.fh.close()
        return max(self.samples) if self.samples else None


def wait_watching(proc: Proc, runner: Runner, marks: Watermark, *, every: int, until, grace: int = 0,
                  fail_when=None) -> str:
    """Poll a launched process: a watermark sample each tick, and every `every`
    seconds `until()`; returns 'target' when `until()` held (the process is
    killed `grace` seconds later), 'failed' when `fail_when()` held, 'exited'
    when the process ended by itself."""
    elapsed = 0.0
    try:
        while proc.poll() is None:
            runner.sleep(WATERMARK_EVERY)
            elapsed += WATERMARK_EVERY
            marks.sample()
            if fail_when is not None and fail_when():
                return "failed"
            if elapsed % every < WATERMARK_EVERY and until():
                if grace:
                    runner.sleep(grace)
                proc.kill_group()
                return "target"
    except BaseException:
        # The watcher is leaving (a SIGINT, a SIGTERM that cli.main turns into
        # SystemExit, a bug): the process it launched goes with it. Left
        # alone, a trainer keeps writing into a directory the next launch
        # removes from under it.
        proc.kill_group()
        raise
    return "exited"


def retrain_cleanup(step: Round, ctx: Context) -> None:
    """A retrained round gets fresh reads: the grid and the codas skip a run
    whose name is scored, and the names carry the checkpoint number, not the weights."""
    out = ctx.root / step.out_dir
    if out.exists():
        shutil.rmtree(out)
    patterns = [f"{step.prefix}{g}" for g in RETRAIN_GLOBS]
    if ctx.transfer:
        patterns.append(f"{step.prefix}{ctx.transfer.transfer_token}-*--*")
    for pat in patterns:
        for d in glob.glob(str(ctx.root / "data/runs" / pat)):
            shutil.rmtree(d, ignore_errors=True)


def supervised_round(step: Round, ctx: Context, runner: Runner) -> None:
    """The round, relaunched on death until its target checkpoint, resuming
    from the last checkpoint each time; a watcher stops the trainer once the
    target is on disk, since the trainer itself would go on to `final/`."""
    retrain_cleanup(step, ctx)
    out = ctx.root / step.out_dir
    if out.exists():
        raise Refused(f"{step.out_dir} exists: a stale directory would silently resume; remove it first")
    cmd = [runner.python(), "scripts/train_traj_grpo.py", *trainer_args(ctx, step.stage)]
    marks = Watermark(runner, ctx.root / "logs" / f"{step.prefix}_r_watermarks.log")
    say(f"=== ROUND {step.stage} start {runner.now()} target {step.target}\n    {shown(cmd)}")
    target = out / f"checkpoint-{step.target}"

    def reached() -> bool:
        return (out / "final").exists() or target.exists()

    try:
        for attempt in range(1, ROUND_ATTEMPTS + 1):
            if reached():
                say(f"SUPERVISOR: target reached before attempt {attempt}")
                break
            runner.reap()
            runner.sleep(5)
            log = ctx.root / "logs" / f"{step.prefix}_r_a{attempt}.log"
            say(f"SUPERVISOR: attempt {attempt} starts {runner.now()} (logs/{log.name})")
            proc = runner.start(cmd, log)
            outcome = wait_watching(proc, runner, marks, every=WATCH_EVERY, until=target.exists, grace=WATCH_EVERY)
            rc = proc.poll()
            cks = sorted(int(p.name.split("-")[1]) for p in out.glob("checkpoint-*")) if out.exists() else []
            say(f"SUPERVISOR: attempt {attempt} {outcome} exit={rc} last_checkpoint={cks[-1] if cks else 'none'}")
            if (out / "final").exists():
                say("SUPERVISOR: final saved")
                break
            if log_lines(log, "|".join(REFUSALS)):
                say(f"SUPERVISOR: the trainer refused the run; read logs/{log.name}")
                break
    finally:
        peak = marks.close()
    say(f"=== ROUND {step.stage} supervisor exited {runner.now()}; peak memory {peak or '?'} MiB")
    if not reached():
        raise Failed(f"the {step.stage} round did not reach checkpoint-{step.target} "
                     f"after {ROUND_ATTEMPTS} attempts (logs/{step.prefix}_r_a*.log)")
    say("checkpoints: " + " ".join(p.name for p in sorted(out.glob("checkpoint-*"))))


def smoke(stage: str, out_dir: str, ctx: Context, runner: Runner) -> None:
    """A GRPO stage's three-step twin under the step timers: killed after
    SMOKE_STEPS `t_gen` lines, failed on a traceback, an OOM or a refusal."""
    name = f"{Path(out_dir).name}_smoke"
    log = ctx.root / "logs" / f"{name}.log"
    cmd = [runner.python(), "scripts/train_traj_grpo.py", *trainer_args(ctx, stage), "--smoke"]
    runner.reap()
    say(f"=== SMOKE {name} start {runner.now()} (kill after {SMOKE_STEPS} steps)\n    {shown(cmd)}")
    marks = Watermark(runner, ctx.root / "logs" / f"{name}_watermarks.log")
    proc = runner.start(cmd, log)
    steps_seen = lambda: len(log_lines(log, "t_gen"))     # noqa: E731
    outcome = wait_watching(proc, runner, marks, every=30, until=lambda: steps_seen() >= SMOKE_STEPS,
                            fail_when=lambda: bool(log_lines(log, SMOKE_FAILURE.pattern)))
    if proc.poll() is None:
        proc.kill_group()
    runner.sleep(5)
    peak = marks.close()
    runner.reap()
    for line in log_lines(log, r"\{[^{]*'t_gen'[^}]*\}", last=SMOKE_STEPS):
        say(line[line.find("{'loss'"):] if "{'loss'" in line else line)
    say(f"--- {name} peak memory {peak or '?'} MiB")
    ok = outcome == "target" or (outcome == "exited" and steps_seen() >= SMOKE_STEPS)
    say(f"=== SMOKE {name} {'end' if ok else 'FAILED'} {runner.now()}"
        + ("" if ok else f" ({outcome}, {steps_seen()} of {SMOKE_STEPS} steps; logs/{name}.log)"))
    if not ok:
        raise Failed(f"the {stage} smoke failed ({outcome}; logs/{name}.log)")


def sft_smoke(stage: str, out_dir: str, ctx: Context, runner: Runner, *, remove: bool = False) -> int | None:
    """An SFT stage's three-step twin under the watermark; returns the peak
    memory. With `remove`, the twin's directory goes afterwards (verify)."""
    name = f"{Path(out_dir).name}_smoke"
    log = ctx.root / "logs" / f"{name}.log"
    cmd = [runner.python(), "scripts/train_sft.py", *trainer_args(ctx, stage), "--smoke"]
    runner.reap()
    say(f"=== SMOKE {name} start {runner.now()}\n    {shown(cmd)}")
    marks = Watermark(runner, ctx.root / "logs" / f"{name}_watermarks.log")
    proc = runner.start(cmd, log)
    wait_watching(proc, runner, marks, every=WATCH_EVERY, until=lambda: False)
    rc = proc.poll()
    peak = marks.close()
    for line in log_lines(log, r"transcripts|trainable params|[Ff]alling back|Adapter saved|train_runtime|Error|OOM|out of memory|missing its inputs", last=5):
        say(line[:160])
    say(f"SFT smoke exit {rc}, peak GPU memory {peak or '?'} MiB")
    if remove:
        shutil.rmtree(ctx.root / f"{out_dir}_smoke", ignore_errors=True)
    if rc != 0:
        raise Failed(f"the {stage} smoke exited {rc} (logs/{name}.log)")
    return peak


# ---------------------------------------------------------------- the loop

def execute_step(step, ctx: Context, runner: Runner) -> None:
    if isinstance(step, Read):
        eval_read(step, ctx, runner)
    elif isinstance(step, Train):
        train(step, ctx, runner)
    elif isinstance(step, Smoke):
        smoke(step.stage, step.out_dir, ctx, runner)
    elif isinstance(step, Round):
        supervised_round(step, ctx, runner)
    elif isinstance(step, Build):
        build(step, ctx, runner)
    elif isinstance(step, Band):
        band(step, ctx, runner)
    elif isinstance(step, Gate):
        gate(step, ctx)
    elif isinstance(step, CheckChain):
        check_chain(step, ctx)
    elif isinstance(step, Require):
        require(step, ctx)
    elif isinstance(step, ValidPick):
        valid_pick(step, ctx)
    elif isinstance(step, WritePeak):
        write_peak(step, ctx)
    elif isinstance(step, Report):
        report(step, ctx, runner)
    elif isinstance(step, Later):
        raise Refused(f"cannot continue: {step.text}")
    else:
        raise Refused(f"unknown step {step!r}")


def run_stage(stage: Stage, ctx: Context, runner: Runner) -> None:
    """Run a stage's steps in order, recomputing them from the disk after
    each, so what a step unlocks (a pick, a round's checkpoints) follows."""
    executed: set[str] = set()
    grid_reaped = False
    while True:
        pending = [s for s in stage.steps() if not (s.done(ctx) or s.key in executed)]
        if not pending:
            return
        step = pending[0]
        if isinstance(step, Read) and step.group == "grid" and not grid_reaped:
            runner.reap(everything=True)      # the grid after a round, when nothing else may hold the card
            grid_reaped = True
        execute_step(step, ctx, runner)
        executed.add(step.key)
        produced = ctx.exists(step.out) if isinstance(step, Build) else step.done(ctx)
        if isinstance(step, PRODUCES) and not produced:
            raise Failed(f"{step.key} ran and left no output")


#: The smallest card the recipe was run on, in MiB (an 80 GB card): a run on
#: less is warned, not refused, since a read fits where a GRPO round may not.
CARD_MIB = 74 * 1024


def execute(ctx: Context, runner: Runner, start: str | None = None, stop: str | None = None) -> int:
    """Run the plan's slice; the exit code (0, or a failure's)."""
    L = ctx.layout
    table = stages(ctx)
    planned = plan(ctx, start, stop)
    say(f"=== RUN {L.P} start {runner.now()}: GRAPH={L.graph.name} SIZE={L.spec.size} MODEL={L.spec.base_model} "
        f"TAG={L.spec.tag} TRANSFER={L.transfer.name if L.transfer else 'none'} TARGET={ctx.target} "
        f"TURN_CAP={ctx.turn_cap} stages {planned[0].name}..{planned[-1].name}")
    total = getattr(runner, "gpu_total", lambda: None)()
    if total is None:
        say("card: none found (nvidia-smi)")
    elif total < CARD_MIB:
        say(f"card: {total} MiB, under the {CARD_MIB} MiB the recipe was run on; a GRPO round may not fit "
            f"(launch/README.md, section 1)")
    else:
        say(f"card: {total} MiB")
    for p in planned:
        label = p.name.upper()
        if p.status == "skipped":
            say(f"=== {label} skipped ({p.reason})")
            continue
        say(f"=== {label} start {runner.now()}")
        try:
            run_stage(table[p.name], ctx, runner)
        except Failed as e:
            say(f"!!! {label} {e.message} {runner.now()}")
            say(f"!!! RUN {L.P} STOPPED (exit {e.code}) {runner.now()}")
            return e.code
        say(f"=== {label} done {runner.now()}")
    say(f"=== RUN {L.P} COMPLETE {runner.now()}")
    return 0
