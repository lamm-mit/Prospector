"""A stub `Runner` for the recipe's tests: records every command, writes
what each leaf would write (a read's summary, a trainer's checkpoints and
adapter, a builder's file, a probe's outputs), keeps a clock that `sleep`
advances, and scripts the launched trainers.

`behaviour` bends a GRPO round (`die_at_60`, `refuse`, `never`), a GRPO
smoke (`traceback`) or the machine check (`no_cuda`).
"""
import json
from pathlib import Path

from graph_reasoner.benchmarks import run_dir_name
from graph_reasoner.recipe import Layout, Spec


def arg(cmd, flag, default=None):
    return cmd[cmd.index(flag) + 1] if flag in cmd else default


def write(path: Path, text="{}\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class FakeProc:
    """A launched trainer, scripted: a GRPO round writes a checkpoint every
    ten seconds of the fake clock (resuming from the last one on disk), a
    GRPO smoke a `t_gen` line every ten seconds, an SFT smoke its adapter at
    ten seconds."""
    def __init__(self, runner, cmd, log: Path):
        self.runner, self.cmd, self.log, self.rc, self.t = runner, cmd, log, None, 0.0
        spec = Spec(arg(cmd, "--graph"), arg(cmd, "--size"), arg(cmd, "--tag"), arg(cmd, "--model"))
        self.stage = arg(cmd, "--stage")
        out = Layout.of(spec, root=runner.root).resolved(self.stage)["output_dir"]
        self.smoke = "--smoke" in cmd
        self.out = runner.root / (out + "_smoke" if self.smoke else out)
        self.sft = "train_sft.py" in cmd[1]
        write(log, "")

    def advance(self, seconds):
        if self.rc is not None:
            return
        self.t += seconds
        if self.sft:
            if self.t >= 10:
                write(self.out / "final" / "adapter_config.json")
                self.log.write_text("Adapter saved -> final\n")
                self.rc = 0
        elif self.smoke:
            if self.runner.behaviour == "traceback":
                self.log.write_text("Traceback (most recent call last): boom\n")
                self.rc = 1
                return
            n = int(self.t // 10)
            self.log.write_text("".join(f"{{'loss': 0.1, 't_gen': {i}}}\n" for i in range(1, n + 1)))
        else:
            if self.runner.behaviour == "refuse":
                self.log.write_text("d1 at --graph tsmc --size 2b --tag flash20 is missing its inputs:\n  prefixes: x\n")
                self.rc = 1
                return
            if self.runner.behaviour == "never":
                self.rc = 1
                return
            have = sorted(int(p.name.split("-")[1]) for p in self.out.glob("checkpoint-*")) if self.out.exists() else []
            start = have[-1] if have else 0
            reach = min(start + 20 * int(self.t // 10), 120)
            for ck in range(start + 20, reach + 1, 20):
                write(self.out / f"checkpoint-{ck}" / "adapter_config.json")
                if self.runner.behaviour == "die_at_60" and ck == 60 and not self.runner.died:
                    self.runner.died = True
                    self.rc = 1
                    return

    def poll(self):
        return self.rc

    def kill_group(self):
        self.rc = -9


class FakeRunner:
    def __init__(self, root: Path, gated=0.7, behaviour="ok", fail_reads=None):
        self.root, self.gated, self.behaviour = root, gated, behaviour
        self.fail_reads = dict(fail_reads or {})
        self.calls, self.started, self.procs, self.reaps, self.clock, self.died = [], [], [], [], 0.0, False
        self.env = {"PATH": "/usr/bin", "OPENAI_BASE_URL": "http://proxy"}
        (root / "logs").mkdir(exist_ok=True)

    def python(self):
        return "python"

    def _score(self, name, benchmark, gated, ids=None, model=None):
        d = self.root / "data/runs" / run_dir_name(name, benchmark)
        ids = ids or [f"{name}-{i}" for i in range(3)]
        tiers = ("easy", "medium", "hard")
        write(d / "results.jsonl", "".join(json.dumps({"id": i, "tier": tiers[k % 3], "gated_acc": gated}) + "\n"
                                           for k, i in enumerate(ids)))
        write(d / "summary.json", json.dumps({"aggregate": {"ALL": {"gated_acc": gated, "n_calls": 12.0,
                                                                     "reached_answer": 0.9, "gate_pass": 0.9,
                                                                     "halluc_strict": 0.0}},
                                               "model": model,
                                               "policy": {"turn_max_tokens": 1024}}))

    def run(self, cmd, log=None, env=None):
        self.calls.append((cmd, env))
        script = Path(cmd[1]).name if len(cmd) > 1 else cmd[0]
        if log:
            write(log, "")
        if script == "run_agent_eval.py":
            name = arg(cmd, "--name")
            if self.fail_reads.get(name, 0) > 0:
                self.fail_reads[name] -= 1
                log.write_text("Engine core initialization failed\n")
                return 1
            g = self.gated(name) if callable(self.gated) else self.gated
            shard = arg(cmd, "--shard")
            ids = [f"item-{shard.split('/')[0]}-{i}" for i in range(4)] if shard else None
            if "--ids-file" in cmd:
                ids = (self.root / arg(cmd, "--ids-file")).read_text().split()
            self._score(name, arg(cmd, "--benchmark"), g, ids, arg(cmd, "--model"))
            log.write_text(f"LLM usage: 1 call\nALL 195 {g:.3f}\n")
            return 0
        if script == "train_sft.py":
            spec = Spec(arg(cmd, "--graph"), arg(cmd, "--size"), arg(cmd, "--tag"), arg(cmd, "--model"))
            out = Layout.of(spec, root=self.root).resolved(arg(cmd, "--stage"))["output_dir"]
            write(self.root / out / "final" / "adapter_config.json")
            log.write_text("Adapter saved -> final\ntrain_loss 0.1\n")
            return 0
        if script == "sample_prefixes.py":
            self._score(arg(cmd, "--name"), arg(cmd, "--prefixes"), 0.5)
            log.write_text("in band 42\n")
            return 0
        if script == "verifier_select.py":
            runs = cmd[cmd.index("--runs") + 1:cmd.index("--name")]
            stem = runs[0].split("--", 1)[1]
            write(self.root / "data/runs" / f"{arg(cmd, '--name')}--{stem}" / "summary.json")
            return 0
        if script == "build_sft_rows.py":
            # keep the first two ids of every pass: the rest "failed"
            kept = []
            for i, a in enumerate(cmd):
                if a == "--rollouts":
                    rows = (self.root / cmd[i + 1] / "results.jsonl").read_text().splitlines()
                    kept += [json.loads(r)["id"] for r in rows[:2]]
            write(self.root / arg(cmd, "--out"), "".join(json.dumps({"item_id": k}) + "\n" for k in kept))
            return 0
        if script in ("build_substrate_mix.py", "build_prefixes.py", "build_verifier_data.py", "build_error_pairs.py"):
            write(self.root / arg(cmd, "--out"))
            if log:
                log.write_text(f"{script}: wrote {arg(cmd, '--out')}\n")
            return 0
        if script == "make_probe_lora.py":
            for f in ("adapter_config.json", "tokenizer_config.json"):
                write(self.root / cmd[3] / f)
            log.write_text("probe adapter written\n")
            return 0
        if script == "probe_vllm_rollout.py":
            log.write_text("PASS 120 tok/s\n")
            return 0
        if script == "probe_vllm_sleep.py":
            log.write_text("cycle 3 PASS, 30 GiB while asleep\n")
            return 0
        if script == "extract_states.py":
            from graph_reasoner.recipe.latent import layer_list
            for run_dir in (cmd[2], cmd[3]):
                reads = self.root / arg(cmd, "--out") / Path(run_dir).name
                write(reads / "index.jsonl")
                for n in layer_list(arg(cmd, "--layers")):
                    write(reads / f"L{n}.npy")
            return 0
        if script in ("label_moves.py", "depth_profile.py"):
            write(self.root / arg(cmd, "--out"))
            return 0
        if script == "directions.py":
            for ext in ("json", "npz"):
                write(self.root / arg(cmd, "--out-dir") / f"directions_L{arg(cmd, '--layers')}.{ext}")
            return 0
        if script == "steer.py":
            write(self.root / arg(cmd, "--out-root") / f"seed{arg(cmd, '--seed')}-{Path(arg(cmd, '--benchmark')).stem}" / "knob.json")
            return 0
        raise AssertionError(f"the fake runner knows no {script}")

    def start(self, cmd, log):
        self.started.append(cmd)
        proc = FakeProc(self, cmd, log)
        self.procs.append(proc)
        return proc

    def capture(self, cmd):
        self.calls.append((cmd, None))
        script = Path(cmd[1]).name if len(cmd) > 1 else cmd[0]
        if cmd[1:2] == ["-c"]:
            return (1, "AssertionError: no CUDA") if self.behaviour == "no_cuda" else (0, "{'torch': '2.13'} NVIDIA H100")
        if script == "best_of_n.py":
            return 0, "best-of-1 (gold-free) 0.700\nbest-of-32 (gold-free) 0.800\nrun 0 0.700\n"
        if script == "mix.py":
            return 0, f"mix {cmd[2]}: ok\n"
        if script in ("build_sft_rows.py", "build_substrate_mix.py", "build_error_pairs.py"):
            rc = self.run(cmd)
            return rc, f"{script} ok\n"
        raise AssertionError(f"the fake runner cannot capture {script}")

    def reap(self, everything=False):
        self.reaps.append(everything)

    def gpu_line(self):
        return "40000, 90"

    def sleep(self, seconds):
        self.clock += seconds
        for p in self.procs:
            p.advance(seconds)

    def now(self):
        return "now"

    def scripts(self):
        return [Path(c[1]).name if len(c) > 1 else c[0] for c, _ in self.calls]
