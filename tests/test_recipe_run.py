"""`graph_reasoner.recipe.run`: the whole ladder on CPU, with a stub runner.

On `tests/fake_runner.py`'s `FakeRunner`, which records every command and
writes what each leaf would write, the 2B ladder runs from an empty tree in the recipe's order, a relaunch issues no
command, a round resumes across attempts and stops on a refusal, a retrain
discards the round's own reads alone, a gate under the floor stops the run,
a read that fails twice fails its stage, and the smoke runner stops at three
steps and fails on a traceback.
"""
import re
from pathlib import Path

import pytest

from fake_runner import FakeRunner, arg, write
from graph_reasoner.recipe import Layout, Spec
from graph_reasoner.recipe.plan import Context, Round
from graph_reasoner.recipe.run import Failed, execute, retrain_cleanup, smoke


def ctx_for(root, transfer="biograph", **kw):
    ctx = Context(Layout.of(Spec("tsmc", "2b"), transfer=transfer, root=root), **kw)
    for g in (ctx.graph, ctx.transfer):      # the graph files check_chain looks for
        if g is not None:
            write(root / g.kb, "")
    write(root / ctx.layout.demos_mix, "")   # stage 1's demonstrations, which no stage makes
    return ctx


def test_the_whole_ladder_runs_from_an_empty_tree_in_order(tmp_path):
    ctx = ctx_for(tmp_path)
    L = ctx.layout
    runner = FakeRunner(tmp_path)
    assert execute(ctx, runner) == 0

    names = [arg(c, "--name") for c, _ in runner.calls if Path(c[1]).name == "run_agent_eval.py"]
    assert names[:6] == [f"q2bsft1eval-s{s}" for s in (0, 1, 2)] + [f"q2bsft1bio-s{s}" for s in (0, 1, 2)]
    assert names[6:10] == [f"q2bfullroll-s{s}" for s in (831, 832, 833, 834)]
    grid = [n for n in names if n.startswith("q2bfulleval-ck")]
    assert grid[:3] == ["q2bfulleval-ck20-s0", "q2bfulleval-ck20-s1", "q2bfulleval-ck20-s2"] and len(grid) == 21
    # every checkpoint tied: the later one wins; three extra seeds and the codas on it
    assert grid[-3:] == [f"q2bfulleval-ck120-s{s}" for s in (3, 4, 5)]
    assert [f"q2bfullbio-s{s}" for s in (0, 1, 2)] + [f"q2bfullpass5-s{s}" for s in (1, 2, 3, 4)] == \
        [n for n in names if n.startswith("q2bfullbio") or n.startswith("q2bfullpass5")]
    assert (tmp_path / L.full_dir / "peak.txt").read_text() == "120\n"
    assert [n for n in names if n.startswith("q2bg2roll")] == [f"q2bg2roll-s{s}" for s in (861, 862, 863, 864)]
    d1 = [n for n in names if n.startswith("q2bd1")]
    assert d1[:4] == [f"q2bd1roll-s{s}" for s in (831, 832, 833, 834)]
    leaders = [n for n in d1 if re.match(r"q2bd1eval-ck(120|100)-s[3-7]$", n)]
    assert len(leaders) == 10 and {n.split("-")[1] for n in leaders} == {"ck120", "ck100"}
    assert d1[-7:] == [f"q2bd1bio-ck120-s{s}" for s in (0, 1, 2)] + [f"q2bd1pass5-ck120-s{s}" for s in (1, 2, 3, 4)]
    assert (tmp_path / L.d1_dir / "peak.txt").read_text() == "120\n" and (tmp_path / L.d2_dir / "peak.txt").exists()
    assert [n for n in names if n.startswith("q2bvroll")] == [f"q2bvroll-s{s}" for s in (831, 832, 833, 834)]
    assert len([n for n in names if n.startswith("q2bpool32")]) == 32

    sfts = [arg(c, "--stage") for c, _ in runner.calls if Path(c[1]).name == "train_sft.py"]
    assert sfts == ["sft1", "sft2", "verifier"]
    rounds = [arg(c, "--stage") for c in runner.started if "--smoke" not in c]
    smokes = [arg(c, "--stage") for c in runner.started if "--smoke" in c]
    assert rounds == ["full", "d1", "d2"] and smokes == ["full", "d1", "d2"]
    builds = [Path(c[1]).name for c, _ in runner.calls if Path(c[1]).name.startswith("build_") or Path(c[1]).name == "sample_prefixes.py"]
    assert builds == ["build_sft_rows.py", "build_substrate_mix.py", "build_prefixes.py", "sample_prefixes.py",
                      "build_prefixes.py", "sample_prefixes.py", "build_verifier_data.py"]
    assert runner.scripts().count("verifier_select.py") == 1 and runner.scripts().count("best_of_n.py") == 2

    read = next(c for c, _ in runner.calls if arg(c, "--name") == "q2bsft1bio-s1")
    assert arg(read, "--benchmark") == "data/benchmarks/biograph/biograph_core_agentic.jsonl"
    assert arg(read, "--kb") == "data/raw/biograph/large_graph_simple_giant.graphml" and "--temperature" not in read
    assert read[:2] == ["python", "scripts/run_agent_eval.py"] and arg(read, "--turn-max-tokens") == "1024"
    roll = next(c for c, _ in runner.calls if arg(c, "--name") == "q2bfullroll-s831")
    assert arg(roll, "--temperature") == "0.7" and arg(roll, "--benchmark") == "data/benchmarks/tsmc/tsmc_train_agentic_4x.jsonl"
    assert arg(roll, "--kb") == "data/raw/GRAPHDATA_TSMC_OUTPUT/tsmc_5b10p.graphml"
    pass5 = next(c for c, _ in runner.calls if arg(c, "--name") == "q2bd2pass5-ck120-s1")
    assert arg(pass5, "--temperature") == "0.9" and arg(pass5, "--adapter").count(",") == 2
    grpo = next(c for c in runner.started if arg(c, "--stage") == "d1" and "--smoke" not in c)
    assert grpo[:2] == ["python", "scripts/train_traj_grpo.py"] and arg(grpo, "--kb").endswith("tsmc_5b10p.graphml")
    assert (tmp_path / "logs" / "q2bsft1_train.log").exists() and (tmp_path / "logs" / "q2bd1_r_a1.log").exists()
    assert (tmp_path / "logs" / "q2bd1_stage0.log").exists() and (tmp_path / "logs" / "traj_grpo_2b_flash20d1_smoke.log").exists()

    again = FakeRunner(tmp_path)
    assert execute(ctx, again) == 0
    assert set(again.scripts()) == {"best_of_n.py"} and again.started == []


def test_a_gate_under_the_floor_stops_the_run_unless_forced(tmp_path, capsys):
    ctx = ctx_for(tmp_path)
    assert execute(ctx, FakeRunner(tmp_path, gated=0.3)) == 3
    out = capsys.readouterr().out
    assert "FAILED" in out and "!!! GATE1" in out and "RUN q2b STOPPED (exit 3)" in out
    forced = ctx_for(tmp_path, force_gate=True)
    assert execute(forced, FakeRunner(tmp_path, gated=0.3), stop="gate1") == 0


def test_a_round_resumes_across_attempts_and_stops_on_a_refusal(tmp_path, capsys):
    ctx = ctx_for(tmp_path, transfer=None)
    L = ctx.layout
    write(tmp_path / L.sft1 / "adapter_config.json")
    runner = FakeRunner(tmp_path, behaviour="die_at_60")
    assert execute(ctx, runner, start="band", stop="round1") == 0
    attempts = [c for c in runner.started if "--smoke" not in c]
    assert len(attempts) == 2, "the first attempt died at checkpoint-60, the second resumed"
    assert sorted(int(p.name.split("-")[1]) for p in (tmp_path / L.full_dir).glob("checkpoint-*")) == [20, 40, 60, 80, 100, 120]
    assert (tmp_path / "logs" / "q2bfull_r_a2.log").exists()
    out = capsys.readouterr().out
    assert "attempt 1 exited exit=1 last_checkpoint=60" in out and "attempt 2 target" in out

    refused = FakeRunner(tmp_path, behaviour="refuse")
    write(tmp_path / L.sft2 / "adapter_config.json")
    assert execute(ctx, refused, start="d1", stop="d1") == 1
    assert len([c for c in refused.started if "--smoke" not in c]) == 1, "a refusal is not retried"
    out = capsys.readouterr().out
    assert "the trainer refused the run" in out and "!!! D1 the d1 round did not reach checkpoint-120" in out


def test_a_retrain_discards_the_round_s_own_reads_and_nothing_else(tmp_path):
    ctx = ctx_for(tmp_path)
    L = ctx.layout
    mine = ["q2bfulleval-ck20-s0--tsmc_core_agentic", "q2bfullpass5-s1--tsmc_core_agentic",
            "q2bfullbio-s0--biograph_core_agentic"]
    others = ["q2bsft1eval-s0--tsmc_core_agentic", "q2bfullroll-s831--tsmc_train_agentic_4x",
              "q2bd1eval-ck20-s0--tsmc_core_agentic"]
    for r in mine + others:
        write(tmp_path / "data/runs" / r / "summary.json")
    write(tmp_path / L.full_dir / "checkpoint-60" / "adapter_config.json")
    retrain_cleanup(Round("full", L.full_dir, L.full_prefix, (L.sft1,), 120), ctx)
    assert not any((tmp_path / "data/runs" / r).exists() for r in mine)
    assert all((tmp_path / "data/runs" / r).exists() for r in others)
    assert not (tmp_path / L.full_dir).exists()


def test_a_read_that_fails_twice_fails_its_stage_and_once_is_retried(tmp_path, capsys):
    ctx = ctx_for(tmp_path, transfer=None)
    write(tmp_path / ctx.layout.sft1 / "adapter_config.json")
    once = FakeRunner(tmp_path, fail_reads={"q2bsft1eval-s1": 1})
    assert execute(ctx, once, start="gate1", stop="gate1") == 0
    assert [arg(c, "--name") for c, _ in once.calls].count("q2bsft1eval-s1") == 2
    twice = FakeRunner(tmp_path, fail_reads={"q2bsft1eval-s2": 2})
    (tmp_path / "data/runs/q2bsft1eval-s2--tsmc_core_agentic/summary.json").unlink()
    assert execute(ctx, twice, start="gate1", stop="gate1") == 1
    assert "!!! GATE1 read q2bsft1eval-s2 failed twice" in capsys.readouterr().out


def test_the_smoke_stops_at_three_steps_and_fails_on_a_traceback(tmp_path, capsys):
    ctx = ctx_for(tmp_path, transfer=None)
    L = ctx.layout
    runner = FakeRunner(tmp_path)
    smoke("d1", L.d1_dir, ctx, runner)
    proc = runner.procs[-1]
    assert proc.rc == -9 and len(re.findall("t_gen", proc.log.read_text())) >= 3
    out = capsys.readouterr().out
    assert "=== SMOKE traj_grpo_2b_flash20d1_smoke end" in out and "peak memory 40000 MiB" in out
    with pytest.raises(Failed) as e:
        smoke("d1", L.d1_dir, ctx, FakeRunner(tmp_path, behaviour="traceback"))
    assert "failed" in e.value.message


def test_a_missing_prerequisite_is_a_refusal_before_anything_runs(tmp_path, capsys):
    ctx = ctx_for(tmp_path, transfer=None)
    write(tmp_path / ctx.layout.sft1 / "adapter_config.json")
    runner = FakeRunner(tmp_path)
    assert execute(ctx, runner, start="round1", stop="round1") == 2
    assert runner.started == [] and "the band's four passes" in capsys.readouterr().out


def test_the_best_of_n_report_shows_the_paper_s_pool_sizes_alone(tmp_path, capsys):
    """best_of_n.py prints a line per pool size; the report keeps k = 1, 2, 5,
    10 and 32 and the first run, not 12 or 20 because they begin with 1 or 2.
    The score stage's report names its scoring run, so the verifier's pick
    is asked for and shown beside the vote."""
    from graph_reasoner.recipe.plan import Report
    from graph_reasoner.recipe.run import report

    class Printing(FakeRunner):
        def capture(self, cmd):
            self.captured = cmd
            out = "".join(f"best-of-{k} (gold-free) 0.8\n" for k in range(1, 33)) + "run 0 (k=1) 0.7\n"
            if "--scores" in cmd:
                out += "verifier pick  q2b-verifier32-yn1  0.882  0.9\n"
            return 0, out

    kept = ("best-of", "run 0", "verifier pick")
    runner = Printing(tmp_path)
    report(Report("best_of_n", ("data/runs/a",)), ctx_for(tmp_path), runner)
    shown = [l.split()[0] for l in capsys.readouterr().out.splitlines() if l.startswith(kept)]
    assert shown == ["best-of-1", "best-of-2", "best-of-5", "best-of-10", "best-of-32", "run"]
    assert "--scores" not in runner.captured

    report(Report("best_of_n", ("data/runs/a",), scores="data/runs/s"), ctx_for(tmp_path), runner)
    shown = [l.split()[0] for l in capsys.readouterr().out.splitlines() if l.startswith(kept)]
    assert shown[-1] == "verifier" and runner.captured[-2:] == ["--scores", "data/runs/s"]


def test_an_interrupted_watcher_kills_the_process_it_launched(tmp_path):
    """A SIGINT, or the SIGTERM cli.main turns into SystemExit, unwinds through
    the watcher and takes the trainer with it; left alone, the trainer would
    keep writing into a directory the next launch removes from under it."""
    from graph_reasoner.recipe.run import Watermark, wait_watching
    runner = FakeRunner(tmp_path)
    (tmp_path / "logs").mkdir(exist_ok=True)
    proc = runner.start(["python", "scripts/train_traj_grpo.py", "--graph", "tsmc", "--size", "2b",
                         "--tag", "flash20", "--stage", "full"], tmp_path / "logs" / "round.log")

    def interrupt():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        wait_watching(proc, runner, Watermark(runner, tmp_path / "logs" / "marks.log"), every=60, until=interrupt)
    assert proc.poll() == -9
