"""`prospector teach`, `verify` and `latent` on the stub runner: the teacher
pass's pieces in order with the ids pass 1 failed, the machine check's five
steps and its stop at the first failure, and the probe's six stages with
their skip rules and the arms per seed.
"""
import json

import pytest

from fake_runner import FakeRunner, arg, write
from graph_reasoner.recipe import Layout, Spec
from graph_reasoner.recipe.latent import ARMS_HOME_FIRST, ARMS_HOME_REPEAT, ARMS_TRANSFER, Probe
from graph_reasoner.recipe.plan import Context
from graph_reasoner.recipe.run import Refused
from graph_reasoner.recipe.teach import NOTE, Teach
from graph_reasoner.recipe.verify import verify


def tree_for_teach(root, graph="pfas", tag="t1"):
    layout = Layout.of(Spec(graph, tag=tag), root=root)
    write(root / layout.graph.eval_benchmark, json.dumps({"id": "item-0-0"}) + "\n")   # one evaluation item among the rolled
    write(root / layout.graph.train_benchmark)
    return layout


def test_the_teacher_pass_runs_its_pieces_in_order(tmp_path, capsys):
    layout = tree_for_teach(tmp_path)
    runner = FakeRunner(tmp_path)
    teach = Teach(layout, runner, teacher="google/some-teacher", shards=2)
    assert teach.execute(["0", "1"]) == 0
    names = [arg(c, "--name") for c, _ in runner.calls]
    assert names == ["t1-teach-g0", "t1-teach-g1"]
    cmd, env = runner.calls[0]
    assert cmd[:2] == ["python", "scripts/run_agent_eval.py"] and arg(cmd, "--shard") == "0/2"
    assert arg(cmd, "--graph") == "pfas" and arg(cmd, "--benchmark") == "data/benchmarks/pfas/pfas_train_agentic.jsonl"
    assert arg(cmd, "--turn-max-tokens") == "3000" and arg(cmd, "--user-note") == NOTE and "--no-latency" in cmd
    assert env["OPENAI_BASE_URL"] == "" and env["PATH"] == "/usr/bin"   # set empty, so .env cannot restore it
    assert teach.execute(["0"]) == 0 and "SKIP t1-teach-g0" in capsys.readouterr().out

    assert teach.execute(["ids"]) == 0
    failed = (tmp_path / layout.demos_failed_ids).read_text().split()
    # four items a shard, two kept a shard by the stub, item-0-0 excluded as an evaluation item
    assert failed == ["item-0-2", "item-0-3", "item-1-2", "item-1-3"]
    assert (tmp_path / layout.demos_keeps_pass1).exists()
    out = capsys.readouterr().out
    assert "pass 1: 8 items rolled, 4 kept, 4 failed" in out

    assert teach.execute(["p2", "mix"]) == 0
    p2 = next(c for c, _ in runner.calls if arg(c, "--name") == "t1-teach-p2")
    assert arg(p2, "--ids-file") == layout.demos_failed_ids and "--no-cache" in p2
    assert (tmp_path / layout.demos_keeps).exists() and (tmp_path / layout.demos_detours).exists()
    assert (tmp_path / layout.demos_mix).exists()
    pairs = next(c for c, _ in runner.calls if c[1] == "scripts/traj/build_error_pairs.py")
    assert arg(pairs, "--rows") == layout.demos_keeps and arg(pairs, "--benchmark") == layout.graph.train_benchmark


def test_the_teacher_pass_refuses_out_of_order_and_fails_on_a_failed_run(tmp_path, capsys):
    layout = tree_for_teach(tmp_path)
    teach = Teach(layout, FakeRunner(tmp_path), shards=2)
    assert teach.execute(["p2"]) == 2 and "run 'ids' first" in capsys.readouterr().out
    assert teach.execute(["mix"]) == 2 and "run 'p2' first" in capsys.readouterr().out
    assert teach.execute(["ids"]) == 2 and "shards [0, 1] have not run" in capsys.readouterr().out
    assert teach.execute(["9"]) == 2 and "not a shard" in capsys.readouterr().out
    failing = Teach(layout, FakeRunner(tmp_path, fail_reads={"t1-teach-g0": 1}), shards=2)
    assert failing.execute(["0"]) == 1 and "!!! TEACH t1-teach-g0 failed" in capsys.readouterr().out
    teach.dry_run(["0", "ids", "mix"])
    out = capsys.readouterr().out
    assert "shard 0: need data/runs/t1-teach-g0--pfas_train_agentic/summary.json" in out
    assert "mix:   need data/traj/pfas_t1_keeps.jsonl" in out


def test_a_tag_holds_one_teachers_pass(tmp_path, capsys):
    layout = tree_for_teach(tmp_path)
    stored = tmp_path / "data/runs/t1-teach-g0--pfas_train_agentic/summary.json"
    write(stored, json.dumps({"model": "google/gemini-3.7-flash"}))
    runner = FakeRunner(tmp_path)
    other = Teach(layout, runner, teacher="openai/other-teacher", shards=2)
    assert other.execute(["1"]) == 2 and runner.calls == []       # refused before any shard runs
    out = capsys.readouterr().out
    assert "holds google/gemini-3.7-flash's teacher pass (data/runs/t1-teach-g0--pfas_train_agentic)" in out
    assert "name the new pass with another --tag" in out
    with pytest.raises(Refused, match="another --tag"):
        other.dry_run(["0"])
    # the same teacher carries on under the tag, and the other teacher under a new one
    assert Teach(layout, FakeRunner(tmp_path), teacher="google/gemini-3.7-flash", shards=2).execute(["1"]) == 0
    fresh = Layout.of(Spec("pfas", tag="t2"), root=tmp_path)
    assert Teach(fresh, FakeRunner(tmp_path), teacher="openai/other-teacher", shards=2).execute(["0"]) == 0
    # with no --teacher, a tag's pass continues with the teacher it records
    later = FakeRunner(tmp_path)
    assert Teach(fresh, later, shards=2).execute(["1"]) == 0
    assert arg(later.calls[0][0], "--model") == "openai/other-teacher"
    # with the runs gone, the rows' sidecar still records the teacher
    for summary in (tmp_path / "data/runs").glob("t1-teach-*/summary.json"):
        summary.unlink()
    write(tmp_path / f"{layout.demos_keeps}.provenance.json", json.dumps({"teacher": "google/gemini-3.7-flash"}))
    assert other.execute(["0"]) == 2 and f"({layout.demos_keeps})" in capsys.readouterr().out


def test_the_default_tag_belongs_to_the_default_teacher(tmp_path, capsys):
    layout = tree_for_teach(tmp_path, tag="flash20")               # a new graph: no stored pass to compare
    runner = FakeRunner(tmp_path)
    assert Teach(layout, runner, teacher="openai/other-teacher", shards=2).execute(["0"]) == 2
    assert runner.calls == []
    assert ("the default tag 'flash20' names google/gemini-3.7-flash's pass, and the teacher is "
            "openai/other-teacher: name the new pass with --tag") in capsys.readouterr().out
    started = FakeRunner(tmp_path)
    assert Teach(layout, started, shards=2).execute(["0"]) == 0
    assert arg(started.calls[0][0], "--model") == "google/gemini-3.7-flash"


def test_verify_runs_its_five_steps_and_stops_at_the_first_failure(tmp_path, capsys):
    ctx = Context(Layout.of(Spec("tsmc", "2b"), root=tmp_path))
    runner = FakeRunner(tmp_path)
    assert verify(ctx, runner, gpu_frac=0.4) == 0
    scripts = runner.scripts()
    assert scripts == ["-c", "make_probe_lora.py", "probe_vllm_rollout.py", "probe_vllm_sleep.py"]
    probe = next(c for c, _ in runner.calls if c[1:2] == ["scripts/infra/probe_vllm_sleep.py"])
    assert probe == ["python", "scripts/infra/probe_vllm_sleep.py", "Qwen/Qwen3.5-2B", "models/probe_lora_2b", "0.4"]
    assert [c for c in runner.started][0][1] == "scripts/train_sft.py" and "--smoke" in runner.started[0]
    assert not (tmp_path / "models/traj_sft_2b_flash20_smoke").exists(), "verify removes the smoke's directory"
    out = capsys.readouterr().out
    assert "=== VERIFY 2b COMPLETE" in out and "SFT smoke exit 0, peak GPU memory 40000 MiB" in out
    assert (tmp_path / "logs/verify_2b_rollout.log").exists()

    again = FakeRunner(tmp_path)
    assert verify(ctx, again) == 0
    assert "make_probe_lora.py" not in again.scripts(), "the probe adapter is kept"

    broken = FakeRunner(tmp_path, behaviour="no_cuda")
    assert verify(ctx, broken) == 1
    assert broken.scripts() == ["-c"] and "FAIL: imports" in capsys.readouterr().out


def probe_tree(root):
    ctx = Context(Layout.of(Spec("tsmc", "4b"), transfer="biograph", root=root))
    L = ctx.layout
    write(root / L.sft2 / "adapter_config.json")
    for d in (L.d1_dir, L.d2_dir):
        write(root / d / "checkpoint-120" / "adapter_config.json")
        write(root / d / "peak.txt", "120\n")
    write(root / "data/runs/home-s0--tsmc_core_agentic/trajectories.jsonl")
    write(root / "data/runs/bio-s0--biograph_core_agentic/trajectories.jsonl")
    for g in (ctx.graph, ctx.transfer):      # the graph files check_chain looks for
        write(root / g.kb, "")
    return ctx


def test_the_states_stage_is_skipped_only_when_the_index_and_every_array_are_there(tmp_path):
    ctx = probe_tree(tmp_path)
    reads = tmp_path / "data/runs/_probe/pilot/policy_all"
    for run in ("home-s0--tsmc_core_agentic", "bio-s0--biograph_core_agentic"):
        write(reads / run / "index.jsonl")                      # a copy made without the arrays
    runner = FakeRunner(tmp_path)
    Probe(ctx, runner, "home-s0", "bio-s0", layers="1-2", stop="states").execute()
    assert runner.scripts() == ["extract_states.py"], "the index alone does not make the stage done"
    for run in ("home-s0--tsmc_core_agentic", "bio-s0--biograph_core_agentic"):
        for n in (1, 2):
            write(reads / run / f"L{n}.npy")
    again = FakeRunner(tmp_path)
    Probe(ctx, again, "home-s0", "bio-s0", layers="1-2", stop="states").execute()
    assert again.scripts() == []


def test_the_probe_runs_its_stages_with_their_skips_and_arms(tmp_path, capsys):
    ctx = probe_tree(tmp_path)
    runner = FakeRunner(tmp_path)
    probe = Probe(ctx, runner, "home-s0", "bio-s0", seeds=(0, 2), limit=5)
    assert probe.execute() == 0
    scripts = runner.scripts()
    assert scripts[:4] == ["extract_states.py", "label_moves.py", "label_moves.py", "depth_profile.py"]
    assert scripts[4] == "directions.py" and scripts.count("steer.py") == 4 and scripts.count("mix.py") == 2
    states = runner.calls[0][0]
    assert states[:4] == ["python", "scripts/latent/extract_states.py", "data/runs/home-s0--tsmc_core_agentic",
                          "data/runs/bio-s0--biograph_core_agentic"]
    assert arg(states, "--adapter") == ",".join(ctx.layout.served_stack()) and arg(states, "--limit") == "5"
    assert arg(states, "--out") == "data/runs/_probe_smoke/pilot/policy_all" and arg(states, "--layers") == "0-32"
    steers = [c for c, _ in runner.calls if c[1] == "scripts/latent/steer.py"]
    arms = lambda c: c[c.index("--arms") + 1:c.index("--out-root")]      # noqa: E731
    assert arms(steers[0]) == ARMS_HOME_FIRST and arg(steers[0], "--seed") == "0" and arg(steers[0], "--out-root") == "data/runs/_probe_smoke/knob_vllm"
    assert arms(steers[1]) == ARMS_TRANSFER and arg(steers[1], "--benchmark") == "data/benchmarks/biograph/biograph_core_agentic.jsonl"
    assert arms(steers[2]) == ARMS_HOME_REPEAT and arg(steers[2], "--out-root") == "data/runs/_probe_smoke/knob_vllm_r2"
    assert arg(steers[0], "--directions") == "data/runs/_probe_smoke/pilot/directions_L20.npz"
    out = capsys.readouterr().out
    assert "=== LATENT COMPLETE" in out

    again = FakeRunner(tmp_path)
    assert Probe(ctx, again, "home-s0", "bio-s0", seeds=(0,), limit=5, stop="directions").execute() == 0
    assert again.scripts() == [] and "SKIP STATES" in capsys.readouterr().out
    full = FakeRunner(tmp_path)                 # the probe proper reads afresh after a smoke
    assert Probe(ctx, full, "home-s0", "bio-s0", seeds=(0,), stop="states").execute() == 0
    assert full.scripts() == ["extract_states.py"]


def test_the_probe_refuses_a_missing_run_or_transfer_graph(tmp_path, capsys):
    ctx = probe_tree(tmp_path)
    assert Probe(ctx, FakeRunner(tmp_path), "home-s0", "missing-s0").execute() == 2
    assert "stored run missing: data/runs/missing-s0--biograph_core_agentic" in capsys.readouterr().out
    with pytest.raises(Refused, match="transfer-graph"):
        Probe(Context(Layout.of(Spec("tsmc", "4b"), root=tmp_path)), FakeRunner(tmp_path), "home-s0", "bio-s0")
    with pytest.raises(Refused, match="not a stage"):
        Probe(ctx, FakeRunner(tmp_path), "home-s0", "bio-s0", start="arms")


def test_a_probe_smoke_writes_under_its_own_root(tmp_path):
    """Every probe stage decides it is done by its files, so a smoke's few
    episodes must not land where the full probe looks."""
    ctx = probe_tree(tmp_path)
    runner = FakeRunner(tmp_path)
    assert Probe(ctx, runner, "home-s0", "bio-s0").out == "data/runs/_probe"
    assert Probe(ctx, runner, "home-s0", "bio-s0", limit=5).out == "data/runs/_probe_smoke"
    assert Probe(ctx, runner, "home-s0", "bio-s0", limit=5, out="data/x").out == "data/x"


def test_a_probe_slice_that_starts_after_it_stops_is_refused(tmp_path):
    """`prospector run` refuses the same mistake; the probe would otherwise
    run no stage and report success."""
    ctx = probe_tree(tmp_path)
    with pytest.raises(Refused, match="--start-at steer comes after --stop-after states"):
        Probe(ctx, FakeRunner(tmp_path), "home-s0", "bio-s0", start="steer", stop="states")
