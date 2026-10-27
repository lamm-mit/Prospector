"""`graph_reasoner.recipe.plan`: the stages of a run computed from the disk.

On fake trees under `tmp_path`: an empty tree plans every stage and names
every read; a tree laid out like the release is all done or skipped; a round
resumes mid-way; the full round's pick is made once over the grid; a decision
round reads its two leaders before it picks; the slice flags; another graph's
names; the dry run's text.
"""
import json

import pytest

from graph_reasoner.recipe import Layout, Spec
from graph_reasoner.recipe.plan import (STAGES, Band, Build, CheckChain, Context, Gate, Later, Read, Require,
                                        Round, Smoke, Train, ValidPick, WritePeak, plan, render, slice_of)


def touch(root, rel, text="{}\n"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def scored(root, run_dir, gated=0.7, rows=3):
    """A scored read: `results.jsonl` rows with gated_acc and a `summary.json`."""
    touch(root, f"{run_dir}/results.jsonl", "".join(json.dumps({"gated_acc": gated}) + "\n" for _ in range(rows)))
    touch(root, f"{run_dir}/summary.json", json.dumps({
        "aggregate": {"ALL": {"gated_acc": gated, "n_calls": 12.0, "reached_answer": 0.9, "gate_pass": 0.9,
                              "halluc_strict": 0.0}},
        "policy": {"turn_max_tokens": 1024}}))


def ctx_for(root, size="2b", graph="tsmc", transfer="biograph", **kw):
    return Context(Layout.of(Spec(graph, size), transfer=transfer, root=root), **kw)


def steps_of(ctx, name):
    return {p.name: p for p in plan(ctx)}[name]


def released(root, ctx, picks=((100, 100, 120))):
    """A tree laid out like the release: every adapter, every round with its
    checkpoints and its pick, the rows, the mix, the states, the bands, and
    every read the plan names, scored."""
    L = ctx.layout
    for final in (L.sft1, L.sft2, L.verifier):
        touch(root, f"{final}/adapter_config.json")
    for d, pk in zip((L.full_dir, L.d1_dir, L.d2_dir), picks):
        for ck in range(20, 121, 20):
            touch(root, f"{d}/checkpoint-{ck}/adapter_config.json")
        touch(root, f"{d}/peak.txt", f"{pk}\n")
    for f in (L.sft2_mix, L.verifier_rows, L.d1_prefixes, L.d2_prefixes):
        touch(root, f)
    for r in (*L.full_band_runs, *L.g2_roll_runs, *L.v_roll_runs, L.d1_band_run, L.d2_band_run):
        scored(root, r)
    for _ in range(4):            # the reads a pick unlocks appear on the next pass
        for p in plan(ctx):
            for s in p.steps:
                if isinstance(s, Read) and not s.done(ctx):
                    scored(root, s.run_dir)
    scored(root, f"data/runs/{L.verifier_select_name}--{ctx.graph.eval_stem}")


def test_an_empty_tree_plans_every_stage_and_names_every_read(tmp_path):
    ctx = ctx_for(tmp_path)
    L = ctx.layout
    planned = plan(ctx)
    assert [p.name for p in planned] == list(STAGES)
    assert {p.status for p in planned} == {"todo"}

    sft1 = steps_of(ctx, "sft1").steps
    # Stage 1 first requires the demonstrations, the one input no stage makes.
    assert isinstance(sft1[0], Require) and sft1[0].paths == (L.demos_mix,)
    assert sft1[1:] == [Train("sft1", L.sft1_dir, L.sft1, "q2bsft1_train")]

    gate1 = steps_of(ctx, "gate1").steps
    assert isinstance(gate1[0], CheckChain) and gate1[0].chain == (L.sft1,)
    reads = [s for s in gate1 if isinstance(s, Read)]
    assert [r.name for r in reads] == [f"q2bsft1eval-s{s}" for s in (0, 1, 2)] + [f"q2bsft1bio-s{s}" for s in (0, 1, 2)]
    assert reads[0].benchmark == "data/benchmarks/tsmc/tsmc_core_agentic.jsonl" and reads[0].temperature is None
    assert reads[3].benchmark == "data/benchmarks/biograph/biograph_core_agentic.jsonl"
    assert reads[3].kb == "data/raw/biograph/large_graph_simple_giant.graphml"
    assert reads[0].run_dir == "data/runs/q2bsft1eval-s0--tsmc_core_agentic"
    assert isinstance(gate1[-1], Gate) and gate1[-1].floor == 0.40

    band = steps_of(ctx, "band").steps
    rolls = [s for s in band if isinstance(s, Read)]
    assert [r.name for r in rolls] == [f"q2bfullroll-s{s}" for s in (831, 832, 833, 834)]
    assert rolls[0].benchmark == "data/benchmarks/tsmc/tsmc_train_agentic_4x.jsonl" and rolls[0].temperature == 0.7

    round1 = steps_of(ctx, "round1").steps
    assert [type(s) for s in round1] == [Require, CheckChain, Smoke, Round, Later]
    assert round1[0].paths == tuple(f"{r}/results.jsonl" for r in L.full_band_runs)
    assert round1[3] == Round("full", L.full_dir, "q2bfull", (L.sft1,), 120)

    assert [type(s) for s in steps_of(ctx, "rftroll").steps] == [ValidPick, Later]
    rftbuild = steps_of(ctx, "rftbuild").steps
    assert isinstance(rftbuild[1], Build) and rftbuild[1].out == L.g2_keeps
    assert rftbuild[1].command[0] == "scripts/traj/build_sft_rows.py" and "--cap" in rftbuild[1].command
    assert rftbuild[2].command[:3] == ("scripts/traj/build_substrate_mix.py", "--demos", L.demos_mix)

    d1 = steps_of(ctx, "d1").steps
    kinds = [type(s) for s in d1]
    assert kinds[0] is CheckChain and kinds[-3:] == [Smoke, Round, Later]
    build = next(s for s in d1 if isinstance(s, Build))
    assert build.always and build.out == L.d1_prefixes and "--kb" in build.command
    band_step = next(s for s in d1 if isinstance(s, Band))
    assert band_step == Band(L.d1_prefixes, "q2bd1-s911", L.d1_band_run, (L.sft2,))
    assert [type(s) for s in steps_of(ctx, "d2").steps] == [ValidPick, Later]
    assert [type(s) for s in steps_of(ctx, "final").steps] == [ValidPick, ValidPick, Later]
    assert isinstance(steps_of(ctx, "score").steps[0], Require)


def test_a_tree_laid_out_like_the_release_is_all_done_or_skipped(tmp_path):
    ctx = ctx_for(tmp_path)
    released(tmp_path, ctx)
    planned = {p.name: p for p in plan(ctx)}
    assert {n for n, p in planned.items() if p.status == "skipped"} == {"band", "rftroll", "rftbuild", "rows"}
    assert {n for n, p in planned.items() if p.status == "done"} == set(STAGES) - {"band", "rftroll", "rftbuild", "rows"}
    assert not any(isinstance(s, Later) for p in planned.values() for s in p.steps)
    round1 = planned["round1"].steps
    assert WritePeak(ctx.layout.full_dir, 100) in round1
    assert [r.name for r in round1 if isinstance(r, Read) and r.group == "post"] == [f"q2bfulleval-ck100-s{s}" for s in (3, 4, 5)]
    assert [r.name for r in round1 if isinstance(r, Read) and r.group == "pass5"] == [f"q2bfullpass5-s{s}" for s in (1, 2, 3, 4)]
    # an existing pick is final: no leader reads, the codas on it under its checkpoint
    d1 = [r for r in planned["d1"].steps if isinstance(r, Read)]
    assert {r.group for r in d1} == {"grid", "transfer", "pass5"}
    assert [r.name for r in d1 if r.group == "pass5"] == [f"q2bd1pass5-ck100-s{s}" for s in (1, 2, 3, 4)]
    assert [r.name for r in d1 if r.group == "transfer"] == [f"q2bd1bio-ck100-s{s}" for s in (0, 1, 2)]
    final = planned["final"].steps
    assert {r.group for r in final if isinstance(r, Read)} == {"pool"}     # the verifier roll is skipped: the rows exist
    assert final[0].chain == tuple(ctx.layout.served_stack())


def test_the_full_round_s_pick_is_made_once_over_the_grid(tmp_path):
    ctx = ctx_for(tmp_path, transfer=None)
    L = ctx.layout
    touch(tmp_path, f"{L.sft1}/adapter_config.json")
    for ck in (100, 120):
        touch(tmp_path, f"{L.full_dir}/checkpoint-{ck}/adapter_config.json")
    r1 = steps_of(ctx, "round1")
    assert r1.status == "todo" and isinstance(r1.steps[-1], Later)
    assert [r.name for r in r1.steps if isinstance(r, Read)] == [f"q2bfulleval-ck{ck}-s{s}" for ck in (100, 120) for s in (0, 1, 2)]
    for s in (0, 1, 2):
        scored(tmp_path, f"data/runs/q2bfulleval-ck100-s{s}--tsmc_core_agentic", 0.76)
        scored(tmp_path, f"data/runs/q2bfulleval-ck120-s{s}--tsmc_core_agentic", 0.80)
    r1 = steps_of(ctx, "round1")
    assert WritePeak(L.full_dir, 120) in r1.steps
    extra = [r for r in r1.steps if isinstance(r, Read) and r.group == "post"]
    assert [r.name for r in extra] == [f"q2bfulleval-ck120-s{s}" for s in (3, 4, 5)]
    for r in extra:                                   # poor extra seeds do not move the pick
        scored(tmp_path, r.run_dir, 0.40)
    assert WritePeak(L.full_dir, 120) in steps_of(ctx, "round1").steps


def test_a_decision_round_reads_its_two_leaders_before_it_picks(tmp_path):
    ctx = ctx_for(tmp_path)
    L = ctx.layout
    touch(tmp_path, f"{L.sft2}/adapter_config.json")
    for ck in (80, 100, 120):
        touch(tmp_path, f"{L.d1_dir}/checkpoint-{ck}/adapter_config.json")
    for ck, g in ((80, 0.70), (100, 0.75), (120, 0.74)):
        for s in (0, 1, 2):
            scored(tmp_path, f"data/runs/q2bd1eval-ck{ck}-s{s}--tsmc_core_agentic", g)
    d1 = steps_of(ctx, "d1")
    post = [r for r in d1.steps if isinstance(r, Read) and r.group == "post"]
    assert [r.name for r in post] == [f"q2bd1eval-ck{ck}-s{s}" for ck in (100, 120) for s in range(8)]
    assert isinstance(d1.steps[-1], Later)
    for r in post:
        scored(tmp_path, r.run_dir, 0.75 if "ck100" in r.name else 0.74)
    d1 = steps_of(ctx, "d1")
    assert WritePeak(L.d1_dir, 120) in d1.steps              # tied within 0.03, equal seeds: the later checkpoint
    tail = [r.name for r in d1.steps if isinstance(r, Read) and r.group in ("transfer", "pass5")]
    assert tail == [f"q2bd1bio-ck120-s{s}" for s in (0, 1, 2)] + [f"q2bd1pass5-ck120-s{s}" for s in (1, 2, 3, 4)]
    assert all(r.chain == (L.sft2, f"{L.d1_dir}/checkpoint-120") for r in d1.steps if isinstance(r, Read) and r.group == "pass5")


def test_the_slice_flags():
    assert slice_of(None, None) == STAGES
    assert slice_of("d1", "d2") == ("d1", "d2")
    assert slice_of("final", None) == ("final", "rows", "verifier", "score")
    with pytest.raises(SystemExit, match="not a stage"):
        slice_of("sft3", None)
    with pytest.raises(SystemExit, match="comes after"):
        slice_of("d2", "d1")


def test_another_graph_takes_its_own_names_and_no_transfer(tmp_path):
    ctx = ctx_for(tmp_path, graph="pfas", transfer=None)
    gate1 = steps_of(ctx, "gate1").steps
    reads = [s for s in gate1 if isinstance(s, Read)]
    assert [r.name for r in reads] == [f"q2bpfassft1eval-s{s}" for s in (0, 1, 2)]
    assert reads[0].benchmark == "data/benchmarks/pfas/pfas_core_agentic.jsonl"
    assert reads[0].kb == "data/raw/pfas/PFAS_KG.graphml"
    assert steps_of(ctx, "sft1").steps[-1].final == "models/traj_sft_2b_pfas_flash20/final"


def test_the_dry_run_has_one_line_per_stage(tmp_path):
    ctx = ctx_for(tmp_path)
    text = render(ctx, plan(ctx))
    lines = text.splitlines()
    assert lines[0].startswith("=== plan q2b: GRAPH=tsmc SIZE=2b") and "  P=q2b" in lines
    for name in STAGES:
        assert any(l.startswith(f"{name}:") for l in lines), name
    assert "demos:    need data/traj/tsmc_sft_mix_flash20.jsonl" in text
    released(tmp_path, ctx)
    text = render(ctx, plan(ctx, "band", "rftbuild"), names=False)
    assert "band:     skipped (models/traj_grpo_2b_flash20full is on disk)" in text
    assert "grid 0 of 18 to read" in text and "pick checkpoint-100" in text


@pytest.mark.parametrize("target", [0, 50, 140])
def test_a_target_no_round_saves_is_refused_before_anything_runs(tmp_path, target):
    """A round that never writes checkpoint-TARGET runs on to final/, counts as
    leaving no output, and the next launch would remove it and train it again."""
    with pytest.raises(SystemExit) as e:
        plan(ctx_for(tmp_path, target=target))
    assert e.value.code == 2 and f"TARGET={target}" in str(e.value)
    assert plan(ctx_for(tmp_path, target=60))


def test_a_stored_read_at_another_turn_budget_is_refused(tmp_path):
    """A different --turn-max-tokens is a different policy, and a read's name
    carries no budget, so a relaunch under another TURN_CAP may not reuse it."""
    ctx = ctx_for(tmp_path, turn_cap=2048)
    first = next(s for s in steps_of(ctx, "gate1").steps if isinstance(s, Read))
    assert plan(ctx)                                   # nothing stored yet
    scored(tmp_path, first.run_dir)                    # recorded at 1024
    with pytest.raises(SystemExit) as e:
        plan(ctx)
    assert e.value.code == 2 and "1024 tokens a turn and TURN_CAP is 2048" in str(e.value)
    assert plan(ctx_for(tmp_path))                     # the default budget takes it


def test_a_recorded_pick_of_a_checkpoint_not_on_disk_is_refused(tmp_path):
    """A recorded pick is final, so one that names a missing checkpoint is
    refused before anything runs rather than silently picked again."""
    ctx = ctx_for(tmp_path)
    L = ctx.layout
    touch(tmp_path, f"{L.full_dir}/checkpoint-60/adapter_config.json")
    touch(tmp_path, f"{L.full_dir}/peak.txt", "100\n")
    with pytest.raises(SystemExit) as e:
        plan(ctx)
    assert e.value.code == 2 and "checkpoint-100, which is not on disk" in str(e.value)
    touch(tmp_path, f"{L.full_dir}/peak.txt", "60\n")
    assert plan(ctx) and WritePeak(L.full_dir, 60).done(ctx)


def test_stage_3_rows_are_not_rebuilt_once_its_adapter_is_saved(tmp_path):
    """A release that ships the stage-3 adapter without its mix need not roll
    four passes to rebuild a mix nothing will train on."""
    ctx = ctx_for(tmp_path)
    L = ctx.layout
    for final in (L.sft1, L.sft2):
        touch(tmp_path, f"{final}/adapter_config.json")
    touch(tmp_path, f"{L.full_dir}/checkpoint-100/adapter_config.json")
    touch(tmp_path, f"{L.full_dir}/peak.txt", "100\n")
    planned = {p.name: p for p in plan(ctx)}
    assert planned["rftroll"].status == "skipped" and planned["rftbuild"].status == "skipped"
    assert planned["rftroll"].reason == f"{L.sft2} is saved"


def test_a_summary_cut_short_is_not_a_scored_read(tmp_path):
    """A kill mid-write leaves a summary.json that exists and does not parse;
    the plan reads it as not done, so the read runs again."""
    ctx = ctx_for(tmp_path)
    L = ctx.layout
    touch(tmp_path, f"{L.sft1}/adapter_config.json")
    gate = [s for s in steps_of(ctx, "gate1").steps if isinstance(s, Read)][0]
    scored(tmp_path, gate.run_dir)
    assert gate.done(ctx)
    touch(tmp_path, f"{gate.run_dir}/summary.json", '{"aggregate": {"ALL": {"gated_acc": 0.')
    assert not gate.done(ctx)


def test_a_checkpoint_counts_once_its_adapter_config_is_written(tmp_path):
    """A `checkpoint-N` directory the trainer is still saving has no
    adapter_config.json yet, and is neither a grid checkpoint nor the target."""
    from graph_reasoner.recipe.plan import checkpoints, round_done
    ctx = ctx_for(tmp_path)
    L = ctx.layout
    (tmp_path / L.full_dir / "checkpoint-120").mkdir(parents=True)
    touch(tmp_path, f"{L.full_dir}/checkpoint-100/adapter_config.json")
    assert checkpoints(ctx, L.full_dir) == [100]
    assert not round_done(ctx, L.full_dir)
    touch(tmp_path, f"{L.full_dir}/checkpoint-120/adapter_config.json")
    assert checkpoints(ctx, L.full_dir) == [100, 120] and round_done(ctx, L.full_dir)


def test_the_full_round_picks_the_best_mean_not_the_later_checkpoint_within_the_band(tmp_path):
    """The released full rounds were picked by the best three-seed mean (the
    2B's checkpoint 100 over its 120, 0.697 against 0.668); a decision round
    picks by the band rule, which would take the later checkpoint here."""
    ctx = ctx_for(tmp_path, transfer=None)
    L = ctx.layout
    touch(tmp_path, f"{L.sft1}/adapter_config.json")
    for ck, g in ((100, 0.80), (120, 0.79)):
        touch(tmp_path, f"{L.full_dir}/checkpoint-{ck}/adapter_config.json")
        for s in (0, 1, 2):
            scored(tmp_path, f"data/runs/q2bfulleval-ck{ck}-s{s}--tsmc_core_agentic", g)
    assert WritePeak(L.full_dir, 100) in steps_of(ctx, "round1").steps
