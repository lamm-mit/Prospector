"""The provenance record of a generated artifact, `provenance.py`: the teacher
label comes from the source runs' own summaries, and two builds from the same
inputs give the same record."""
import json

from conftest import ROOT
from graph_reasoner import provenance


def test_teacher_is_read_from_the_run_not_guessed_from_its_name(tmp_path):
    """The teacher and the turn budget come from the run's summary.json, not
    from the directory's name."""
    run = tmp_path / "flash-teach-4x-r--tsmc_train_agentic_4x"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({
        "model": "google/gemini-3.7-flash", "backend": "tsmc",
        "policy": {"max_turns": 14, "relation_vocab": True}}))

    info = provenance.describe_run(run)
    assert info["model"] == "google/gemini-3.7-flash"
    assert info["max_turns"] == 14
    assert provenance.teacher_label([info]) == "google/gemini-3.7-flash"


def test_distillation_and_expert_iteration_get_different_labels(tmp_path):
    """A teacher's rollouts and the policy's own get different labels: each is
    the model its run records."""
    labels = []
    for name, model in (("teach", "google/gemini-3.7-flash"),
                        ("self", "hf:models/traj_grpo_2b_flash20full/step60")):
        run = tmp_path / name
        run.mkdir()
        (run / "summary.json").write_text(json.dumps({"model": model}))
        labels.append(provenance.teacher_label([provenance.describe_run(run)]))
    assert labels[0] != labels[1]
    assert labels == ["google/gemini-3.7-flash", "hf:models/traj_grpo_2b_flash20full/step60"]


def test_a_blended_dataset_names_every_teacher(tmp_path):
    """A dataset built from runs of more than one model is labelled `mixed:`
    and names each of them."""
    runs = []
    for name, model in (("a", "google/gemini-3.7-flash"), ("b", "hf:models/x")):
        r = tmp_path / name
        r.mkdir()
        (r / "summary.json").write_text(json.dumps({"model": model}))
        runs.append(provenance.describe_run(r))
    label = provenance.teacher_label(runs)
    assert label.startswith("mixed:")
    assert "google/gemini-3.7-flash" in label and "hf:models/x" in label


def test_an_unreadable_run_is_reported_not_silently_dropped(tmp_path):
    """A run with no readable summary is labelled 'unknown', never given a
    guessed teacher."""
    run = tmp_path / "no-summary"
    run.mkdir()
    info = provenance.describe_run(run)
    assert info["summary"] == "unreadable"
    assert provenance.teacher_label([info]) == "unknown"


def test_block_records_the_commit_and_whether_the_tree_was_dirty():
    """The block records the commit, the branch and whether the tree was
    dirty. A dirty build is recorded, not refused."""
    prov = provenance.block("tests", ["--x"])
    assert prov["schema"] == "provenance_v1"
    assert prov["tool"] == "tests" and prov["argv"] == ["--x"]
    assert set(prov["git"]) == {"sha", "branch", "dirty"}


def test_the_stamp_carries_no_wall_clock():
    """Two builds from identical inputs produce identical provenance: no
    timestamp is recorded unless the caller passes one in `extra`."""
    a = provenance.block("t", [])
    b = provenance.block("t", [])
    assert a == b
    assert a, "an empty block would satisfy both checks here"
    assert not any("time" in k.lower() or "date" in k.lower() for k in a)


def test_sidecar_round_trips(tmp_path):
    out = tmp_path / "rows.jsonl"
    out.write_text("{}\n")
    prov = provenance.block("scripts/traj/build_sft_rows.py", ["--out", str(out)])
    p = provenance.write_sidecar(out, prov)
    assert p == provenance.sidecar_path(out)
    assert json.loads(p.read_text()) == prov


def test_the_shipped_substrate_has_a_committed_lineage():
    """`data/traj/` is gitignored except for its provenance sidecars, so a
    fresh clone can still say which teacher produced the stage-1 training
    data: keeps from nine gemini-3.7-flash teacher passes, blended into the
    mix the first SFT stage reads."""
    keeps = provenance.sidecar_path(ROOT / "data/traj/tsmc_flash20_keeps.jsonl")
    assert keeps.exists(), "the flash20 keeps' provenance sidecar must stay committed"
    prov = json.loads(keeps.read_text())
    assert prov["teacher"] == "google/gemini-3.7-flash"
    assert len(prov["source_runs"]) == 9
    assert all(r["run"].startswith("flash20-teach-4x-") for r in prov["source_runs"])
    assert all(r["model"] == "google/gemini-3.7-flash" for r in prov["source_runs"])

    mix = provenance.sidecar_path(ROOT / "data/traj/tsmc_sft_mix_flash20.jsonl")
    assert mix.exists(), "the stage-1 mix's provenance sidecar must stay committed"
    mprov = json.loads(mix.read_text())
    assert mprov["components"]["keeps"]["path"] == "data/traj/tsmc_flash20_keeps.jsonl"
    assert mprov["rows_total"] == 39_695


def test_a_file_a_build_writes_under_data_does_not_make_the_tree_dirty():
    """A builder writes its sidecar under data/traj/, a tracked location; the
    next build in the same session must not record its code as dirty for it."""
    from conftest import ROOT
    from graph_reasoner.provenance import git_state

    before = git_state()["dirty"]
    probe = ROOT / "data" / "traj" / "_dirty_probe.jsonl.provenance.json"
    try:
        probe.write_text("{}\n")
        assert git_state()["dirty"] == before
    finally:
        probe.unlink(missing_ok=True)
