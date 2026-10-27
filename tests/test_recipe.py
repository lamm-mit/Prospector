"""`graph_reasoner.recipe`: the names the recipe derives from (graph, size,
tag) reproduce the released names, the stages resolve to the dicts the
trainers read, a hyperparameter file carries no derived key, and the CLI
prints every name.

The golden names below are the released artefacts' names at the three sizes,
which the layout must reproduce.
"""
import json
import shlex
import subprocess
import sys

import pytest
import yaml

from conftest import ROOT
from graph_reasoner import recipe
from graph_reasoner.recipe import (BAND_SEEDS, TRAINING, Graph, Layout, Spec, load_hyperparameters,
                                   read_manifest, smoke_config, training_config, write_manifest)

RELEASED = {
    "2b": {
        "sft1": "models/traj_sft_2b_flash20/final",
        "full_dir": "models/traj_grpo_2b_flash20full",
        "full_band_runs": [f"data/runs/q2bfullroll-s{s}--tsmc_train_agentic_4x" for s in BAND_SEEDS],
        "g2_keeps": "data/traj/tsmc_2b_flash20g2_keeps.jsonl",
        "g2_keeps_name": "2b_g2_keeps",
        "sft2_mix": "data/traj/tsmc_sft_mix_2b_flash20g2.jsonl",
        "sft2": "models/traj_sft_2b_flash20g2/final",
        "d1_dir": "models/traj_grpo_2b_flash20d1",
        "d1_prefixes": "data/traj/tsmc_prefixes_q2bd1.jsonl",
        "d1_band_run": "data/runs/q2bd1-s911--tsmc_prefixes_q2bd1",
        "d2_prefixes": "data/traj/tsmc_prefixes_q2bd2.jsonl",
        "verifier_rows": "data/traj/tsmc_verifier_yesno_2b_flash20v.jsonl",
        "verifier": "models/verifier_sft_2b_flash20v/final",
        "pool_prefix": "q2bpool32",
        "verifier_select_name": "q2b-verifier32-yn1",
        "gate1_prefix": "q2bsft1", "g2_prefix": "q2bg2", "g2_roll_prefix": "q2bg2roll",
        "v_roll_prefix": "q2bvroll", "teach_run_prefix": "flash20-teach-4x",
        "demos_mix": "data/traj/tsmc_sft_mix_flash20.jsonl",
        "demos_keeps": "data/traj/tsmc_flash20_keeps.jsonl",
        "demos_detours": "data/traj/tsmc_error_pairs_flash20.jsonl",
    },
    "0.8b": {
        "sft1": "models/traj_sft_0.8b_flash20/final",
        "full_dir": "models/traj_grpo_0.8b_flash20full",
        "d2_dir": "models/traj_grpo_0.8b_flash20d2",
        "d1_band_run": "data/runs/q0.8bd1-s911--tsmc_prefixes_q0.8bd1",
        "verifier_rows": "data/traj/tsmc_verifier_yesno_0.8b_flash20v.jsonl",
        "pool_prefix": "q0.8bpool32",
    },
    "4b": {
        "sft1": "models/traj_sft_4b_flash20/final",
        "sft2": "models/traj_sft_4b_flash20g2/final",
        "sft2_mix": "data/traj/tsmc_sft_mix_4b_flash20g2.jsonl",
        "g2_keeps": "data/traj/tsmc_4b_flash20g2_keeps.jsonl",
        "d1_prefixes": "data/traj/tsmc_prefixes_q4bd1.jsonl",
        "d2_prefixes": "data/traj/tsmc_prefixes_q4bd2.jsonl",
        "verifier_rows": "data/traj/tsmc_verifier_yesno_4b_flash20v.jsonl",
        "verifier": "models/verifier_sft_4b_flash20v/final",
        "full_dir": "models/traj_grpo_4b_flash20full",
        "d1_dir": "models/traj_grpo_4b_flash20d1",
        "d2_dir": "models/traj_grpo_4b_flash20d2",
        "pool_prefix": "q4bpool32",
    },
}


@pytest.mark.parametrize("size", sorted(RELEASED))
def test_the_layout_reproduces_the_released_names(size):
    lay = Layout.of(Spec("tsmc", size))
    for key, want in RELEASED[size].items():
        assert getattr(lay, key) == want, key


def test_the_model_a_size_implies_is_the_same_spec_passed_or_not():
    """The runner always passes --model, a user rarely does; both name one
    run, so the manifest's spec is the same."""
    assert Spec("tsmc", "2b", model="Qwen/Qwen3.5-2B") == Spec("tsmc", "2b")
    assert Spec("tsmc", "2b", model="Qwen/Qwen3.5-2B").model is None
    assert Spec("tsmc", "2b", model="Qwen/Qwen3.5-2B-Base").model == "Qwen/Qwen3.5-2B-Base"
    assert Spec("tsmc", "7b", model="Qwen/Qwen3.5-7B").base_model == "Qwen/Qwen3.5-7B"


def test_the_base_model_follows_the_size():
    assert Spec("tsmc", "0.8b").base_model == "Qwen/Qwen3.5-0.8B"
    assert Spec("tsmc", "2b").base_model == "Qwen/Qwen3.5-2B"
    assert Spec("tsmc", "4b").base_model == "Qwen/Qwen3.5-4B"
    assert Spec("tsmc", "4b", model="org/other").base_model == "org/other"
    with pytest.raises(SystemExit, match="--model"):
        Spec("tsmc", "9b").base_model


def test_the_run_prefix_rule_and_its_collision_freedom():
    """The released graph and tag add nothing to a name; every other graph
    and tag adds its token, so (graph, size, tag) never share a name."""
    assert Spec("tsmc", "2b").run_prefix == "q2b"
    assert Spec("tsmc", "2b", tag="exp1").run_prefix == "q2bexp1"
    assert Spec("pfas", "2b").run_prefix == "q2bpfas"
    assert Layout.of(Spec("pfas", "2b")).sft2_dir == "models/traj_sft_2b_pfas_flash20g2"
    assert Layout.of(Spec("tsmc", "2b", tag="exp1")).sft2_dir == "models/traj_sft_2b_exp1g2"
    names = {}
    for spec in (Spec("tsmc", "2b"), Spec("tsmc", "2b", tag="exp1"), Spec("tsmc", "0.8b"),
                 Spec("pfas", "2b")):
        env = Layout.of(spec).env()
        for k, v in env.items():
            if k in ("TAG", "EVAL_BENCH", "EVAL_STEM", "TRAIN_BENCH", "TRAIN_STEM", "KB", "GRAPH",
                     "XFER", "XFER_GRAPH", "XFER_BENCH", "XFER_STEM", "XFER_KB", "SIZE", "MODEL",
                     "BAND_SEEDS", "G2_ROLL_SEEDS", "GATE_SEEDS", "POOL_SEEDS", "PREFIX_BAND_SEED",
                     "FULL_PICK", "D1_PICK", "D2_PICK", "DEMOS", "DEMO_KEEPS", "DEMO_KEEPS1",
                     "DEMO_FAILED", "DETOURS", "TEACH", "G2_KEEPS_NAME") or not v:
                continue
            assert v not in names, f"{k}={v} of {spec} collides with {names[v]}"
            names[v] = spec
    for bad in ("a-b", "a_b", "A", ""):
        with pytest.raises(SystemExit, match="--tag"):
            Spec("tsmc", "2b", tag=bad)


def test_every_shipped_substrate_declares_resolvable_question_files():
    for name in ("tsmc", "biograph", "pfas", "matprop"):
        g = Graph.of(name)
        assert (ROOT / g.eval_benchmark).exists(), g.eval_benchmark
        assert g.eval_stem.split("_")[0] == name
    assert Graph.of("tsmc").train_benchmark == "data/benchmarks/tsmc/tsmc_train_agentic_4x.jsonl"
    assert Graph.of("biograph").transfer_token == "bio" and Graph.of("pfas").transfer_token == "pfas"
    with pytest.raises(SystemExit, match="registered substrate"):
        Graph.of("nosuch")


def _tree(tmp_path, pick=100):
    for p in ("models/traj_sft_2b_flash20/final", "models/traj_sft_2b_flash20g2/final",
              "data/traj", "data/runs/q2bd1-s911--tsmc_prefixes_q2bd1",
              "data/runs/q2bd2-s911--tsmc_prefixes_q2bd2", f"models/traj_grpo_2b_flash20d1/checkpoint-{pick}",
              "data/benchmarks/tsmc", "data/raw/GRAPHDATA_TSMC_OUTPUT", "configs"):
        (tmp_path / p).mkdir(parents=True, exist_ok=True)
    for p in ("models/traj_sft_2b_flash20/final/adapter_config.json",
              "models/traj_sft_2b_flash20g2/final/adapter_config.json",
              "data/traj/tsmc_sft_mix_flash20.jsonl", "data/traj/tsmc_sft_mix_2b_flash20g2.jsonl",
              "data/traj/tsmc_prefixes_q2bd1.jsonl", "data/traj/tsmc_prefixes_q2bd2.jsonl",
              "data/traj/tsmc_verifier_yesno_2b_flash20v.jsonl",
              "data/runs/q2bd1-s911--tsmc_prefixes_q2bd1/results.jsonl",
              "data/runs/q2bd2-s911--tsmc_prefixes_q2bd2/results.jsonl",
              "data/benchmarks/tsmc/tsmc_core_agentic.jsonl", "data/benchmarks/tsmc/tsmc_train_agentic_4x.jsonl",
              "data/raw/GRAPHDATA_TSMC_OUTPUT/tsmc_5b10p.graphml"):
        (tmp_path / p).write_text("{}\n")
    for s in BAND_SEEDS:
        d = tmp_path / f"data/runs/q2bfullroll-s{s}--tsmc_train_agentic_4x"
        d.mkdir(parents=True); (d / "results.jsonl").write_text("")
    (tmp_path / "models/traj_grpo_2b_flash20d1/peak.txt").write_text(f"{pick}\n")
    (tmp_path / f"models/traj_grpo_2b_flash20d1/checkpoint-{pick}/adapter_config.json").write_text("{}")
    for f in ("sft1", "sft2", "verifier", "grpo_full", "grpo_decision"):
        (tmp_path / "configs" / f"{f}.yaml").write_text((ROOT / "configs" / f"{f}.yaml").read_text())
    return Layout.of(Spec("tsmc", "2b"), root=tmp_path)


def test_the_resolved_dicts_validate_and_read_the_pick(tmp_path):
    from graph_reasoner.train.grpo import validate_config_keys
    from graph_reasoner.train.sft import validate_sft_config_keys

    lay = _tree(tmp_path)
    assert lay.d2_substrate() == ["models/traj_sft_2b_flash20g2/final",
                                  "models/traj_grpo_2b_flash20d1/checkpoint-100"]
    for stage, st in TRAINING.items():
        config, manifest = training_config(lay, stage, argv=["--graph", "tsmc"])
        (validate_sft_config_keys if st.trainer == "sft" else validate_config_keys)(config)
        assert config["base_model"] == "Qwen/Qwen3.5-2B"
        assert manifest["schema"] == "recipe_run_v1" and manifest["stage"] == stage
        assert manifest["config_file"]["path"] == st.config and len(manifest["config_file"]["sha256"]) == 64
        assert lay.check_inputs(stage) == [], stage
    full, _ = training_config(lay, "full")
    assert full["informative_band"] == {"lo": 0.05, "hi": 0.95, "rollouts": lay.full_band_runs}
    assert full["rollout_mode"] == "episode" and full["tiers"] == ["medium", "hard"]
    assert full["substrate_adapters"] == ["models/traj_sft_2b_flash20/final"]
    d1, _ = training_config(lay, "d1")
    assert d1["prefix_band"] == {"lo": 0.05, "hi": 0.95, "rollouts": [lay.d1_band_run]}
    assert d1["rollout_mode"] == "prefix" and d1["prefixes"] == lay.d1_prefixes and "tiers" not in d1
    assert d1["lr"] == pytest.approx(1.5e-5) and full["lr"] == pytest.approx(3e-5)
    (tmp_path / "models/traj_sft_2b_flash20g2/final/adapter_config.json").unlink()
    assert lay.check_inputs("d1") == ["adapter: models/traj_sft_2b_flash20g2/final"]


def test_a_missing_or_absent_pick_is_refused(tmp_path):
    lay = _tree(tmp_path, pick=100)
    (tmp_path / "models/traj_grpo_2b_flash20d1/peak.txt").write_text("120\n")
    with pytest.raises(SystemExit, match="not on disk"):
        lay.d2_substrate()
    (tmp_path / "models/traj_grpo_2b_flash20d1/peak.txt").unlink()
    with pytest.raises(SystemExit, match="no pick"):
        lay.d2_substrate()
    assert not lay.has_pick(lay.d1_dir)


def test_asking_whether_a_round_has_a_pick_prints_nothing(tmp_path, capsys):
    """The runner asks on every stage recompute; only a refusal prints."""
    lay = _tree(tmp_path, pick=100)
    (tmp_path / "models/traj_grpo_2b_flash20d1/peak.txt").unlink()
    assert not lay.has_pick(lay.d1_dir) and lay.pick_problem(lay.d1_dir).startswith("no pick")
    assert capsys.readouterr().err == ""
    with pytest.raises(SystemExit):
        lay.pick(lay.d1_dir)
    assert capsys.readouterr().err.count("no pick") == 1


def test_the_hyperparameter_files_carry_no_derived_key(tmp_path):
    for stage, st in TRAINING.items():
        cfg = load_hyperparameters(ROOT / st.config, st.trainer)
        assert "stage" not in cfg and "output_dir" not in cfg and "base_model" not in cfg
    bad = tmp_path / "bad.yaml"
    for key, value in (("dataset", "x.jsonl"), ("output_dir", "models/x"), ("base_model", "m")):
        bad.write_text(yaml.safe_dump({key: value, "lr": 1e-4}))
        with pytest.raises(SystemExit, match="derives"):
            load_hyperparameters(bad, "sft")
    bad.write_text(yaml.safe_dump({"informative_band": {"lo": 0.1, "hi": 0.9, "rollouts": ["r"]}}))
    with pytest.raises(SystemExit, match="informative_band.rollouts"):
        load_hyperparameters(bad, "grpo")
    bad.write_text(yaml.safe_dump({"nosuch_key": 1}))
    with pytest.raises(SystemExit, match="unknown config key"):
        load_hyperparameters(bad, "grpo")


def test_smoke_config_differs_in_exactly_three_keys():
    config = {"output_dir": "models/x/", "max_steps": 120, "save_steps": 20, "lr": 1e-5}
    twin = smoke_config(config)
    assert twin == {"output_dir": "models/x_smoke", "max_steps": 3, "save_steps": 100, "lr": 1e-5}


def test_the_manifest_is_written_once_and_resumes_or_refuses(tmp_path):
    lay = _tree(tmp_path)
    config, manifest = training_config(lay, "sft1", argv=["a"])
    out = tmp_path / config["output_dir"]
    path = write_manifest(out, manifest)
    again = dict(manifest, launch={**manifest["launch"], "argv": ["b"]})
    write_manifest(out, again)
    have = json.loads(path.read_text())
    assert have["launch"]["argv"] == ["a"] and [r["argv"] for r in have["relaunches"]] == [["b"]]
    assert read_manifest(out / "final") == have and read_manifest(out / "checkpoint-50") == have
    other = dict(manifest, hyperparameters={**manifest["hyperparameters"], "lr": 1.0})
    with pytest.raises(SystemExit, match="hyperparameters differs"):
        write_manifest(out, other)
    assert read_manifest(tmp_path / "models/nowhere") is None


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "graph_reasoner.recipe", "names", *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_the_cli_prints_every_name_one_name_or_json():
    env = _cli("--graph", "tsmc", "--size", "2b", "--transfer-graph", "biograph", "--env")
    assert env.returncode == 0, env.stderr
    pairs = {}
    for line in env.stdout.splitlines():
        words = shlex.split(line)
        assert words[0] == "export" and "=" in words[1], line
        k, v = words[1].split("=", 1); pairs[k] = v
    for k in ("P", "SFT1", "SFT2", "D1_V", "D2_V", "PREFIXES_D1", "D1_BAND_RUN", "ROWS", "VER",
              "POOL", "VSEL", "EVAL_STEM", "TRAIN_BENCH", "KB", "XFER", "XFER_BENCH", "LOG_PREFIX",
              "DEMOS", "TEACH", "BAND_SEEDS"):
        assert pairs[k], k
    assert pairs["P"] == "q2b" and pairs["XFER"] == "bio" and pairs["LOG_PREFIX"] == "logs/q2b"
    one = _cli("--graph", "tsmc", "--size", "2b", "sft1")
    assert one.stdout.strip() == "models/traj_sft_2b_flash20/final"
    lower = _cli("--graph", "tsmc", "--size", "2b", "d1_band_run")
    assert lower.stdout.strip() == pairs["D1_BAND_RUN"]
    sizeless = _cli("--graph", "tsmc", "TEACH")
    assert sizeless.stdout.strip() == "flash20-teach-4x"
    js = _cli("--graph", "pfas", "--size", "2b", "--json")
    data = json.loads(js.stdout)
    assert data["spec"]["graph"] == "pfas" and data["P"] == "q2bpfas"
    assert data["DEMOS"] == "data/traj/pfas_sft_mix_flash20.jsonl"
    bad = _cli("--graph", "tsmc", "--size", "2b", "nosuch")
    assert bad.returncode != 0 and "unknown key" in bad.stderr
    nograph = _cli("--graph", "nosuch", "--size", "2b", "P")
    assert nograph.returncode != 0 and "registered substrate" in nograph.stderr


def test_the_served_stack_reads_the_two_picks(tmp_path):
    lay = _tree(tmp_path)
    for d, ck in (("models/traj_grpo_2b_flash20d2", 120),):
        (tmp_path / d / f"checkpoint-{ck}").mkdir(parents=True)
        (tmp_path / d / "peak.txt").write_text(f"{ck}\n")
    assert lay.served_stack() == ["models/traj_sft_2b_flash20g2/final",
                                  "models/traj_grpo_2b_flash20d1/checkpoint-100",
                                  "models/traj_grpo_2b_flash20d2/checkpoint-120"]
    assert lay.value("stack") == ",".join(lay.served_stack())
    assert lay.env()["D1_PICK"] == "100" and lay.env()["D2_PICK"] == "120" and lay.env()["FULL_PICK"] == ""


def test_the_app_serves_the_recipe_s_4b_stack():
    """The app's default chain is the layout's served 4B stack at the paper's
    picks (configs/README.md), spelled once there and derived here."""
    from app.live import DEFAULT_ADAPTER, DEFAULT_MODEL

    lay = Layout.of(Spec("tsmc", "4b"))
    assert DEFAULT_ADAPTER == ",".join([lay.sft2, f"{lay.d1_dir}/checkpoint-120", f"{lay.d2_dir}/checkpoint-120"])
    assert DEFAULT_MODEL == lay.spec.base_model


def test_the_recipe_module_imports_no_torch():
    out = subprocess.run([sys.executable, "-c",
                          "import sys, graph_reasoner.recipe; "
                          "print(sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'trl', 'peft', 'vllm')))"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
    assert recipe.MANIFEST == "run.json"


def _load_script(name):
    import importlib.util

    from conftest import script

    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), script(name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name, stage, trainer", [("train_sft.py", "sft1", "run_sft"),
                                                  ("train_traj_grpo.py", "d2", "run_grpo")])
def test_the_trainer_scripts_write_the_manifest_then_train(tmp_path, monkeypatch, name, stage, trainer):
    """Resolution, the input check, the manifest, --dry-run and --smoke,
    with the trainer itself replaced: it only has to receive the dict."""
    _tree(tmp_path)
    (tmp_path / "models/traj_grpo_2b_flash20d1/checkpoint-100/adapter_config.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    mod = _load_script(name)
    got = []
    monkeypatch.setattr(mod, trainer, lambda config: got.append(config))
    base = ["--graph", "tsmc", "--size", "2b", "--stage", stage]

    assert mod.main(base + ["--dry-run"]) == 0 and not got
    assert not list(tmp_path.glob("models/**/run.json"))

    assert mod.main(base) == 0 and len(got) == 1
    config = got[0]
    out = tmp_path / config["output_dir"]
    manifest = json.loads((out / "run.json").read_text())
    assert manifest["stage"] == stage and manifest["launch"]["argv"] == base
    if stage == "sft1":
        assert config["dataset"] == "data/traj/tsmc_sft_mix_flash20.jsonl"
        assert config["output_dir"] == "models/traj_sft_2b_flash20"
    else:
        assert config["substrate_adapters"] == ["models/traj_sft_2b_flash20g2/final",
                                                "models/traj_grpo_2b_flash20d1/checkpoint-100"]
        assert config["prefix_band"]["rollouts"] == ["data/runs/q2bd2-s911--tsmc_prefixes_q2bd2"]
        assert config["backend"] == "tsmc" and config["kb"].endswith("tsmc_5b10p.graphml")

    stale = tmp_path / (config["output_dir"] + "_smoke")
    stale.mkdir(); (stale / "old").write_text("x")
    assert mod.main(base + ["--smoke"]) == 0
    assert got[-1]["output_dir"] == config["output_dir"] + "_smoke" and got[-1]["max_steps"] == 3
    assert not (stale / "old").exists() and (stale / "run.json").exists()

    (tmp_path / "data/traj/tsmc_sft_mix_flash20.jsonl").unlink()
    (tmp_path / "data/traj/tsmc_prefixes_q2bd2.jsonl").unlink()
    with pytest.raises(SystemExit, match="missing its inputs"):
        mod.main(base)
