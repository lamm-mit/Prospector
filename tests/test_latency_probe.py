"""Every eval run records its own latency.

A run's `wall_ms` is a latency only under `--sequential --no-cache`. Any other
run ends with `run_latency_probe`, unless `--no-latency` skips it: the first
five items of each tier of the whole benchmark, one at a time, cache off,
written to `<run>/latency/`.

The probe drives the engine the main pass built. A vLLM engine holds the GPU
until the process exits, so `build_generate_seam` is called once per run.
"""
import json

import pytest

from graph_reasoner.agent import eval as ev
from graph_reasoner.agent.eval import run_agentic_eval
from graph_reasoner.agent.loop import EpisodeConfig

@pytest.fixture
def toy(tmp_path):
    """A text KB and a six-item benchmark, two items in each of three tiers."""
    kb = tmp_path / "kb.txt"
    kb.write_text("Film A|written_by|Jane Doe\nFilm B|written_by|Jane Doe\n",
                  encoding="utf-8")
    bench = tmp_path / "toy_bench.jsonl"
    with bench.open("w", encoding="utf-8") as f:
        for tier in ("easy", "medium", "hard"):
            for n in range(2):
                f.write(json.dumps({
                    "id": f"{tier}-{n:04d}", "tier": tier,
                    "question": "who wrote Film A", "gold_answer": "Jane Doe",
                    "gold_path": {"nodes": ["Film A", "Jane Doe"],
                                  "edges": [{"source": "Film A", "relation": "written_by",
                                             "target": "Jane Doe"}]},
                    "subgraph_edges": [], "subgraph_text": "",
                    "meta": {"anchor": "Film A", "kb_answer_set": ["Jane Doe"],
                             "answer_type": "person"}}) + "\n")
    return {"kb": str(kb), "bench": str(bench), "out": str(tmp_path / "runs")}


def _run(toy, model="hf:fake", **kw):
    return run_agentic_eval(toy["bench"], model, kb_path=toy["kb"], backend="triples",
                            out_root=toy["out"], run_name="t", cfg=EpisodeConfig(),
                            **kw)


def test_every_run_writes_a_sequential_uncached_probe(toy, stub_local_hf, tmp_path):
    log = stub_local_hf
    s = _run(toy)
    # The main pass ran as one batch; the probe ran one question at a time.
    assert s["timing"]["sequential"] is False
    assert s["latency"]["dir"] == "latency" and s["latency"]["n"] == 6
    assert "p50_s" in s["latency"] and "questions_per_minute" in s["latency"]

    probe = tmp_path / "runs" / "t--toy_bench" / "latency"
    ps = json.loads((probe / "summary.json").read_text())
    assert ps["timing"]["sequential"] is True
    assert ps["timing"]["wall_ms_semantics"] == "per_episode"
    assert ps["timing"]["concurrency"] == 1 and ps["timing"]["n_episodes"] == 6
    assert ps["probe"] == {"per_tier": 5, "cache": False,
                           "items": ["easy-0000", "easy-0001", "medium-0000",
                                     "medium-0001", "hard-0000", "hard-0001"]}
    rows = [json.loads(l) for l in (probe / "trajectories.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ps["probe"]["items"]
    # One question per batch during the probe, all six together before it.
    assert log["widths"][0] == 6 and set(log["widths"][1:]) == {1}


def test_the_probe_drives_the_engine_the_main_pass_built(toy, stub_local_hf):
    log = stub_local_hf
    _run(toy)
    assert log["built"] == 1, "a second engine for the probe would not fit beside the first"


def test_the_probe_times_the_whole_benchmarks_first_items_not_the_runs_subset(
        toy, stub_local_hf, tmp_path):
    s = _run(toy, limit=1)                      # main pass: three items
    assert s["timing"]["n_episodes"] == 3
    assert s["latency"]["n"] == 6               # probe: every item, same slice for every run
    ps = json.loads((tmp_path / "runs" / "t--toy_bench" / "latency" / "summary.json").read_text())
    assert ps["probe"]["items"] == ["easy-0000", "easy-0001", "medium-0000", "medium-0001",
                                    "hard-0000", "hard-0001"]


def test_a_sequential_uncached_main_pass_is_its_own_latency(toy, stub_local_hf, tmp_path):
    log = stub_local_hf
    s = _run(toy, sequential=True)
    assert s["latency"]["source"] == "main run" and s["latency"]["n"] == 6
    assert not (tmp_path / "runs" / "t--toy_bench" / "latency").exists()
    assert set(log["widths"]) == {1}


def test_opting_out_records_why_the_probe_was_skipped(toy, stub_local_hf):
    s = run_agentic_eval(toy["bench"], "hf:fake", kb_path=toy["kb"], backend="triples",
                         out_root=toy["out"], run_name="t", cfg=EpisodeConfig(), latency=False)
    assert "--no-latency" in s["latency"]["skipped"]


def test_a_failed_probe_is_recorded_and_the_scored_run_survives(toy, stub_local_hf, monkeypatch):

    def boom(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(ev, "run_latency_probe", boom)
    s = _run(toy)
    assert s["latency"]["failed"].startswith("RuntimeError: provider down")
    assert s["timing"]["n_episodes"] == 6 and s["aggregate"]
