"""The static scorer's answer sets follow the run it is scoring.

Scoring one graph's arm against another graph's answer sets would report a
near-zero arm as if the model had failed, so the scorer takes the answer sets
from the run's own graph and refuses a mismatch.

The static arms' files are built, not tracked, so the tests that need one build
it in a scratch copy of the layout: a graph's evaluation file under
`<root>/<graph>/` and its closed-book arm beside it.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys

import pytest

from conftest import ROOT, script
from graph_reasoner import benchmarks

REAL = ROOT / "data/benchmarks"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, script(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """`<tmp>/benchmarks/{biograph,tsmc}/`: each graph's evaluation file and the
    closed-book arm built from it, with the package's benchmark root set there."""
    root = tmp_path / "benchmarks"
    closed_book = _load("build_closed_book").closed_book
    for kind in ("biograph", "tsmc"):
        (root / kind).mkdir(parents=True)
        src = root / kind / f"{kind}_core_agentic.jsonl"
        shutil.copy(REAL / kind / src.name, src)
        with (root / kind / f"{kind}_core__closed_book.jsonl").open("w", encoding="utf-8") as f:
            for it in benchmarks.load_benchmark(src):
                f.write(json.dumps(closed_book(it).model_dump(), ensure_ascii=False) + "\n")
    monkeypatch.setattr(benchmarks, "BENCHMARK_ROOT", root)
    return root


@pytest.mark.parametrize("kind", ["biograph", "tsmc"])
def test_answers_follow_the_static_benchmark(layout, kind):
    static = layout / kind / f"{kind}_core__closed_book.jsonl"
    got = benchmarks.answers_for(static)
    assert got.name == f"{kind}_core_agentic.jsonl"
    assert got.parent == static.parent, "answer sets must stay in the substrate"


def test_a_non_static_benchmark_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="not a static benchmark"):
        benchmarks.answers_for(REAL / "tsmc/tsmc_core_agentic.jsonl")


def test_substrate_of_reads_the_layout():
    assert benchmarks.substrate_of(REAL / "biograph/biograph_core_agentic.jsonl") == "biograph"
    assert benchmarks.substrate_of(REAL / "tsmc/tsmc_core_agentic.jsonl") == "tsmc"
    assert benchmarks.substrate_of(ROOT / "README.md") is None


def _biograph_run(layout, tmp_path):
    """A one-completion static run on the biograph closed-book arm."""
    arm = layout / "biograph" / "biograph_core__closed_book.jsonl"
    run = tmp_path / "arm--biograph_core__closed_book"
    run.mkdir()
    first = json.loads(arm.read_text(encoding="utf-8").splitlines()[0])
    (run / "completions.jsonl").write_text(json.dumps(
        {"id": first["id"], "completion": "ANSWER: heavy metal ions"}) + "\n")
    return run, arm


def _score(monkeypatch, run, arm, *extra):
    monkeypatch.setattr(sys, "argv", ["score_static_gated.py", str(run),
                                      "--subgraphs", str(arm), *extra])
    return _load("score_static_gated").main()


def test_the_scorer_takes_the_answer_sets_of_the_runs_own_graph(layout, tmp_path, monkeypatch, capsys):
    """No --benchmark: the answer sets come from the run's substrate."""
    run, arm = _biograph_run(layout, tmp_path)
    assert not _score(monkeypatch, run, arm)
    assert "(answers: biograph_core_agentic.jsonl;" in capsys.readouterr().out
    scored = json.loads((run / "score_static_gated.json").read_text())
    assert scored["ALL"]["n"] == 1 and scored["skipped"] == 0


def test_the_scorer_refuses_a_cross_substrate_answer_set(layout, tmp_path, monkeypatch):
    run, arm = _biograph_run(layout, tmp_path)
    with pytest.raises(SystemExit, match="refusing to score a biograph run against tsmc answer sets"):
        _score(monkeypatch, run, arm, "--benchmark", str(layout / "tsmc" / "tsmc_core_agentic.jsonl"))


def test_a_stored_record_names_repository_paths_relative_and_others_as_given(tmp_path):
    """A score file goes out with the release, so it must not name the
    directories of the machine that scored it."""
    inside = REAL / "tsmc" / "tsmc_core_agentic.jsonl"
    assert benchmarks.repo_path(inside) == "data/benchmarks/tsmc/tsmc_core_agentic.jsonl"
    assert benchmarks.repo_path(tmp_path / "x.jsonl") == str(tmp_path / "x.jsonl")
