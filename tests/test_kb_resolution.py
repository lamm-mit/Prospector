"""The graph file follows the substrate, so a benchmark is never scored
against another graph's file.

A substrate registers its default graph file, `resolve_kb` returns an explicit
path or that default, and no script carries a `--kb` default or a graph path
of its own.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from conftest import ROOT
from graph_reasoner.agent.backend import (available_backends, default_kb,
                                          register_backend, resolve_kb)

#: Every registered graph but the toy text-triples backend, which has no file
#: of its own (the tests write one).
GRAPHS = sorted(set(available_backends()) - {"triples"})


def test_every_shipped_substrate_declares_its_graph():
    assert {"tsmc", "biograph", "pfas", "matprop"} <= set(GRAPHS)
    for kind in GRAPHS:
        path = default_kb(kind)
        assert path, f"{kind} declares no default graph file"
        assert path.startswith("data/raw/") and kind[:4] in path.lower(), \
            f"{kind}'s declared file does not look like its own: {path}"


def test_no_two_substrates_share_a_file():
    files = [default_kb(k) for k in GRAPHS]
    assert len(set(files)) == len(files), files


def test_explicit_path_wins():
    assert resolve_kb("biograph", "somewhere/else.graphml") == "somewhere/else.graphml"


def test_omitted_path_follows_the_substrate():
    assert resolve_kb("biograph", None) == default_kb("biograph")
    assert resolve_kb("tsmc", None) == default_kb("tsmc")


def test_a_substrate_without_a_declared_file_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="declares no default graph file"):
        resolve_kb("triples", None)


def test_registration_accepts_a_declared_file():
    from graph_reasoner.agent import backend as B

    register_backend("_probe_kb", lambda path: None, default_kb="data/raw/probe.graphml")
    try:
        assert default_kb("_probe_kb") == "data/raw/probe.graphml"
        assert resolve_kb("_probe_kb", None) == "data/raw/probe.graphml"
    finally:
        # The registry is module state: the probe must not outlive the test.
        B._BACKENDS.pop("_probe_kb", None)
        B._DEFAULT_KB.pop("_probe_kb", None)


def test_a_substrate_declares_or_defaults_its_question_files():
    """The recipe reads each graph's evaluation and training questions from
    the registry: a declared file wins, else the default path under the
    graph's own directory; an unregistered name is refused."""
    from graph_reasoner.agent import backend as B
    from graph_reasoner.agent.backend import eval_benchmark, train_benchmark

    register_backend("_probe_bm", lambda path: None,
                     train_benchmark="data/benchmarks/_probe_bm/_probe_bm_train_agentic_9x.jsonl")
    try:
        assert eval_benchmark("_probe_bm") == "data/benchmarks/_probe_bm/_probe_bm_core_agentic.jsonl"
        assert train_benchmark("_probe_bm") == "data/benchmarks/_probe_bm/_probe_bm_train_agentic_9x.jsonl"
    finally:
        B._BACKENDS.pop("_probe_bm", None)
        B._BENCHMARKS.pop("_probe_bm", None)
    assert train_benchmark("tsmc") == "data/benchmarks/tsmc/tsmc_train_agentic_4x.jsonl"
    assert eval_benchmark("pfas") == "data/benchmarks/pfas/pfas_core_agentic.jsonl"
    for kind in ("tsmc", "biograph", "pfas", "matprop"):
        assert (ROOT / eval_benchmark(kind)).exists(), kind
    with pytest.raises(ValueError, match="unknown backend"):
        eval_benchmark("nosuch")


#: Every script that takes a graph file, found by the flag it declares. Each
#: infers the substrate (from the benchmark, the source runs or --graph) and
#: resolves the file with it.
GRAPH_SCRIPTS = sorted(p.relative_to(ROOT / "scripts").as_posix()
                       for p in ROOT.glob("scripts/**/*.py") if '"--kb"' in p.read_text())


def test_no_entry_point_carries_a_graph_file_as_its_own_default():
    """Every script's `--kb` defaults to None, and the file is resolved with
    `resolve_kb`, directly or through the recipe's `Layout.of`, which refuses
    an unregistered graph first."""
    assert len(GRAPH_SCRIPTS) >= 6, GRAPH_SCRIPTS
    offenders = []
    for script in GRAPH_SCRIPTS:
        src = (ROOT / "scripts" / script).read_text()
        i = src.index('"--kb"')
        tail = src[i:i + 200]
        # run_agent_eval resolves inside run_agentic_eval (the last test)
        resolves = (script == "run_agent_eval.py" or "resolve_kb(" in src
                    or ("Layout.of(" in src and "kb=args.kb" in src))
        if "default=None" not in tail or not resolves:
            offenders.append(script)
    assert not offenders, f"--kb carries a hardcoded default in: {offenders}"


def test_no_script_names_a_graph_file_of_its_own():
    """A literal path under data/raw/ in a script is a default that does not
    follow the substrate; the registry's `default_kb` owns every one."""
    offenders = [str(p.relative_to(ROOT)) for p in sorted(ROOT.glob("scripts/**/*.py"))
                 if any("data/raw/" in line and not line.lstrip().startswith("#")
                        and ".graphml" in line
                        for line in _code_lines(p))]
    assert not offenders, f"a graph file is named in: {offenders}"


def _code_lines(path: Path) -> list[str]:
    """The lines of a script outside its module docstring."""
    import ast
    src = path.read_text()
    tree = ast.parse(src)
    doc = tree.body[0] if tree.body and isinstance(tree.body[0], ast.Expr) \
        and isinstance(getattr(tree.body[0], "value", None), ast.Constant) else None
    lines = src.splitlines()
    if doc is not None:
        lines = lines[:doc.lineno - 1] + lines[doc.end_lineno:]
    return lines


def test_the_eval_resolves_the_file_through_the_substrate(monkeypatch):
    """A biograph benchmark with no --kb loads biograph's own graph file: the
    substrate comes from the benchmark and the file from the substrate."""
    from graph_reasoner.agent import eval as ev

    loaded = []

    def stop(kind, path):
        loaded.append((kind, path))
        raise RuntimeError("stop before any model is built")

    monkeypatch.setattr(ev, "load_backend", stop)
    bench = ROOT / "data/benchmarks/biograph/biograph_core_agentic.jsonl"
    with pytest.raises(RuntimeError, match="stop before"):
        ev.run_agentic_eval(str(bench), "gpt-x", kb_path=None, limit=1)
    assert loaded == [("biograph", default_kb("biograph"))]
