"""The retrieval baseline's files are what the builder writes.

`scripts/arms/build_retrieval_benchmark.py` cuts each question's 400 edges out
of the graph by a fixed rule, starting from the question's entity. The files
are built, not tracked, so `SHOWN` holds the SHA-256 of each file the paper's
retrieval runs read, and a build must reproduce it.

Two builds run here, one per rule: k-hop on the semiconductor graph and
personalized PageRank on a transfer graph. PFAS and MatProp take a minute each
and are left to a manual check: build, then `shasum -a 256` against `SHOWN`.
"""
import hashlib
import subprocess
import sys

import pytest

from conftest import ROOT, script
from graph_reasoner.agent.backend import default_kb

BUILDER = script("build_retrieval_benchmark.py")

# (graph, rule) -> SHA-256 of `<graph>_core__<rule>400anchor.jsonl` as the
# paper's runs read it.
SHOWN = {
    ("tsmc", "khop"): "677cd9b1c8dcbbade566d1e1523f3ee3fdc7928e9e6553741e466c1b05ebe080",
    ("biograph", "ppr"): "c7903b4e7104f419973e1b4b9ff4693dd53837e56db0a4f83f3fbb4c21c7e524",
    ("pfas", "ppr"): "1a1e4f730add0baac746c358f57f6cd70a41fd1f13c2705a49239d1d21f9dc52",
    ("matprop", "ppr"): "b56d36a164ce24c65c34e0c67aca7664d066628e273d35f50723eae0d05a1b9b",
}


@pytest.mark.slow
@pytest.mark.parametrize("kind, rule", [("tsmc", "khop"), ("biograph", "ppr")])
def test_a_build_is_the_file_the_runs_were_shown(kind, rule, tmp_path):
    graph = ROOT / default_kb(kind)
    if not graph.exists():
        pytest.skip(f"{kind} graph absent from {graph.parent}")
    out = tmp_path / "built.jsonl"
    p = subprocess.run(
        [sys.executable, str(BUILDER), "--graph", kind, "--rule", rule,
         "--budget", "400", "--out", str(out)],
        capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr
    assert hashlib.sha256(out.read_bytes()).hexdigest() == SHOWN[kind, rule]
