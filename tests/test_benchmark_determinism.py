"""The benchmark config's `seed:` reproduces a build across processes.

`sample_gold_path` and `build_subgraph` walk `undirected_neighbors`, which
returns a set, and set iteration order follows string hashes, which Python
randomises per process. `kg/sample.py` sorts the neighbours before they reach
`rng.choice`. The test samples in two subprocesses with different
PYTHONHASHSEED values and requires identical output; repeating the draw inside
one process shares one hash seed and cannot detect the dependence.
"""
import json
import pathlib
import subprocess
import sys

import networkx as nx

SNIPPET = """
import json, random, sys
import networkx as nx
from graph_reasoner.kg.load import classify_nodes
from graph_reasoner.kg.sample import sample_gold_path, build_subgraph

G = nx.read_graphml(sys.argv[1])
roles = classify_nodes(G)
rng = random.Random(7)
out = []
for _ in range(6):
    gold = sample_gold_path(G, roles, 2, rng)
    SG = build_subgraph(G, roles, gold, rng, branch_per_node=2, branch_depth=1,
                        decoy_walks=1, decoy_len=2, biblio_attach=0)
    out.append({"nodes": gold.nodes,
                "sub": sorted(map(str, SG.edges()))})
print(json.dumps(out, sort_keys=True))
"""


def _star_graph(tmp_path: pathlib.Path) -> str:
    """A graph with real branching, so set order would show if it leaked:
    every interior node has several same-role neighbours to choose among."""
    G = nx.DiGraph()
    for i in range(8):
        for j in range(8):
            u, v = f"n{i}", f"m{i}{j}"
            G.add_node(u, type="concept")
            G.add_node(v, type="concept")
            G.add_edge(u, v, relation=f"rel{j}")
        if i:
            G.add_edge(f"n{i-1}", f"n{i}", relation="chain")
    p = tmp_path / "toy.graphml"
    nx.write_graphml(G, p)
    return str(p)


def _sample(graphml: str, hashseed: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", SNIPPET, graphml],
        capture_output=True, text=True, timeout=60,
        env={"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_same_seed_same_build_across_processes(tmp_path):
    graphml = _star_graph(tmp_path)
    a = _sample(graphml, "1")
    b = _sample(graphml, "424242")
    assert json.loads(a), "sampler produced nothing — the fixture is broken"
    assert a == b, (
        "identical RNG seed, different PYTHONHASHSEED, different benchmark — "
        "a set order is feeding rng.choice (kg/sample.py)")
