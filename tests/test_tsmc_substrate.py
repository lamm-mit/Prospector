"""The semiconductor backend, `substrates/tsmc/backend.py`, built through the
registry on a toy GraphML in the release's shape.

The release graph is not in the repository, and without it no other test
constructs this backend. The tests cover its startup, the system prompt, and
a find and an expand through the tool layer.
"""
import random

import networkx as nx
import pytest

from graph_reasoner.agent.backend import load_backend
from graph_reasoner.agent.prompts import build_agent_system
from graph_reasoner.agent.tools import ToolConfig, dispatch
from graph_reasoner.kg.serialize import parse_edge_line


@pytest.fixture
def tsmc(tmp_path):
    """Directed, typed nodes, `relation` on edges with chunk/DOI provenance as
    the release carries it; one edge with no relation, one pair stored in both
    directions, an author node, and an untyped figure reference."""
    G = nx.DiGraph()
    for node, kind in [("Plasma Etching", "process"), ("Silicon Nitride", "material"),
                       ("CF4", "chemical"), ("Etch Rate", "property"),
                       ("A. Author", "author")]:
        G.add_node(node, type=kind)
    G.add_node("Fig. 3")
    for u, rel, v in [("Plasma Etching", "removes", "Silicon Nitride"),
                      ("CF4", "used in", "Plasma Etching"),
                      ("Plasma Etching", "controls", "Etch Rate"),
                      ("Etch Rate", "depends on", "Plasma Etching"),
                      ("A. Author", "authored", "Plasma Etching")]:
        G.add_edge(u, v, relation=rel, chunk_id="ab" * 16, DOI="10.1/x")
    G.add_edge("Fig. 3", "Etch Rate", chunk_id="cd" * 16)          # no relation
    path = tmp_path / "toy_tsmc.graphml"
    nx.write_graphml(G, path)
    return load_backend("tsmc", str(path))


def test_the_registry_builds_the_backend_from_its_file(tsmc):
    assert type(tsmc).__name__ == "TSMCBackend" and tsmc.name == "tsmc"
    assert len(tsmc.fingerprint()) == 16
    # Each relation once; an edge with no relation gets the default,
    # "related to".
    vocab = tsmc.relation_vocab()
    assert len(vocab) == len(set(vocab)) == 6
    assert set(vocab) == {"removes", "used in", "controls", "depends on",
                          "authored", "related to"}
    assert tsmc.node_type("CF4") == "chemical"
    assert tsmc.node_type("Fig. 3") == "biblio"          # untyped: its role
    assert tsmc.degree("Plasma Etching") == 5


def test_edges_between_answers_both_orientations(tsmc):
    one = tsmc.edges_between("Silicon Nitride", "Plasma Etching")
    assert one == tsmc.edges_between("Plasma Etching", "Silicon Nitride")
    assert [(e.source, e.relation, e.target) for e in one] == [
        ("Plasma Etching", "removes", "Silicon Nitride")]
    both = {(e.source, e.relation, e.target)
            for e in tsmc.edges_between("Plasma Etching", "Etch Rate")}
    assert both == {("Plasma Etching", "controls", "Etch Rate"),
                    ("Etch Rate", "depends on", "Plasma Etching")}


def test_the_system_prompt_builds_on_it(tsmc):
    system = build_agent_system(tsmc, ToolConfig())
    assert "removes" in system and "depends on" in system   # the relation list


def test_find_and_expand_return_what_they_render(tsmc):
    found = dispatch(tsmc, ToolConfig(), "find", {"name": "plasma etch"})
    assert found.ok and "Plasma Etching" in found.text and found.edges == ()

    r = dispatch(tsmc, ToolConfig(), "expand", {"node": "Plasma Etching"},
                 rng=random.Random(0))
    returned = {(e.source, e.relation, e.target) for e in r.edges}
    rendered = {tuple(p.strip() for p in parse_edge_line(line.strip()))
                for line in r.text.splitlines() if parse_edge_line(line.strip())}
    assert r.ok and returned == rendered            # the seen contract
    assert ("CF4", "used in", "Plasma Etching") in returned          # incoming
    assert ("Plasma Etching", "removes", "Silicon Nitride") in returned  # outgoing
    assert len(returned) == 5
