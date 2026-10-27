"""The two GraphAgents substrates (pfas, matprop) share one loader. Its
normalisation is tested on a synthetic graph with every shape the releases
have: an empty `type`, a self-loop, a relation over the cap, and a reciprocal
pair stored in both orientations. The release tests need the graph files
under `data/raw/<kg>/` and check each graph's size as read.
"""
import pathlib
import random

import networkx as nx
import pytest

from graph_reasoner.agent.backend import available_backends, default_kb, load_backend
from graph_reasoner.substrates._graphagents import canonical_type, load_graphagents
from graph_reasoner.substrates._graphml import RELATION_CHAR_CAP

ROOT = pathlib.Path(__file__).resolve().parents[1]
RELEASES = {
    # substrate -> (nodes as read, edges as read, self-loops, min distinct relations)
    "pfas": (144_416, 459_342, 453, 50_000),
    "matprop": (317_800, 774_009, 1_279, 100_000),
}


def _release_graphml(tmp_path: pathlib.Path) -> str:
    """A tiny directed graph in the GraphAgents attribute layout."""
    D = nx.DiGraph()
    D.add_node("PFOA", type="chemical", group=1, color="#000", size=3)
    D.add_node("drinking water", type="location", group=2, color="#000", size=2)
    D.add_node("liver", type="", group=3, color="#000", size=1)        # empty type
    D.add_node("GenX")                                                   # no attrs at all
    long_rel = "is detected at elevated concentrations in " + "very " * 40 + "many samples"
    D.add_edge("PFOA", "drinking water", relation="detected in", chunk_id="aa" * 16, DOI="10.1/x")
    D.add_edge("drinking water", "PFOA", relation="contains", chunk_id="ab" * 16, DOI="10.1/x")
    D.add_edge("PFOA", "liver", relation=long_rel, chunk_id="bb" * 16, DOI="10.1/y")
    D.add_edge("liver", "liver", relation="self", chunk_id="cc" * 16, DOI="10.1/y")
    D.add_edge("GenX", "PFOA", relation="  replaces \n ", chunk_id="dd" * 16, DOI="10.1/z")
    p = tmp_path / "toy_ga.graphml"
    nx.write_graphml(D, p)
    return str(p)


def test_canonical_type_degrades_empty_and_absent():
    assert canonical_type("chemical") == "chemical"
    assert canonical_type(" biological \n process ") == "biological process"
    assert canonical_type("") == "Unknown"
    assert canonical_type(None) == "Unknown"


def test_loader_normalises_the_release_shapes(tmp_path):
    G = load_graphagents(_release_graphml(tmp_path))
    assert G.is_directed()
    assert G.graph["n_self_loops"] == 1
    assert not any(u == v for u, v in G.edges)
    rels = {(u, v): d["relation"] for u, v, d in G.edges(data=True)}
    assert all(len(r) <= RELATION_CHAR_CAP for r in rels.values())
    assert rels[("GenX", "PFOA")] == "replaces"                  # whitespace collapsed
    # A reciprocal pair stays two directed edges in the file's orientation.
    assert rels[("PFOA", "drinking water")] == "detected in"
    assert rels[("drinking water", "PFOA")] == "contains"
    assert G.nodes["PFOA"]["type"] == "chemical"
    assert G.nodes["liver"]["type"] == "Unknown"
    assert G.nodes["GenX"]["type"] == "Unknown"
    assert G.edges["PFOA", "liver"]["doi"] == "10.1/y"


@pytest.mark.parametrize("kind", ["pfas", "matprop"])
def test_backends_register_and_answer_both_orientations(kind, tmp_path):
    assert kind in available_backends()
    assert default_kb(kind) == f"data/raw/{kind}/{'PFAS_KG' if kind == 'pfas' else 'MatProp_KG'}.graphml"
    backend = load_backend(kind, _release_graphml(tmp_path))
    assert backend.name == kind
    assert backend.fingerprint().endswith(f"+r{RELATION_CHAR_CAP}")
    # One stored orientation: both query orders surface it, as stored.
    fwd = backend.edges_between("GenX", "PFOA")
    assert fwd == backend.edges_between("PFOA", "GenX")
    assert {(e.source, e.relation, e.target) for e in fwd} == {("GenX", "replaces", "PFOA")}
    # A reciprocal pair: both stored edges, each once.
    both = backend.edges_between("PFOA", "drinking water")
    assert {(e.source, e.relation, e.target) for e in both} == {
        ("PFOA", "detected in", "drinking water"), ("drinking water", "contains", "PFOA")}
    assert backend.node_type("liver") == "Unknown"
    assert backend.node_type("drinking water") == "location"
    assert backend.node_type("not a node") == "Unknown"


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["pfas", "matprop"])
def test_release_measures_as_documented(kind):
    """The released file's node, edge and self-loop counts, its relation
    vocabulary, its biblio share and a sampled gold path."""
    path = ROOT / default_kb(kind)
    if not path.exists():
        pytest.skip(f"{kind} release absent from {path.parent}")
    nodes, edges, loops, min_rels = RELEASES[kind]
    backend = load_backend(kind, str(path))
    G = backend.digraph()
    assert G.number_of_nodes() == nodes
    assert G.graph["n_self_loops"] == loops
    assert G.number_of_edges() == edges - loops
    vocab = backend.relation_vocab()
    assert len(vocab) > min_rels, "free-text vocabulary is the deciding property"
    assert all(isinstance(r, str) and len(r) <= RELATION_CHAR_CAP for r in vocab)
    assert all(G.nodes[n]["type"] for n in G.nodes), "every node types, if only Unknown"
    from graph_reasoner.kg.load import classify_nodes
    from graph_reasoner.kg.sample import sample_gold_path
    roles = classify_nodes(G)
    # The default biblio type vocabulary (author, reference, location,
    # organization, image, publication) occurs here as real node types, on a
    # small share of the nodes.
    bib = sum(1 for r in roles.values() if r == "biblio")
    assert 0.005 * len(roles) < bib < 0.15 * len(roles), f"{bib} biblio of {len(roles)}"
    gold = sample_gold_path(G, roles, 3, random.Random(0), forbid_endpoint_in_relation=True)
    assert len(gold.edges) == 3
    assert all(G.degree(n) <= 150 for n in gold.nodes), "hubs stay off gold paths"


def test_a_substrate_declares_the_window_its_conversations_need():
    """PFAS conversations outgrow the engines' default of 16,384 tokens, so
    its substrate declares 32,768 and every local engine built for it uses
    that unless a caller names another; the other graphs keep the default."""
    from graph_reasoner.agent.backend import context_window
    assert context_window("pfas") == 32768
    assert context_window("tsmc") is None and context_window("matprop") is None and context_window(None) is None
