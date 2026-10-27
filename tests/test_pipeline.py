"""The benchmark-construction pipeline: `kg/` (classify, sample, subgraph,
serialize) and the answer parsers it is scored through, `eval/parse.py` and
`eval/metrics.py`.

The gold path must be the unique simple path through the sampled subgraph, so
an answer set is mechanically computable. Everything runs on a toy graph;
`test_real_kg_sampling` re-checks uniqueness on the real graph and skips when
the file is absent.
"""
import json
import os
import random
from pathlib import Path

import networkx as nx
import pytest

from conftest import ROOT

from graph_reasoner.eval.metrics import mechanical_metrics
from graph_reasoner.eval.parse import extract_final_answer, extract_graph_json
from graph_reasoner.kg import build_subgraph, classify_nodes, sample_gold_path, serialize_subgraph
from graph_reasoner.schema import CitedGraph, Edge

#: The graph the gold-path uniqueness invariant is checked against; set
#: `GRAPH_REASONER_KG` to check a different one.
REAL_KG = Path(os.environ.get(
    "GRAPH_REASONER_KG", ROOT / "data/raw/GRAPHDATA_TSMC_OUTPUT/tsmc_5b10p.graphml"))


@pytest.fixture
def toy_graph():
    """Chain a->b->c->d with side noise, a hub, and a biblio node."""
    G = nx.DiGraph()
    chain = [("a", "causes", "b"), ("b", "increases", "c"), ("c", "controls", "d")]
    for u, r, v in chain:
        G.add_edge(u, v, relation=r)
    # side noise off every chain node
    for n in "abcd":
        for i in range(3):
            G.add_edge(n, f"{n}_side{i}", relation="involves")
    # a biblio author node and a related-to edge (blocked from gold paths)
    G.add_node("Dr. Smith", type="author")
    G.add_edge("Dr. Smith", "a", relation="studied")
    G.add_edge("a", "weakly", relation="related to")
    for n, d in G.nodes(data=True):
        d.setdefault("type", "concept")
    return G


def test_classify_nodes(toy_graph):
    roles = classify_nodes(toy_graph)
    assert roles["Dr. Smith"] == "biblio"
    assert roles["a"] == "domain"


def test_gold_path_sampling(toy_graph):
    roles = classify_nodes(toy_graph)
    rng = random.Random(0)
    for _ in range(10):
        gold = sample_gold_path(
            toy_graph, roles, 2, rng,
            relation_blocklist={"related to"}, max_degree=50,
        )
        assert len(gold.edges) == 2
        assert len(set(gold.nodes)) == 3  # self-avoiding
        for e in gold.edges:
            assert e.relation != "related to"
            assert roles[e.source] == "domain" and roles[e.target] == "domain"


def test_subgraph_unique_path(toy_graph):
    """The gold path is the only simple path between its endpoints in the
    sampled subgraph."""
    roles = classify_nodes(toy_graph)
    for seed in range(20):
        rng = random.Random(seed)
        gold = sample_gold_path(toy_graph, roles, 2, rng,
                                relation_blocklist={"related to"}, max_degree=50)
        SG = build_subgraph(toy_graph, roles, gold, rng,
                            branch_per_node=2, branch_depth=2,
                            decoy_walks=2, decoy_len=2, biblio_attach=1)
        a, b = gold.endpoints
        paths = list(nx.all_simple_paths(SG.to_undirected(), a, b))
        assert len(paths) == 1, f"seed {seed}: expected unique path, got {len(paths)}"
        assert paths[0] == gold.nodes


def test_serialization_roundtrip(toy_graph):
    roles = classify_nodes(toy_graph)
    rng = random.Random(1)
    gold = sample_gold_path(toy_graph, roles, 2, rng,
                            relation_blocklist={"related to"}, max_degree=50)
    SG = build_subgraph(toy_graph, roles, gold, rng)
    text, edges = serialize_subgraph(SG, rng)
    assert len(text.splitlines()) == len(edges) == SG.number_of_edges()
    for e in edges:
        assert f"{e['source']} --[{e['relation']}]--> {e['target']}" in text


def test_parse_graph_json():
    gj = {"nodes": ["a", {"id": "b"}], "edges": [{"source": "a", "relation": "causes", "target": "b"}]}
    text = f"<think>reasoning <graph_json>{json.dumps(gj)}</graph_json></think>\nANSWER: b happens."
    cited = extract_graph_json(text)
    assert cited is not None
    assert cited.nodes == ["a", "b"]
    assert cited.edges[0].relation == "causes"
    assert extract_final_answer(text) == "b happens."


def test_parse_no_answer_marker():
    text = "<think>hmm</think>The result is X."
    assert extract_final_answer(text) == "The result is X."
    assert extract_graph_json(text) is None


def test_mechanical_metrics():
    sub = [Edge(source="a", relation="causes", target="b"),
           Edge(source="b", relation="increases", target="c"),
           Edge(source="x", relation="involves", target="y")]
    gold = sub[:2]
    # cites both gold edges + one fabricated edge
    cited = CitedGraph(edges=[
        {"source": "A", "relation": "Causes", "target": "b"},   # case-insensitive match
        {"source": "b", "relation": "increases", "target": "c"},
        {"source": "b", "relation": "eliminates", "target": "z"},  # hallucinated
    ])
    m = mechanical_metrics(cited, sub, gold)
    assert m["format_ok"]
    assert m["n_cited"] == 3
    assert abs(m["halluc_strict"] - 1 / 3) < 1e-9
    assert m["path_recall"] == 1.0
    # a reversed edge is not the edge that was shown: direction is part of it
    rev = CitedGraph(edges=[{"source": "b", "relation": "causes", "target": "a"}])
    m2 = mechanical_metrics(rev, sub, gold)
    assert m2["halluc_strict"] == 1.0


def test_no_citation():
    m = mechanical_metrics(None, [Edge(source="a", relation="r", target="b")], [])
    assert not m["format_ok"]
    assert m["halluc_strict"] is None
    assert m["path_recall"] == 0.0


@pytest.mark.slow
@pytest.mark.skipif(not REAL_KG.exists(), reason="the semiconductor graph is not in data/raw/")
def test_real_kg_sampling():
    from graph_reasoner.kg import load_kg
    from graph_reasoner.kg.load import DEFAULT_GOLD_RELATION_BLOCKLIST

    G = load_kg(str(REAL_KG))
    roles = classify_nodes(G)
    rng = random.Random(42)
    for length in (2, 3, 4):
        gold = sample_gold_path(G, roles, length, rng,
                                relation_blocklist=DEFAULT_GOLD_RELATION_BLOCKLIST)
        SG = build_subgraph(G, roles, gold, rng,
                            branch_per_node=3, branch_depth=3,
                            decoy_walks=4, decoy_len=4, biblio_attach=4)
        a, b = gold.endpoints
        paths = list(nx.all_simple_paths(SG.to_undirected(), a, b, cutoff=12))
        assert len(paths) == 1 and paths[0] == gold.nodes


def test_the_sampler_never_routes_a_gold_path_through_a_hub():
    """No node above `max_degree` enters a gold path, so every gold node stays
    expandable without a relation filter (`bench.agentic.hub_violations`
    checks this on a written benchmark). Here the only length-2 route runs
    through a hub, so the sampler must fail rather than take it."""
    import random as _random

    G = nx.DiGraph()
    for i in range(3):
        G.add_edge(f"s{i}", "HUB", relation="feeds")
        G.add_edge("HUB", f"t{i}", relation="drives")
        G.add_edge(f"s{i}", f"s{i}x", relation="involves")   # start degree >= 2
    for i in range(3):                                        # HUB degree = 9 > 5
        G.add_edge("HUB", f"pad{i}", relation="pads")
    for _, d in G.nodes(data=True):
        d.setdefault("type", "concept")
    roles = classify_nodes(G)

    with pytest.raises(RuntimeError):
        sample_gold_path(G, roles, 2, _random.Random(0), max_degree=5,
                         max_tries=50)
