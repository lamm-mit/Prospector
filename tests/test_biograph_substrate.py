"""The biograph substrate: the loader's three normalisations.

bio-graph-1K has edges whose relation is a boolean or absent, relations that
run to whole sentences, and no node types. The loader normalises all three;
these tests pin that on a synthetic graph that reproduces each defect, so they
run without the release. The release test is skipped unless the file is in
`data/raw/biograph/`.
"""
import pathlib

import networkx as nx
import pytest

from graph_reasoner.agent.backend import load_backend
from graph_reasoner.substrates._graphml import RELATION_CHAR_CAP, canonical_relation
from graph_reasoner.substrates.biograph.loader import cluster_label, load_biograph

ROOT = pathlib.Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data/raw/biograph/large_graph_simple_giant.graphml"


def _defective_graphml(tmp_path: pathlib.Path) -> str:
    """A tiny undirected graph carrying every defect the release has."""
    U = nx.Graph()
    for n, g in [("silk", 3), ("spider", 3), ("collagen", 7), ("bone", 7),
                 ("enamel", None)]:
        if g is None:
            U.add_node(n)
        else:
            U.add_node(n, group=g, color=g, size=1.0)
    long_rel = "provides " + "very " * 60 + "strong anchoring"   # > cap
    U.add_edge("silk", "spider", title="is produced by", chunk_id="aa" * 16)
    U.add_edge("collagen", "bone", title=long_rel, chunk_id="bb" * 16)
    U.add_edge("bone", "enamel", title=True, chunk_id="cc" * 16)  # boolean
    U.add_edge("silk", "collagen", chunk_id="dd" * 16)            # absent
    U.add_edge("bone", "bone", title="self", chunk_id="ee" * 16)  # loop
    p = tmp_path / "toy_bio.graphml"
    nx.write_graphml(U, p)
    return str(p)


def test_canonical_relation_normalises_the_three_shapes():
    assert canonical_relation("binds  to\ncalcium") == "binds to calcium"
    assert canonical_relation(True) == "related to"
    assert canonical_relation(None) == "related to"
    assert canonical_relation("   ") == "related to"
    capped = canonical_relation("word " * 100)
    assert len(capped) <= RELATION_CHAR_CAP
    assert not capped.endswith(" ") and capped.endswith("word")


def test_cluster_label_survives_graphml_type_dance():
    assert cluster_label(28) == "cluster-28"
    assert cluster_label("28") == "cluster-28"
    assert cluster_label(28.0) == "cluster-28"
    assert cluster_label(None) == "cluster-unknown"


def test_loader_normalises_the_defective_release_shapes(tmp_path):
    G = load_biograph(_defective_graphml(tmp_path))
    assert G.is_directed()
    assert G.graph["n_self_loops"] == 1
    assert not any(u == v for u, v in G.edges)
    rels = {(u, v): d["relation"] for u, v, d in G.edges(data=True)}
    # Boolean and absent titles both degrade to the tree-wide default.
    assert all(isinstance(r, str) for r in rels.values())
    assert sum(1 for r in rels.values() if r == "related to") == 2
    # The long relation is capped at a word boundary.
    assert all(len(r) <= RELATION_CHAR_CAP for r in rels.values())
    # Types synthesised from the community id; the untyped node degrades.
    assert G.nodes["silk"]["type"] == "cluster-3"
    assert G.nodes["enamel"]["type"] == "cluster-unknown"


def test_backend_answers_both_edge_orientations(tmp_path):
    backend = load_backend("biograph", _defective_graphml(tmp_path))
    fwd = backend.edges_between("silk", "spider")
    rev = backend.edges_between("spider", "silk")
    # The file stores one orientation; both query orders must surface it, in
    # the stored orientation — that is what the seen contract records.
    assert fwd and fwd == rev
    assert {(e.source, e.relation, e.target) for e in fwd} == \
        {("silk", "is produced by", "spider")}
    assert backend.node_type("collagen") == "cluster-7"
    assert backend.fingerprint().endswith(f"+r{RELATION_CHAR_CAP}")


@pytest.mark.slow
@pytest.mark.skipif(not RELEASE.exists(),
                    reason="bio-graph-1K absent from data/raw/biograph/")
def test_release_measures_as_documented():
    """The real file's node and edge counts and its free-text vocabulary, and
    that classification, gold-path sampling and search work on it."""
    backend = load_backend("biograph", str(RELEASE))
    G = backend.digraph()
    assert G.number_of_nodes() == 33_159
    assert G.number_of_edges() == 48_753 - G.graph["n_self_loops"]
    vocab = backend.relation_vocab()
    assert len(vocab) > 20_000, "free-text vocabulary is the deciding property"
    assert all(len(r) <= RELATION_CHAR_CAP for r in vocab)
    assert all(isinstance(r, str) for r in vocab)
    # The generator's inputs work unchanged: classification and a gold path.
    import random
    from graph_reasoner.kg.load import classify_nodes
    from graph_reasoner.kg.sample import sample_gold_path
    roles = classify_nodes(G)
    # Nodes have no types, but the default name patterns classify a few
    # citation-shaped names as biblio. Pin the magnitude, not the count.
    bib = sum(1 for r in roles.values() if r == "biblio")
    assert 0 < bib < 0.01 * len(roles), f"{bib} biblio of {len(roles)}"
    gold = sample_gold_path(G, roles, 3, random.Random(0))
    assert len(gold.edges) == 3
    # And the search index resolves a real node by fuzzy lookup.
    hits = backend.search("mineralized collagen", k=5)
    assert any("collagen" in n for n, _t, _d in hits)
