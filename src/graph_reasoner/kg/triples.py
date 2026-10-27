"""A knowledge graph stored as `head|relation|tail` text triples.

The cheapest graph to construct, a few lines in a temporary file, which makes
it the fixture behind the seen-contract and rollout-loop tests.

`load_triples_kb` builds a DiGraph, which collapses parallel relations between
the same pair of nodes: topology is exact, and the full edge labels come from
`load_triples_relations`. Every lookup here keys on the raw string and is
case-sensitive; the case-folded and fuzzy layer is
`agent.backend.EntityIndex`. The test fixtures' expected observations rest on
both behaviours.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

# Relation -> the type of its object, in a movie graph's vocabulary (a subject
# is always typed `movie`, an object of an unlisted relation `attr`). The types
# only label nodes in `find` output; the gate and the metrics do not read them.
OBJECT_TYPE = {
    "directed_by": "person",
    "written_by": "person",
    "starred_actors": "person",
    "has_genre": "genre",
    "release_year": "year",
    "in_language": "language",
    "has_tags": "tag",
    "has_imdb_rating": "rating",
    "has_imdb_votes": "votes",
}


def load_triples_kb(path: str) -> nx.DiGraph:
    """Load `subject|relation|object` lines into a typed DiGraph.

    Every node gets a `type` attribute and every edge a `relation`. Parallel
    relations between the same pair keep the first seen, so the collapse loses
    labels, never topology; `load_triples_relations` returns the full label
    set. The graph must stay a DiGraph: `kg/sample.py` reads a pair's relation
    as `G.edges[u, v]`, which a MultiDiGraph breaks.
    """
    G = nx.DiGraph()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) != 3:
                continue
            s, r, o = (p.strip() for p in parts)
            if not (s and r and o):
                continue
            G.add_node(s, type="movie")
            if o not in G:
                G.add_node(o, type=OBJECT_TYPE.get(r, "attr"))
            if not G.has_edge(s, o):
                G.add_edge(s, o, relation=r)
    # A name used as both subject and object is typed 'movie'. The loop below
    # only fills in a missing type.
    for n, d in G.nodes(data=True):
        d.setdefault("type", "attr")
    return G


def node_types(G: nx.DiGraph) -> dict[str, str]:
    return {n: d.get("type", "attr") for n, d in G.nodes(data=True)}


@dataclass(eq=False)
class RelationIndex:
    """Every relation of every triple, including those `load_triples_kb`
    collapses.

    A label-only supplement to the DiGraph: the pair set is identical, so it
    changes what a pair is called, never reachability. The triples backend
    reads it so that `expand` shows every relation on a pair.
    """

    pairs: dict[tuple[str, str], tuple[str, ...]]

    def relations(self, u: str, v: str) -> tuple[str, ...]:
        """Every relation on the pair (u, v), in file order. Direction-sensitive."""
        return self.pairs.get((u, v), ())


def load_triples_relations(path: str) -> RelationIndex:
    """Every triple in the file as a `RelationIndex`, including the labels the
    DiGraph drops."""
    pairs: dict[tuple[str, str], tuple[str, ...]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) != 3:
                continue
            s, r, o = (p.strip() for p in parts)
            if not (s and r and o):
                continue
            prev = pairs.get((s, o), ())
            if r not in prev:
                pairs[(s, o)] = prev + (r,)
    return RelationIndex(pairs=pairs)

