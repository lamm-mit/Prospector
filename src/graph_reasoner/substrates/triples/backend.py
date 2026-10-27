"""A KG stored as `head|relation|tail` text triples.

The cheapest backend to construct, a few lines in a temp file, so the tests of
the tools and the rollout loop use it as their fixture.
"""
from __future__ import annotations

from .._base import SubstrateBase
from ...agent.backend import EntityIndex, SeenEdge, register_backend
from ...kg.triples import (
    RelationIndex,
    load_triples_kb,
    load_triples_relations,
    node_types,
)


class TextKBBackend(SubstrateBase):
    """The triples DiGraph plus the full relation index.

    Topology comes from `load_triples_kb`, which keeps one relation per node
    pair. Edge labels come from `load_triples_relations`, so `expand` shows
    every relation on a pair that carries more than one.
    """

    name = "triples"

    def __init__(self, kb_path: str) -> None:
        self._path = kb_path
        self._G = load_triples_kb(kb_path)
        self._rel: RelationIndex = load_triples_relations(kb_path)
        self._types = node_types(self._G)
        self._index = EntityIndex(self._G.nodes, rank=self._G.degree)
        self._vocab = tuple(dict.fromkeys(
            r for rels in self._rel.pairs.values() for r in rels
        ))

    def node_type(self, node: str) -> str:
        return self._types.get(node, "attr")

    def edges_between(self, u: str, v: str) -> tuple[SeenEdge, ...]:
        out = [SeenEdge(u, r, v) for r in self._rel.relations(u, v)]
        out += [SeenEdge(v, r, u) for r in self._rel.relations(v, u)]
        return tuple(out)

register_backend("triples", TextKBBackend)
