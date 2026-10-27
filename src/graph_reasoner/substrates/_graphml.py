"""The shared body of the two LLM-extracted graphml formats: bio-graph-1K's
(`substrates/biograph/`) and GraphAgents' (`substrates/_graphagents.py`, behind
pfas and matprop). Not a substrate itself.

Both formats carry free-text relations, cut here to one length so a citation
can copy them back, and both are loaded as one directed edge per stored edge in
the file's own orientation. `PairIndexedBackend` holds what the backends
share: the pair index that answers a query in either order, the relation
vocabulary, and a fingerprint that names the relation cap.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import networkx as nx

from ._base import SubstrateBase
from ..agent.backend import EntityIndex, SeenEdge
from ..kg.load import file_fingerprint

# Relations longer than this are cut at the previous word boundary. The cap
# is part of the backend fingerprint.
RELATION_CHAR_CAP = 160


def canonical_relation(title: object) -> str:
    """The graphml relation attribute -> the relation string everything
    downstream sees.

    A non-string or empty value becomes `related to`, the default of
    `kg/load.py`. A string is whitespace-collapsed and cut at the last word
    boundary within RELATION_CHAR_CAP.
    """
    if not isinstance(title, str) or not title.strip():
        return "related to"
    t = " ".join(title.split())
    if len(t) > RELATION_CHAR_CAP:
        cut = t[:RELATION_CHAR_CAP]
        t = cut.rsplit(" ", 1)[0] if " " in cut else cut
    return t


class PairIndexedBackend(SubstrateBase):
    """A graph loaded as one directed edge per stored edge, queried in either
    order.

    A subclass sets `name` and `_load = staticmethod(<loader>)`, a path ->
    DiGraph loader that gives every edge a `relation` from
    `canonical_relation` and every node a `type`, and registers itself with
    its own default path. It inherits `__init__`.
    """

    name = "graphml"            # overridden by each registered subclass
    _load: Callable[[str], nx.DiGraph]

    def __init__(self, graphml_path: str) -> None:
        self._path = graphml_path
        self._G = self._load(graphml_path)
        self._index = EntityIndex(self._G.nodes, rank=self._G.degree)
        # Keyed on the stored orientation. `edges_between` reads both orders,
        # so `expand` shows each stored edge once, as the file stores it.
        pairs: dict[tuple[str, str], list[str]] = defaultdict(list)
        for u, v, d in self._G.edges(data=True):
            r = d["relation"]
            if r not in pairs[(u, v)]:
                pairs[(u, v)].append(r)
        self._pairs = {k: tuple(v) for k, v in pairs.items()}
        self._vocab = tuple(dict.fromkeys(
            d["relation"] for _, _, d in self._G.edges(data=True)))

    def node_type(self, node: str) -> str:
        return self._G.nodes[node].get("type", "Unknown") if node in self._G else "Unknown"

    def edges_between(self, u: str, v: str) -> tuple[SeenEdge, ...]:
        out = [SeenEdge(u, r, v) for r in self._pairs.get((u, v), ())]
        out += [SeenEdge(v, r, u) for r in self._pairs.get((v, u), ())]
        return tuple(out)

    def fingerprint(self) -> str:
        # The cap changes the relation strings the model sees, so it is part
        # of the graph's identity.
        return f"{file_fingerprint(self._path)}+r{RELATION_CHAR_CAP}"


__all__ = ["PairIndexedBackend", "RELATION_CHAR_CAP", "canonical_relation"]
