"""The GraphAgents graphml format: one loader and one backend for the two
graphs released with that paper (lamm-mit/GraphAgents, Stewart et al. 2026),
`PFAS_KG.graphml` behind `substrates/pfas/` and `MatProp_KG.graphml` behind
`substrates/matprop/`. Not a substrate itself. The backend body is
`PairIndexedBackend` (`_graphml.py`), shared with biograph.

Both files are directed graphml. Nodes carry `type` (free text, sometimes
absent), `group`, `color` and `size`; edges carry `relation`, `chunk_id` and
`DOI`. The loader normalises four things:

1. Self-loops are dropped and counted in `G.graph["n_self_loops"]`.
2. Relations go through `canonical_relation` (`_graphml.py`):
   whitespace-collapsed and cut at a word boundary under RELATION_CHAR_CAP.
3. An absent or empty node `type` becomes `Unknown`; any other is kept,
   whitespace-collapsed.
4. A reciprocal pair, u->v and v->u both stored, stays two directed edges.
   `edges_between` answers both query orders, so `expand` shows each stored
   edge once.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx

from ._graphml import PairIndexedBackend, canonical_relation


def canonical_type(value: object) -> str:
    """The graphml `type` -> the node type `find` prints. Empty or absent
    becomes `Unknown`."""
    if not isinstance(value, str) or not value.strip():
        return "Unknown"
    return " ".join(value.split())


def load_graphagents(path: str | Path) -> nx.DiGraph:
    """A GraphAgents graphml -> a DiGraph in the tree's edge conventions.

    One directed edge per stored edge, `relation` from `canonical_relation`,
    node `type` from `canonical_type`; `chunk_id` and `doi` are copied onto
    each edge.
    """
    R = nx.read_graphml(path)
    G = nx.DiGraph()
    for n, d in R.nodes(data=True):
        G.add_node(n, type=canonical_type(d.get("type")))
    loops = 0
    for u, v, d in R.edges(data=True):
        if u == v:
            loops += 1
            continue
        G.add_edge(u, v,
                   relation=canonical_relation(d.get("relation")),
                   chunk_id=str(d.get("chunk_id", "")),
                   doi=str(d.get("DOI", "")))
    G.graph["n_self_loops"] = loops
    return G


class GraphAgentsBackend(PairIndexedBackend):
    """The backend both GraphAgents substrates share.

    A subclass sets `name` and registers itself with its own default path. It
    inherits `__init__` and the loader.
    """

    name = "graphagents"        # overridden by each registered subclass
    _load = staticmethod(load_graphagents)


__all__ = ["GraphAgentsBackend", "load_graphagents", "canonical_type"]
