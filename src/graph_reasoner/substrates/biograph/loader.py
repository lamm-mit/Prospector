"""bio-graph-1K loader.

The release is lamm-mit/bio-graph-1K's `large_graph_simple_giant.graphml`
(Hugging Face, Apache-2.0): the giant component of a knowledge graph
LLM-extracted from about 1,000 bio-inspired-materials papers. The loader
normalises three things:

1. The relation is the graphml `title`. Where it is absent, empty or not a
   string it becomes `related to`.
2. Relations are free text, some of them whole sentences. They are
   whitespace-collapsed and cut at the last word boundary under
   `RELATION_CHAR_CAP` (`canonical_relation`, `substrates/_graphml.py`).
3. Nodes carry no type, only `group` (a community id), `color` and `size`.
   The node type is synthesised from the community id as `cluster-<id>`.

The file is undirected. Each edge is loaded once, in the orientation the file
stores; `PairIndexedBackend.edges_between` answers both query orders, so
`expand` never shows the same edge twice. `chunk_id` is copied onto each
loaded edge and nothing reads it: the chunk corpus is not published.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx

from .._graphml import canonical_relation


def cluster_label(group: object) -> str:
    """`group` community id -> the synthesised node type, e.g. `cluster-28`.

    A group that arrives as a float (`28.0`) reads as its integer. An absent
    group gives `cluster-unknown`.
    """
    if group is None:
        return "cluster-unknown"
    try:
        return f"cluster-{int(float(group))}"
    except (TypeError, ValueError):
        return f"cluster-{group}"


def load_biograph(path: str | Path) -> nx.DiGraph:
    """The graphml release -> a DiGraph in the tree's edge conventions.

    One directed edge per stored edge (file orientation), `relation` from
    `canonical_relation`, node `type` from `cluster_label`. Self-loops are
    dropped and counted in `G.graph["n_self_loops"]`.
    """
    U = nx.read_graphml(path)
    G = nx.DiGraph()
    for n, d in U.nodes(data=True):
        G.add_node(n, type=cluster_label(d.get("group")))
    loops = 0
    for u, v, d in U.edges(data=True):
        if u == v:
            loops += 1
            continue
        G.add_edge(u, v,
                   relation=canonical_relation(d.get("title")),
                   chunk_id=str(d.get("chunk_id", "")))
    G.graph["n_self_loops"] = loops
    return G


__all__ = ["load_biograph", "cluster_label"]
