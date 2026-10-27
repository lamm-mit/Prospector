"""The TSMC semiconductor-manufacturing KG: 10,056 nodes / 24,324 edges in
`data/raw/GRAPHDATA_TSMC_OUTPUT/tsmc_5b10p.graphml`.
"""
from __future__ import annotations

from .._base import SubstrateBase
from ...agent.backend import EntityIndex, SeenEdge, register_backend
from ...kg.load import classify_nodes, load_kg


class TSMCBackend(SubstrateBase):
    """The TSMC GraphML KG. Edges carry `chunk_id`/`DOI` provenance; the
    passages they were extracted from are publisher text and are not part of
    the release."""

    name = "tsmc"

    def __init__(self, graphml_path: str) -> None:
        self._path = graphml_path
        self._G = load_kg(graphml_path)
        self._roles = classify_nodes(self._G)
        self._index = EntityIndex(self._G.nodes, rank=self._G.degree)
        self._vocab = tuple(dict.fromkeys(
            str(d.get("relation", "related to")) for _, _, d in self._G.edges(data=True)
        ))

    def node_type(self, node: str) -> str:
        if node not in self._G:
            return "Unknown"            # as the other graphml backends spell it
        return str(self._G.nodes[node].get("type") or self._roles.get(node, "domain"))

    def edges_between(self, u: str, v: str) -> tuple[SeenEdge, ...]:
        out = []
        for a, b in ((u, v), (v, u)):
            if self._G.has_edge(a, b):
                out.append(SeenEdge(a, str(self._G.edges[a, b].get("relation", "related to")), b))
        return tuple(out)


register_backend("tsmc", TSMCBackend,
                 default_kb="data/raw/GRAPHDATA_TSMC_OUTPUT/tsmc_5b10p.graphml",
                 train_benchmark="data/benchmarks/tsmc/tsmc_train_agentic_4x.jsonl")
