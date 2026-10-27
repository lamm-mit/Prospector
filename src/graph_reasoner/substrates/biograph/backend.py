"""bio-graph-1K, the biomaterials KG from SciAgents (lamm-mit, Apache-2.0).

The format and its normalisation are in `loader.py`. The pair index, the
relation vocabulary and the capped fingerprint are `PairIndexedBackend`'s
(`substrates/_graphml.py`), shared with the GraphAgents graphs.
"""
from __future__ import annotations

from .._graphml import PairIndexedBackend
from ...agent.backend import register_backend
from .loader import load_biograph


class BiographBackend(PairIndexedBackend):
    """33,159 nodes / 48,753 edges as released, with free-text relations.
    Edges carry a `chunk_id`; the chunk corpus is not published."""

    name = "biograph"
    _load = staticmethod(load_biograph)


register_backend("biograph", BiographBackend,
                 default_kb="data/raw/biograph/large_graph_simple_giant.graphml")
