"""MatProp_KG, the Material Properties Knowledge Graph of GraphAgents
(lamm-mit, Stewart, Hage, Hsu and Buehler, arXiv 2602.07491).

Extracted from 63,222 Web of Science abstracts returned by 51
material-property keywords (that paper's section 4.4 and Table 4). The release
holds 317,800 nodes / 774,009 edges. The format and its normalisation are in
`substrates/_graphagents.py`.
"""
from __future__ import annotations

from .._graphagents import GraphAgentsBackend
from ...agent.backend import register_backend


class MatpropBackend(GraphAgentsBackend):
    """Edges carry a `DOI` and a `chunk_id`; the abstracts themselves are not
    released."""

    name = "matprop"


register_backend("matprop", MatpropBackend, default_kb="data/raw/matprop/MatProp_KG.graphml")
