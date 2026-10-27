"""PFAS_KG, the PFAS-Specific Knowledge Graph of GraphAgents (lamm-mit,
Stewart, Hage, Hsu and Buehler, arXiv 2602.07491).

Extracted from 4,716 full-text papers returned by a Web of Science search for
"Per- and polyfluoroalkyl substances" (section 4.4 of that paper). The release
holds 144,416 nodes / 459,342 edges. The format and its normalisation are in
`substrates/_graphagents.py`.
"""
from __future__ import annotations

from .._graphagents import GraphAgentsBackend
from ...agent.backend import register_backend


class PfasBackend(GraphAgentsBackend):
    """Edges carry a `chunk_id` and a `DOI`; the corpus is publisher full
    text and is not released."""

    name = "pfas"


# Its conversations outgrow the engines' default window of 16,384 tokens; the
# paper's PFAS runs read at 32,768.
register_backend("pfas", PfasBackend, default_kb="data/raw/pfas/PFAS_KG.graphml", max_model_len=32768)
