"""Faithfulness metrics, mechanical (no model involved):

- hallucination rate: fraction of cited edges not in `subgraph_edges`
- path recall: fraction of gold-path edges the model cited

`subgraph_edges` is whatever the caller was allowed to cite from: a provided
subgraph, or the seen set of an agentic episode (agent/metrics.py passes
that). Correctness is not scored here: the headline `gated_acc` is built in
agent/metrics.py.
"""
from __future__ import annotations

from ..kg.serialize import edge_key
from ..schema import CitedGraph, Edge


def mechanical_metrics(
    cited: CitedGraph | None,
    subgraph_edges: list[Edge],
    gold_edges: list[Edge],
) -> dict:
    sub_keys = {edge_key(e.source, e.relation, e.target) for e in subgraph_edges}
    gold_keys = {edge_key(e.source, e.relation, e.target) for e in gold_edges}

    out = {
        "format_ok": cited is not None and len(cited.edges) > 0,
        "n_cited": 0,
        "halluc_strict": None,   # None when nothing was cited
        "path_recall": 0.0,
    }
    if not out["format_ok"]:
        return out

    cited_keys = [edge_key(e.source, e.relation, e.target) for e in cited.edges]
    out["n_cited"] = len(cited_keys)

    n = len(cited_keys)
    out["halluc_strict"] = 1.0 - sum(k in sub_keys for k in cited_keys) / n
    hit = len(gold_keys & set(cited_keys))
    out["path_recall"] = hit / len(gold_keys) if gold_keys else 0.0
    return out
