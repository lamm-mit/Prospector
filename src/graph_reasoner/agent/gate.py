"""The assertion gate: groundedness checked by set membership, with no judge.

An answer that abstains is a voluntary abstention and does not pass. Any other
final answer passes iff all three hold:

    1. it cites at least one edge,
    2. every cited edge was seen (cited ⊆ seen),
    3. the name of some cited node appears in the answer.

Condition 2 makes hallucination among passing answers zero by construction.
Whatever does not pass counts as an abstention (`gated_abstained`).

Everything here is a pure function of strings and edge dicts, so a stored
trajectory record scores exactly as a live episode does.
"""
from __future__ import annotations

from ..eval.parse import detect_abstention
from ..kg.serialize import edge_key, normalize

GATE_OK = "ok"
GATE_VOLUNTARY = "voluntary_abstention"
GATE_NO_CITATION = "no_citation"
GATE_HALLUCINATED = "hallucinated_citation"
GATE_UNSUPPORTED = "answer_not_in_cited"


def _keys(edges) -> set[tuple[str, str, str]]:
    return {edge_key(e["source"], e["relation"], e["target"]) for e in edges}


def _supporting_nodes(answer: str, cited) -> set[str]:
    """Cited nodes whose normalized name appears in the normalized answer, the
    substring rule `score_answer` uses for correctness."""
    na = normalize(answer or "")
    return {n for e in cited for n in (e["source"], e["target"])
            if normalize(n) and normalize(n) in na}


def _answer_supported(answer: str, cited) -> bool:
    return bool(_supporting_nodes(answer, cited))


def edge_dicts(edges) -> list[dict[str, str]]:
    """Edge objects -> the dict shape this module consumes.

    Takes anything with `.source/.relation/.target`: `SeenEdge`
    (agent/backend.py) and the pydantic `Edge` (schema.py). `None` gives an
    empty list, since `extract_graph_json` returns None when there is no block.
    """
    return [{"source": e.source, "relation": e.relation, "target": e.target}
            for e in edges or ()]


def gate_assertion(answer: str, cited_edges, seen_edges) -> dict:
    """Classify one final turn. Returns the verdict (`asserted`, `gate_pass`,
    `gate_reason`); the caller joins correctness."""
    if detect_abstention(answer or ""):
        return {"asserted": False, "gate_pass": False, "gate_reason": GATE_VOLUNTARY}

    cited = list(cited_edges or [])
    if not cited:
        return {"asserted": True, "gate_pass": False, "gate_reason": GATE_NO_CITATION}
    if not _keys(cited) <= _keys(seen_edges or []):
        return {"asserted": True, "gate_pass": False, "gate_reason": GATE_HALLUCINATED}
    if not _answer_supported(answer, cited):
        return {"asserted": True, "gate_pass": False, "gate_reason": GATE_UNSUPPORTED}
    return {"asserted": True, "gate_pass": True, "gate_reason": GATE_OK}


def gate_record(rec: dict) -> dict:
    """Apply the gate to a persisted trajectory record (no Episode needed)."""
    return gate_assertion(rec.get("answer") or "",
                          rec.get("cited_edges") or [],
                          rec.get("seen_edges") or [])


def gated_fields(verdict: dict) -> dict:
    """The per-item gate block: whether the answer asserted, whether it
    passed, the reason, and `gated_abstained`, the gate not passed, by
    abstention or by refusal. Correctness is scored apart (agent/metrics.py),
    and the headline `gated_acc` joins the two.
    """
    gate_pass = bool(verdict["gate_pass"])
    return {
        "asserted": bool(verdict["asserted"]),
        "gate_pass": gate_pass,
        "gate_reason": verdict["gate_reason"],
        "gated_abstained": not gate_pass,
    }


__all__ = ["gate_assertion", "gate_record", "gated_fields",
           "GATE_OK", "GATE_VOLUNTARY", "GATE_NO_CITATION",
           "GATE_HALLUCINATED", "GATE_UNSUPPORTED"]
