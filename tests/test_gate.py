"""The assertion gate, `agent/gate.py`, on synthetic records: no backend, no model.

An assertion stands only if it cites at least one edge, every cited edge was
seen (`cited ⊆ seen`, by set membership) and the answer names a node of a
cited edge. These tests cover the three conditions, and that there is no
fourth.
"""
from graph_reasoner.agent.gate import (
    GATE_HALLUCINATED,
    GATE_NO_CITATION,
    GATE_OK,
    GATE_UNSUPPORTED,
    GATE_VOLUNTARY,
    gate_assertion,
    gate_record,
    gated_fields,
)

E1 = {"source": "The Giant Spider Invasion", "relation": "directed_by", "target": "Bill Rebane"}
E2 = {"source": "The Giant Spider Invasion", "relation": "has_genre", "target": "Horror"}
FAKE = {"source": "The Giant Spider Invasion", "relation": "has_genre", "target": "Comedy"}

SEEN = [E1, E2]


def test_clean_assertion_stands():
    v = gate_assertion("Horror", [E1, E2], SEEN)
    assert v == {"asserted": True, "gate_pass": True, "gate_reason": GATE_OK}


def test_fabricated_citation_is_converted():
    """A cited edge that was never seen fails the gate."""
    v = gate_assertion("Comedy", [E1, FAKE], SEEN)
    assert v["gate_reason"] == GATE_HALLUCINATED
    assert v["asserted"] and not v["gate_pass"]


def test_parametric_answer_is_converted_even_when_correct():
    """A correct answer that no cited edge mentions fails the gate and
    counts as an abstention."""
    v = gate_assertion("Horror", [E1], SEEN)          # cites only the director hop
    assert v["gate_reason"] == GATE_UNSUPPORTED
    f = gated_fields(v)
    assert f["asserted"] and f["gated_abstained"] and not f["gate_pass"]


def test_no_citation_is_converted():
    v = gate_assertion("Horror", [], SEEN)
    assert v["gate_reason"] == GATE_NO_CITATION


def test_voluntary_abstention_passes_through():
    v = gate_assertion("insufficient information in the graph", [], SEEN)
    assert v["gate_reason"] == GATE_VOLUNTARY
    assert not v["asserted"]
    f = gated_fields(v)
    assert f["gated_abstained"] and not f["asserted"]


def test_gate_record_reads_persisted_trajectories():
    rec = {"answer": "Horror", "cited_edges": [E1, E2], "seen_edges": SEEN}
    assert gate_record(rec)["gate_pass"]
    rec = {"answer": "Comedy", "cited_edges": [FAKE], "seen_edges": SEEN}
    assert gate_record(rec)["gate_reason"] == GATE_HALLUCINATED


def test_case_differences_do_not_defeat_membership():
    """The answer line is free text, so 'horror' still counts as supported by
    the node 'Horror'."""
    v = gate_assertion("horror", [E1, E2], SEEN)
    assert v["gate_pass"]


def test_the_gate_is_three_conditions_and_no_more():
    """Two seen edges that never touch each other still stand when one names
    the answer: the gate does not check that the cited chain is connected."""
    loose = {"source": "Bill Rebane", "relation": "directed", "target": "The Alpha Incident"}
    v = gate_assertion("Horror", [loose, E2], [loose, E2])
    assert v == {"asserted": True, "gate_pass": True, "gate_reason": GATE_OK}
