"""The miss-bucket rule, `graph_reasoner.eval.misses`, which the paper's
miss-bucket figure reads through scripts/analysis/diagnose_miss_buckets.py.

`no_citation` is not the gate bucket, and reachedness is recomputed from
`seen_edges` against the current answer set, not read from stored flags.
`classify` is pure, so the fixtures are synthetic rows.
"""
from graph_reasoner.eval import misses as dmb


def _traj(*nodes):
    edges = [{"source": a, "relation": "r", "target": b}
             for a, b in zip(nodes, nodes[1:])]
    return {"seen_edges": edges}


def _res(answer, gate_pass=True, gate_reason="ok", status="answered",
         gold_complete_at_call=None):
    return {"answer": answer, "gate_pass": gate_pass, "gate_reason": gate_reason,
            "status": status, "gold_complete_at_call": gold_complete_at_call}


def test_hit_is_recomputed_not_read_from_stored_flags():
    # The row carries no em fields: a hit comes from the claim against the
    # current answer set.
    r = dmb.classify(_res("so the answer is Cu"), _traj("anchor", "Cu"),
                     ["copper", "Cu"])
    assert r["hit"] is True and r["bucket"] is None


def test_gate_bucket_is_the_two_honesty_reasons_only():
    r = dmb.classify(
        _res("the answer is copper", gate_pass=False,
             gate_reason="hallucinated_citation"),
        _traj("anchor", "copper"), ["copper"])
    assert r["bucket"] == "gate"


def test_no_citation_is_not_gate_it_falls_through_by_reachedness():
    seen_it = dmb.classify(
        _res("the answer is iron", gate_pass=False, gate_reason="no_citation"),
        _traj("anchor", "copper"), ["copper"])
    assert seen_it["bucket"] == "rival"
    never_saw = dmb.classify(
        _res("the answer is iron", gate_pass=False, gate_reason="no_citation"),
        _traj("anchor", "iron"), ["copper"])
    assert never_saw["bucket"] == "search"


def test_search_vs_rival_split_on_recomputed_reachedness():
    r = dmb.classify(_res("the answer is iron", status="budget"),
                     _traj("anchor", "tungsten"), ["copper"])
    assert r["bucket"] == "search" and r["at_cap"] is True
    r = dmb.classify(
        _res("the answer is iron", gold_complete_at_call=5),
        _traj("anchor", "copper", "iron"), ["copper"])
    assert r["bucket"] == "rival" and r["ev_complete"] is True


def test_stable_misses_keep_a_strict_majority_with_its_majority_bucket():
    """Three reads: an item missed twice is stable, under the bucket it
    missed in most; an item missed once is noise; an item one read never
    scored is out (only the items every read scored are compared)."""
    hit = {"hit": True, "bucket": None}
    search = {"hit": False, "bucket": "search"}
    rival = {"hit": False, "bucket": "rival"}
    per_run = {
        "s0": {"a": search, "b": rival, "c": hit, "d": search},
        "s1": {"a": search, "b": hit, "c": rival, "d": search},
        "s2": {"a": rival, "b": hit, "c": rival},
    }
    assert dmb.stable_misses(per_run) == {"a": "search", "c": "rival"}
