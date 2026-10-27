"""The search cut (`graph_reasoner.traj.prefixes`): the prefix ends on the
observation of the first call whose seen set holds a neighbour of a gold node,
and its state (calls, seen edges) is truncated to that call."""
from graph_reasoner.traj.prefixes import search_cut, to_search_prefix


def _edge(a, b):
    return {"source": a, "relation": "rel", "target": b}


def _rec():
    # three calls; "b", a neighbour of the gold node, is first seen at call 1,
    # and "Gold Node" itself enters the seen set at call 2
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<tool_call>1</tool_call>"},
        {"role": "user", "content": "OBSERVATION 1"},
        {"role": "assistant", "content": "<tool_call>2</tool_call>"},
        {"role": "user", "content": "OBSERVATION 2"},
        {"role": "assistant", "content": "<tool_call>3</tool_call>"},
        {"role": "user", "content": "OBSERVATION 3"},
        {"role": "assistant", "content": "ANSWER: rival"},
    ]
    seen = [_edge("a", "b"), _edge("b", "Gold Node"), _edge("Gold Node", "c"), _edge("c", "d")]
    calls = [{"turn": 1, "tool": "expand", "args": {}, "cum_edges_seen": 1, "n_new_gold": 0},
             {"turn": 2, "tool": "expand", "args": {}, "cum_edges_seen": 3, "n_new_gold": 1},
             {"turn": 3, "tool": "expand", "args": {}, "cum_edges_seen": 4, "n_new_gold": 0}]
    return {"id": "hard-0001a", "tier": "hard", "messages": msgs, "seen_edges": seen,
            "calls": calls, "kb_answer_set": ["Gold Node"], "gold_answer": "Gold Node",
            "answer": "rival", "final_text": "ANSWER: rival", "cited_edges": [],
            "n_edges_seen": 4, "status": "answered", "gold_complete_at_call": 2}


def test_search_prefix_truncates_state_and_ends_on_the_observation():
    rec = _rec()
    # "c" is first on the table at call 2, so the cut is before turn 3 and the
    # source spent one more call from there
    cut = search_cut(rec, {"c"})
    assert cut == (3, 1)
    p = to_search_prefix(rec, run_tag="roll-s861", cut_turn=cut[0], hops_short=cut[1],
                         extra={})
    assert p["id"] == "roll-s861#hard-0001a#c3"
    assert p["class"] == "search" and p["search_hops_short"] == 1
    assert p["n_prefix_turns"] == 2
    assert [m["role"] for m in p["messages"]] == ["system", "user", "assistant", "user",
                                                  "assistant", "user"]
    assert p["messages"][-1]["content"] == "OBSERVATION 2"
    assert [c["turn"] for c in p["calls"]] == [1, 2]
    assert len(p["seen_edges"]) == p["calls"][-1]["cum_edges_seen"] == 3
    assert p["n_edges_seen"] == 3
    for k in ("final_text", "answer", "cited_edges"):
        assert k not in p
    assert search_cut(rec, {"absent"}) is None


def test_search_prefix_ids_and_class_are_the_search_record():
    p = to_search_prefix(_rec(), run_tag="r", cut_turn=2, hops_short=2, extra={})
    assert p["id"] == "r#hard-0001a#c2" and p["class"] == "search"
