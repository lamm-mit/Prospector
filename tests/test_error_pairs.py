"""Detour-recovery rows: real observations, a real recovery, no label leak.

Three properties of an error-correction row (see `traj/error_pairs.py`): the
wrong node is never gold or an answer, the injected observation is what the
backend returns, rendered by the loop's own renderer, and the completion is
the episode's own next call.
"""
import json
import pathlib

import networkx as nx

from graph_reasoner.agent.backend import load_backend
from graph_reasoner.agent.loop import parse_tool_call
from graph_reasoner.agent.tools import ToolConfig
from graph_reasoner.kg.serialize import render_edge_line
from graph_reasoner.traj.error_pairs import RECOGNITIONS, build_error_pair_rows


def _toy_backend(tmp_path: pathlib.Path):
    U = nx.Graph()
    for a, b, rel in [
        ("anchor", "mid", "feeds"),
        ("mid", "answer", "controls"),
        ("anchor", "stray", "borders"),      # the plausible wrong turn
        ("stray", "far", "touches"),
        ("mid", "other", "mentions"),
    ]:
        U.add_node(a, group=1)
        U.add_node(b, group=1)
        U.add_edge(a, b, title=rel, chunk_id="ab" * 16)
    p = tmp_path / "toy.graphml"
    nx.write_graphml(U, p)
    return load_backend("biograph", str(p))


def _obs(text: str, call_no: int, max_turns: int = 20) -> dict:
    return {"role": "user",
            "content": f"OBSERVATION (call {call_no}/{max_turns}, expand):\n"
                       f"{text}\n[2 new edges; 2 edges seen; "
                       f"{max_turns - call_no} tool calls left]"}


def _keeps_rows() -> list[dict]:
    """Two per-turn rows of one successful episode, in the stored format."""
    call1 = ('<think>Expand the anchor first.</think>\n'
             '<tool_call>{"tool": "expand", "args": {"node": "anchor"}}</tool_call>')
    obs1 = _obs(render_edge_line("anchor", "feeds", "mid") + "\n"
                + render_edge_line("anchor", "borders", "stray"), 1)
    call2 = ('<think>mid looks on-path; expand it.</think>\n'
             '<tool_call>{"tool": "expand", "args": {"node": "mid"}}</tool_call>')
    base = [{"role": "system", "content": "tools..."},
            {"role": "user", "content": "Question: what does anchor control?"}]
    return [
        {"schema": "agent_sft_v1", "id": "easy-0001#toy#t1", "item_id": "easy-0001",
         "tier": "easy", "turn": 1, "prompt": base,
         "completion": [{"role": "assistant", "content": call1}], "source": "rft"},
        {"schema": "agent_sft_v1", "id": "easy-0001#toy#t2", "item_id": "easy-0001",
         "tier": "easy", "turn": 2,
         "prompt": base + [{"role": "assistant", "content": call1}, obs1],
         "completion": [{"role": "assistant", "content": call2}], "source": "rft"},
    ]


def test_builds_one_recovery_row_with_real_observation(tmp_path):
    backend = _toy_backend(tmp_path)
    gold = {"easy-0001": {"anchor", "mid", "answer"}}
    rows, tally = build_error_pair_rows(
        _keeps_rows(), backend, ToolConfig(), gold, seed=0)
    assert tally.get("built") == 1, tally
    (row,) = rows

    # The detour expands the only legal candidate: not gold, not an answer,
    # not already expanded — `stray`.
    wrong_turn = row["prompt"][-2]
    assert wrong_turn["role"] == "assistant"
    tool, args = parse_tool_call(wrong_turn["content"])
    assert (tool, args["node"]) == ("expand", "stray")

    # The observation is the backend's own output in the loop's rendering: it
    # shows stray's second edge, which no stored message mentions.
    obs = row["prompt"][-1]
    assert obs["role"] == "user"
    assert "OBSERVATION (call 2/20, expand):" in obs["content"]
    assert render_edge_line("stray", "touches", "far") in obs["content"]

    # The completion is the episode's own next call with a recognition
    # spliced into the think, never a new tool choice.
    comp = row["completion"][0]["content"]
    assert comp.startswith("<think>")
    assert any(r in comp for r in RECOGNITIONS)
    assert parse_tool_call(comp) == ("expand", {"node": "mid"})

    # Bookkeeping: same item, distinct id, provenance labelled.
    assert row["item_id"] == "easy-0001"
    assert row["id"].endswith("#ec")
    assert row["source"] == "error_pair"


def test_refuses_when_every_candidate_is_gold(tmp_path):
    backend = _toy_backend(tmp_path)
    gold = {"easy-0001": {"anchor", "mid", "answer", "stray", "far", "other"}}
    rows, tally = build_error_pair_rows(
        _keeps_rows(), backend, ToolConfig(), gold, seed=0)
    assert rows == []
    assert tally.get("no_viable_detour") == 1


def test_determinism_across_calls(tmp_path):
    backend = _toy_backend(tmp_path)
    gold = {"easy-0001": {"anchor", "mid", "answer"}}
    a, _ = build_error_pair_rows(_keeps_rows(), backend, ToolConfig(), gold, seed=3)
    b, _ = build_error_pair_rows(_keeps_rows(), backend, ToolConfig(), gold, seed=3)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
