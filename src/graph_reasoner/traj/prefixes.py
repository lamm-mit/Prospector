"""Stored rollouts -> decision states: episodes cut before a turn worth training.

This module writes `agent_prefix_v1` records, `train/prefixes.py` reads them
and `scripts/traj/build_prefixes.py` is the CLI. A record is the trajectory
record with `messages` cut before an assistant turn, the seen set and call log
truncated to that state, and a class label saying why the state is here.

An episode's final turn is a decision worth training only where a correct
final answer was available from that state. The gate credits a claim only for
a node the policy has seen, so the criterion is mechanical (`classify`):

  fixable   the episode failed (gated_acc false) and a gold node is among the
            seen nodes. Carries the source's gate reason and status.
  success   the episode passed; the CLI keeps a share, so the resumed policy
            is also trained to keep a right answer.
  dropped   failed with no gold node seen: nothing at the final turn could
            have been right.
  search    a dropped episode that at some call had already seen a graph
            neighbour of a gold node (`search_cut`). The prefix is cut before
            the assistant turn that followed the first such observation, so
            the continuation's first decision is which node to expand next.
            `search_hops_short` records how many calls the source spent after
            that state.

`chain_from_anchor` records the stricter condition (a seen path from the
question's entity to the gold node) for analysis; it is not the criterion,
because the gate does not require it.
"""
from __future__ import annotations

import collections

from ..agent.metrics import reached
from ..kg.serialize import normalize
from ..train.prefixes import PREFIX_SCHEMA

# Fields of a finished episode that a mid-episode state does not have yet.
_FINAL_FIELDS = ("final_text", "answer", "cited_edges", "messages", "wall_ms")


def answer_set(rec: dict) -> set[str]:
    """The gold answers a record is judged against: its full-graph answer set,
    or its single gold answer when it carries none."""
    return {a for a in (rec.get("kb_answer_set") or [rec.get("gold_answer")]) if a}


def _seen_graph(rec: dict) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = collections.defaultdict(set)
    for e in rec.get("seen_edges", []):
        adj[e["source"]].add(e["target"])
        adj[e["target"]].add(e["source"])
    return adj


def _reachable(adj: dict[str, set[str]], start: str, goals: set[str]) -> bool:
    if start not in adj:
        return False
    seen, stack = {start}, [start]
    while stack:
        n = stack.pop()
        if n in goals:
            return True
        for m in adj[n]:
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return False


def classify(rec: dict, res: dict) -> tuple[str, dict]:
    """`(class, analysis fields)` for one trajectory record and its results
    row: success, fixable or dropped (the CLI turns some dropped into search)."""
    answers = answer_set(rec)
    norm_answers = {normalize(a) for a in answers}
    seen_edges = rec.get("seen_edges", [])
    gold_seen = reached(answers, seen_edges)
    gold_nodes = {n for e in seen_edges for n in (e["source"], e["target"])
                  if normalize(n) in norm_answers}
    chain = bool(gold_nodes) and _reachable(_seen_graph(rec), rec.get("anchor") or "",
                                            gold_nodes)
    extra = {"gold_seen": gold_seen, "chain_from_anchor": chain,
             "source_status": rec.get("status"),
             "source_gate_reason": res.get("gate_reason"),
             "source_gated_acc": bool(res.get("gated_acc")),
             "source_answer": rec.get("answer")}
    if res.get("gated_acc"):
        return "success", extra
    if gold_seen:
        return "fixable", extra
    return "dropped", extra


def gold_neighbours(graph, answers: set[str]) -> set[str]:
    """Every node joined by an edge, either way, to a gold answer, through the
    substrate's own `neighbors` (an answer absent from the graph has none)."""
    out: set[str] = set()
    for a in answers:
        out |= graph.neighbors(a)
    return out


def search_cut(rec: dict, neighbours_of_gold: set[str]) -> tuple[int, int] | None:
    """(assistant-turn number to cut before, calls spent after it) for a
    never-reached-gold episode whose seen set touched a gold neighbour, else
    None. The seen set is replayed call by call from `cum_edges_seen`, so the
    cut is the first state in which a neighbour of gold was on the table."""
    calls = rec.get("calls", [])
    seen = rec.get("seen_edges", [])
    for c in calls:
        upto = int(c.get("cum_edges_seen", 0))
        nodes = {n for e in seen[:upto] for n in (e["source"], e["target"])}
        if nodes & neighbours_of_gold:
            turn = int(c["turn"])
            return turn + 1, len(calls) - calls.index(c) - 1
    return None


def _n_chars(messages: list[dict]) -> int:
    return sum(len(m.get("content", "")) for m in messages)


def to_search_prefix(rec: dict, *, run_tag: str, cut_turn: int, hops_short: int,
                     extra: dict) -> dict | None:
    """The `search` record: `rec` cut before assistant turn `cut_turn`, with
    its calls and seen edges truncated to that state; None when the cut does
    not land after an observation."""
    msgs = rec["messages"]
    positions = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    if not 1 <= cut_turn <= len(positions):
        return None
    prefix = msgs[:positions[cut_turn - 1]]
    if not prefix or prefix[-1].get("role") != "user":
        return None
    out = {k: v for k, v in rec.items() if k not in _FINAL_FIELDS}
    # State is mid-episode: seen edges and calls beyond the cut do not exist yet.
    n_calls = cut_turn - 1
    calls = [c for c in rec.get("calls", []) if int(c["turn"]) <= n_calls]
    n_seen = calls[-1]["cum_edges_seen"] if calls else 0
    out["calls"] = calls
    out["seen_edges"] = rec["seen_edges"][:n_seen]
    out["n_edges_seen"] = n_seen
    out.update({
        "schema": PREFIX_SCHEMA,
        "id": f"{run_tag}#{rec['id']}#c{cut_turn}",
        "item_id": rec["id"],
        "source_run": run_tag,
        "cut": cut_turn,
        "n_prefix_turns": cut_turn - 1,
        "class": "search",
        "search_hops_short": hops_short,
        "messages": prefix,
        "n_chars": _n_chars(prefix),
    })
    out.update(extra)
    return out


def to_prefix(rec: dict, *, run_tag: str, cls: str, extra: dict) -> dict | None:
    """The decision record: `rec` cut before its final assistant turn; None
    when that turn does not follow an observation."""
    msgs = rec["messages"]
    positions = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    if not positions:
        return None
    prefix = msgs[:positions[-1]]
    if not prefix or prefix[-1].get("role") != "user":
        return None
    out = {k: v for k, v in rec.items() if k not in _FINAL_FIELDS}
    out.update({
        "schema": PREFIX_SCHEMA,
        "id": f"{run_tag}#{rec['id']}",
        "item_id": rec["id"],
        "source_run": run_tag,
        "cut": len(positions),
        "n_prefix_turns": len(positions) - 1,
        "class": cls,
        "messages": prefix,
        "n_chars": _n_chars(prefix),
    })
    out.update(extra)
    return out


__all__ = ["answer_set", "classify", "gold_neighbours", "search_cut", "to_prefix",
           "to_search_prefix"]
