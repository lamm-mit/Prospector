"""Per-call edge attribution for a stored trajectory record, shared by the
replay endpoint and the live stream.

Edges are parsed from the recorded observation text; no tool is re-run. Every
seen edge was rendered into an observation (the seen contract, agent/tools.py),
so parsing can recover them all. The result is checked against the record's
own `seen_edges`, and any shortfall is reported to the UI.
"""
from __future__ import annotations

from graph_reasoner.agent.prompts import parse_observation_header
from graph_reasoner.kg.serialize import edge_key as _canonical_edge_key
from graph_reasoner.kg.serialize import parse_edge_line


def edge_key(e: dict) -> tuple[str, str, str]:
    """The scorer's membership key (kg.serialize.edge_key) for an edge dict.
    It normalises whitespace and case, so membership is tested on this key,
    never on raw strings."""
    return _canonical_edge_key(e["source"], e["relation"], e["target"])


def attribute_turn_edges(
    t: dict,
) -> tuple[list[list[dict]], dict, list[list[str]]]:
    """Per-call new edges, parsed from recorded observations, plus checks.

    Returns `(turn_edges, checks, call_focus)`; the two lists are indexed by
    call (observation headers number calls from 1, these from 0):
      turn_edges  the edges first rendered on that call
      checks      the reconstruction compared with the record's seen_edges;
                  seen edges never parsed are attached to the final call
      call_focus  the node or nodes the call operated on
    """
    n_calls = len(t.get("calls", []))
    per_call: list[list[dict]] = [[] for _ in range(n_calls)]
    cum: set[tuple[str, str, str]] = set()
    # The record's seen set decides membership; parsing only decides on which
    # call an edge appeared. A line that only looks like an edge is dropped.
    recorded = {edge_key(e) for e in t.get("seen_edges", [])}
    names = {e["source"] for e in t.get("seen_edges", [])} | \
            {e["target"] for e in t.get("seen_edges", [])}
    by_lower = {n.lower(): n for n in names}
    calls = t.get("calls", [])
    # Per call: the node or nodes the call operated on, read from the record
    # without a backend lookup.
    call_focus: list[list[str]] = [[] for _ in range(n_calls)]
    per_call_all: list[list[dict]] = [[] for _ in range(n_calls)]
    for m in t.get("messages", []):
        if m.get("role") != "user":
            continue
        content = m.get("content", "")
        head = parse_observation_header(content)
        if not head:
            continue
        k = head[0] - 1
        if not (0 <= k < n_calls):
            continue
        for line in content.splitlines():
            parsed = parse_edge_line(line.strip())
            if not parsed:
                continue
            e = {"source": parsed[0].strip(), "relation": parsed[1].strip(),
                 "target": parsed[2].strip()}
            key = edge_key(e)
            if key not in recorded:
                continue
            per_call_all[k].append(e)
            if key not in cum:
                cum.add(key)
                per_call[k].append(e)
        tool = (calls[k] or {}).get("tool")
        args = (calls[k] or {}).get("args") or {}
        if tool == "expand":
            ends = [x for e in per_call_all[k] for x in (e["source"], e["target"])]
            if ends:
                # the expanded node is an endpoint of every returned edge
                call_focus[k] = [max(set(ends), key=ends.count)]
        elif tool == "connect":
            got = [by_lower[str(args.get(a, "")).lower()]
                   for a in ("a", "b") if str(args.get(a, "")).lower() in by_lower]
            call_focus[k] = got
        # find takes a string, not a node: no focus.
        if not call_focus[k] and tool == "expand":
            # A call with no edges: match the argument against the recorded
            # names, case-insensitively.
            v = str(args.get("node", "")).lower()
            if v in by_lower:
                call_focus[k].append(by_lower[v])
    # Parsed edges were filtered to the seen set, so the reconstruction can
    # only fall short of it.
    missing = recorded - cum          # in seen, never parsed from an obs
    if missing and n_calls:
        by_key = {edge_key(e): e for e in t.get("seen_edges", [])}
        per_call[-1].extend(by_key[k] for k in sorted(missing))
    checks = {
        "attribution_exact": not missing,
        "n_recorded": len(recorded), "n_parsed": len(cum),
        "n_unattributed": len(missing),
    }
    return per_call, checks, call_focus
