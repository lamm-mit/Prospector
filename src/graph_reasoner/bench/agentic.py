"""Add the fields agentic scoring needs to a curated benchmark.

The one that matters is `kb_answer_set`: every answer reachable from the anchor
by the gold path's own relation sequence over the full graph. Several paths can
join the same anchor to different answers by the same relations, and a policy
that finds another of them is right, so an item is scored against the set
rather than one gold string.

The walk is undirected, because `expand` shows a node's edges both ways and the
policy may cite either. It follows relation names exactly, so the answer set is
computed mechanically, with no judge.

An item whose question names a member of that set other than the anchor is
dropped (`names_an_answer`): the curation filter (`curate.py`) catches the
gold answer spelled as its node label, and this catches a second member or
another spelling, which a model is credited for restating.

`hub_violations` lists the gold nodes above `ToolConfig.hub_threshold`, which
a policy cannot expand without a relation filter; a config that raises
`max_node_degree`, or a changed graph, can produce them.
"""
from __future__ import annotations

import collections
import re

import networkx as nx

from ..kg.serialize import normalize

def names_an_answer(question: str, anchor: str, answers) -> list[str]:
    """The members of an answer set, other than the anchor, that the question
    names (a normalized substring): answers the question states."""
    q, a = normalize(question), normalize(anchor)
    return [x for x in answers if normalize(x) != a and normalize(x) in q]


def relation_index(G: nx.DiGraph) -> dict[tuple[str, str], set[str]]:
    """(node, relation) -> neighbours reachable by that relation, either way."""
    idx: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for u, v, d in G.edges(data=True):
        r = str(d.get("relation", "related to"))
        idx[(u, r)].add(v)
        idx[(v, r)].add(u)
    return idx

def walk(idx, start: str, relations: list[str], cap: int = 200) -> set[str]:
    """Every node reachable from `start` by following `relations` in order;
    empty when a step fans out to more than `cap` nodes."""
    frontier = {start}
    for r in relations:
        nxt: set[str] = set()
        for node in frontier:
            nxt |= idx.get((node, r), set())
            if len(nxt) > cap:
                return set()                 # pathological fan-out: refuse to guess
        frontier = nxt
        if not frontier:
            return set()
    return frontier

def oriented_relations(item: dict, anchor: str) -> list[str]:
    """The gold relation names in traversal order starting from `anchor`.

    `gold_path.nodes` is in traversal order and the edge list is aligned with
    it, so anchoring at the far end reverses the list.
    """
    rels = [e["relation"] for e in item["gold_path"]["edges"]]
    return rels if item["gold_path"]["nodes"][0] == anchor else list(reversed(rels))

def surface_form(question: str, anchor: str) -> str:
    """The anchor as the question writes it (its casing there), or the graph
    key when the question does not contain it."""
    m = re.search(re.escape(anchor), question, re.I)
    return m.group(0) if m else anchor

def pick_anchor(item: dict) -> tuple[str, str] | None:
    """(anchor, answer): the anchor is the gold endpoint the question names and
    the answer is the other one. None when it names both or neither."""
    q = normalize(item["question"])
    a, z = item["gold_path"]["nodes"][0], item["gold_path"]["nodes"][-1]
    a_in, z_in = normalize(a) in q, normalize(z) in q
    if a_in and not z_in:
        return a, z
    if z_in and not a_in:
        return z, a
    return None


def augment(rows: list[dict], G: nx.DiGraph, *, source: str,
            banned: set[frozenset] | None = None
            ) -> tuple[list[dict], collections.Counter, list[int]]:
    """rows -> (augmented, drop tally, kb_answer_set sizes). Pure; no I/O."""
    banned = banned or set()
    idx = relation_index(G)
    out, drop, sizes = [], collections.Counter(), []
    for item in rows:
        if frozenset(item["gold_path"]["nodes"]) in banned:
            drop["gold path collides with an excluded benchmark"] += 1
            continue
        picked = pick_anchor(item)
        if not picked:
            drop["anchor not identifiable in question"] += 1
            continue
        anchor, answer = picked
        if anchor not in G:
            drop["anchor absent from the KG"] += 1
            continue
        answers = walk(idx, anchor, oriented_relations(item, anchor))
        if answer not in answers:
            # The gold endpoint must be reachable by its own relation sequence.
            # If it is not, the item's gold path disagrees with the graph and no
            # mechanical answer set can be trusted for it.
            drop["gold answer not reachable by its own relations"] += 1
            continue
        if names_an_answer(item["question"], anchor, answers):
            drop["an answer-set member is named in the question"] += 1
            continue
        meta = dict(item.get("meta") or {})
        meta.update({
            "source": source,
            "anchor": anchor,
            "anchor_query": surface_form(item["question"], anchor),
            "kb_answer_set": sorted(answers),
            "kb_answer_set_source": "relation_walk",
            "answer_set": [answer],
            "path_len": len(item["gold_path"]["edges"]),
        })
        sizes.append(len(answers))
        out.append({**item, "gold_answer": answer, "meta": meta})
    return out, drop, sizes


def hub_violations(rows: list[dict], G: nx.DiGraph) -> list[tuple[str, str, int]]:
    """Gold-path nodes a policy could not expand without a relation filter."""
    from ..agent.tools import ToolConfig
    hub = ToolConfig().hub_threshold
    return [(r["id"], n, G.degree(n)) for r in rows
            for n in r["gold_path"]["nodes"] if n in G and G.degree(n) > hub]
