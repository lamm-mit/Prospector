"""Curate raw generation down to items that are actually engineering questions.

`reject` applies five filters, in this order:

1. Graph-navigation question: "according to the graph, how is X connected to
   Y?" is a query about the data structure, not a question an engineer asks.
2. Bibliographic endpoint: a gold endpoint like `28` or `25(3)`, a figure or
   classification identifier rather than an engineering term.
3. Degenerate endpoint: a gold endpoint of one or two characters (`D`, `Al`).
   Scoring is a normalised substring test, which so short a string matches
   almost anywhere.
4. Answer named in the question: with both endpoints known, `connect(a, b)`
   returns the chain in one call and nothing is left to search for.
5. No anchor in the question: neither gold endpoint appears in the text, so
   `find` has nothing to resolve.

Every dropped item is returned with its reason, so the cut is auditable.
"""
from __future__ import annotations

import collections
import re

from ..kg.serialize import normalize

NAV = re.compile(r"according to the graph|describe the full chain", re.I)
BIB = re.compile(
    r"^(\d+|\d+\(\d+\)|[\w\-]*\.png|\d+_image_\d+|(fig|figs|eq|eqs|table)[\s.~(].*)$",
    re.I)


def endpoints(row: dict) -> tuple[str, str]:
    nodes = row["gold_path"]["nodes"]
    return nodes[0].strip(), nodes[-1].strip()


def reject(row: dict) -> str | None:
    """The first filter an item trips, or None if it is kept. Order matters only
    for the reported tally, never for the keep/drop decision."""
    a, z = endpoints(row)
    na, nz = normalize(a), normalize(z)
    q = normalize(row["question"])
    if NAV.search(row["question"]):
        return "graph-navigation question"
    if any(BIB.match(x) for x in (a, z)):
        return "bibliographic endpoint"
    if any(len(x) <= 2 for x in (na, nz)):
        return "degenerate 1-2 char endpoint"
    if na in q and nz in q:
        return "answer named in the question"
    if not (na in q or nz in q):
        return "no anchor present in question"
    return None


def curate(rows: list[dict]) -> tuple[list[dict], list[dict], collections.Counter]:
    """rows -> (kept, dropped-with-reasons, tally). Pure; no I/O."""
    keep, dropped, why = [], [], collections.Counter()
    for r in rows:
        reason = reject(r)
        if reason:
            why[reason] += 1
            dropped.append({**r, "drop_reason": reason})
        else:
            keep.append(r)
    return keep, dropped, why
