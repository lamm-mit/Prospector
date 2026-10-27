"""Subgraph -> prompt text, plus the canonical edge normalization used everywhere.

The same normalization must be used when serializing subgraphs, when matching
a model's cited edges, and when computing path recall, so it all lives here.
"""
from __future__ import annotations

import random
import re
import unicodedata

import networkx as nx


def normalize(s: str) -> str:
    """Canonical form for entity/relation matching: NFKC, casefold, collapse ws."""
    s = unicodedata.normalize("NFKC", str(s))
    return re.sub(r"\s+", " ", s).strip().casefold()


def edge_key(source: str, relation: str, target: str) -> tuple[str, str, str]:
    return (normalize(source), normalize(relation), normalize(target))


def render_edge_line(source: str, relation: str, target: str) -> str:
    """The one edge rendering: `source --[relation]--> target`.

    The model reads edges in this form and cites them back, and the gate checks
    the citation by set membership, so every producer of edge lines must render
    through this function. A caller that needs a prefix (an indent, a line
    number) adds it around the result.
    """
    return f"{source} --[{relation}]--> {target}"


_EDGE_LINE_RE = re.compile(r"^(.+?) --\[(.+?)\]--> (.+)$")


def parse_edge_line(line: str) -> tuple[str, str, str] | None:
    """`render_edge_line` read back: `(source, relation, target)` for an edge
    line, None for any other line of an observation. The line is matched as
    given; a caller that wants surrounding whitespace ignored strips it."""
    m = _EDGE_LINE_RE.match(line)
    return (m.group(1), m.group(2), m.group(3)) if m else None


# How close two already-normalized strings must be to count as the same thing:
# `agent/backend.py`'s EntityIndex thresholds fuzzy entity-name lookup with it.
FUZZY_THRESHOLD = 0.85


def serialize_subgraph(
    SG: nx.DiGraph, rng: random.Random
) -> tuple[str, list[dict]]:
    """Render a subgraph as a shuffled edge list.

    Returns (text_block, edge_dicts). Shuffling scatters gold-path edges so
    position carries no signal.
    """
    edges = [
        {"source": str(u), "relation": str(d.get("relation", "related to")), "target": str(v)}
        for u, v, d in SG.edges(data=True)
    ]
    rng.shuffle(edges)
    lines = [render_edge_line(e["source"], e["relation"], e["target"])
             for e in edges]
    return "\n".join(lines), edges
