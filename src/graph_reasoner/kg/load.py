"""KG loading and node-role classification (domain vs. bibliographic)."""
from __future__ import annotations

import hashlib
import re

import networkx as nx

# Node types that are bibliographic or presentational, not domain knowledge;
# the default of a benchmark config's `biblio_types` (bench/qa_gen.py).
DEFAULT_BIBLIO_TYPES = {
    "author", "image", "figure", "table", "reference", "publication",
    "journal", "conference", "person", "email", "location", "organization",
}

# Junk nodes caught by name shape, typed or not: figure/table refs, citation strings, emails.
DEFAULT_BIBLIO_NAME_PATTERNS = [
    r"^(fig|figs|figure|table|eq|eqs|equation)[\s.~(]",
    r"\(\d{4}\)$",          # citation strings ending in a year: '14, 3005 (2021)'
    r"@",                    # email addresses
]

# Relations that should never appear on a *gold* reasoning path (bibliographic
# plumbing or semantically weak links). They remain in the graph as noise.
DEFAULT_GOLD_RELATION_BLOCKLIST = {
    "authored", "authored_by", "author_of", "published in", "published_in",
    "illustrates", "illustrated in", "shows", "cited", "cites",
    "related to", "related_to", "is related to", "abbreviation_of",
}


def load_kg(path: str) -> nx.DiGraph:
    """Load a GraphML knowledge graph. Ensures every edge has a 'relation'."""
    G = nx.read_graphml(path)
    if not G.is_directed():
        G = G.to_directed()
    for _, _, d in G.edges(data=True):
        d.setdefault("relation", "related to")
    return G


def classify_nodes(
    G: nx.DiGraph,
    biblio_types: set[str] | None = None,
    biblio_name_patterns: list[str] | None = None,
) -> dict[str, str]:
    """Classify every node as 'domain' or 'biblio'.

    A node is biblio if its 'type' attribute is bibliographic, or if its name
    matches a bibliographic pattern, whatever its type.
    """
    btypes = {t.casefold() for t in (biblio_types or DEFAULT_BIBLIO_TYPES)}
    patterns = [re.compile(p, re.IGNORECASE)
                for p in (biblio_name_patterns or DEFAULT_BIBLIO_NAME_PATTERNS)]

    roles: dict[str, str] = {}
    for n, d in G.nodes(data=True):
        ntype = str(d.get("type", "")).casefold()
        if ntype in btypes:
            roles[n] = "biblio"
        elif any(p.search(str(n)) for p in patterns):
            roles[n] = "biblio"
        else:
            roles[n] = "domain"
    return roles


def file_fingerprint(path: str) -> str:
    """Short content hash of a graph file: 16 hex characters of its SHA-256.

    Backends build their `fingerprint()` from it and every trajectory records
    that as `kg_fingerprint`, so runs on different graph files can be told
    apart.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]
