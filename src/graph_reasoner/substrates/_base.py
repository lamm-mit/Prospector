"""The graph-access methods every substrate implements the same way. Not a
substrate itself: the leading underscore keeps it out of the registration
imports in `substrates/__init__.py`.

A subclass defines `__init__(self, path)` and sets `_G`, `_index`, `_vocab`
and `_path`; this class defines no `__init__` of its own. A subclass also
defines `node_type` and `edges_between`, which differ per graph and have no
default here.
"""
from __future__ import annotations

import networkx as nx

from ..kg.load import file_fingerprint
from ..kg.sample import undirected_neighbors


class SubstrateBase:
    """Defaults over `self._G` / `self._index` / `self._vocab` / `self._path`."""

    def canonical(self, name: str) -> str | None:
        return self._index.canonical(name)

    def search(self, query: str, k: int = 10) -> list[tuple[str, str, int]]:
        return [(n, self.node_type(n), self.degree(n))
                for n in self._index.search(query, k)]

    def degree(self, node: str) -> int:
        return self._G.degree(node) if node in self._G else 0

    def importance(self, node: str) -> float:
        """Query-independent node importance, by which `expand` ranks a hub's
        first page. The node's degree."""
        return float(self.degree(node))

    def neighbors(self, node: str) -> set[str]:
        return undirected_neighbors(self._G, node) if node in self._G else set()

    def relation_vocab(self) -> tuple[str, ...]:
        return self._vocab

    def digraph(self) -> nx.DiGraph:
        return self._G

    def fingerprint(self) -> str:
        return file_fingerprint(self._path)
