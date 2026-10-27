"""Path search over a KG, independent of any substrate. `find_paths` backs the
`connect` tool.
"""
from __future__ import annotations

import networkx as nx

from .sample import undirected_neighbors


def find_paths(
    G: nx.DiGraph,
    src: str,
    dst: str,
    hops: int,
    *,
    max_expand: int = 400,
    limit: int = 8,
) -> list[list[str]]:
    """Self-avoiding undirected paths of exactly `hops` edges from src to dst,
    over any relation; at most `limit` are returned.

    Bounded DFS: a node with more than `max_expand` neighbours is not expanded,
    except for a final step straight to `dst`, so a hub cannot blow up the
    search. Neighbours are visited in name order, so which `limit` paths come
    back does not depend on the per-process string hash.
    """
    if src not in G or dst not in G:
        return []
    found: list[list[str]] = []

    def dfs(path: list[str], seen: set[str]) -> None:
        if len(found) >= limit:
            return
        cur = path[-1]
        depth = len(path) - 1
        if depth == hops:
            if cur == dst:
                found.append(list(path))
            return
        nbrs = undirected_neighbors(G, cur)
        if len(nbrs) > max_expand:
            # Hub: only follow the edge that can still reach dst in one step.
            nbrs = {dst} if dst in nbrs and depth + 1 == hops else set()
        for v in sorted(nbrs):
            if v in seen:
                continue
            # prune: must still be able to arrive exactly at depth `hops`
            if v == dst and depth + 1 != hops:
                continue
            path.append(v)
            seen.add(v)
            dfs(path, seen)
            path.pop()
            seen.discard(v)

    dfs([src], {src})
    return found
