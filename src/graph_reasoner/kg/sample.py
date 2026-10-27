"""Gold-path sampling and noisy-subgraph construction.

Invariant guaranteed by construction: within a built subgraph, the gold path is
the only simple path between its endpoints. All noise is attached as trees
(every noise node enters the subgraph through exactly one edge), so no noise
can create an alternative route between gold nodes. This is what makes
traversal metrics (path recall, hallucination) mechanically decidable.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import networkx as nx

from .serialize import normalize


@dataclass(frozen=True)
class PathEdge:
    source: str
    relation: str
    target: str


@dataclass
class GoldPath:
    nodes: list[str]                 # traversal order
    edges: list[PathEdge] = field(default_factory=list)  # true KG direction

    @property
    def endpoints(self) -> tuple[str, str]:
        return self.nodes[0], self.nodes[-1]


def undirected_neighbors(G: nx.DiGraph, n: str) -> set:
    """Every node joined to `n` by an edge in either direction."""
    return set(G.successors(n)) | set(G.predecessors(n))


def _true_edge(G: nx.DiGraph, a: str, b: str) -> PathEdge:
    """The KG edge between a and b in its true direction (prefer a->b)."""
    if G.has_edge(a, b):
        return PathEdge(a, G.edges[a, b].get("relation", "related to"), b)
    return PathEdge(b, G.edges[b, a].get("relation", "related to"), a)


def sample_gold_path(
    G: nx.DiGraph,
    roles: dict[str, str],
    length: int,
    rng: random.Random,
    *,
    relation_blocklist: set[str] | None = None,
    max_degree: int = 150,
    max_tries: int = 2000,
    forbid_endpoint_in_relation: bool = False,
) -> GoldPath:
    """Self-avoiding walk of `length` edges over domain nodes only.

    Blocked relations are skipped, and no node above `max_degree` enters a
    path at any position, so a sampled benchmark holds no path through a hub.
    A path of three or more edges that uses one relation throughout is
    rejected.

    `forbid_endpoint_in_relation` (config `gold_leak_filter`) keeps an edge off
    gold paths when its relation text contains either endpoint's name: such an
    edge gives the next hop away, in the question the teacher writes from it
    and in what `expand` prints. Noise edges are not filtered. Off by default.
    """
    blocked = {normalize(r) for r in (relation_blocklist or set())}
    candidates = [
        n for n in G.nodes
        if roles.get(n) == "domain" and 2 <= G.degree(n) <= max_degree
    ]
    if not candidates:
        raise RuntimeError("No eligible start nodes for gold-path sampling.")

    for _ in range(max_tries):
        start = rng.choice(candidates)
        nodes, edges = [start], []
        dead = False
        for _step in range(length):
            cur = nodes[-1]
            opts = []
            # sorted() because set iteration order varies between processes;
            # rng.choice over it would not reproduce a build from its seed.
            for v in sorted(undirected_neighbors(G, cur)):
                if v in nodes or roles.get(v) != "domain":
                    continue
                if G.degree(v) > max_degree:
                    continue
                e = _true_edge(G, cur, v)
                if normalize(e.relation) in blocked:
                    continue
                if forbid_endpoint_in_relation:
                    rel_n = normalize(e.relation)
                    if (normalize(cur) and normalize(cur) in rel_n) or \
                       (normalize(v) and normalize(v) in rel_n):
                        continue
                opts.append((v, e))
            if not opts:
                dead = True
                break
            v, e = rng.choice(opts)
            nodes.append(v)
            edges.append(e)
        if dead:
            continue
        # Reject monotone paths (all steps the same relation) for length >= 3.
        if length >= 3 and len({normalize(e.relation) for e in edges}) == 1:
            continue
        return GoldPath(nodes, edges)

    raise RuntimeError(f"Failed to sample a length-{length} gold path in {max_tries} tries.")


def build_subgraph(
    G: nx.DiGraph,
    roles: dict[str, str],
    gold: GoldPath,
    rng: random.Random,
    *,
    branch_per_node: int = 2,
    branch_depth: int = 2,
    decoy_walks: int = 2,
    decoy_len: int = 3,
    biblio_attach: int = 2,
    exclude: set[str] | None = None,
) -> nx.DiGraph:
    """Build the context subgraph: gold path + tree-shaped noise.

    - branches: short chains hung off every gold node (local noise)
    - decoys: longer chains from the two endpoints (misleading directions)
    - biblio: bibliographic attachments (authors, figures, venues)

    All noise nodes are new to the subgraph when added (tree property), which
    preserves the gold path as the unique simple endpoint-to-endpoint route.

    `exclude` keeps the given nodes out of the noise. When a question has
    several true answers in the full KG, the others must not enter as noise,
    or the item would acquire a second correct answer.
    """
    blocked = set(exclude or ())
    SG = nx.DiGraph()
    for n in gold.nodes:
        SG.add_node(n, **G.nodes[n])
    for e in gold.edges:
        if G.has_edge(e.source, e.target):
            SG.add_edge(e.source, e.target, **G.edges[e.source, e.target])
        else:  # defensive: store as given
            SG.add_edge(e.source, e.target, relation=e.relation)

    members = set(gold.nodes)

    def grow_walk(root: str, depth: int, biblio_only: bool = False) -> None:
        cur = root
        for _ in range(depth):
            opts = []
            # sorted() for the same reason as in the gold-path walk.
            for v in sorted(undirected_neighbors(G, cur)):
                if v in members or v in blocked:
                    continue  # tree property: never reattach to the subgraph
                if biblio_only and roles.get(v) != "biblio":
                    continue
                opts.append(v)
            if not opts:
                return
            v = rng.choice(opts)
            e = _true_edge(G, cur, v)
            SG.add_node(v, **G.nodes[v])
            SG.add_edge(e.source, e.target, **G.edges[e.source, e.target])
            members.add(v)
            cur = v

    for n in list(gold.nodes):
        for _ in range(branch_per_node):
            grow_walk(n, rng.randint(1, max(1, branch_depth)))

    a, b = gold.endpoints
    for i in range(decoy_walks):
        grow_walk((a, b)[i % 2], decoy_len)

    for _ in range(biblio_attach):
        grow_walk(rng.choice(gold.nodes), rng.randint(1, 2), biblio_only=True)

    return SG
