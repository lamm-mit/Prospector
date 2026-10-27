#!/usr/bin/env python
"""Build the retrieval baseline: the subgraph a GraphRAG- or HippoRAG-style
pipeline would put in the prompt, written as a static benchmark.

    # containment sweep over both rules and two budgets; writes no benchmark
    python scripts/arms/build_retrieval_benchmark.py --graph tsmc --stats-only

    # the semiconductor graph's arm: k-hop from the question's entity, 400 edges
    python scripts/arms/build_retrieval_benchmark.py --graph tsmc --rule khop --budget 400
    # -> data/benchmarks/tsmc/tsmc_core__khop400anchor.jsonl, unless --out

    # a transfer graph's arm: personalized PageRank
    python scripts/arms/build_retrieval_benchmark.py --graph biograph \\
        --rule ppr --budget 400

Retrieval starts from the question's entity (`meta.anchor`), which is given,
ranks the graph's edges by one rule on the undirected graph, and keeps the top
`--budget`. There are no text embeddings:

    ppr    personalized PageRank from the entity (HippoRAG's rule)
    khop   hop distance from the entity, then question-term overlap
           (GraphRAG-local's shape, a lexical stand-in for its embedding
           ranking)

`--benchmark` defaults to the substrate's evaluation file
(`<backend>_core_agentic`); one from another substrate's directory is refused.
The output goes beside it as `<stem>__<rule><budget>anchor.jsonl`, the name
`benchmarks.answers_for` maps back to the answer sets, with ids
`<base>#<rule><budget>` and a `<out>.provenance.json` sidecar. It is built, not
tracked; `tests/test_retrieval_rebuild.py` holds the checksum of each file the
paper's runs read. No model is called here: scripts/arms/run_static_api.py
answers the file and scripts/arms/score_static_gated.py scores it.

`--stats-only` reports, for each rule and budget, how often the retrieved set
contains the whole gold chain, and how often it contains an answer node. The
gate credits only edges that were shown, so chain containment bounds the arm's
gated score. Every run first applies both checks to each item's own gold
subgraph and exits if either fails.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import sys
from collections import deque
from pathlib import Path

import numpy as np

from graph_reasoner import benchmarks, provenance
from graph_reasoner.agent.backend import available_backends, load_backend, resolve_kb
from graph_reasoner.agent.metrics import answer_set_of
from graph_reasoner.kg.serialize import edge_key, normalize, render_edge_line

# The stopword list and token rule of the k-hop ordering: what counts as a
# term when an edge's overlap with the question is counted.
STOP = {"the", "a", "an", "of", "in", "to", "is", "are", "which", "what", "for",
        "and", "or", "by", "on", "at", "with", "that", "this", "it", "its",
        "be", "can", "has", "have", "when", "during", "from", "as", "was"}


def terms(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").casefold())
            if w not in STOP and len(w) > 2}


# ---------------------------------------------------------------- graph index

class GraphIndex:
    """The KG flattened to numpy once: nodes, directed edges, undirected arcs."""

    def __init__(self, G):
        self.nodes = list(G.nodes())
        self.idx = {n: i for i, n in enumerate(self.nodes)}
        # Directed edges as stored (the citable universe).
        self.edges = [(u, v, str(d.get("relation", "related to")))
                      for u, v, d in G.edges(data=True)]
        self.e_src = np.array([self.idx[u] for u, _, _ in self.edges], dtype=np.int64)
        self.e_dst = np.array([self.idx[v] for _, v, _ in self.edges], dtype=np.int64)
        # Undirected arcs (each unordered pair both ways) for PPR/BFS.
        pairs = {(self.idx[u], self.idx[v]) for u, v, _ in self.edges}
        arcs = sorted(pairs | {(b, a) for a, b in pairs})
        self.a_src = np.array([a for a, _ in arcs], dtype=np.int64)
        self.a_dst = np.array([b for _, b in arcs], dtype=np.int64)
        n = len(self.nodes)
        self.out_deg = np.bincount(self.a_src, minlength=n).astype(np.float64)
        self.adj: list[list[int]] = [[] for _ in range(n)]
        for a, b in arcs:
            self.adj[a].append(b)

    def ppr(self, seeds: dict[int, float], alpha: float = 0.85,
            tol: float = 1e-10, max_iter: int = 200) -> np.ndarray:
        n = len(self.nodes)
        p = np.zeros(n)
        for i, w in seeds.items():
            p[i] = w
        p /= p.sum()
        x = p.copy()
        safe_deg = np.where(self.out_deg > 0, self.out_deg, 1.0)
        for _ in range(max_iter):
            contrib = x / safe_deg
            spread = np.bincount(self.a_dst, weights=contrib[self.a_src], minlength=n)
            dangling = x[self.out_deg == 0].sum()
            x_new = (1 - alpha) * p + alpha * (spread + dangling * p)
            if np.abs(x_new - x).sum() < tol:
                x = x_new
                break
            x = x_new
        return x

    def bfs_dist(self, seeds: list[int]) -> np.ndarray:
        dist = np.full(len(self.nodes), np.iinfo(np.int32).max, dtype=np.int32)
        dq = deque()
        for s in seeds:
            dist[s] = 0
            dq.append(s)
        while dq:
            u = dq.popleft()
            for v in self.adj[u]:
                if dist[v] > dist[u] + 1:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        return dist


# ---------------------------------------------------------------- retrieval

def retrieve(gi: GraphIndex, rule: str, seeds: dict[str, float],
             question: str, budget: int) -> list[dict]:
    """One fixed procedure: seeds -> ranked edges -> top-`budget`."""
    sid = {gi.idx[n]: w for n, w in seeds.items() if n in gi.idx}
    if not sid:
        return []
    if rule == "ppr":
        x = gi.ppr(sid)
        score = x[gi.e_src] + x[gi.e_dst]
        order = np.argsort(-score, kind="stable")
    elif rule == "khop":
        dist = gi.bfs_dist(list(sid))
        e_dist = np.maximum(dist[gi.e_src], dist[gi.e_dst])
        q = terms(question)
        overlap = np.array([len(terms(f"{u} {r} {v}") & q)
                            for u, v, r in gi.edges], dtype=np.int64)
        order = np.lexsort((np.arange(len(gi.edges)), -overlap, e_dist))
    else:
        raise ValueError(rule)
    out = []
    for j in order[:budget]:
        u, v, r = gi.edges[j]
        out.append({"source": u, "relation": r, "target": v})
    return out


def contains(edges: list[dict], item: dict) -> tuple[bool, bool]:
    """(full gold chain in set, any answer-set node among endpoints)."""
    have = {edge_key(e["source"], e["relation"], e["target"]) for e in edges}
    gold = [edge_key(e["source"], e["relation"], e["target"])
            for e in item["gold_path"]["edges"]]
    chain = all(g in have for g in gold)
    nodes = {normalize(e[k]) for e in edges for k in ("source", "target")}
    answers = {normalize(a) for a in answer_set_of(item.get("meta"), item.get("gold_answer"))}
    return chain, bool(nodes & answers)


# ---------------------------------------------------------------- main

def variant_tag(rule: str, budget: int) -> str:
    """The id suffix of one arm, e.g. `khop400`; the file name adds `anchor`."""
    return f"{rule}{budget}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # The graph and the questions both follow --graph: the graph from the
    # registry's `default_kb`, the questions from `<graph>_core_agentic`.
    ap.add_argument("--graph", required=True, choices=available_backends())
    ap.add_argument("--benchmark", type=Path, default=None,
                    help="default: the substrate's agentic benchmark, "
                         "<graph>_core_agentic")
    ap.add_argument("--kb", default=None,
                    help="graph file; defaults to the substrate's own "
                         "(register_backend default_kb)")
    ap.add_argument("--stats-only", action="store_true",
                    help="containment sweep over both rules and two budgets")
    # khop is the semiconductor graph's rule; the transfer graphs' arms use ppr.
    ap.add_argument("--rule", choices=["ppr", "khop"], default="khop")
    ap.add_argument("--budget", type=int, default=400,
                    help="edge cap; 400 = the agent's own max_edges_seen")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <benchmark dir>/<stem>__<rule><budget>anchor.jsonl")
    ap.add_argument("--stats-json", type=Path, default=None,
                    help="write per-item sweep results here (stats-only mode)")
    a = ap.parse_args()

    bench = a.benchmark or benchmarks.resolve(f"{a.graph}_core_agentic")
    shown = benchmarks.repo_path(bench)
    there = benchmarks.substrate_of(bench)
    if there and there != a.graph:
        raise SystemExit(f"refusing to build a {a.graph} retrieval arm from {there} "
                         f"questions: {bench}")
    a.kb = resolve_kb(a.graph, a.kb)

    with open(bench, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    be = load_backend(a.graph, a.kb)
    G = be.digraph()
    gi = GraphIndex(G)
    print(f"{len(items)} items from {shown}; graph {len(gi.nodes)} nodes / "
          f"{len(gi.edges)} directed edges")

    # -- self-test: the checker on the gold subgraph itself must be perfect
    bad = [it["id"] for it in items
           if contains(it["subgraph_edges"], it) != (True, True)]
    if bad:
        print(f"SELF-TEST FAILED on {len(bad)} items, e.g. {bad[:5]}", file=sys.stderr)
        return 1
    print("self-test: gold subgraph -> chain 1.000, answer 1.000  [ok]")

    def seeds_of(it: dict) -> dict[str, float]:
        """The one seed: the entity the question is about."""
        return {it["meta"]["anchor"]: 1.0}

    if a.stats_only:
        budgets = (100, 400)
        rows, per_item = [], []
        for rule in ("ppr", "khop"):
            for budget in budgets:
                chain_n = ans_n = 0
                for it in items:
                    edges = retrieve(gi, rule, seeds_of(it), it["question"], budget)
                    chain, ans = contains(edges, it)
                    chain_n += chain
                    ans_n += ans
                    if budget == max(budgets):
                        per_item.append({"id": it["id"], "rule": rule, "budget": budget,
                                         "chain": chain, "answer": ans})
                rows.append((rule, budget, chain_n, ans_n))
                print(f"{rule:5s} budget {budget:4d}: "
                      f"chain {chain_n}/{len(items)} = {chain_n/len(items):.3f}   "
                      f"answer-node {ans_n}/{len(items)} = {ans_n/len(items):.3f}")
        if a.stats_json:
            a.stats_json.parent.mkdir(parents=True, exist_ok=True)
            a.stats_json.write_text(json.dumps(
                {"rows": [dict(zip(("rule", "budget", "chain", "answer"), r))
                          for r in rows],
                 "per_item": per_item}, indent=1))
            print(f"per-item results -> {a.stats_json}")
        return 0

    # -- emit one benchmark file
    tag = variant_tag(a.rule, a.budget)
    stem = bench.stem.removesuffix("_agentic")
    out = a.out or bench.with_name(f"{stem}__{tag}anchor.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    chain_n = ans_n = 0
    sizes = []
    with out.open("w") as fh:
        for it in items:
            seeds = seeds_of(it)
            edges = retrieve(gi, a.rule, seeds, it["question"], a.budget)
            chain, ans = contains(edges, it)
            chain_n += chain
            ans_n += ans
            sizes.append(len(edges))
            rng = random.Random(f"{it['id']}:{a.rule}:{a.budget}")
            edges = list(edges)
            rng.shuffle(edges)  # edge order carries no signal
            row = dict(it)
            # The variant is part of the id; scorers join on id.split("#")[0].
            row["id"] = f"{it['id']}#{tag}"
            row["subgraph_edges"] = edges
            row["subgraph_text"] = "\n".join(
                render_edge_line(e["source"], e["relation"], e["target"])
                for e in edges)
            # Record how the subgraph was retrieved.
            row["meta"] = dict(it.get("meta") or {},
                               retrieval={"rule": a.rule, "seeding": "anchor",
                                          "budget": a.budget,
                                          **({"alpha": 0.85} if a.rule == "ppr" else {}),
                                          "seeds": sorted(seeds),
                                          "n_edges": len(edges),
                                          "diag_chain_contained": chain,
                                          "diag_answer_present": ans})
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = len(items)
    provenance.write_sidecar(out, provenance.block(
        "scripts/arms/build_retrieval_benchmark.py", sys.argv[1:],
        extra={"source_benchmark": shown, "backend": a.graph, "kb": a.kb,
               "rule": a.rule, "seeding": "anchor", "budget": a.budget,
               "n_items": n, "chain_contained": chain_n, "answer_present": ans_n}))
    print(f"{n} items -> {out}")
    print(f"  edges/item mean {st.mean(sizes):.0f} (max {max(sizes)}); "
          f"chain contained {chain_n}/{n} = {chain_n/n:.3f}; "
          f"answer node present {ans_n}/{n} = {ans_n/n:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
