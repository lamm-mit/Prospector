"""Benchmark construction: gold paths -> subgraphs -> (teacher or template) QA.

Gold answers exist by construction: the question is generated from a sampled
path, so the path is the ground truth. The teacher only phrases the question;
it cannot inject knowledge the graph lacks.
"""
from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

from ..kg import (
    GoldPath,
    build_subgraph,
    classify_nodes,
    normalize,
    sample_gold_path,
    serialize_subgraph,
)
from ..kg.load import (
    DEFAULT_BIBLIO_NAME_PATTERNS,
    DEFAULT_BIBLIO_TYPES,
    DEFAULT_GOLD_RELATION_BLOCKLIST,
)
from ..kg.serialize import render_edge_line
from ..eval.parse import extract_json_object
from ..llm import LLMClient, gather_with_progress
from ..schema import BenchmarkItem, Edge, GoldPathT

def teacher_system(substrate: str) -> str:
    """The teacher persona, owned by the substrate: `TEACHER_SYSTEM` in
    `substrates/<kg>/qa_style.py`. Exits when that module is missing, because
    another domain's persona would phrase fluent nonsense."""
    import importlib
    try:
        mod = importlib.import_module(
            f"graph_reasoner.substrates.{substrate}.qa_style")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"substrate {substrate!r} has no qa_style.py — benchmark generation "
            f"needs a teacher persona it owns. Add "
            f"src/graph_reasoner/substrates/{substrate}/qa_style.py with a "
            f"TEACHER_SYSTEM string.") from exc
    return mod.TEACHER_SYSTEM


def _chain_text(gold: GoldPath) -> str:
    return "\n".join(
        f"{i+1}. " + render_edge_line(e.source, e.relation, e.target)
        for i, e in enumerate(gold.edges)
    )


def _teacher_prompt(gold: GoldPath, feedback: str = "") -> str:
    src, dst = gold.endpoints
    intermediates = gold.nodes[1:-1]
    inter_txt = ", ".join(repr(n) for n in intermediates) if intermediates else "(none)"
    fb = f"\nPrevious attempt was rejected: {feedback}\n" if feedback else ""
    return f"""Reasoning chain from {src!r} to {dst!r}:
{_chain_text(gold)}

Rules:
- The question must be answerable ONLY by following the whole chain in order.
- Mention {src!r} in the question; the answer resolves at {dst!r}.
- Do NOT name any intermediate entity in the question: {inter_txt}.
- The question should read like a domain exam question, not mention "the graph".
- Gold answer: 2-4 sentences that walk the chain to the conclusion.
{fb}
Return JSON: {{"question": "...", "answer": "..."}}"""


def _leaks_intermediates(question: str, gold: GoldPath) -> str | None:
    q = normalize(question)
    for n in gold.nodes[1:-1]:
        if normalize(n) and normalize(n) in q:
            return n
    return None


def offline_qa(gold: GoldPath) -> tuple[str, str]:
    """Template QA for offline/plumbing runs — no API needed."""
    src, dst = gold.endpoints
    question = (
        f"According to the graph, how is '{src}' connected to '{dst}'? "
        f"Describe the full chain of relationships."
    )
    steps = "; ".join(
        f"'{e.source}' {e.relation} '{e.target}'" for e in gold.edges
    )
    answer = f"The chain is: {steps}."
    return question, answer


async def _teacher_qa(client: LLMClient, gold: GoldPath,
                      system: str) -> tuple[str, str]:
    feedback = ""
    for _attempt in range(3):
        raw = await client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": _teacher_prompt(gold, feedback)},
            ],
            json_mode=True,
            max_tokens=4000,
        )
        obj = extract_json_object(raw) or {}
        q, a = str(obj.get("question", "")).strip(), str(obj.get("answer", "")).strip()
        if len(q) < 20 or len(a) < 20:
            feedback = "question or answer was empty/too short"
            continue
        leaked = _leaks_intermediates(q, gold)
        if leaked:
            feedback = f"the question named the intermediate entity {leaked!r}"
            continue
        return q, a
    # Teacher kept failing — fall back to template so the item isn't lost.
    return offline_qa(gold)


def build_benchmark(
    config: dict,
    *,
    offline: bool = False,
    limit_per_tier: int | None = None,
    out_path: str | None = None,
) -> Path:
    """Sample tiered items from the KG and write a benchmark JSONL."""
    kg_path = config["kg_path"]
    # The substrate, and with it the teacher persona, comes from the config:
    # its `substrate` key, else the `data/benchmarks/<substrate>/` directory of
    # its `output`. Never from `out_path`, which may point at scratch output.
    substrate = str(config.get("substrate") or Path(config["output"]).parent.name)
    # Resolved even when `offline`, which never uses the persona, so a missing
    # qa_style.py fails on the free run rather than on a paid teacher pass.
    system = teacher_system(substrate)
    seed = int(config.get("seed", 0))
    rng = random.Random(seed)

    # The graph loads through the substrate's own backend, which knows its
    # file format.
    from ..agent.backend import load_backend
    G = load_backend(substrate, kg_path).digraph()
    roles = classify_nodes(
        G,
        set(config.get("biblio_types", DEFAULT_BIBLIO_TYPES)),
        list(config.get("biblio_name_patterns", DEFAULT_BIBLIO_NAME_PATTERNS)),
    )
    blocklist = set(config.get("gold_relation_blocklist", DEFAULT_GOLD_RELATION_BLOCKLIST))
    # A gold node must be expandable without a relation filter, so the
    # sampler's degree cap defaults to the tools' hub threshold. A config
    # may set a lower cap; a higher one breaks the property, which
    # `bench.agentic.hub_violations` reports.
    from ..agent.tools import ToolConfig
    max_degree = int(config.get("max_node_degree", ToolConfig().hub_threshold))

    items: list[BenchmarkItem] = []
    golds: list[GoldPath] = []
    for tier, params in config["tiers"].items():
        n_items = limit_per_tier or int(params["n_items"])
        seen: set[frozenset] = set()
        made = 0
        attempts = 0
        while made < n_items and attempts < n_items * 50:
            attempts += 1
            try:
                gold = sample_gold_path(
                    G, roles, int(params["path_len"]), rng,
                    relation_blocklist=blocklist, max_degree=max_degree,
                    forbid_endpoint_in_relation=bool(
                        config.get("gold_leak_filter", False)),
                )
            except RuntimeError:
                break
            key = frozenset(gold.nodes)
            if key in seen:
                continue
            seen.add(key)
            SG = build_subgraph(
                G, roles, gold, rng,
                branch_per_node=int(params.get("branch_per_node", 2)),
                branch_depth=int(params.get("branch_depth", 2)),
                decoy_walks=int(params.get("decoy_walks", 2)),
                decoy_len=int(params.get("decoy_len", 3)),
                biblio_attach=int(params.get("biblio_attach", 2)),
            )
            text, edge_dicts = serialize_subgraph(SG, rng)
            items.append(
                BenchmarkItem(
                    id=f"{tier}-{made:04d}",
                    tier=tier,
                    question="",  # filled below
                    gold_answer="",
                    gold_path=GoldPathT(
                        nodes=list(gold.nodes),
                        edges=[Edge(**e.__dict__) for e in gold.edges],
                    ),
                    subgraph_edges=[Edge(**e) for e in edge_dicts],
                    subgraph_text=text,
                    meta={
                        "seed": seed,
                        "n_nodes": SG.number_of_nodes(),
                        "n_edges": SG.number_of_edges(),
                        "path_len": len(gold.edges),
                    },
                )
            )
            golds.append(gold)
            made += 1
        if made < n_items:
            print(f"[warn] tier {tier}: only sampled {made}/{n_items} items")

    # Fill in questions/answers.
    if offline:
        for item, gold in zip(items, golds):
            item.question, item.gold_answer = offline_qa(gold)
    else:
        client = LLMClient(config.get("teacher_model", "gpt-5-mini"))
        print(f"Teacher: {client.model} — generating {len(items)} QA pairs...")

        async def _run():
            return await gather_with_progress(
                [_teacher_qa(client, g, system) for g in golds], desc="teacher QA"
            )

        results = asyncio.run(_run())
        for item, (q, a) in zip(items, results):
            item.question, item.gold_answer = q, a
        print(client.usage_summary())

    out = Path(out_path or config["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")

    # Stats
    by_tier: dict[str, list[BenchmarkItem]] = {}
    for it in items:
        by_tier.setdefault(it.tier, []).append(it)
    print(f"\nWrote {len(items)} items -> {out}")
    for tier, its in by_tier.items():
        avg_e = sum(i.meta["n_edges"] for i in its) / len(its)
        avg_tok = sum(len(i.subgraph_text) for i in its) / len(its) / 4
        print(f"  {tier}: {len(its)} items, avg {avg_e:.0f} edges, ~{avg_tok:.0f} tok subgraph")
    return out
