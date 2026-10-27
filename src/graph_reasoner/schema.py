"""Pydantic schemas: benchmark items and model-cited graphs.

`BenchmarkItem.meta` is an untyped dict, but the rest of the tree reads
some of its keys by name, and no validator catches a typo in one:

  * `kb_answer_set` / `answer_set`: the gold answers, in that precedence,
    ahead of the single `gold_answer` (`answer_set_of` below)
  * `anchor`: the question's entity

`bench/agentic.py` writes them, and beside them `anchor_query`, the anchor as
the question spells it. `subgraph_text` is the edge list
`eval/static_prompts.build_eval_prompt` puts in the static arms' prompt,
and `subgraph_edges` is what `scripts/arms/score_static_gated.py` checks their
citations against. The agentic prompt (`agent/prompts.build_agent_user`) reads
neither, and an agentic item may leave both empty.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class Edge(BaseModel):
    source: str
    relation: str
    target: str


class GoldPathT(BaseModel):
    nodes: list[str]
    edges: list[Edge]


def answer_set_of(meta: dict | None, gold_answer=None) -> list[str]:
    """Gold answers under the benchmark's precedence: `kb_answer_set`, then
    `answer_set`, then the single `gold_answer`. `BenchmarkItem.answer_set`
    applies it to an item and `benchmarks.gold_answers` to raw rows."""
    meta = meta or {}
    return [str(a) for a in (meta.get("kb_answer_set")
                             or meta.get("answer_set")
                             or [gold_answer]) if a]


class BenchmarkItem(BaseModel):
    id: str
    tier: str
    question: str
    gold_answer: str
    gold_path: GoldPathT
    subgraph_edges: list[Edge]
    subgraph_text: str
    meta: dict = Field(default_factory=dict)

    def gold_pathset(self) -> frozenset[str]:
        """The eval-leak key: the set of gold nodes, not the id.

        The node set identifies a question across a benchmark rebuild, so an
        eval item cannot re-enter training under a new id; `train/rollout.py`
        excludes eval items by it. `scripts/bench/merge_train.py` keeps
        the same key for raw rows, and `tests/test_item_ids.py` asserts the
        two agree.
        """
        return frozenset(self.gold_path.nodes)

    def answer_set(self) -> list[str]:
        """This item's gold answers, under `answer_set_of`'s precedence."""
        return answer_set_of(self.meta, self.gold_answer)


class CitedGraph(BaseModel):
    """The <graph_json> block a model emits. Tolerant of node shapes."""
    nodes: list[str] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)

    @field_validator("nodes", mode="before")
    @classmethod
    def _coerce_nodes(cls, v: Any) -> list[str]:
        out = []
        for item in v or []:
            if isinstance(item, dict):
                out.append(str(item.get("id", item.get("name", ""))))
            else:
                out.append(str(item))
        return [x for x in out if x]

    @field_validator("edges", mode="before")
    @classmethod
    def _coerce_edges(cls, v: Any) -> list[dict]:
        out = []
        for item in v or []:
            if isinstance(item, dict):
                src = item.get("source", item.get("from", item.get("src")))
                tgt = item.get("target", item.get("to", item.get("dst")))
                rel = item.get("relation", item.get("rel", item.get("label", "related to")))
                if src is not None and tgt is not None:
                    out.append({"source": str(src), "relation": str(rel), "target": str(tgt)})
        return out
