"""Agentic metrics.

`mechanical_metrics` takes the edges the model was allowed to cite from. In the
agentic task that is the episode's seen set instead of a provided subgraph, so
`halluc_strict` is the share of cited edges no tool returned. This module adds
the exploration measures (whether the search reached the evidence, and how
many calls and edges it used) and the headline `gated_acc`.
"""
from __future__ import annotations

import statistics

from ..eval.metrics import mechanical_metrics
from ..eval.parse import (detect_abstention, extract_claim, extract_final_answer,
                          extract_graph_json)
from ..kg.serialize import edge_key, normalize
from ..schema import BenchmarkItem, Edge, answer_set_of
from .gate import edge_dicts, gate_assertion, gated_fields
from .loop import Episode


def seen_nodes(edges) -> set[str]:
    """The normalized endpoints of a set of edges, as `SeenEdge`s or as the
    stored `{source, relation, target}` dicts."""
    out: set[str] = set()
    for e in edges:
        s, t = (e["source"], e["target"]) if isinstance(e, dict) else (e.source, e.target)
        out.add(normalize(s))
        out.add(normalize(t))
    return out


def reached(answers: list[str], edges) -> bool:
    """Did the search reach an answer: some gold answer is a normalized
    endpoint of an edge it saw. The `reached_answer` column, and the miss
    buckets' search/rival split (`eval/misses.py`)."""
    nodes = seen_nodes(edges)
    return any(normalize(a) in nodes for a in answers)


def score_answer(answer: str, answers: list[str]) -> bool:
    """EM, judge-free: some gold answer is a normalized substring of `answer`.
    `em_claim_of` applies it to the claim alone; the headline is that under
    the gate."""
    na = normalize(answer or "")
    return any(normalize(a) in na for a in answers)


def em_claim_of(answer: str, answers: list[str]) -> bool:
    """EM on what the answer asserts: the text after its last "the answer is"
    (eval/parse.extract_claim), or the whole answer when the marker is absent.
    `answer` is what `eval/parse.extract_final_answer` returns: everything
    from `ANSWER:` to the end of the final turn, the `<graph_json>` block
    removed, which can run past the ANSWER line. The headline accuracy is
    gated_acc = em_claim and gate_pass."""
    return score_answer(extract_claim(answer), answers)


def agentic_metrics(ep: Episode, item: BenchmarkItem) -> dict:
    seen = [Edge(source=e.source, relation=e.relation, target=e.target)
            for e in ep.seen.values()]
    cited = extract_graph_json(ep.final_text)
    out = dict(mechanical_metrics(cited, seen, item.gold_path.edges))

    answers = item.answer_set()
    answer = extract_final_answer(ep.final_text)
    em_claim = em_claim_of(answer, answers)

    gold_keys = {edge_key(e.source, e.relation, e.target) for e in item.gold_path.edges}
    seen_keys = set(ep.seen)

    out.update({
        "status": ep.status,
        "answer": answer,
        "em_claim": em_claim,
        "abstained": detect_abstention(answer),
        "evidence_recall": len(gold_keys & seen_keys) / len(gold_keys) if gold_keys else 0.0,
        "reached_answer": reached(answers, ep.seen.values()),
        "n_turns": ep.n_turns,
        "n_calls": ep.n_calls,
        "n_tool_errors": ep.n_tool_errors,
        "n_malformed": ep.n_malformed,
        "n_repeats": ep.n_repeats,
        "n_edges_seen": len(ep.seen),
        "gold_complete_at_call": ep.gold_complete_at_call,
    })

    # The gate (agent/gate.py) reads the final turn whatever the episode's
    # status: a forced final answer that cites seen edges can still pass.
    cited_dicts = edge_dicts(cited.edges if cited else None)
    seen_dicts = edge_dicts(seen)
    out.update(gated_fields(gate_assertion(answer, cited_dicts, seen_dicts)))
    # The headline: the gate passed and the asserted claim names a gold answer.
    out["gated_acc"] = bool(out["gate_pass"] and em_claim)
    return out


_MEAN_KEYS = (
    "gated_acc", "gate_pass", "em_claim",
    "gated_abstained", "abstained",
    "halluc_strict", "path_recall", "evidence_recall", "reached_answer",
    "n_calls", "n_edges_seen", "n_repeats", "n_malformed", "n_tool_errors",
)
_MEDIAN_KEYS = ("n_edges_seen", "n_calls")


def _mean(values):
    vals = [float(v) for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def _median(values):
    vals = [float(v) for v in values if v is not None]
    return statistics.median(vals) if vals else None


def aggregate_agentic(results: list[dict]) -> dict:
    """Per tier plus ALL. The cost columns get medians as well as means: cost is
    long-tailed and censored at the budget, so compare policies on the median."""
    out: dict[str, dict] = {}
    tiers = sorted({r["tier"] for r in results})
    for tier in [*tiers, "ALL"]:
        rows = results if tier == "ALL" else [r for r in results if r["tier"] == tier]
        if not rows:
            continue
        agg = {"n": len(rows)}
        for key in _MEAN_KEYS:
            agg[key] = _mean([r.get(key) for r in rows])
        for key in _MEDIAN_KEYS:
            agg[f"median_{key}"] = _median([r.get(key) for r in rows])
        out[tier] = agg
    return out


def warn_on_starved_turns(episodes, *, threshold: float = 0.05) -> str | None:
    """A warning when at least `threshold` of the assistant turns were empty,
    else None. An empty turn means the model spent its token budget before
    emitting anything, which the metrics alone show as protocol failure."""
    turns = sum(ep.n_turns for ep in episodes)
    empty = sum(ep.n_empty for ep in episodes)
    if not turns or empty / turns < threshold:
        return None
    return (f"WARNING: {empty}/{turns} assistant turns ({empty / turns:.0%}) were empty — "
            f"the model is running out of tokens before it emits anything. "
            f"Raise --turn-max-tokens (reasoning models need 3000+) and re-run; "
            f"these results understate the model.")


def print_agentic_summary(model: str, agg: dict) -> None:
    """One line per tier, led by the headline gated_acc. `prospector` echoes the
    `ALL` line whole from a run's log (recipe/run.py after each read,
    recipe/teach.py after each teacher pass); nothing parses its columns."""
    print(f"\n=== {model} ===")
    header = (f"{'tier':8s} {'n':>4s} {'acc':>6s} {'gate':>6s} {'claim':>6s} {'halluc':>7s} "
              f"{'recall':>7s} {'evid':>6s} {'reach':>6s} {'calls':>6s} {'edges':>7s}")
    print(header)
    for tier, a in agg.items():
        def fmt(key, spec=".3f"):
            v = a.get(key)
            return format(v, spec) if v is not None else "  -  "
        print(f"{tier:8s} {a['n']:4d} {fmt('gated_acc'):>6s} {fmt('gate_pass'):>6s} "
              f"{fmt('em_claim'):>6s} "
              f"{fmt('halluc_strict'):>7s} {fmt('path_recall'):>7s} "
              f"{fmt('evidence_recall'):>6s} {fmt('reached_answer'):>6s} "
              f"{fmt('median_n_calls', '.1f'):>6s} {fmt('median_n_edges_seen', '.1f'):>7s}")


__all__ = ["agentic_metrics", "aggregate_agentic", "answer_set_of",
           "em_claim_of", "print_agentic_summary", "reached", "score_answer",
           "seen_nodes", "warn_on_starved_turns"]
