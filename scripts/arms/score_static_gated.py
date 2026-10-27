#!/usr/bin/env python
"""Score a static run (subgraph in the prompt, one call per question) under the
agentic runs' rule: the citation gate plus claim-scoped EM, `gated_acc`.

    python scripts/arms/score_static_gated.py data/runs/gpt6astra-cb2--tsmc_core__closed_book
    python scripts/arms/score_static_gated.py data/runs/gpt6astra-raganchor-label--tsmc_core__khop400anchor

The gate and the EM rule are the agentic scorer's own functions:
`gate_assertion`, with the prompt's `subgraph_edges` in place of an episode's
seen set, and `em_claim_of` against `answer_set_of(meta)`. Each completion is
joined, by the id before the `#`, to the evaluation file of the run's own
substrate (`benchmarks.answers_for`); `--benchmark` names another answer file
and is refused if it belongs to a different substrate. Completions whose id is
not in that file are skipped and counted. The shown edges come from the
benchmark recorded in the run's `summary.json`, or from `--subgraphs`.

Prints a per-tier table and writes `<run>/score_static_gated.json`.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from graph_reasoner import benchmarks
from graph_reasoner.agent.gate import edge_dicts, gate_assertion
from graph_reasoner.agent.metrics import answer_set_of, em_claim_of
from graph_reasoner.eval.parse import extract_final_answer, extract_graph_json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--benchmark", type=Path, default=None,
                    help="answer-set source of truth; defaults to the agentic "
                         "benchmark of the substrate the run was scored on")
    ap.add_argument("--subgraphs", type=Path, default=None,
                    help="benchmark the run consumed (for the shown edges); "
                         "default: the `benchmark` path in the run's summary.json")
    a = ap.parse_args()

    sub_path = a.subgraphs
    if sub_path is None:
        # The path recorded in summary.json may not exist in this checkout;
        # the benchmark is then resolved by name.
        summ = json.loads((a.run / "summary.json").read_text())
        recorded = benchmarks.ROOT / str(summ["benchmark"]).replace("\\", "/")
        sub_path = recorded if recorded.exists() else benchmarks.resolve(recorded)

    # The answer sets follow the run's substrate; an explicit --benchmark from
    # another substrate is refused.
    bench_path = a.benchmark or benchmarks.answers_for(sub_path)
    if a.benchmark is not None:
        here, there = (benchmarks.substrate_of(sub_path),
                       benchmarks.substrate_of(a.benchmark))
        if here and there and here != there:
            raise SystemExit(
                f"refusing to score a {here} run against {there} answer sets: "
                f"{a.benchmark}")
    with bench_path.open() as f:
        current = {r["id"]: r for r in map(json.loads, f)}
    with sub_path.open() as f:
        shown = {r["id"]: r.get("subgraph_edges") or [] for r in map(json.loads, f)}
    with (a.run / "completions.jsonl").open() as f:
        completions = f.readlines()

    per_tier: dict[str, list[dict]] = {}
    skipped = 0
    reasons: Counter = Counter()
    for line in completions:
        rec = json.loads(line)
        rid = rec["id"]
        base = current.get(rid.split("#")[0])
        if base is None:
            skipped += 1
            continue
        answers = answer_set_of(base.get("meta"), base.get("gold_answer"))
        answer = extract_final_answer(rec.get("completion") or "")
        cited = extract_graph_json(rec.get("completion") or "")
        cited_dicts = edge_dicts(cited.edges if cited else None)
        if rid not in shown:
            # An empty seen set would fail every citation and score the arm
            # 0.000, so an id missing from the subgraphs file is an error.
            raise SystemExit(
                f"item {rid!r} is absent from {sub_path} — that file is not the "
                f"benchmark this run consumed. Pass the right --subgraphs; "
                f"scoring against the wrong one silently zeroes the arm.")
        verdict = gate_assertion(answer, cited_dicts, shown[rid])
        em_claim = em_claim_of(answer, answers)
        reasons[verdict["gate_reason"]] += 1
        per_tier.setdefault(base["tier"], []).append({
            "id": rid.split("#")[0],
            "em_claim": em_claim,
            "gate_pass": verdict["gate_pass"],
            "gated_acc": bool(verdict["gate_pass"] and em_claim),
        })

    def agg(rows):
        n = len(rows)
        return {k: sum(r[k] for r in rows) / n if n else 0.0
                for k in ("em_claim", "gate_pass", "gated_acc")} | {"n": n}

    all_rows = [r for rows in per_tier.values() for r in rows]
    print(f"{a.run.name}  (answers: {bench_path.name}; shown edges: {sub_path.name}; "
          f"{skipped} completions outside the current benchmark skipped)")
    hdr = f"{'tier':8s} {'n':>4s} {'em_claim':>9s} {'gate_pass':>10s} {'gated_acc':>9s}"
    print(hdr)
    for tier in sorted(per_tier):
        s = agg(per_tier[tier])
        print(f"{tier:8s} {s['n']:4d} {s['em_claim']:9.3f} "
              f"{s['gate_pass']:10.3f} {s['gated_acc']:9.3f}")
    s = agg(all_rows)
    print(f"{'ALL':8s} {s['n']:4d} {s['em_claim']:9.3f} "
          f"{s['gate_pass']:10.3f} {s['gated_acc']:9.3f}")
    print("gate reasons:", dict(reasons.most_common()))

    out = a.run / "score_static_gated.json"
    out.write_text(json.dumps(
        {"benchmark": benchmarks.repo_path(bench_path), "subgraphs": benchmarks.repo_path(sub_path),
         "skipped": skipped, "ALL": s,
         "tiers": {t: agg(rows) for t, rows in per_tier.items()},
         "gate_reasons": dict(reasons),
         "items": sorted(all_rows, key=lambda r: r["id"])}, indent=1))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
