#!/usr/bin/env python
"""Stored train-side rollouts -> yes/no verifier SFT rows.

    python scripts/select/build_verifier_data.py \\
        --rollouts data/runs/q4bvroll-s831--tsmc_train_agentic_4x \\
        --rollouts data/runs/q4bvroll-s832--tsmc_train_agentic_4x ... \\
        --exclude data/benchmarks/tsmc/tsmc_core_agentic.jsonl \\
        --out data/traj/tsmc_verifier_yesno_4b_flash20v.jsonl

Writes the rows and `<out>.provenance.json` beside them (the arguments, the
source runs as their summaries describe them, and the counts printed).

A row is one asserted, gate-passing rollout: its question, answer and cited
chain, labelled yes when the rollout is `gated_acc` (gate_pass and the
claim-scoped em_claim), a label every results.jsonl already stores. No judge
and no API call is involved. Abstentions and gate-fails are left out because
the selector (scripts/select/verifier_select.py) drops them before scoring
whenever a gate-passing candidate remains. The label rule and the pool are
defined in `graph_reasoner.select.verifier_data`.

The prompt is rendered by `graph_reasoner.select.prompts`, the code the scorer
renders with, so the training and scoring prompts cannot differ.

Exact (prompt, label) duplicates are collapsed; a prompt that occurs with both
labels is counted and kept. A rollout on an `--exclude` item aborts the build.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from graph_reasoner import provenance
from graph_reasoner.benchmarks import excluded_ids
from graph_reasoner.files import write_jsonl_atomic
from graph_reasoner.select.verifier_data import build_rows, dedupe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", action="append", type=Path, required=True,
                    help="run dir with results.jsonl + trajectories.jsonl (repeatable)")
    ap.add_argument("--exclude", action="append", type=Path, default=[],
                    help="benchmark whose item ids must not appear (repeatable); "
                         "a rollout on one aborts the build")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rows, stats = build_rows(a.rollouts, excluded_ids(a.exclude))
    if not rows:
        raise SystemExit(f"no rows: {stats['seen']} rollouts read, none asserted, "
                         f"gate-passing and with a question")
    rows, dups, conflicts = dedupe(rows)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(a.out, rows)

    n_yes = sum(1 for r in rows if r["label"])
    max_chars = max(len(r["prompt"][0]["content"]) for r in rows)
    prov = provenance.block(
        "scripts/select/build_verifier_data.py", sys.argv[1:],
        source_runs=a.rollouts,
        extra={"exclude": [str(p) for p in a.exclude], "label": "gated_acc",
               "rows": len(rows), "positive": n_yes,
               "positive_share": round(n_yes / len(rows), 4),
               "duplicates_collapsed": dups, "conflicting_prompts": conflicts,
               "counts": dict(stats)})
    sidecar = provenance.write_sidecar(a.out, prov)

    print(f"{len(rows)} rows -> {a.out}  ({n_yes} yes / {len(rows) - n_yes} no, "
          f"{n_yes / len(rows):.3f} positive)")
    print(f"  from {stats['seen']} rollouts: {stats['skip_abstained']} abstained "
          f"skipped, {stats['skip_gate_fail']} gate-fail skipped, "
          f"{stats['skip_no_question']} question-less skipped, "
          f"{dups} exact duplicates collapsed, {conflicts} conflicting-label "
          f"prompts KEPT (real label noise)")
    print(f"  max prompt {max_chars} chars — check the config's max_length "
          f"covers it; run_sft drops a longer row and prints how many")
    print(f"provenance -> {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
