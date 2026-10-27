"""Stored rollouts -> yes/no verifier rows, labelled without a judge.

A row's label is `gated_acc`, which the eval already stored: `gate_pass` and the
claim-scoped `em_claim`, computed by set membership and string match against
the gold answers.

Rows come only from asserted, gate-passing rollouts, the candidates the
selector offers the verifier (`pool.offered` drops gate-fails whenever any
candidate passes). `banned` holds the held-out eval items, and a rollout on one
aborts the build (`benchmarks.assert_no_leak`), so the verifier is scored on
items it never trained on.

Nothing here reweights the labels. The label prior should equal the base rate
the verifier meets at selection, so the classes are not balanced.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..benchmarks import assert_no_leak, split_run_name
from .prompts import NO, VERIFIER_SCHEMA, YES, render_prompt


def build_rows(run_dirs: list[Path], banned: set[str]) -> tuple[list[dict], Counter]:
    """One row per asserted, gate-passing rollout with a question, labelled
    YES when it is `gated_acc`; `stats` counts what was skipped and why."""
    from . import pool as bn
    stats: Counter = Counter()
    rows: list[dict] = []
    ids_read: list[str] = []
    for d in run_dirs:
        tag = split_run_name(d.name)[0]
        for rec in bn.load(d).values():
            stats["seen"] += 1
            ids_read.append(rec["id"])
            if rec["id"] in banned:
                continue
            if not rec["asserted"]:
                stats["skip_abstained"] += 1
                continue
            if not rec["gate_pass"]:
                stats["skip_gate_fail"] += 1
                continue
            if not rec["question"]:
                # A blank question renders a prompt with nothing to verify
                # against; refuse rather than train on it.
                stats["skip_no_question"] += 1
                continue
            good = bool(rec["gated_acc"])
            label = YES if good else NO
            stats[f"label_{label}"] += 1
            rows.append({
                "schema": VERIFIER_SCHEMA,
                "id": f"{rec['id']}#{tag}",
                "item_id": rec["id"],
                "tier": rec["tier"],
                "turn": 1,
                "prompt": [{"role": "user", "content": render_prompt(
                    rec["question"], rec["answer"], rec["cited"])}],
                "completion": [{"role": "assistant", "content": label}],
                "label": good,
                "source": "verifier",
            })
    assert_no_leak(ids_read, banned)
    return rows, stats


def dedupe(rows: list[dict]) -> tuple[list[dict], int, int]:
    """Collapse exact (prompt, label) duplicates; keep and count conflicts."""
    seen: dict[tuple[str, str], dict] = {}
    prompts: dict[str, set[str]] = {}
    kept = []
    dups = 0
    for r in rows:
        key = (r["prompt"][0]["content"], r["completion"][0]["content"])
        if key in seen:
            dups += 1
            continue
        seen[key] = r
        prompts.setdefault(key[0], set()).add(key[1])
        kept.append(r)
    conflicts = sum(1 for labels in prompts.values() if len(labels) > 1)
    return kept, dups, conflicts
