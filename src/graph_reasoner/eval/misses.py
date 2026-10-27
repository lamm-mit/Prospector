"""The three buckets a headline miss falls into.

A miss is `not gated_acc`, recomputed from the stored answer and gate verdict
against the answer set passed in. The buckets, tested in this order:

    gate    `gate_reason` is in GATE_REASONS, a citation-honesty failure.
            `no_citation` is not one of them: a final answer with no cited
            block falls into one of the other two buckets.
    search  the episode never reached the answer: no member of the answer set
            is a normalized endpoint of a seen edge (recomputed from the
            trajectory, not read from a stored flag).
    rival   the rest: the episode saw an answer-set member and still missed.

The paper names the three buckets citation, retrieval and decision misses,
in that order. `scripts/analysis/diagnose_miss_buckets.py` joins a run's two
files and calls `classify` per item.
"""
from __future__ import annotations

from collections import Counter

from ..agent.metrics import em_claim_of, reached

#: The gate reasons that are citation-honesty failures.
GATE_REASONS = {"hallucinated_citation", "answer_not_in_cited"}


def classify(res: dict, traj: dict, answers: list[str]) -> dict:
    """One item's stored results row and trajectory -> `{hit, bucket,
    reached, ev_complete, at_cap}`. `bucket` is None for a hit. Pure."""
    hit = bool(res.get("gate_pass")) and em_claim_of(res.get("answer") or "", answers)
    saw = reached(answers, traj.get("seen_edges", []))
    if hit:
        bucket = None
    elif res.get("gate_reason") in GATE_REASONS:
        bucket = "gate"
    elif not saw:
        bucket = "search"
    else:
        bucket = "rival"
    return {"hit": hit, "bucket": bucket, "reached": saw,
            "ev_complete": res.get("gold_complete_at_call") is not None,
            "at_cap": res.get("status") == "budget"}


def stable_misses(per_run: dict[str, dict[str, dict]]) -> dict[str, str]:
    """item id -> its majority bucket, for the items every run scored and a
    strict majority of runs missed. `per_run` maps a run to its `classify`
    rows by item id. Greedy decoding flips items run to run, so this is the
    miss set that holds across runs."""
    ids = set.intersection(*(set(rows) for rows in per_run.values()))
    out: dict[str, str] = {}
    for iid in sorted(ids):
        reads = [rows[iid] for rows in per_run.values()]
        if sum(not r["hit"] for r in reads) * 2 > len(reads):
            buckets = Counter(r["bucket"] for r in reads if not r["hit"])
            out[iid] = buckets.most_common(1)[0][0]
    return out


__all__ = ["GATE_REASONS", "classify", "stable_misses"]
