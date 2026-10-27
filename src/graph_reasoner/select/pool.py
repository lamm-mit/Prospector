"""Load a stored eval run into the rows the selectors and scorers read, and
define the offered pool and the margin pick.

`load_pair` joins the two files an eval writes, `results.jsonl` (metrics) and
`trajectories.jsonl` (answers and citations), by item id; `load` turns the pair
into one dict per item. `em_claim` and `gated_acc` depend only on the stored
answer text, the stored gate flag and the benchmark's answer sets, so `load`
can recompute them for any stored run.
"""
from __future__ import annotations

import json
from pathlib import Path


def _rows_by_id(run: Path, name: str) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    with (run / name).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in by_id:
                raise ValueError(
                    f"{run.name}/{name} repeats id {row['id']!r}: a pool is one "
                    f"run directory per seed (the 32-seed pool the recipe's `final` stage draws), each holding one "
                    f"row per item, and a directory with several rows per item "
                    f"would keep only the last of them")
            by_id[row["id"]] = row
    return by_id


def load_pair(run: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """(results rows, trajectory rows), each keyed by item id in file order.

    An id repeated within a file, or present in one file and not the other,
    raises: skipping it would shrink the item set every rate is computed over.
    """
    res = _rows_by_id(run, "results.jsonl")
    traj = _rows_by_id(run, "trajectories.jsonl")
    if res.keys() != traj.keys():
        only_res = sorted(res.keys() - traj.keys())
        only_traj = sorted(traj.keys() - res.keys())
        raise ValueError(
            f"{run.name}: {len(only_res)} id(s) only in results.jsonl "
            f"{only_res[:3]}, {len(only_traj)} only in trajectories.jsonl "
            f"{only_traj[:3]}; refusing a partial join")
    return res, traj


def load(run: Path, benchmark: str | None = None,
         recompute: bool = False) -> dict[str, dict]:
    """id -> the fields the selectors and the scorer need, joined across the two
    files an eval writes.

    Stored `em_claim`/`gated_acc` are used when the run has them. Otherwise, or
    with `recompute=True`, they are recomputed from the stored answer text
    against the benchmark the run directory's name carries (`benchmark`
    overrides it), and a row whose item is not in that benchmark is dropped.
    Stored labels are the labels at eval time; pass `recompute=True` to score
    several runs against the same, current answer sets."""
    from ..agent.metrics import em_claim_of
    from ..benchmarks import gold_answers, split_run_name

    res, traj = load_pair(run)
    bench = benchmark if benchmark is not None else split_run_name(run.name)[1]
    out = {}
    for iid, t in traj.items():
        r = res[iid]
        row = {
            "id": iid, "tier": t["tier"], "answer": t.get("answer") or "",
            # The verifier prompt shows the question.
            "question": t.get("question") or "",
            "cited": t.get("cited_edges") or [],
            "gate_pass": bool(r["gate_pass"]), "reached": bool(r["reached_answer"]),
            "abstained": bool(r["abstained"]), "n_calls": int(r["n_calls"]),
            # select/verifier_data.py builds rows from asserted rollouts only.
            "asserted": bool(r.get("asserted")),
        }
        if "em_claim" in r and not recompute:
            row["em_claim"] = bool(r["em_claim"])
            row["gated_acc"] = bool(r["gated_acc"])
        else:
            if bench is None:
                raise RuntimeError(
                    f"{run} stores no claim-scoped scores and its dir name carries no "
                    f"'--<benchmark>' suffix to recompute them from")
            # The benchmark is opened only here, to label rows for scoring;
            # no selector reads these labels.
            answers = gold_answers(bench).get(iid)
            if answers is None:
                # The item is not in the benchmark: drop its row, so every
                # consumer sees the benchmark's current item set.
                continue
            row["em_claim"] = em_claim_of(row["answer"], answers)
            row["gated_acc"] = bool(row["gate_pass"] and row["em_claim"])
        out[iid] = row
    return out


# ---------------------------------------------------------------- the offered pool

def live_indices(cands: list[dict]) -> list[int]:
    """Indices of candidates the gate passed and that did not abstain. No
    fallback: the list is empty when nothing asserted cleanly, which
    `vote.select` answers by returning the reference rollout."""
    return [n for n, c in enumerate(cands)
            if c["gate_pass"] and not c["abstained"]]


def offered(cands: list[dict]) -> list[dict]:
    """The pool a selector is offered: the gate-passing, non-abstaining
    candidates, or all of them when there are none. A verifier's stored
    `cand_idx` is a position in this list."""
    return [cands[n] for n in (live_indices(cands) or range(len(cands)))]


def pick_by_margin(margins: list[float]) -> int:
    """Index of the largest verifier margin; the earliest wins a tie."""
    if not margins:
        raise ValueError("no margins to pick from")
    best = 0
    for n, m in enumerate(margins):
        if m > margins[best]:
            best = n
    return best
