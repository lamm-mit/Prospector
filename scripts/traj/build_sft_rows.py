#!/usr/bin/env python
"""Verified rollouts -> SFT rows: the keep filter over stored trajectories.

    python scripts/traj/build_sft_rows.py \\
        --rollouts data/runs/flash20-teach-4x-g0--tsmc_train_agentic_4x ... (g0..g7) \\
        --rollouts data/runs/flash20-teach-4x-p2--tsmc_train_agentic_4x \\
        --exclude data/benchmarks/tsmc/tsmc_core_agentic.jsonl \\
        --cap 1 --strip-user-note --out data/traj/tsmc_flash20_keeps.jsonl   # sidecar written alongside

(the demonstrations' keeps, as `prospector teach mix` builds them; the stage-3
keeps take the policy's own rollouts with `--cap 2 --abstain-ratio 0.1`, as
`prospector run` builds them).

A rollout is kept when it is well-formed, gate-passing and right under the
claim rule (`em_claim AND gate_pass`). The filter, the abstention class and
the row shape are defined in `graph_reasoner.traj.sft_rows`. Every row carries
the producing model's id in `source`, read from the source runs' summaries,
and `<out>.provenance.json` names the source runs, their models and the git
state.

`--cap` bounds the trajectories kept per item, shortest first: easy items
succeed on most samples, and uncapped they would dominate the rows.
`--abstain-ratio` adds honest abstentions, up to that fraction of the kept
answers. `--strip-user-note` removes each run's generation-time user note
from the prompts. A rollout on an `--exclude` item aborts the build.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from graph_reasoner import provenance
from graph_reasoner.benchmarks import assert_no_leak, excluded_ids, split_run_name
from graph_reasoner.files import write_jsonl_atomic
from graph_reasoner.traj.sft_rows import (honest_abstention, strip_user_note,
                                          to_rows, verdict)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", action="append", required=True,
                    help="run directory holding trajectories.jsonl (repeatable)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="benchmark whose ids must not appear (repeatable)")
    ap.add_argument("--cap", type=int, default=2,
                    help="max retained trajectories per item (default 2)")
    ap.add_argument("--abstain-ratio", type=float, default=0.0,
                    help="honest abstentions to retain per verified answer; "
                         "0 keeps none (see honest_abstention)")
    ap.add_argument("--strip-user-note", action="store_true",
                    help="remove each run's generation-time user note (its "
                         "summary.json policy.user_note) from the prompt side "
                         "of every row, so training prompts equal deployed ones")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    banned = excluded_ids(args.exclude)

    kept_by_item: dict[str, list] = collections.defaultdict(list)
    abst_by_item: dict[str, list] = collections.defaultdict(list)
    reasons = collections.Counter()
    seen_total = notes_stripped = 0
    ids_read: list[str] = []
    for d in args.rollouts:
        p = Path(d) / "trajectories.jsonl"
        if not p.exists():
            raise SystemExit(f"no trajectories.jsonl in {d}")
        tag = split_run_name(Path(d).name)[0]
        note = None
        if args.strip_user_note:
            sp = Path(d) / "summary.json"
            note = (json.loads(sp.read_text()).get("policy") or {}).get("user_note") if sp.exists() else None
            if not note:
                print(f"note: {d} recorded no user_note; nothing to strip there")
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                seen_total += 1
                if note and strip_user_note(rec, note):
                    notes_stripped += 1
                ids_read.append(rec["id"])
                if rec["id"] in banned:
                    continue
                ok, why = verdict(rec)
                reasons[why] += 1
                if ok:
                    kept_by_item[rec["id"]].append((rec, tag))
                elif args.abstain_ratio > 0 and honest_abstention(rec)[0]:
                    abst_by_item[rec["id"]].append((rec, tag))
                    reasons["ok_abstain"] += 1

    assert_no_leak(ids_read, banned)

    rows, kept = [], 0
    for item, recs in kept_by_item.items():
        # Shortest first: of two verified trajectories for one item, the shorter
        # reached the answer in fewer turns.
        recs.sort(key=lambda r: len(r[0]["messages"]))
        for rec, tag in recs[:args.cap]:
            rows.extend(to_rows(rec, tag))
            kept += 1

    kept_abstain = 0
    if args.abstain_ratio > 0:
        # Items with no verified answer first, then the longest abstention: a
        # long one searched before it declined.
        budget = int(round(kept * args.abstain_ratio))
        order = sorted(
            ((rec, tag) for recs in abst_by_item.values() for rec, tag in recs),
            key=lambda rt: (rt[0]["id"] in kept_by_item, -len(rt[0]["messages"])))
        per_item: collections.Counter = collections.Counter()
        for rec, tag in order:
            if kept_abstain >= budget:
                break
            if per_item[rec["id"]] >= args.cap:
                continue
            per_item[rec["id"]] += 1
            rows.extend(to_rows(rec, f"{tag}-ab"))
            kept_abstain += 1

    # The producing model is read from each run's summary.json, never from
    # the directory name.
    prov = provenance.block("scripts/traj/build_sft_rows.py", sys.argv[1:],
                            source_runs=args.rollouts,
                            extra={"cap": args.cap,
                                   "abstain_ratio": args.abstain_ratio,
                                   "exclude": args.exclude,
                                   "strip_user_note": args.strip_user_note,
                                   "notes_stripped": notes_stripped})
    source = prov.get("teacher", "unknown")
    if source == "unknown":
        print("WARNING: no source run recorded a model — rows will be stamped "
              "'unknown'. Check that each --rollouts dir has a readable summary.json.",
              file=sys.stderr)
    for r in rows:
        r["source"] = source

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.out, rows)

    sidecar = provenance.write_sidecar(args.out, prov)
    print(f"teacher: {source}")
    print(f"provenance -> {sidecar}")

    tiers = collections.Counter(r["tier"] for r in rows)
    print(f"{seen_total} rollouts over {len(args.rollouts)} pass(es)"
          + (f"; user note stripped from {notes_stripped}" if args.strip_user_note else ""))
    for why, n in reasons.most_common():
        print(f"  {why:28s} {n:5d}  ({n/seen_total:.1%})")
    print(f"\nverified on {len(kept_by_item)} distinct items; "
          f"kept {kept} answered trajectories at cap {args.cap}")
    if args.abstain_ratio > 0:
        print(f"kept {kept_abstain} honest abstentions "
              f"(ratio {kept_abstain / max(1, kept):.2f} of answered)")
    print(f"{len(rows)} turn-rows -> {args.out}")
    print(f"rows by tier: {dict(tiers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
