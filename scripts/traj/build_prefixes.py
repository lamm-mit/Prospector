#!/usr/bin/env python
"""Stored rollouts -> prefix records: the decision states a decision round trains on.

    python scripts/traj/build_prefixes.py \\
        --rollouts data/runs/q4bd1roll-s831--tsmc_train_agentic_4x \\
        --rollouts data/runs/q4bd1roll-s832--tsmc_train_agentic_4x \\
        --rollouts data/runs/q4bd1roll-s833--tsmc_train_agentic_4x \\
        --rollouts data/runs/q4bd1roll-s834--tsmc_train_agentic_4x \\
        --exclude data/benchmarks/tsmc/tsmc_core_agentic.jsonl \\
        --kb data/raw/GRAPHDATA_TSMC_OUTPUT/tsmc_5b10p.graphml \\
        --success-share 0.25 --seed 7 \\
        --out data/traj/tsmc_prefixes_q4bd1.jsonl    # sidecar written alongside

(the 4B's first decision round, as `prospector run` builds them).

The classes (fixable, success, search) and the cut are defined in
`graph_reasoner.traj.prefixes`. This script pairs each run's results and
trajectories, keeps `--success-share` of the passing episodes, drawn with one
seeded generator in trajectory order, and writes the records with their
`<out>.provenance.json` sidecar. The search class needs the neighbours of each
gold answer; they come from the graph the source runs recorded (`--kb`
overrides its file).

A prefix longer than `--max-chars` is dropped, so that a resumed conversation
still fits the sampler's context window. A rollout on an `--exclude` item
aborts the build; the trainer applies its own leak filter as well
(`train/rollout.py`, `load_train_prefixes`).
"""
from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path

from graph_reasoner import provenance
from graph_reasoner.agent.backend import load_backend, resolve_kb
from graph_reasoner.benchmarks import assert_no_leak, excluded_ids, split_run_name
from graph_reasoner.files import write_jsonl_atomic
from graph_reasoner.select.pool import load_pair
from graph_reasoner.traj.prefixes import (answer_set, classify, gold_neighbours,
                                          search_cut, to_prefix, to_search_prefix)


def source_backend(runs: list[str]) -> str:
    """The one substrate the source runs recorded in their summaries."""
    kinds = {provenance.describe_run(r).get("backend") for r in runs}
    if len(kinds) != 1 or None in kinds:
        raise SystemExit(f"the --rollouts record backends {sorted(map(str, kinds))}; "
                         f"one build reads one graph")
    return kinds.pop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", action="append", required=True,
                    help="run dir with trajectories.jsonl + results.jsonl (repeatable)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="benchmark whose item ids must not appear (repeatable)")
    ap.add_argument("--success-share", type=float, default=0.25,
                    help="fraction of passing episodes kept as 'success' prefixes")
    ap.add_argument("--max-chars", type=int, default=48000,
                    help="drop prefixes longer than this many characters (~12k tokens)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kb", default=None,
                    help="graph file for the 'search' class; defaults to the one "
                         "the source runs' substrate declares")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    kind = source_backend(args.rollouts)
    kb = resolve_kb(kind, args.kb)
    graph = load_backend(kind, kb)
    neigh_cache: dict[str, set[str]] = {}

    def neighbours(answers: set[str]) -> set[str]:
        key = "|".join(sorted(answers))
        if key not in neigh_cache:
            neigh_cache[key] = gold_neighbours(graph, answers)
        return neigh_cache[key]

    banned = excluded_ids(args.exclude)
    rng = random.Random(args.seed)

    counts = collections.Counter()
    sub = collections.Counter()
    records = []
    ids_read: list[str] = []
    for d in args.rollouts:
        run = Path(d)
        tag = split_run_name(run.name)[0]
        # Paired strictly by id, and walked in trajectory order: the success
        # draw below consumes the seeded generator in that order.
        res, traj = load_pair(run)
        for rec in traj.values():
            r = res[rec["id"]]
            counts["episodes"] += 1
            ids_read.append(rec["id"])
            if rec["id"] in banned:
                continue
            cls, extra = classify(rec, r)
            if cls == "dropped":
                cut = search_cut(rec, neighbours(answer_set(rec)))
                if cut is None:
                    counts["dropped: failed, never adjacent to gold"] += 1
                    continue
                p = to_search_prefix(rec, run_tag=tag, cut_turn=cut[0],
                                     hops_short=cut[1], extra=extra)
                if p is None or p["n_chars"] > args.max_chars:
                    counts["dropped: search prefix unusable/over max-chars"] += 1
                    continue
                counts["kept search"] += 1
                records.append(p)
                continue
            if cls == "success" and rng.random() >= args.success_share:
                counts["success not sampled"] += 1
                continue
            p = to_prefix(rec, run_tag=tag, cls=cls, extra=extra)
            if p is None:
                counts["unusable shape"] += 1
                continue
            if p["n_chars"] > args.max_chars:
                counts["dropped: over max-chars"] += 1
                continue
            counts[f"kept {cls}"] += 1
            if cls == "fixable":
                sub[f"{extra['source_gate_reason']}/{extra['source_status']}"] += 1
            records.append(p)
    assert_no_leak(ids_read, banned)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.out, records)
    prov = provenance.block("scripts/traj/build_prefixes.py", sys.argv[1:],
                            source_runs=args.rollouts,
                            extra={"kb": kb, "success_share": args.success_share,
                                   "max_chars": args.max_chars, "exclude": args.exclude,
                                   "counts": dict(counts)})
    sidecar = provenance.write_sidecar(args.out, prov)

    for k, v in counts.most_common():
        print(f"  {v:6d}  {k}")
    print("fixable by source gate/status:")
    for k, v in sub.most_common():
        print(f"  {v:6d}  {k}")
    items = {p["item_id"] for p in records}
    print(f"{len(records)} prefixes over {len(items)} items -> {args.out}")
    print(f"provenance -> {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
