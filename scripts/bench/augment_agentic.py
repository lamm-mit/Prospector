#!/usr/bin/env python
"""Add the fields agentic scoring needs to a curated benchmark, on any graph.

    # the evaluation questions, written to a scratch path
    python scripts/bench/augment_agentic.py \\
        --benchmark data/benchmarks/tsmc/tsmc_core.jsonl --out /tmp/tsmc_core_agentic.jsonl

    # a training batch: --exclude the evaluation file and name the batch's source
    python scripts/bench/augment_agentic.py \\
        --benchmark <train raw>.jsonl --source tsmc/train-<batch> \\
        --exclude data/benchmarks/tsmc/tsmc_core_agentic.jsonl \\
        --out data/benchmarks/tsmc/tsmc_train_agentic_<batch>.jsonl

    # a transfer graph
    python scripts/bench/augment_agentic.py \\
        --benchmark data/benchmarks/pfas/pfas_core.jsonl --out /tmp/pfas_core_agentic.jsonl

Each item gains its anchor, the anchor's surface form in the question, and
`kb_answer_set`, every answer the gold relation sequence reaches over the full
graph (`graph_reasoner.bench.agentic`). The substrate is inferred from the
benchmark and its graph file resolved with it (`--kb` overrides). An item is
dropped, and counted by reason, when its anchor cannot be found in the question
or the graph, its gold answer is not reachable by its own relations, its
question names another member of its answer set, or its gold path appears in
an `--exclude` file. The run fails if a gold-path node has
more connections than the tools' hub threshold.

The tracked `tsmc_core_agentic.jsonl` is this script's output amended by hand:
one item removed, 18 answer sets widened with another spelling of the gold
answer or with another node of the graph that answers the question (11 of
them carry a dated record under `meta.amendments`, seven carry none) and one
gold answer corrected; tests/test_eval_benchmark_integrity.py pins every
amendment. So the script refuses to overwrite a tracked benchmark without
`--force`. Merge training batches with
scripts/bench/merge_train.py, which checks disjointness by gold node set.
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
from pathlib import Path

from graph_reasoner.agent.backend import load_backend, resolve_kb
from graph_reasoner.agent.eval import infer_backend_kind
from graph_reasoner.agent.tools import ToolConfig
from graph_reasoner.bench.agentic import augment, hub_violations
from graph_reasoner.benchmarks import ROOT
from graph_reasoner.schema import BenchmarkItem


def is_tracked(path: Path) -> bool:
    """Whether git tracks `path` in this repository."""
    try:
        p = subprocess.run(["git", "ls-files", "--error-unmatch", str(path.resolve())],
                           cwd=ROOT, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return p.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, help="the curated benchmark")
    ap.add_argument("--kb", "--kg", dest="kb", default=None,
                    help="graph file; defaults to the one the inferred substrate "
                         "declares (--kg is an alias)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true",
                    help="overwrite a benchmark git tracks (the shipped eval is "
                         "hand-amended; a rebuild discards the amendments)")
    ap.add_argument("--source", default=None,
                    help="meta.source of every item (default <substrate>/core)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="benchmark(s) whose gold paths must not appear here. "
                         "Required when building a TRAIN split — a leaked gold "
                         "path turns every trained number into a memorisation "
                         "measurement, and the sampler cannot know about it.")
    args = ap.parse_args()

    dest = Path(args.out)
    if dest.exists() and is_tracked(dest) and not args.force:
        raise SystemExit(f"refusing to overwrite {dest}, a tracked benchmark: the "
                         f"shipped files carry hand amendments a rebuild discards. "
                         f"Write elsewhere, or pass --force.")

    banned: set[frozenset] = set()
    for path in args.exclude:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    banned.add(frozenset(json.loads(line)["gold_path"]["nodes"]))

    with open(args.benchmark, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    # The graph is loaded by the substrate's own backend, which knows its file format.
    kind = infer_backend_kind(args.benchmark, [BenchmarkItem(**r) for r in rows[:1]])
    G = load_backend(kind, resolve_kb(kind, args.kb)).digraph()

    out, drop, sizes = augment(rows, G, source=args.source or f"{kind}/core",
                               banned=banned)

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{len(rows)} -> {len(out)} agentic items")
    for reason, n in drop.most_common():
        print(f"   dropped {n:4d}  {reason}")
    if sizes:
        sizes.sort()
        uniq = sum(1 for s in sizes if s == 1)
        print(f"\nkb_answer_set size: median {sizes[len(sizes)//2]}, "
              f"max {sizes[-1]}, unique in {uniq}/{len(sizes)} ({uniq/len(sizes):.0%})")
    print(f"tier: {dict(sorted(collections.Counter(r['tier'] for r in out).items()))}")

    if banned:
        overlap = {frozenset(r["gold_path"]["nodes"]) for r in out} & banned
        assert not overlap, f"{len(overlap)} gold paths leaked from an excluded set"
        print(f"disjointness verified against {len(banned)} excluded gold paths")

    # Every gold node must be expandable without a relation filter
    # (`bench.agentic.hub_violations`).
    hub = ToolConfig().hub_threshold
    over = hub_violations(out, G)
    if over:
        worst = sorted(over, key=lambda x: -x[2])[:5]
        raise SystemExit(
            f"{len(over)} gold-path nodes exceed hub_threshold={hub}, so an "
            f"unfiltered expand on them returns a ranked page rather than the "
            f"whole neighbourhood: "
            + ", ".join(f"{n!r} deg {d}" for _i, n, d in worst)
            + "\nEither lower max_node_degree in the benchmark config or raise "
              "ToolConfig.hub_threshold — deliberately, and re-run every arm.")
    print(f"gold-node degree <= hub_threshold ({hub}) verified for {len(out)} items")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
