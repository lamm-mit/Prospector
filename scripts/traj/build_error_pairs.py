#!/usr/bin/env python
"""Build detour-recovery SFT rows from stored keeps rows.

    python scripts/traj/build_error_pairs.py \\
        --rows data/traj/tsmc_flash20_keeps.jsonl \\
        --benchmark data/benchmarks/tsmc/tsmc_train_agentic_4x.jsonl \\
        --out data/traj/tsmc_error_pairs_flash20.jsonl

At most one extra row per episode: its prompt ends in an expansion of an
off-path node, with the observation the graph backend returns for it, and its
completion recognises the dead end and takes the turn the episode took next.
The mechanism is in `graph_reasoner.traj.error_pairs`. The detour's
observation comes from the local graph backend; no model is called.

`--benchmark` must be the file the rows were rolled out from: it supplies each
item's gold path and answer set, the nodes a detour must never expand. The
substrate is inferred from it and the graph file resolved from the substrate
(`--kb` overrides).
"""
import argparse
import json
from pathlib import Path

from graph_reasoner.agent.backend import load_backend, resolve_kb
from graph_reasoner.agent.eval import infer_backend_kind
from graph_reasoner.agent.tools import ToolConfig
from graph_reasoner.files import write_jsonl_atomic
from graph_reasoner.schema import BenchmarkItem
from graph_reasoner.traj.error_pairs import build_error_pair_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, help="keeps JSONL (agent_sft_v1)")
    ap.add_argument("--benchmark", required=True,
                    help="the agentic benchmark these rows rolled out from")
    ap.add_argument("--kb", "--kg", dest="kb", default=None,
                    help="graph file; defaults to the one the inferred substrate "
                         "declares (--kg is an alias)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.rows, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    with open(args.benchmark, encoding="utf-8") as f:
        items = [BenchmarkItem(**json.loads(l)) for l in f if l.strip()]
    gold = {}
    for it in items:
        nodes = set(it.gold_path.nodes)
        nodes |= set(map(str, it.meta.get("kb_answer_set") or ()))
        gold[it.id] = nodes

    kind = infer_backend_kind(args.benchmark, items[:1])
    backend = load_backend(kind, resolve_kb(kind, args.kb))

    # The detour observation renders under the default ToolConfig, as
    # `run_agent_eval.py` serves by default. Rows rolled out under other tool
    # settings need the same settings here.
    out, tally = build_error_pair_rows(rows, backend, ToolConfig(), gold,
                                       seed=args.seed)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(dest, out)
    print(f"wrote {len(out)} rows -> {dest}")
    for k, v in sorted(tally.items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
