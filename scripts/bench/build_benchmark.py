#!/usr/bin/env python
"""Build a tiered benchmark from the KG: sample gold paths and have a teacher
model phrase a question for each. The first of three steps (README, "Training on
your own graph"); scripts/bench/curate_core.py and
scripts/bench/augment_agentic.py are the other two.

Usage:
    # template questions, no API calls: the plumbing check
    python scripts/bench/build_benchmark.py --config configs/benchmark/tsmc_v0.yaml \\
        --offline --limit 4 --out /tmp/v0_smoke.jsonl
    # the teacher writes the questions (spends API credit)
    python scripts/bench/build_benchmark.py --config configs/benchmark/tsmc_v0.yaml \\
        --out /tmp/v0.jsonl

Without `--out` the file goes to the config's `output`, the graph's raw
question file (`<graph>_v0.jsonl`, not tracked) that the curation step reads.
An `--offline` build writes template questions and needs `--out`, so it never
writes them to that path by default.

Without `--offline` this calls the config's `teacher_model` for every item, up
to three times each, and spends API credit.
"""
import argparse

import yaml

from graph_reasoner.bench import build_benchmark


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the graph's build, configs/benchmark/<kg>_v0.yaml")
    ap.add_argument("--offline", action="store_true",
                    help="Template questions, no API calls (plumbing tests); "
                         "needs --out")
    ap.add_argument("--limit", type=int, default=None, help="Items per tier (overrides config)")
    ap.add_argument("--out", default=None, help="Output JSONL path (overrides config)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.offline and not args.out:
        ap.error(f"--offline writes template questions; pass --out rather than "
                 f"overwrite the config's output {config.get('output')!r}")

    build_benchmark(config, offline=args.offline, limit_per_tier=args.limit, out_path=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
