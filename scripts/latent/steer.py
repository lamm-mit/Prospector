#!/usr/bin/env python
"""Run a benchmark through the served policy with a direction added at one
decoder layer, one arm per dose, each an ordinary eval run. A GPU job on vLLM.

    python scripts/latent/steer.py --benchmark data/benchmarks/tsmc/tsmc_core_agentic.jsonl \\
        --model Qwen/Qwen3.5-4B --adapter <stack> \\
        --directions data/runs/_probe/pilot/directions_L20.npz --layer 20 \\
        --arms dose0 fol:+1 fol:-1 ans:+0.5 rndf:+1 --out-root data/runs/_probe/knob_vllm

The arms, the seed rule, the hook and what it rests on are described in
`graph_reasoner.latent.steer`. `--directions` is the npz
scripts/latent/directions.py wrote; its json of gaps sits beside it. The
served stack is merged once into `<last adapter>/vllm_merged`, the merge the
vllm: driver serves. Each arm writes `<out-root>/knob-<arm>--<benchmark>/`
with the eval's files and knob.json; an arm whose directory holds a summary
is skipped, so a stopped invocation resumes. The engine's window is
`--max-model-len`, else the one the benchmark's substrate declares (32,768 for
PFAS), else 16,384.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from graph_reasoner.agent.backend import context_window
from graph_reasoner.benchmarks import substrate_of
from graph_reasoner.latent.steer import SteeredVLLM, parse_arm, read_seed, run_arms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, help="the evaluation questions (data/benchmarks/<kg>/<stem>.jsonl)")
    ap.add_argument("--model", required=True, help="the base model")
    ap.add_argument("--adapter", default=None, help="the served adapter chain (comma-separated)")
    ap.add_argument("--chat-template", default=None, help="a named template to install (plain_chatml)")
    ap.add_argument("--directions", type=Path, required=True, help="directions_L<layer>.npz; its .json sits beside it")
    ap.add_argument("--layer", type=int, default=20, help="the decoder layer the vector is added after (1-based)")
    ap.add_argument("--arms", nargs="+", required=True, help="dose0, fol:<d>, ans:<d>, rnd:<d>, rndf:<d>, rnd<k>:<d>, rndf<k>:<d>")
    ap.add_argument("--out-root", type=Path, required=True, help="where the arms' run directories go")
    ap.add_argument("--prefix", default="knob-", help="the run directories' prefix")
    ap.add_argument("--limit", type=int, default=None, help="items per tier, as run_agent_eval.py's --limit")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="the engine's context window (default: the substrate's, else 16384)")
    ap.add_argument("--gpu-frac", type=float, default=0.85, help="vLLM's gpu_memory_utilization")
    ap.add_argument("--seed", type=int, default=None, help="the episode seed (default: N from an out-root ending in _r<N>, else 0)")
    a = ap.parse_args()
    arms = [parse_arm(s) for s in a.arms]
    seed = read_seed(a.seed, a.out_root)
    directions_json = a.directions.with_suffix(".json")
    if not directions_json.exists():
        raise SystemExit(f"{directions_json} is missing; it is written beside the npz by scripts/latent/directions.py")
    engine = SteeredVLLM(a.model, a.layer, adapter=a.adapter, chat_template=a.chat_template,
                         max_model_len=a.max_model_len or context_window(substrate_of(a.benchmark)) or 16384,
                         gpu_frac=a.gpu_frac, seed=seed)
    return run_arms(engine, arms, benchmark=a.benchmark, directions=a.directions, directions_json=directions_json,
                    out_root=a.out_root, seed=seed, prefix=a.prefix, limit=a.limit, model=a.model, adapter=a.adapter,
                    chat_template=a.chat_template, gpu_frac=a.gpu_frac, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    raise SystemExit(main())
