#!/usr/bin/env python
"""Sample continuations of stored prefixes with a local policy: the band
measurement a decision round starts from.

    python scripts/traj/sample_prefixes.py \\
        --prefixes data/traj/tsmc_prefixes_q2bd1.jsonl \\
        --benchmark data/benchmarks/tsmc/tsmc_train_agentic_4x.jsonl \\
        --model vllm:Qwen/Qwen3.5-2B --adapter models/traj_sft_2b_flash20g2/final \\
        --temperature 0.7 --n 8 --seed 911 \\
        --turn-max-tokens 1024 --name q2bd1-s911 [--limit 200]

(the 2B's first decision band, as `prospector run` draws it).

Each prefix is resumed `--n` times through the rollout loop (`agent/loop.py`:
`seed_episode`, `resume_episodes_batched`): the policy sees the transcript up
to the cut and may answer at once or keep calling tools against the graph,
under the same budgets. Every continuation is scored by `agentic_metrics`,
like an evaluation episode, and written to
`<out-root>/<name>--<prefixes stem>/` as results.jsonl, trajectories.jsonl,
completions.jsonl and summary.json. Rows carry the prefix id as `id` and the
draw as `sample`; `train/rollout.py` (`load_train_prefixes`) reads their
`gated_acc` column as the prefix band.

The call budget is read from the prefixes, which must all record the same
one. `--turn-max-tokens` is a turn's token budget (default 1,024, the trained
policies'): an answer cut off by it can fail the gate, so the budget affects
which prefixes fall in the band. Only the `vllm:` engine is supported.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import time
from pathlib import Path

from graph_reasoner.agent.backend import available_backends, context_window, load_backend, resolve_kb
from graph_reasoner.agent.eval import build_generate_seam, infer_backend_kind
from graph_reasoner.agent.loop import (EpisodeConfig, resume_episodes_batched,
                                       seed_episode, trajectory_record)
from graph_reasoner.agent.metrics import agentic_metrics
from graph_reasoner.benchmarks import load_benchmark, run_dir_name
from graph_reasoner.files import write_text_atomic
from graph_reasoner.train.prefixes import load_prefixes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefixes", required=True)
    ap.add_argument("--benchmark", required=True,
                    help="the items the prefixes were rolled on (gold for scoring)")
    ap.add_argument("--model", required=True, help="vllm:<base> (the only engine wired here)")
    ap.add_argument("--adapter", default=None, help="comma-separated LoRA chain")
    ap.add_argument("--chat-template", default=None)
    ap.add_argument("--kb", default=None,
                    help="graph file; defaults to the one the substrate declares")
    ap.add_argument("--graph", default=None, choices=available_backends(),
                    help="the substrate; inferred from --benchmark when absent, as the eval does")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--n", type=int, default=8, help="continuations per prefix")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="first N prefixes (smoke)")
    ap.add_argument("--chunk", type=int, default=512, help="episodes per driver batch")
    ap.add_argument("--turn-max-tokens", type=int, default=EpisodeConfig.turn_max_tokens,
                    help="tokens one assistant turn may use (default: the trained "
                         f"policies' budget, {EpisodeConfig.turn_max_tokens:,})")
    ap.add_argument("--final-max-tokens", type=int, default=EpisodeConfig.final_max_tokens)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-root", default="data/runs")
    args = ap.parse_args()

    if not args.model.startswith("vllm:"):
        raise SystemExit("only the vllm: engine is wired here (prefix caching is the point)")
    prefixes = load_prefixes(args.prefixes)
    if args.limit:
        prefixes = prefixes[:args.limit]
    if not prefixes:
        raise SystemExit("no prefixes selected")
    budgets = {int((p.get("policy") or {}).get("max_turns", 20)) for p in prefixes}
    if len(budgets) != 1:
        raise SystemExit(f"prefixes announce several budgets {sorted(budgets)}; one run, one budget")
    max_turns = budgets.pop()
    bench_items = load_benchmark(args.benchmark)
    items = {i.id: i for i in bench_items}
    missing = [p["id"] for p in prefixes if p["item_id"] not in items]
    if missing:
        raise SystemExit(f"{len(missing)} prefixes name items absent from {args.benchmark}")

    kind = (infer_backend_kind(args.benchmark, bench_items) if args.graph is None
            else args.graph)
    kb = resolve_kb(kind, args.kb)
    graph = load_backend(kind, kb)
    cfg = EpisodeConfig(max_turns=max_turns, temperature=args.temperature, seed=args.seed,
                        turn_max_tokens=args.turn_max_tokens,
                        final_max_tokens=args.final_max_tokens)
    # The evaluation's generate seam: one engine and one generate contract
    # for a run and a resume.
    generate = build_generate_seam(args.model, cfg, adapter=args.adapter,
                                   chat_template=args.chat_template, gpu_frac=args.gpu_frac,
                                   max_model_len=context_window(kind))

    print(f"{len(prefixes)} prefixes x {args.n} draws | max_turns={max_turns} | "
          f"T={args.temperature} | model={args.model} adapter={args.adapter}")
    out_dir = Path(args.out_root) / run_dir_name(args.name, args.prefixes)
    out_dir.mkdir(parents=True, exist_ok=True)
    f_res = (out_dir / "results.jsonl").open("w", encoding="utf-8")
    f_traj = (out_dir / "trajectories.jsonl").open("w", encoding="utf-8")
    f_comp = (out_dir / "completions.jsonl").open("w", encoding="utf-8")

    started = time.monotonic()
    per_prefix: dict[str, list[bool]] = collections.defaultdict(list)
    by_class: dict[str, list[bool]] = collections.defaultdict(list)
    new_turns: list[int] = []
    answered_at_once = 0
    n_done = 0
    # Prefix-major, draws adjacent (a,a,...,b,b,...) so a chunk shares its
    # prefixes' KV cache across the n draws.
    work = [(p, k) for p in prefixes for k in range(args.n)]
    for start in range(0, len(work), args.chunk):
        batch = work[start:start + args.chunk]
        seeded = []
        for p, k in batch:
            per_draw = EpisodeConfig(max_turns=max_turns, temperature=args.temperature,
                                     seed=args.seed * 1000 + k,
                                     turn_max_tokens=args.turn_max_tokens,
                                     final_max_tokens=args.final_max_tokens)
            seeded.append(seed_episode(p, items[p["item_id"]], per_draw, episode_id=p["id"]))
        episodes = resume_episodes_batched(seeded, graph, generate, cfg)
        for (p, k), ep in zip(batch, episodes):
            item = items[p["item_id"]]
            m = agentic_metrics(ep, item)
            n_new = ep.n_turns - ep.resumed_turns
            row = {"id": p["id"], "sample": k, "tier": ep.tier, "class": p.get("class"),
                   "item_id": p["item_id"], "n_new_turns": n_new,
                   "n_new_calls": ep.n_calls - len([c for c in p.get("calls", [])
                                                    if int(c["turn"]) <= p["n_prefix_turns"]])}
            row.update(m)
            f_res.write(json.dumps(row, ensure_ascii=False) + "\n")
            rec = trajectory_record(ep, item, graph, cfg, model=args.model, adapter=args.adapter)
            rec.update({"sample": k, "class": p.get("class"), "item_id": p["item_id"],
                        "prefix_id": p["id"], "resumed_turns": ep.resumed_turns,
                        "source_run": p.get("source_run")})
            f_traj.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_comp.write(json.dumps({"id": p["id"], "sample": k, "completion": ep.final_text},
                                    ensure_ascii=False) + "\n")
            ok = bool(m["gated_acc"])
            per_prefix[p["id"]].append(ok)
            by_class[p.get("class") or "?"].append(ok)
            new_turns.append(n_new)
            answered_at_once += int(n_new == 1)
        n_done += len(batch)
        el = time.monotonic() - started
        print(f"  {n_done}/{len(work)} continuations | {el/60:.1f} min | "
              f"pass so far {sum(b for v in per_prefix.values() for b in v) / n_done:.3f}")
        for fh in (f_res, f_traj, f_comp):
            fh.flush()
    for fh in (f_res, f_traj, f_comp):
        fh.close()

    rates = {pid: sum(v) / len(v) for pid, v in per_prefix.items()}
    hist = collections.Counter(min(int(r * 5), 4) for r in rates.values())
    cls_rates = {c: (sum(v) / len(v), len(v)) for c, v in by_class.items()}
    band = sum(1 for r in rates.values() if 0.05 <= r <= 0.95)
    summary = {
        "model": args.model, "adapter": args.adapter, "chat_template": args.chat_template,
        "prefixes": args.prefixes, "benchmark": args.benchmark, "kb_path": kb,
        "backend": graph.name, "kg_fingerprint": graph.fingerprint(),
        "max_model_len": getattr(generate, "max_model_len", None),
        "policy": cfg.policy_dict(), "n": args.n, "temperature": args.temperature,
        "seed": args.seed, "limit": args.limit,
        "n_prefixes": len(prefixes), "n_continuations": len(work),
        "wall_s": round(time.monotonic() - started, 1),
        "aggregate": {
            "pass_rate": sum(b for v in per_prefix.values() for b in v) / max(1, len(work)),
            "by_class": {c: {"pass_rate": r, "n": n} for c, (r, n) in cls_rates.items()},
            "prefix_rate_hist_fifths": {f"[{b/5:.1f},{(b+1)/5:.1f})": hist[b] for b in range(5)},
            "in_band_0.05_0.95": band,
            "answered_at_once_share": answered_at_once / max(1, len(work)),
            "new_turns_mean": statistics.mean(new_turns) if new_turns else 0.0,
        },
    }
    # Written last and whole: the recipe takes this file as the band's completion.
    write_text_atomic(out_dir / "summary.json", json.dumps(summary, indent=2))
    a = summary["aggregate"]
    print(f"\n{out_dir}")
    print(f"pass {a['pass_rate']:.3f} over {len(work)} continuations; "
          f"answered at once {a['answered_at_once_share']:.2f}; new turns mean {a['new_turns_mean']:.2f}")
    for c, d in a["by_class"].items():
        print(f"  class {c:9s} pass {d['pass_rate']:.3f}  (n={d['n']})")
    print(f"  per-prefix rate fifths: {a['prefix_rate_hist_fifths']}  in band: {band}/{len(rates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
