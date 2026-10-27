#!/usr/bin/env python
"""Answer a static benchmark (edges in the prompt, no tools) with an API model.

    python scripts/arms/run_static_api.py --model gpt-6-astra --label \\
        --benchmark data/benchmarks/tsmc/tsmc_core__khop400anchor.jsonl --name gpt6astra-raganchor-label

Spends API credit: one chat call per item, plus the latency probe's. `--limit N`
takes the first N items of each tier, for a cheap trial.

The run is written to `<out-root>/<name>--<benchmark stem>/` as
`completions.jsonl` and `summary.json`. This script does not score:
scripts/arms/score_static_gated.py does, with the gate the agentic runs use.

The prompt comes from `static_prompts.build_eval_prompt`, which gives a
closed-book item the question alone and any other item its edges. `--max-tokens`
defaults to 3000, and no temperature is sent unless `--temperature` is given,
because some models reject an explicit value.

Every run ends with a latency probe: the first 5 items of each tier of the
whole benchmark, one call at a time, cache off, written to `<run>/latency/`.
`--no-latency` skips it, and a `--sequential --no-cache` run is its own
latency and runs no separate probe.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from graph_reasoner.agent.eval import LATENCY_PROBE_PER_TIER, latency_quantiles, select_items
from graph_reasoner.benchmarks import load_benchmark, run_dir_name
from graph_reasoner.eval.static_prompts import build_eval_prompt
from graph_reasoner.llm import LLMClient, gather_with_progress
from graph_reasoner.llm.client import EmptyChoices
from graph_reasoner.provenance import env_state, git_state


def latency_stats(walls_ms):
    """The agentic probe's quartiles (`agent/eval.latency_quantiles`) over one
    call per question, and the questions a minute at the median."""
    walls = [w / 1000 for w in walls_ms]
    if not walls:
        return {"n": 0}
    p50 = statistics.median(walls)
    return {"n": len(walls), **latency_quantiles(walls),
            "questions_per_minute": round(60 / p50, 2) if p50 > 0 else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="OpenAI-compatible model name, e.g. gpt-5-mini")
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-tokens", type=int, default=3000)
    # None omits the parameter from the request, which is not the same as
    # 0.0; some models reject an explicit value.
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="N items PER TIER — keeps the tier balance while probing")
    # LLMClient caches on (model, messages, params), so a repeated run replays
    # from disk: the completions are the model's, the times are the cache's.
    ap.add_argument("--no-cache", dest="cache", action="store_false",
                    help="bypass the response cache; REQUIRED when timing")
    ap.add_argument("--sequential", action="store_true",
                    help="one call at a time, so wall_ms is a single request's "
                         "latency rather than a batch's share of one")
    ap.add_argument("--no-think", dest="think", action="store_false",
                    help="drop the 'reason inside <think>' instruction from the "
                         "retrieved-subgraph prompt (static_prompts.EVAL_PROMPT_NOTHINK); "
                         "for providers that restrict reasoning disclosure "
                         "(Anthropic). PROTOCOL-CHANGING: recorded in summary.json")
    ap.add_argument("--label", action="store_true",
                    help="append the label rule to the retrieval prompt (static_prompts.LABEL_RULE): "
                         "the static twin of the tool arms' agent_v1_label")
    ap.add_argument("--no-latency", dest="latency", action="store_false",
                    help="skip the latency probe every run ends with (first "
                         f"{LATENCY_PROBE_PER_TIER} items per tier, one call at "
                         "a time, cache off, in <run>/latency/)")
    ap.set_defaults(latency=True)
    ap.add_argument("--out-root", type=Path, default=Path("data/runs"),
                    help="where the run directory goes, as run_agent_eval.py's")
    a = ap.parse_args()

    all_items = load_benchmark(a.benchmark)
    # `--limit` and the probe take `select_items`' slice, the agentic runner's:
    # the first N of each tier in file order.
    items = select_items(all_items, a.limit)

    client = LLMClient(a.model, concurrency=a.concurrency, cache=a.cache)
    print(f"{len(items)} items | {a.model} | SPENDING CREDIT", flush=True)

    # Under --sequential each call is awaited alone, so wall_ms is one
    # request's latency. Otherwise it is that call's elapsed time while
    # `concurrency` others were in flight, which is not a latency; summary.json
    # says which (`timing.wall_ms_semantics`).
    # `blocked` is the pass's own list, so the main run's provider-blocked
    # items and the latency probe's are recorded apart.
    async def _one(cl, it, blocked):
        t0 = time.monotonic()
        try:
            c = await cl.chat([{"role": "user", "content": build_eval_prompt(it, think=a.think, label=a.label)}],
                              temperature=a.temperature, max_tokens=a.max_tokens)
        except EmptyChoices as e:
            # A provider that returns no completion (a content filter, for
            # example) does not end the run: the item is recorded as an
            # empty completion and its id is written to summary.json.
            print(f"\n  {it.id}: provider returned no completion ({e})")
            blocked.append(it.id)
            c = ""
        return c, int((time.monotonic() - t0) * 1000)

    async def _gen(cl, its, sequential, blocked):
        if sequential:
            out = []
            for i, it in enumerate(its, 1):
                out.append(await _one(cl, it, blocked))
                print(f"  {i}/{len(its)}", end="\r", flush=True)
            return out
        return await gather_with_progress([_one(cl, it, blocked) for it in its],
                                          desc=f"static:{a.model}")

    def write_run(d, its, pairs, total_wall_ms, sequential, cache, extra, blocked):
        extra = {**extra, "blocked_ids": list(blocked)} if blocked else extra
        d.mkdir(parents=True, exist_ok=True)
        with (d / "completions.jsonl").open("w") as fh:
            for it, (comp, w) in zip(its, pairs):
                fh.write(json.dumps({"id": it.id, "completion": comp, "wall_ms": w},
                                    ensure_ascii=False) + "\n")
        # score_static_gated.py reads `benchmark` when --subgraphs is not
        # given.
        summary = {"model": a.model, "benchmark": str(a.benchmark), "n": len(pairs),
                   "think": a.think,
                   "label": a.label,
                   "max_tokens": a.max_tokens,
                   "temperature": a.temperature,
                   "timing": {"driver": "async", "sequential": bool(sequential),
                              "wall_ms_semantics": ("per_call" if sequential
                                                    else "per_call_under_concurrency"),
                              "concurrency": 1 if sequential else a.concurrency,
                              "n_calls_per_item": 1, "total_wall_ms": total_wall_ms,
                              "cache": bool(cache)},
                   "git": git_state(), "env": env_state()}
        summary.update(extra)
        (d / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    started = time.monotonic()
    blocked: list[str] = []
    pairs = asyncio.run(_gen(client, items, a.sequential, blocked))
    total_wall_ms = int((time.monotonic() - started) * 1000)
    walls = [w for _, w in pairs]

    out_dir = a.out_root / run_dir_name(a.name, a.benchmark)
    print(client.usage_summary())

    # The latency probe: the first N items per tier of the whole benchmark,
    # one call at a time, through a client with the cache off. The main run is
    # written first, so a probe failure leaves the completions on disk with
    # the failure recorded.
    if not a.latency:
        lat = {"skipped": "--no-latency"}
    elif a.sequential and not a.cache:
        lat = {"source": "main run", **latency_stats(walls)}
    else:
        probe_items = select_items(all_items, LATENCY_PROBE_PER_TIER)
        write_run(out_dir, items, pairs, total_wall_ms, a.sequential, a.cache, {}, blocked)
        print(f"latency probe: {len(probe_items)} items one call at a time, cache off",
              flush=True)
        try:
            probe_client = LLMClient(a.model, concurrency=1, cache=False)
            t0 = time.monotonic()
            probe_blocked: list[str] = []
            probe_pairs = asyncio.run(_gen(probe_client, probe_items, True, probe_blocked))
            stats = latency_stats([w for _, w in probe_pairs])
            write_run(out_dir / "latency", probe_items, probe_pairs,
                      int((time.monotonic() - t0) * 1000), True, False,
                      {"probe": {"per_tier": LATENCY_PROBE_PER_TIER,
                                 "items": [it.id for it in probe_items], "cache": False},
                       "latency": stats}, probe_blocked)
            lat = {"dir": "latency", **stats}
            print(probe_client.usage_summary())
        except Exception as e:  # noqa: BLE001 -- the answered run must survive its probe
            lat = {"failed": f"{type(e).__name__}: {e}"}
            print(f"LATENCY PROBE FAILED: {lat['failed']}; the run is complete but "
                  f"carries no latency. Re-run with --limit {LATENCY_PROBE_PER_TIER} "
                  f"--sequential --no-cache.")
    write_run(out_dir, items, pairs, total_wall_ms, a.sequential, a.cache,
              {"latency": lat}, blocked)
    if "p50_s" in lat:
        print(f"latency: {lat['n']} items, p50 {lat['p50_s']} s, "
              f"{lat['questions_per_minute']} questions/min")
    print(f"-> {out_dir}")
    print(f"score it with:\n"
          f"  python scripts/arms/score_static_gated.py {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
