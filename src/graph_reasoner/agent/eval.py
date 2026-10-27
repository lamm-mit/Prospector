"""Run a model over a benchmark agentically and write its run directory.

`data/runs/<tag>--<benchmark>/` (`benchmarks.run_dir_name`) holds
`completions.jsonl`, `results.jsonl` (the per-question metrics),
`trajectories.jsonl` (the replayable transcript plus the explicit seen-edge set
that makes every metric recomputable offline), `summary.json` and the latency
probe's `latency/`.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import statistics
import time
from pathlib import Path

from .. import __version__
from ..benchmarks import run_dir_name
from ..eval.local_model import engine_class, split_model
from ..files import write_jsonl_atomic, write_text_atomic
from ..llm import LLMClient
from ..provenance import env_state, git_state
from ..schema import BenchmarkItem
from ..train.generate import TOOL_CALL_STOP
from .backend import GraphBackend, context_window, load_backend, resolve_kb
from .loop import (
    Episode,
    EpisodeConfig,
    run_episodes_async,
    run_episodes_batched,
    trajectory_record,
)
from .metrics import (
    aggregate_agentic,
    agentic_metrics,
    print_agentic_summary,
    warn_on_starved_turns,
)


def infer_backend_kind(benchmark_path: str, items: list[BenchmarkItem]) -> str:
    """Which substrate a benchmark belongs to, from three signals in order: the
    first item's `meta["substrate"]`, the `data/benchmarks/<substrate>/`
    directory the file sits in, then a registered name in the filename stem.
    Raises ValueError when none of them names a registered substrate."""
    from .backend import available_backends

    registered = available_backends()
    declared = str((items[0].meta.get("substrate") or "") if items else "")
    if declared in registered:
        return declared
    path = Path(benchmark_path)
    if path.parent.name in registered:
        return path.parent.name
    stem = path.stem.lower()
    for name in registered:
        if name in stem:
            return name
    raise ValueError(
        f"cannot tell which substrate {benchmark_path!r} belongs to. Declare it "
        f'as meta["substrate"], put the file under data/benchmarks/<substrate>/, '
        f"or name it after one. Registered: {', '.join(registered)}")


def select_items(items: list[BenchmarkItem], limit: int | None,
                 shard: str | None = None) -> list[BenchmarkItem]:
    """`limit` keeps the first `limit` items of each tier.

    `shard` is "k/n": within each tier, in file order, every n-th item
    starting at offset k. The n shards partition the benchmark, each with the
    tier balance of the whole. Applied after `limit`."""
    if limit:
        out, counts = [], {}
        for item in items:
            if counts.get(item.tier, 0) < limit:
                out.append(item)
                counts[item.tier] = counts.get(item.tier, 0) + 1
        items = out
    if shard:
        try:
            k, n = (int(x) for x in shard.split("/"))
        except ValueError:
            raise ValueError(f"shard {shard!r}: expected 'k/n'") from None
        if not (0 <= k < n):
            raise ValueError(f"shard {shard!r}: need 0 <= k < n")
        out, counts = [], {}
        for item in items:
            i = counts.get(item.tier, 0)
            counts[item.tier] = i + 1
            if i % n == k:
                out.append(item)
        items = out
    return items


def _effective_chat_template(model: str, chat_template: str | None,
                             adapter: str | None = None) -> str | None:
    """The chat template a run rendered with, for the run record.

    For a local model an explicit `chat_template` wins; otherwise the last
    adapter's record (train/adapters.py, `recorded_chat_template`), and
    otherwise the tokenizer directory the engine loads is read back for a
    named template, because a trained adapter directory carries the template
    its run installed. None means no named template: the model's native one,
    and always for an API model.
    """
    engine, path = split_model(model)
    if engine is not None:
        if chat_template:
            return chat_template
        from ..train.tokenize import template_name
        from ..train.adapters import recorded_chat_template, resolve_base, tokenizer_source
        base, chain = resolve_base(path, adapter)
        recorded = recorded_chat_template(chain)
        if recorded:
            return recorded
        tok_src = Path(tokenizer_source(base, chain))
        jinja = tok_src / "chat_template.jinja"
        if jinja.exists():
            return template_name(jinja.read_text(encoding="utf-8"))
        tok_cfg = tok_src / "tokenizer_config.json"
        if tok_cfg.exists():
            try:
                tpl = json.loads(tok_cfg.read_text(encoding="utf-8")).get("chat_template")
            except (OSError, ValueError):
                return None
            return template_name(tpl if isinstance(tpl, str) else None)
    return None


def driver_kind(model: str) -> str:
    """Which driver a model string selects: "batched" for a local engine prefix
    (`split_model`), "async" for an API model. `run_episodes` branches on this
    function, so the label in summary.json and the driver that ran agree."""
    return "batched" if split_model(model)[0] is not None else "async"


def wall_ms_semantics(kind: str, sequential: bool) -> str:
    """What a stored `wall_ms` measures, from the driver and `sequential`:

        per_episode         that episode's own duration
        since_batch_start   elapsed since a batch of different questions began

    The static arms' runner (scripts/arms/run_static_api.py) writes its own
    labels, `per_call` and `per_call_under_concurrency`.
    """
    if kind == "batched" and not sequential:
        return "since_batch_start"
    return "per_episode"


def local_generate_seam(local, cfg: EpisodeConfig):
    """The generate seam over a local engine (hf:, vllm:, mlx:). `run_episodes`
    hands it to the batched driver and the live app (app/live.py) drives the
    same call, with the same stop and temperature source."""
    def generate(conversations, max_new_tokens):
        return local.generate_chat(
            conversations, max_new_tokens=max_new_tokens,
            temperature=cfg.temperature, stop=[TOOL_CALL_STOP], desc="rollout turn",
        )
    # The engine's context window, exposed for the run record.
    generate.max_model_len = getattr(local, "max_model_len", None)
    return generate


def build_generate_seam(model: str, cfg: EpisodeConfig, *, adapter: str | None = None,
                        batch_size: int = 8, chat_template: str | None = None,
                        max_model_len: int | None = None, gpu_frac: float | None = None):
    """The generate seam of a local engine (`eval/local_model.ENGINES`), or
    None for an API model, which has no seam.

    Build it once per process and hand it to `run_episodes`, so a run and its
    latency probe drive one engine: a vLLM engine holds the GPU until the
    process exits, and a second one may not fit. `batch_size` reaches hf: only;
    `max_model_len` is a vllm: and mlx: option (the two engines that cut an
    over-long conversation) and `gpu_frac` a vllm: one; each is refused
    elsewhere. `cfg.seed` seeds the engine's sampler.
    """
    engine, path = split_model(model)
    if max_model_len is not None and engine not in ("vllm", "mlx"):
        raise SystemExit(f"--max-model-len applies to the vllm: and mlx: drivers only, and --model is {model!r}.")
    if gpu_frac is not None and engine != "vllm":
        raise SystemExit(f"gpu_frac applies to the vllm: driver only, and --model is {model!r}.")
    if engine is None:
        return None
    cls = engine_class(engine)
    package = getattr(cls, "needs", None)
    if package and importlib.util.find_spec(package) is None:
        raise SystemExit(f"the {engine}: driver needs the package {package}, which this environment lacks: "
                         f"{ENGINE_INSTALL[engine]}")
    kwargs: dict = {"adapter": adapter, "chat_template": chat_template, "seed": cfg.seed}
    if engine == "hf":
        kwargs["batch_size"] = batch_size
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    if gpu_frac is not None:
        kwargs["gpu_frac"] = gpu_frac
    return local_generate_seam(cls(path, **kwargs), cfg)


# How to get the package a local engine imports (its class's `needs`), checked
# before anything is merged or loaded: without the check the driver fails
# inside the engine's constructor, after a merge that takes minutes and writes
# gigabytes.
ENGINE_INSTALL = {
    "vllm": "launch/setup.sh installs it on a Linux machine with a CUDA card; on a Mac use mlx: (quickstart.sh) or hf:",
    "hf": "install the train extra, uv pip install -e '.[train]'",
    "mlx": "install the train and mlx extras on Apple silicon, uv pip install -e '.[train,mlx]' -c launch/constraints-mac.txt, as quickstart.sh does",
}


def run_episodes(
    items: list[BenchmarkItem], backend: GraphBackend, model: str, cfg: EpisodeConfig,
    *, generate=None, concurrency: int = 8, cache: bool = True,
    usage: dict | None = None, sequential: bool = False,
    api_timeout: float | None = None,
) -> list[Episode]:
    """Every item's episode, in the items' order. A local model is driven
    through `generate`, the seam `build_generate_seam` built; an API model
    through an `LLMClient`, whose counters are copied into `usage`.

    With `sequential`, one question runs at a time and its `wall_ms` is that
    question's latency; without it every item starts together and `wall_ms`
    is not a latency. Best-of-n is separate sampled runs (the 32-seed pool the recipe's `final` stage draws),
    never several draws inside one run.
    """
    if driver_kind(model) == "batched":
        if generate is None:
            raise ValueError(f"{model!r} is a local model: pass the seam "
                             f"build_generate_seam built for it")
        if not sequential:
            return run_episodes_batched(items, backend, generate, cfg)
        # One question per batch: the batched driver starts every episode it is
        # handed together, so only then is `wall_ms` that question's own time.
        out: list[Episode] = []
        for item in items:
            out.extend(run_episodes_batched([item], backend, generate, cfg))
        return out

    client = LLMClient(model, concurrency=1 if sequential else concurrency,
                       cache=cache,
                       **({"timeout": api_timeout} if api_timeout else {}))
    if sequential:
        # `concurrency` gates in-flight calls, not episodes: `run_episodes_async`
        # starts every episode as a task, so under a wide concurrency each
        # `wall_ms` would include waiting on the others. One question per
        # call instead, all in one event loop, since the client's semaphore
        # binds to the loop it first waits in.
        async def one_question_at_a_time():
            out = []
            for item in items:
                out.extend(await run_episodes_async([item], backend, client, cfg))
            return out
        episodes = asyncio.run(one_question_at_a_time())
    else:
        episodes = asyncio.run(run_episodes_async(items, backend, client, cfg))
    print(f"LLM usage: {client.usage}")
    # The client's counters, which the episodes do not carry. `cached` counts
    # cache replays, which take no time and must not be read as latency.
    if usage is not None:
        usage.update(client.usage)
    return episodes


# ---------------------------------------------------------------- the record

def _score(episodes: list[Episode], items: list[BenchmarkItem]) -> list[dict]:
    """One results row per episode, `items` in the order the driver ran them,
    so each episode is scored against its own item."""
    rows = []
    for ep, item in zip(episodes, items):
        row = {"id": ep.item_id, "tier": ep.tier}
        row.update(agentic_metrics(ep, item))
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows, mode: str = "w") -> None:
    """One JSON object per line, unicode kept. A whole file is written under a
    temporary name and renamed into place; `mode="a"` appends (the live app
    adds one episode at a time)."""
    if mode == "w":
        write_jsonl_atomic(path, rows)
        return
    with Path(path).open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_run(out_dir: Path, episodes: list[Episode], items: list[BenchmarkItem],
               results: list[dict], graph: GraphBackend, cfg: EpisodeConfig, *,
               model: str, adapter: str | None, completions: bool) -> None:
    """The per-episode files of a run (or of its latency probe)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if completions:
        write_jsonl(out_dir / "completions.jsonl",
                    ({"id": ep.item_id, "completion": ep.final_text} for ep in episodes))
    write_jsonl(out_dir / "results.jsonl", results)
    write_jsonl(out_dir / "trajectories.jsonl",
                (trajectory_record(ep, item, graph, cfg, model=model, adapter=adapter)
                 for ep, item in zip(episodes, items)))


def timing_block(kind: str, *, sequential: bool, concurrency,
                 batch_size, n_episodes: int, total_wall_ms: int | None = None,
                 usage: dict | None = None) -> dict:
    """The `timing` block of a summary.json: what the run's `wall_ms` measures.

      driver,             from `driver_kind` and `wall_ms_semantics`
      wall_ms_semantics
      sequential          one question at a time, which a latency needs
      concurrency,        as passed; None where the knob does not apply
      batch_size          (`batch_size` applies to hf: only)
      total_wall_ms       the caller's wall clock; the main run's includes
                          model construction, which the drivers' timers exclude
      cache_hits,         from the API client's usage. A cached reply takes no
      api_calls           time, so check `cache_hits` before reading a latency
    """
    usage = usage or {}
    return {
        "driver": kind,
        "sequential": bool(sequential),
        "wall_ms_semantics": wall_ms_semantics(kind, bool(sequential)),
        "concurrency": concurrency,
        "batch_size": batch_size,
        "n_episodes": n_episodes,
        "total_wall_ms": total_wall_ms,
        "cache_hits": usage.get("cached"),
        "api_calls": usage.get("calls"),
    }


def _usage(usage: dict, *, outcomes: bool) -> dict:
    """What an API arm was billed for: the client's token counts, including
    reasoning tokens, which the transcripts do not show. Every value is None
    for a local model. `outcomes` adds the soft failures the client counts
    instead of raising (truncated, refused)."""
    out = {key: usage.get(key) for key in
           ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_prompt_tokens")}
    if outcomes:
        out.update(truncated=usage.get("truncated"), refused=usage.get("refused"))
    return out


# ---------------------------------------------------------------- latency

#: Items per tier the latency probe times: the first five of each tier of the
#: whole benchmark, one question at a time, cache off. The same items for
#: every arm on a benchmark, which makes the medians comparable across arms.
LATENCY_PROBE_PER_TIER = 5


def latency_quantiles(walls_s: list[float]) -> dict:
    """The quartiles of per-question seconds, rounded to 0.1 s: p25 and p75 by
    index (`walls[n // 4]`, `walls[3n // 4]` of the sorted list), p50 the
    median. Shared with the static probe (scripts/arms/run_static_api.py)."""
    walls = sorted(walls_s)
    n = len(walls)
    return {"p25_s": round(walls[n // 4], 1),
            "p50_s": round(statistics.median(walls), 1),
            "p75_s": round(walls[(3 * n) // 4], 1)}


def latency_stats(episodes: list[Episode]) -> dict:
    """Median seconds per question and its reciprocal in questions per minute,
    plus the quartiles, turns per episode and median seconds per turn."""
    walls = [ep.wall_ms / 1000 for ep in episodes]
    n = len(walls)
    if n == 0:
        return {"n": 0}
    p50 = statistics.median(walls)
    return {
        "n": n,
        **latency_quantiles(walls),
        "questions_per_minute": round(60 / p50, 2) if p50 > 0 else None,
        "turns_per_episode": round(sum(ep.n_turns for ep in episodes) / n, 1),
        "s_per_turn": round(statistics.median(
            ep.wall_ms / 1000 / max(1, ep.n_turns) for ep in episodes), 2),
    }


def run_latency_probe(
    benchmark_items: list[BenchmarkItem], graph: GraphBackend, model: str,
    cfg: EpisodeConfig, out_dir: Path, *, adapter: str | None = None,
    chat_template: str | None = None, generate=None,
    api_timeout: float | None = None, benchmark_path: str | None = None,
    per_tier: int = LATENCY_PROBE_PER_TIER,
) -> dict:
    """Time this arm one question at a time and write `<run>/latency/`.

    A run's own `wall_ms` is a latency only when its questions ran one at a
    time with nothing replayed from the cache (`--sequential --no-cache`). Any
    other eval carries this separate measurement of the same policy: the first
    `per_tier` items of each tier of the whole benchmark (not the run's
    `--limit`, `--shard` or `--ids-file` subset), drawn fresh with the response
    cache off. The probe's accuracy is recorded but is not the arm's score.
    """
    items = select_items(benchmark_items, per_tier)
    kind = driver_kind(model)
    print(f"\nlatency probe: {len(items)} items one at a time, cache off "
          f"(first {per_tier} per tier of the whole benchmark)", flush=True)
    usage: dict = {}
    started = time.monotonic()
    episodes = run_episodes(items, graph, model, cfg, generate=generate,
                            cache=False, concurrency=1, usage=usage,
                            sequential=True, api_timeout=api_timeout)
    total_wall_ms = int((time.monotonic() - started) * 1000)

    results = _score(episodes, items)
    probe_dir = Path(out_dir) / "latency"
    _write_run(probe_dir, episodes, items, results, graph, cfg,
               model=model, adapter=adapter, completions=False)
    stats = latency_stats(episodes)
    summary = {"model": model, "benchmark": benchmark_path, "agentic": True,
               "probe": {"per_tier": per_tier, "items": [it.id for it in items],
                         "cache": False},
               "timing": timing_block(kind, sequential=True,
                                      concurrency=1, batch_size=None,
                                      n_episodes=len(episodes),
                                      total_wall_ms=total_wall_ms, usage=usage),
               "usage": _usage(usage, outcomes=False),
               "latency": stats,
               "policy": cfg.policy_dict(), "aggregate": aggregate_agentic(results),
               "git": git_state(), "env": env_state(), "version": __version__}
    (probe_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"dir": "latency", **stats}


# ---------------------------------------------------------------- the eval

def run_agentic_eval(
    benchmark_path: str,
    model: str,
    *,
    kb_path: str | None,
    backend: str = "auto",
    adapter: str | None = None,
    limit: int | None = None,
    run_name: str | None = None,
    cfg: EpisodeConfig | None = None,
    concurrency: int = 8,
    batch_size: int = 8,
    chat_template: str | None = None,
    out_root: str | Path = "data/runs",
    cache: bool = True,
    sequential: bool = False,
    shard: str | None = None,
    api_timeout: float | None = None,
    ids_file: str | None = None,
    latency: bool = True,
    max_model_len: int | None = None,
    generate=None,
) -> dict:
    """Load, run, score, write, probe: the run directory of one eval, with its
    summary.json written once, after the latency probe. The scored files are
    on disk before the probe starts, so a probe failure costs only the probe;
    it is recorded in the summary. `generate` is a ready generate seam over an
    engine the caller built (`local_generate_seam`), used in place of the one
    `build_generate_seam` would build from `model`, as the latency probe
    already allows; the steered engine of `graph_reasoner.latent.steer` passes
    one."""
    from ..benchmarks import load_benchmark

    cfg = cfg or EpisodeConfig()
    # The whole benchmark is kept: the latency probe times the same first
    # items of every tier whatever subset this run scores.
    items = benchmark_items = load_benchmark(benchmark_path)
    if ids_file:
        # A listed subset of the benchmark, one id per line.
        with open(ids_file) as f:
            wanted = {ln.strip() for ln in f if ln.strip()}
        items = [it for it in items if it.id in wanted]
        missing = wanted - {it.id for it in items}
        if missing:
            raise SystemExit(f"{len(missing)} ids in {ids_file} are not in "
                             f"{benchmark_path}, e.g. {sorted(missing)[:3]}")
    items = select_items(items, limit, shard)
    kind = infer_backend_kind(benchmark_path, items) if backend == "auto" else backend
    # The substrate and its graph file are resolved together, and the summary
    # records the resolved path (`kb_path`).
    kb = resolve_kb(kind, kb_path)
    graph = load_backend(kind, kb)
    print(f"{len(items)} items | backend={graph.name} fingerprint={graph.fingerprint()} "
          f"| model={model} | max_turns={cfg.max_turns}")

    # Timed from before the model is built: the drivers' own timers start
    # after it, so they miss an adapter merge and an engine boot.
    usage: dict = {}
    driver = driver_kind(model)
    started = time.monotonic()
    if max_model_len is None and split_model(model)[0] in ("vllm", "mlx"):
        max_model_len = context_window(kind)       # the window the graph's substrate declares, if any
    generate = generate or build_generate_seam(model, cfg, adapter=adapter,
                                               batch_size=batch_size, chat_template=chat_template,
                                               max_model_len=max_model_len)
    episodes = run_episodes(items, graph, model, cfg, generate=generate,
                            cache=cache, concurrency=concurrency, usage=usage,
                            sequential=sequential, api_timeout=api_timeout)
    total_wall_ms = int((time.monotonic() - started) * 1000)

    results = _score(episodes, items)
    name = run_name or split_model(model)[1].replace("/", "_")
    out_dir = Path(out_root) / run_dir_name(name, benchmark_path)
    _write_run(out_dir, episodes, items, results, graph, cfg,
               model=model, adapter=adapter, completions=True)

    agg = aggregate_agentic(results)
    summary = {"model": model, "benchmark": benchmark_path, "agentic": True,
               # "k/n" when this run is one tier-balanced group of the
               # benchmark (select_items), else None. The run directory's
               # name does not carry it.
               "shard": shard,
               "ids_file": ids_file,
               # The LoRA chain passed as `adapter`, or None.
               "adapter": adapter,
               "backend": graph.name, "kg_fingerprint": graph.fingerprint(),
               "kb_path": str(kb),
               # The template the run rendered with, resolved rather than
               # echoed from the argument: two runs of one checkpoint under
               # different templates are not comparable.
               "chat_template": _effective_chat_template(model, chat_template,
                                                         adapter=adapter),
               # The context window of the engine that served the run: a
               # conversation past it gets an empty turn. None for a driver
               # without one.
               "max_model_len": getattr(generate, "max_model_len", None),
               "timing": timing_block(
                   driver, sequential=sequential,
                   concurrency=((1 if sequential else concurrency) if driver == "async"
                                else (1 if sequential else None)),
                   batch_size=batch_size if split_model(model)[0] == "hf" else None,
                   n_episodes=len(episodes), total_wall_ms=total_wall_ms, usage=usage),
               "usage": _usage(usage, outcomes=True),
               "policy": cfg.policy_dict(), "aggregate": agg,
               "git": git_state(), "env": env_state(), "version": __version__}

    print_agentic_summary(name, agg)
    starved = warn_on_starved_turns(episodes)
    if starved:
        print(f"\n{starved}")
        summary["warning"] = starved
    # An episode whose call failed past the retries is scored as a failure and
    # listed here, so the run is complete and the loss is visible.
    failed = [(ep.item_id, ep.error) for ep in episodes if ep.status == "error"]
    if failed:
        print(f"\nWARNING: {len(failed)} episode(s) ended in an error and are scored "
              f"as failures; the first: {failed[0][0]}: {failed[0][1]}")
        summary["errors"] = {"n": len(failed), "items": [i for i, _ in failed]}

    # Every run records a latency, or the reason it has none.
    if not latency:
        lat: dict = {"skipped": "--no-latency"}
    elif sequential and (driver == "batched" or not cache):
        # The main pass already ran one question at a time with nothing
        # replayed, so its own walls are the latency.
        lat = {"source": "main run", **latency_stats(episodes)}
    else:
        try:
            lat = run_latency_probe(benchmark_items, graph, model, cfg, out_dir,
                                    adapter=adapter, chat_template=chat_template,
                                    generate=generate, api_timeout=api_timeout,
                                    benchmark_path=benchmark_path)
        except Exception as e:  # noqa: BLE001 -- the scored run must survive its probe
            lat = {"failed": f"{type(e).__name__}: {e}"}
            print(f"\nLATENCY PROBE FAILED: {lat['failed']}\nThe run above is "
                  f"complete and scored but carries no latency. Re-run the "
                  f"probe alone with --limit {LATENCY_PROBE_PER_TIER} "
                  f"--sequential --no-cache and another --name: the same "
                  f"name would overwrite this run.")
    summary["latency"] = lat
    # Written last and whole: the recipe takes this file as the run's completion.
    write_text_atomic(out_dir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    if "p50_s" in lat:
        print(f"latency: {lat['n']} items one at a time, p50 {lat['p50_s']} s, "
              f"{lat['questions_per_minute']} questions/min "
              f"({'latency/' if 'dir' in lat else 'from the main run'})")
    elif "skipped" in lat:
        print(f"latency: skipped ({lat['skipped']})")
    print(f"\n-> {out_dir}")
    return summary


__all__ = ["run_agentic_eval", "run_episodes", "run_latency_probe", "latency_stats",
           "latency_quantiles", "build_generate_seam", "local_generate_seam",
           "driver_kind", "select_items", "timing_block", "write_jsonl",
           "LATENCY_PROBE_PER_TIER"]
