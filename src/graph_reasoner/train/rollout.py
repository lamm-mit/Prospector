"""Config -> the rollout environment, the objective and the training set,
for the GRPO trainer.

Every cross-package import here is function-scoped, so importing this module
does not load `agent`, `eval` or `select`, and the objective's tests run
CPU-only in seconds (`tests/test_module_health.py`).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable


# The config keys that set the rollout environment, each named as the field it
# sets. A key a config leaves out takes the dataclass's default, the one
# evaluation uses (tests/test_traj_grpo.py holds TOOL_KEYS to every ToolConfig
# field). `tools` is the tool surface offered (ToolConfig.tools, the eval's
# --tools), not TRL's trainer argument of that name.
EPISODE_KEYS = ("max_turns", "turn_max_tokens")
TOOL_KEYS = ("page_size", "hub_threshold", "hub_summary_limit", "connect_max_hops",
             "connect_limit", "find_k", "tools")


def episode_config(config: dict):
    """The rollout environment: call budget, token budget, tool surface, seed.

    A key the config omits takes the default `scripts/run_agent_eval.py`
    uses. Every GRPO config sets `max_turns` and `turn_max_tokens`.
    """
    from ..agent.loop import EpisodeConfig
    from ..agent.tools import ToolConfig

    tool = {k: config[k] for k in TOOL_KEYS if k in config}
    if tool.get("tools") is not None:
        tool["tools"] = tuple(tool["tools"])
    episode = {k: config[k] for k in EPISODE_KEYS if k in config}
    if "env_seed" in config:
        episode["seed"] = config["env_seed"]
    return EpisodeConfig(temperature=config["temperature"],
                         tool_cfg=ToolConfig(**tool), **episode)


def build_objective(config: dict) -> Callable:
    """The trained objective. A key the config leaves out takes the recipe's
    value (`reward.PARTIAL`, `PARTIAL_ON`); the objective in force is printed
    at launch. Correctness is the claim rule, with no switch.
    """
    from .reward import PARTIAL, PARTIAL_ON, make_gated_reward

    partial = float(config.get("partial_credit", PARTIAL))
    partial_on = str(config.get("partial_on", PARTIAL_ON))
    reward_fn = make_gated_reward(partial=partial, partial_on=partial_on)
    print(f"objective: 1 for a gated success, else {partial} x {partial_on}")
    return reward_fn


def band_keep(hits: list[bool] | None, lo: float, hi: float) -> bool:
    """Whether an item's stored outcomes place it in the informative band.

    A GRPO group gives gradient only when its rollouts disagree, so an item
    stays when its measured pass rate lies in [lo, hi]. An item with no
    stored outcome is kept; `load_train_prefixes` drops such a prefix itself.
    """
    if not hits:
        return True
    return lo <= sum(hits) / len(hits) <= hi


def _tiers(config: dict) -> set[str] | None:
    """The `tiers` filter both loaders apply: the listed tiers, or every tier
    when the key is absent or empty."""
    return set(config["tiers"]) if config.get("tiers") else None


def load_train_items(config: dict) -> list:
    """Benchmark -> the items this run trains on: tier filter, gold-path leak
    exclusion, then the optional informative band.

    The leak key is the gold node set (`BenchmarkItem.gold_pathset`), which
    identifies a question across benchmark rebuilds, so an eval item cannot
    re-enter as a train item under a new id.
    """
    from ..benchmarks import load_benchmark

    items = load_benchmark(config["benchmark"])
    tiers = _tiers(config)
    if tiers:
        items = [i for i in items if i.tier in tiers]
    blocked = set()
    for path in config.get("exclude", []):
        blocked |= {i.gold_pathset() for i in load_benchmark(path)}
    items = [i for i in items if i.gold_pathset() not in blocked]
    print(f"{len(items)} train items (tiers={sorted(tiers) if tiers else 'all'}) "
          f"after leak filter")

    # Informative band (off unless configured): tier is a poor proxy for
    # within-group disagreement, so items are filtered by their measured
    # success rate in stored rollout passes, under `band_keep`:
    # `informative_band: {rollouts: [dirs], lo: 0.05, hi: 0.95}`.
    band = config.get("informative_band")
    if band:
        from pathlib import Path

        from ..select import load as _bn_load
        lo = float(band.get("lo", 0.05))
        hi = float(band.get("hi", 0.95))
        hits: dict[str, list[bool]] = defaultdict(list)
        for d in band["rollouts"]:
            for rid, rec in _bn_load(Path(d), recompute=True).items():
                hits[rid].append(bool(rec["gated_acc"]))

        before = len(items)
        items = [i for i in items if band_keep(hits.get(i.id), lo, hi)]
        print(f"informative band [{lo}, {hi}] over {len(band['rollouts'])} "
              f"stored pass(es): {before} -> {len(items)} items "
              f"({before - len(items)} always-solved or never-solved dropped)")
    return items


def load_train_prefixes(config: dict) -> list[tuple[dict, object]]:
    """Prefix records -> the (prefix, item) pairs a `rollout_mode: prefix` run
    trains on: leak filter by the item's gold-path set, the optional tier
    filter, then the optional band over stored continuation passes.

    A prefix's item supplies gold for the reward, so an eval item's prefix
    leaks as the item would, and the same `gold_pathset` exclusion applies.
    The band reads `results.jsonl` of runs written by
    scripts/traj/sample_prefixes.py, whose ids are prefix ids:
    `prefix_band: {rollouts: [dirs], lo: 0.05, hi: 0.95}`.
    """
    from ..benchmarks import load_benchmark
    from .prefixes import load_prefixes

    prefixes = load_prefixes(config["prefixes"])
    items = {i.id: i for i in load_benchmark(config["benchmark"])}
    blocked = set()
    for path in config.get("exclude", []):
        blocked |= {i.gold_pathset() for i in load_benchmark(path)}
    tiers = _tiers(config)

    pairs = []
    unknown = leaked = 0
    for p in prefixes:
        item = items.get(p["item_id"])
        if item is None:
            unknown += 1
            continue
        if item.gold_pathset() in blocked:
            leaked += 1
            continue
        if tiers and item.tier not in tiers:
            continue
        pairs.append((p, item))
    if unknown:
        raise SystemExit(f"{unknown} prefixes name items absent from "
                         f"{config['benchmark']} — wrong benchmark for these prefixes")
    print(f"{len(pairs)} train prefixes from {len(prefixes)} "
          f"(leak-excluded {leaked}; tiers={sorted(tiers) if tiers else 'all'})")

    band = config.get("prefix_band")
    if band:
        import json as _json
        from pathlib import Path as _Path

        lo = float(band.get("lo", 0.05))
        hi = float(band.get("hi", 0.95))
        hits: dict[str, list[bool]] = defaultdict(list)
        for d in band["rollouts"]:
            with (_Path(d) / "results.jsonl").open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = _json.loads(line)
                        hits[r["id"]].append(bool(r["gated_acc"]))
        before = len(pairs)
        # A prefix with no stored outcome is dropped, unlike an item: the
        # stored passes sample the prefix set, so a missing one was not
        # covered.
        pairs = [(p, it) for p, it in pairs
                 if hits.get(p["id"]) and band_keep(hits[p["id"]], lo, hi)]
        print(f"prefix band [{lo}, {hi}] on gated_acc over {len(band['rollouts'])} "
              f"stored pass(es): {before} -> {len(pairs)} prefixes")
    if not pairs:
        raise SystemExit("no train prefixes survive the filters")
    return pairs
