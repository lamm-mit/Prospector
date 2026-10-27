"""The recipe's pick rules over a round's grid: per-checkpoint means of gated_acc
over the stored evaluation runs `<prefix>-ck<N>-s<K>--<benchmark stem>`, and
the rule that chooses one. Every pick the recipe makes goes through here;
`scripts/analysis/ladder_peak.py` is the command line.

Two rules, the ones the released rounds were picked by. `pick_best` is the
full round's: the checkpoint with the best mean over the grid's three seeds,
the later one on a tie. `pick` is a decision round's, over the grid plus the
eight seeds its two leaders get: checkpoints whose mean is within `band`
(0.03) of the best mean are tied, and a tie goes to the checkpoint with more
seeds, then to the later one.
"""
from __future__ import annotations

import collections
import glob
import json
import re
import statistics as st
from pathlib import Path

BAND = 0.03


def read_grid(prefix: str, bench: str, runs: str | Path = "data/runs", cap: int | None = None,
              seeds=None, checkpoints=None):
    """`({checkpoint: [(seed, gated_acc mean)]}, {checkpoint: Counter(turn cap)})`
    over the runs `<prefix>-ck<N>-s<K>--<bench>`. A run whose summary.json is
    missing or unreadable is not a seed. `cap` keeps only seeds whose summary
    records that `policy.turn_max_tokens`; `seeds` and `checkpoints`, when
    given, keep only those."""
    acc: dict[int, list[tuple[int, float]]] = collections.defaultdict(list)
    caps: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for d in sorted(glob.glob(f"{runs}/{prefix}-ck*-s*--{bench}")):
        m = re.search(r"-ck(\d+)-s(\d+)--", d)
        if not m:
            continue
        ck, seed = int(m.group(1)), int(m.group(2))
        if checkpoints is not None and ck not in checkpoints:
            continue
        if seeds is not None and seed not in seeds:
            continue
        try:
            with open(d + "/summary.json") as fh:
                sm = json.load(fh)
        except (OSError, ValueError):
            continue
        run_cap = (sm.get("policy") or {}).get("turn_max_tokens")
        caps[ck][run_cap] += 1
        if cap is not None and run_cap != cap:
            continue
        with open(d + "/results.jsonl") as fh:
            g = st.mean(json.loads(line)["gated_acc"] for line in fh if line.strip())
        acc[ck].append((seed, g))
    return dict(acc), dict(caps)


def means(acc: dict) -> dict[int, float]:
    return {k: st.mean(g for _, g in v) for k, v in acc.items()}


def tied(acc: dict, band: float = BAND) -> list[int]:
    """The checkpoints within `band` of the best mean."""
    m = means(acc)
    best = max(m.values())
    return [k for k, v in m.items() if v >= best - band]


def pick(acc: dict, band: float = BAND) -> int:
    """A decision round's rule: among the tied, more seeds first, then the later checkpoint."""
    return max(tied(acc, band), key=lambda k: (len(acc[k]), k))


def pick_best(acc: dict) -> int:
    """The full round's rule: the best mean, and the later checkpoint on a tie."""
    m = means(acc)
    return max(m, key=lambda k: (m[k], k))


def leaders(acc: dict, n: int) -> list[int]:
    """The `n` checkpoints with the best means, best first (ties: the later checkpoint)."""
    m = means(acc)
    return sorted(m, key=lambda k: (m[k], k), reverse=True)[:n]
