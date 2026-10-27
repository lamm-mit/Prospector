"""Prefix records: a stored episode cut before one of its assistant turns.

A prefix rollout resumes a stored episode at the cut: the prompt is the
transcript up to it, the policy generates from there (it may answer or keep
searching), and only the new turns carry loss. A GRPO group's variance is
then about that one decision, where a full episode spreads one scalar
advantage over every search turn.

A prefix record is a trajectory record (`agent/loop.py trajectory_record`)
with `messages` cut and a few fields added; `agent/loop.py seed_episode`
turns one back into a live Episode. `traj/prefixes.py` writes them,
`scripts/traj/sample_prefixes.py` samples continuations, and
`rollout.load_train_prefixes` selects the training set.

No torch or trl is imported here.
"""
from __future__ import annotations

import json
from pathlib import Path

PREFIX_SCHEMA = "agent_prefix_v1"


def load_prefixes(path: str | Path) -> list[dict]:
    """Every prefix record in a jsonl file, schema-checked."""
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("schema") != PREFIX_SCHEMA:
                raise ValueError(f"{path}:{n}: schema {rec.get('schema')!r}, "
                                 f"expected {PREFIX_SCHEMA!r}")
            for key in ("id", "item_id", "messages", "seen_edges", "n_prefix_turns"):
                if key not in rec:
                    raise ValueError(f"{path}:{n}: prefix record lacks {key!r}")
            out.append(rec)
    if not out:
        raise ValueError(f"{path}: no prefix records")
    return out


def zero_prefix_turns(mask: list[int], turn_idx: list[int],
                      n_prefix_turns: int) -> list[int]:
    """The loss mask with the seeded prefix's assistant turns zeroed.

    `turn_idx` is `tokenize.token_spans`' per-token assistant-turn index
    (-1 for environment tokens). Turns `0 .. n_prefix_turns-1` came with the
    prefix and are conditioning; only the turns generated after the cut keep
    their loss.
    """
    if len(mask) != len(turn_idx):
        raise ValueError(f"mask/turn_idx length mismatch: {len(mask)} vs {len(turn_idx)}")
    return [0 if 0 <= t < n_prefix_turns else m for m, t in zip(mask, turn_idx)]


__all__ = ["PREFIX_SCHEMA", "load_prefixes", "zero_prefix_turns"]
