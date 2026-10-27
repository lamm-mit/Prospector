"""Trajectory SFT rows -> full transcripts.

Nothing here imports torch or trl.
"""
from __future__ import annotations

import json
from collections import defaultdict


def load_transcripts(path: str, *, verbose: bool = True) -> list[list[dict]]:
    """Per-turn prompt/completion rows -> full transcripts, losslessly.

    The trajectory SFT format stores one row per assistant turn, each carrying
    the whole prefix: row `t` has prompt = messages[:t] and completion =
    [messages[t]]. The last row of a trajectory therefore holds every message,
    and per-token weights over it recover the supervision of all its rows.

    Within an item, a row is dropped when its prompt + completion is a prefix
    of a longer row's or repeats another row's. Keeping only the highest-turn
    row per `item_id` would lose transcripts, because one item can contribute
    several: a detour row (`traj/error_pairs.py`) has a prompt that diverges
    from its episode's. A row already in `messages` form passes through as
    one transcript.
    """
    by_ep: dict[str, list[dict]] = defaultdict(list)
    plain: list[list[dict]] = []
    n_rows = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            n_rows += 1
            if "prompt" in row and "completion" in row:
                by_ep[str(row.get("item_id") or row.get("id") or n_rows)].append(row)
            else:
                plain.append(row["messages"])

    def sig(msgs) -> tuple:
        return tuple((m.get("role"), m.get("content")) for m in msgs)

    n_traj = 0
    for rows in by_ep.values():
        full = [list(r["prompt"]) + list(r["completion"]) for r in rows]
        # longest first, so a shorter prefix is always tested against a kept one
        full.sort(key=len, reverse=True)
        kept: list[list[dict]] = []
        for cand in full:
            s = sig(cand)
            if any(sig(k)[:len(s)] == s for k in kept):
                continue
            kept.append(cand)
        n_traj += len(kept)
        plain.extend(kept)
    if verbose:
        print(f"{n_rows} rows -> {len(plain)} transcripts "
              f"({n_traj} from {len(by_ep)} items)")
    return plain