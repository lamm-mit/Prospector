"""The gold-free vote over a pool of rollouts.

The vote reads only what the rollouts produced, never a gold answer:

  1. keep the rollouts whose assertion passes the gate (cites >= 1 edge,
     `cited subset seen`, the answer names a node of the cited evidence), and
     that did not abstain;
  2. among those, vote on the answer entity: the cited nodes the claim names
     (the text after the last "the answer is", eval/parse.extract_claim),
     under the same normalisation EM uses;
  3. return the earliest rollout supporting a top-voted entity, so a tie goes
     toward the first run (the reference).
"""
from __future__ import annotations

from collections import Counter

from .pool import live_indices


def select(cands: list[dict]) -> dict:
    """Pick one rollout by vote. `cands` is in run order; `cands[0]` is the
    reference, returned when no candidate asserted cleanly."""
    from ..agent.gate import _supporting_nodes
    from ..eval.parse import extract_claim

    live = [cands[n] for n in live_indices(cands)]
    if not live:
        return cands[0]                      # nothing asserted cleanly: keep the reference
    # Vote on the claim, not the full answer: an answer can restate its cited
    # chain, and a vote over that elects whichever node is in the most chains.
    support = [_supporting_nodes(extract_claim(c["answer"]), c["cited"]) for c in live]
    votes = Counter(n for s in support for n in s)
    if not votes:
        return live[0]
    top = max(votes.values())
    # Earliest run supporting a top-voted entity; `live` preserves run order.
    for c, s in zip(live, support):
        if any(votes[n] == top for n in s):
            return c
    return live[0]
