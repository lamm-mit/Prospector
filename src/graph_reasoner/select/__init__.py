"""Inference-time selection: a pool of rollouts for one question -> one answer.

Selection reads no gold. The pool is filtered on `gate_pass` and `abstained`,
both mechanical, then ranked by a vote or by a trained verifier's margins. The
benchmark is opened in one place, `pool.load`, and only to fill in the
`em_claim` and `gated_acc` labels a selection is scored by afterwards; no
selector reads them.

    pool           stored run -> the rows every selector reads; the offered
                   pool and the margin pick
    vote           the entity vote (best-of-n)
    prompts        the verifier prompt, the same for training and scoring
    verifier       an offered pool -> the verifier's margins and their stored rows
    verifier_data  stored rollouts -> verifier SFT rows

The CLIs are `scripts/select/`.
"""
from __future__ import annotations

from .pool import live_indices, load, load_pair, offered, pick_by_margin
from .vote import select

__all__ = ["live_indices", "load", "load_pair", "offered", "pick_by_margin", "select"]
