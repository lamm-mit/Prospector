"""An offered pool -> the trained verifier's margins, and the rows they are
stored as.

A candidate's margin is log P("yes") - log P("no") for the verifier's reply to
the prompt `prompts.render_prompt` builds from the question, the candidate's
answer and its cited chain. Each log-probability is summed over exactly the
token span SFT trained (`train.tokenize.completion_logprob_span`), so the
train-time and score-time tokenizations cannot drift apart. Two forward passes
per candidate, no generation.

The scorer is passed in, so the arithmetic is testable without weights:
`score_ids(ids)` returns log P(ids[j] | ids[:j]) for every position j, None at
0. scripts/select/verifier_select.py builds it from the merged verifier.
"""
from __future__ import annotations

from typing import Callable

from .pool import pick_by_margin
from .prompts import NO, YES, render_prompt

Scorer = Callable[[list[int]], list]


def score_pool(score_ids: Scorer, tok, question: str, pool: list[dict]) -> list[float]:
    """One margin per candidate of `pool`, in pool order."""
    from ..train.tokenize import completion_logprob_span

    margins = []
    for cand in pool:
        prompt = render_prompt(question, cand["answer"], cand["cited"])
        logp = {}
        for label in (YES, NO):
            full, span = completion_logprob_span(
                [{"role": "user", "content": prompt},
                 {"role": "assistant", "content": label}], tok)
            per_token = score_ids(full)
            logp[label] = sum(per_token[j] for j in span)
        margins.append(logp[YES] - logp[NO])
    return margins


def score_rows(item_id: str, pool: list[dict], margins: list[float]) -> list[dict]:
    """The `scores.jsonl` rows for one item, one per candidate. `cand_idx` is
    the candidate's position in the offered pool of the runs in
    `runs_in_order`, the key the paper's figure reads the margins back by;
    `picked` marks `pick_by_margin`'s choice. The labels are diagnostics for
    the AUC readout and are never read by the pick."""
    best = pick_by_margin(margins)
    return [{"id": item_id, "cand_idx": n, "margin": m, "picked": n == best,
             "gated_acc": bool(c["gated_acc"]), "em_claim": bool(c["em_claim"]),
             "gate_pass": bool(c["gate_pass"])}
            for n, (c, m) in enumerate(zip(pool, margins))]
