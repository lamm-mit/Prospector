"""The training reward and the scoring that feeds it, defined once.

`score_episodes` turns episodes into the metric columns the reward and the
monitors read. No torch or trl is imported here, at module scope or inside a
function, so the objective's tests run CPU-only in seconds.

The objective: a rollout whose answer is correct under the claim rule and
whose cited chain passes the gate earns 1. Any other rollout earns a quarter
of its evidence recall, the fraction of the gold path's edges its calls
returned. The defaults below are those values, and every GRPO config states
them again.

The advantages are TRL's: the rewards centred within each group and, under
`scale_rewards: group`, divided by the group's standard deviation
(`grpo._bessel` turns TRL's sample deviation into the population one). A
group whose rollouts all earn the same reward has zero advantage and gives no
gradient; TRL logs the fraction of such groups as `frac_reward_zero_std`.
"""
from __future__ import annotations

__all__ = [
    "PARTIAL",
    "PARTIAL_ON",
    "make_gated_reward",
    "find_rate_monitor",
    "gate_pass_monitor",
    "token_advantages",
    "score_episodes",
]

#: The recipe's objective, the defaults of `make_gated_reward`, `score_episodes`
#: and the config keys `partial_credit` and `partial_on` that
#: `rollout.build_objective` reads.
PARTIAL = 0.25
PARTIAL_ON = "evidence_recall"


# ---------------------------------------------------------------- the reward

def make_gated_reward(partial: float = PARTIAL, partial_on: str = PARTIAL_ON):
    """The training reward: 1.0 for a gated success, else `partial` x recall.

    A gated success is a correct answer (`em`, the claim rule, filled by
    `score_episodes`) whose cited chain passed the gate. Every other
    rollout, a wrong answer, a fabricated citation, a decline, earns the
    dense term, which orders the failures by how much of the gold path the
    search found. `partial` must lie in [0, 1), so no failure out-earns a
    success.

    `partial_on` picks which recall the dense term reads:

        path_recall     = |gold AND cited| / |gold|     what the answer cited
        evidence_recall = |gold AND seen|  / |gold|     what the search found

    A failing episode cites the chain behind its wrong answer, which is not
    the gold chain, so `path_recall` under-counts the progress the search
    made. `evidence_recall` is the default.
    """
    if partial_on not in ("path_recall", "evidence_recall"):
        raise ValueError(f"partial_on must be 'path_recall' or 'evidence_recall', "
                         f"got {partial_on!r}")
    if not 0.0 <= partial < 1.0:
        raise ValueError(f"partial must lie in [0, 1) so a failure never earns "
                         f"what a success does, got {partial!r}")

    def gated_reward(completions=None, em=None, gate_pass=None,
                     path_recall=None, evidence_recall=None, **_):
        dense = path_recall if partial_on == "path_recall" else evidence_recall
        rec = dense if dense is not None else [0.0] * len(em)
        # Partial credit for chain progress: under a 0/1 reward a nearly
        # complete chain earns what an empty one does, which leaves no
        # gradient toward finishing the search.
        return [1.0 if (e and g) else partial * float(r)
                for e, g, r in zip(em, gate_pass, rec)]
    return gated_reward


def find_rate_monitor(completions=None, reached_answer=None, **_):
    """Weight-0 reward: logged per step, contributes nothing to the gradient."""
    return [float(bool(x)) for x in reached_answer]


def gate_pass_monitor(completions=None, gate_pass=None, **_):
    """Weight-0 reward: the gate-pass rate, logged per step."""
    return [float(bool(x)) for x in gate_pass]


# ---------------------------------------------------------------- the oracle

def token_advantages(outcome: float, mask: list[int], *,
                     kl_tok=(), kl_beta: float = 0.0,
                     norm: float = 1.0) -> list[float]:
    """The per-completion-token advantage the parity loss must reproduce,
    written as plain Python: the CPU oracle `tests/test_parity_loss.py` pins
    `grpo._parity_loss` against. The trainer does not call it.

    The environment mask is folded in by multiplication: an advantage of 0
    contributes nothing to the loss, which masks observation tokens exactly.
    `kl_tok` is the per-token k1 gap to the reference, subtracted from the
    advantage at `kl_beta`; `norm` carries the per-sequence normalisation.
    """
    out = []
    for i, m in enumerate(mask):
        a = float(outcome)
        if kl_beta and i < len(kl_tok):
            a -= kl_beta * kl_tok[i]
        out.append(a * float(m) * norm)
    return out


# ---------------------------------------------------------------- scoring

def score_episodes(episodes, items, reward_fn) -> tuple[list[float], dict]:
    """Episodes -> (rewards, per-episode metric columns).

    The columns are what the reward and the monitors read, so no caller
    re-parses completion text: `em` is the claim rule (`em_claim`), and
    `gated_acc` the headline, which the reward bit agrees with.
    """
    from ..agent.metrics import agentic_metrics

    cols: dict[str, list] = {"em": [], "gate_pass": [], "reached_answer": [],
                             "gated_acc": [], "path_recall": [],
                             "evidence_recall": []}
    for ep, item in zip(episodes, items):
        m = agentic_metrics(ep, item)
        cols["em"].append(bool(m["em_claim"]))
        cols["gate_pass"].append(bool(m["gate_pass"]))
        cols["reached_answer"].append(bool(m["reached_answer"]))
        cols["gated_acc"].append(bool(m["gated_acc"]))
        # The share of the gold chain the answer cited and the share the
        # search saw; the dense term reads one of them (`partial_on`).
        cols["path_recall"].append(float(m.get("path_recall") or 0.0))
        cols["evidence_recall"].append(float(m.get("evidence_recall") or 0.0))
    return list(reward_fn(**cols)), cols
