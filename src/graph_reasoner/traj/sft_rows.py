"""Stored rollouts -> supervised rows: the keep filter and the row shape.

`scripts/traj/build_sft_rows.py` is the CLI. The same filter serves a frontier
teacher's rollouts (the demonstrations of stage 1) and the policy's own (the
keeps of stage 3); the CLI records which in each row's `source` and in the
sidecar.

`verdict` keeps an episode that is `em_claim AND gate_pass`, the trained
reward's success condition. Both are mechanical: `gate_pass` needs no gold
(cites at least one edge, `cited ⊆ seen`, the answer names a node of the cited
evidence), and `em_claim` matches the item's full-graph answer set against the
asserted answer (`agent.metrics.em_claim_of`). `honest_abstention` keeps a
second class, declines, which a filter on answered successes alone would leave
out of training entirely.

`to_rows` (schema `agent_sft_v1`) writes one prompt/completion row per
assistant turn: the prompt is every message before the turn, the completion
the turn. The trainer reassembles an episode's rows into one transcript
(`train/transcripts.load_transcripts`) and supervises assistant tokens only.
"""
from __future__ import annotations

from ..agent.gate import gate_record
from ..agent.metrics import em_claim_of, reached
from ..eval.parse import detect_abstention
from ..kg.serialize import edge_key

#: The schema tag of every supervised row: the keeps, the detours
#: (`traj/error_pairs.py`) and the mixes built from them.
TRACE_SCHEMA = "agent_sft_v1"


def verdict(rec: dict) -> tuple[bool, str]:
    """Keep a stored trajectory record or not, and why: well-formed, answered,
    gate-passing, and right under the claim rule."""
    if rec.get("n_malformed"):
        return False, "malformed_call"
    if not (rec.get("answer") or "").strip():
        return False, "no_answer"
    g = gate_record(rec)
    if not g["gate_pass"]:
        return False, f"gate:{g['gate_reason']}"
    if not em_claim_of(rec.get("answer") or "", rec.get("kb_answer_set") or []):
        return False, "wrong_answer"
    return True, "ok"


def abstained(rec: dict) -> bool:
    """The deployed abstention rule (`eval.parse.detect_abstention`, the one
    behind the gate's `voluntary_abstention`)."""
    return bool(detect_abstention(rec.get("answer") or ""))


def honest_abstention(rec: dict) -> tuple[bool, str]:
    """Keep a decline or not, and why: it declined, no gold answer was among
    the nodes it saw, and it cited only edges it saw.

    A decline with a gold answer already among the seen nodes is giving up
    and would teach quitting. The citation check is explicit because the gate
    classifies a decline as `voluntary_abstention` before it reads any
    citation.
    """
    if rec.get("n_malformed"):
        return False, "abstain:malformed"
    if not abstained(rec):
        return False, "abstain:did_not_abstain"
    seen = rec.get("seen_edges") or []
    if reached(rec.get("kb_answer_set") or [], seen):
        return False, "abstain:answer_was_reachable"
    def keys(edges):
        return {edge_key(e["source"], e["relation"], e["target"]) for e in edges}
    if not keys(rec.get("cited_edges") or []) <= keys(seen):
        return False, "abstain:hallucinated_citation"
    return True, "ok_abstain"


def strip_user_note(rec: dict, note: str | None) -> bool:
    """Drop the generation-time user note from the prompt side; return whether
    one was removed.

    `agent/loop.py` appends `cfg.user_note` to the first user message, and the
    deployed policy is prompted without it. Stripping it makes the training
    prompt identical to the deployed one; the completions are untouched.
    """
    if not note:
        return False
    for m in rec.get("messages") or []:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        suffix = "\n\n" + note
        if isinstance(c, str) and c.endswith(suffix):
            m["content"] = c[: -len(suffix)]
            return True
        return False
    return False


def to_rows(rec: dict, tag: str) -> list[dict]:
    """One `TRACE_SCHEMA` row per assistant turn of a trajectory record. The
    caller stamps `source` once it knows who produced the rollouts."""
    msgs, rows = rec["messages"], []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        rows.append({
            "schema": TRACE_SCHEMA,
            "id": f"{rec['id']}#{tag}#t{i}",
            "item_id": rec["id"],
            "tier": rec.get("tier"),
            "turn": sum(1 for x in msgs[:i] if x["role"] == "assistant") + 1,
            "prompt": msgs[:i],
            "completion": [m],
            "source": "unknown",
        })
    return rows


__all__ = ["TRACE_SCHEMA", "abstained", "honest_abstention", "strip_user_note",
           "to_rows", "verdict"]
