"""The keep filter (`graph_reasoner.traj.sft_rows`) and its CLI
(`scripts/traj/build_sft_rows.py`): the retained set is exactly
`em_claim AND gate_pass`, the trained reward's success condition, plus honest
abstentions when asked for, and its prompts are the deployed ones
(`strip_user_note`).
"""
from __future__ import annotations

import json
import subprocess
import sys

from conftest import script

from graph_reasoner.traj import sft_rows as brr

# Found by file name under scripts/ (`conftest.script`), not by a fixed folder.
SCRIPT = script("build_sft_rows.py")


def _rec(**kw):
    seen = [{"source": "A", "relation": "r", "target": "B"},
            {"source": "B", "relation": "s", "target": "C"}]
    base = dict(
        id="easy-0001", tier="easy", kb_answer_set=["C"],
        answer="ANSWER: C", seen_edges=seen, cited_edges=seen, n_malformed=0,
        messages=[{"role": "system", "content": "sys"},
                  {"role": "user", "content": "q"},
                  {"role": "assistant", "content": "<tool_call>{}</tool_call>"},
                  {"role": "user", "content": "OBSERVATION"},
                  {"role": "assistant", "content": "ANSWER: C"}],
    )
    base.update(kw)
    return base


def test_verified_trajectory_is_kept():
    ok, why = brr.verdict(_rec())
    assert ok and why == "ok"


def test_the_filter_rejects_a_chain_mention_without_the_claim():
    """Gold 'C' is named in the because-clause but the claim is 'B', so the
    claim rule rejects it. The answer still names cited node B, so the gate
    passes and only the EM rule decides."""
    rec = _rec(answer="ANSWER: Because A r B and B s C, the answer is B.")
    ok, why = brr.verdict(rec)
    assert ok is False and why == "wrong_answer"
    good = _rec(answer="ANSWER: Because A r B and B s C, the answer is C.")
    assert brr.verdict(good)[0] is True


def test_right_answer_with_a_fabricated_citation_is_rejected():
    """cited ⊄ seen: a correct answer that cites an edge it never saw is
    rejected by the gate, though EM alone would accept it."""
    ok, why = brr.verdict(_rec(cited_edges=[
        {"source": "A", "relation": "invented", "target": "C"}]))
    assert not ok and why.startswith("gate:")


def test_grounded_but_wrong_answer_is_rejected():
    ok, why = brr.verdict(_rec(answer="ANSWER: Z", kb_answer_set=["C"],
                               cited_edges=[{"source": "A", "relation": "r",
                                             "target": "B"}]))
    assert not ok and why in ("wrong_answer", "gate:answer_not_in_cited")


def test_abstention_is_rejected():
    ok, why = brr.verdict(_rec(
        answer="insufficient information in the graph — I could not reach the answer.",
        kb_answer_set=["insufficient"]))
    assert not ok and why.startswith("gate:")


def test_rows_carry_the_teacher_schema_and_one_row_per_assistant_turn():
    rows = brr.to_rows(_rec(), "s11")
    assert len(rows) == 2, "one row per assistant turn"
    assert all(r["schema"] == "agent_sft_v1" for r in rows)
    assert rows[0]["prompt"][-1]["role"] == "user"
    assert rows[0]["completion"][0]["role"] == "assistant"
    assert rows[1]["turn"] == 2


def test_leak_guard_aborts_rather_than_dropping(tmp_path):
    """A rollout on an `--exclude` item aborts the build; it is not dropped
    silently."""
    run = tmp_path / "roll--bench"
    run.mkdir()
    (run / "trajectories.jsonl").write_text(json.dumps(_rec()) + "\n")
    bench = tmp_path / "eval.jsonl"
    bench.write_text(json.dumps({"id": "easy-0001"}) + "\n")
    p = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--rollouts", str(run), "--exclude", str(bench),
         "--out", str(tmp_path / "out.jsonl")],
        capture_output=True, text=True)
    assert p.returncode != 0 and "ABORT" in (p.stdout + p.stderr)


def test_cap_bounds_retained_trajectories_per_item(tmp_path):
    """`--cap` bounds the trajectories kept per item, so items that verify on
    most samples do not dominate the rows."""
    run = tmp_path / "roll--bench"
    run.mkdir()
    with open(run / "trajectories.jsonl", "w") as f:
        for _ in range(5):
            f.write(json.dumps(_rec()) + "\n")
    out = tmp_path / "out.jsonl"
    p = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--rollouts", str(run), "--cap", "2", "--out", str(out)],
        capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "kept 2 answered trajectories" in p.stdout
    assert len(out.read_text().splitlines()) == 4, "2 trajectories x 2 assistant turns"


def test_strip_user_note_removes_only_the_appended_note():
    """The generation-time note appended to the first user turn is removed,
    and nothing else changes."""
    note = "Before every tool call, first write one or two sentences inside <think>."
    rec = {"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Question: q\n\nBudget: 20 tool calls.\n\n" + note},
        {"role": "assistant", "content": "<think>x</think><tool_call>{}</tool_call>"},
        {"role": "user", "content": "OBSERVATION: ... " + note},   # a later user turn is untouched
    ]}
    assert brr.strip_user_note(rec, note) is True
    assert rec["messages"][1]["content"] == "Question: q\n\nBudget: 20 tool calls."
    assert rec["messages"][3]["content"].endswith(note)
    assert brr.strip_user_note(rec, note) is False               # idempotent
    assert brr.strip_user_note(rec, None) is False


def test_honest_abstention_accepted_only_when_the_answer_was_unreachable():
    """A decline is kept only when it is honest: no gold answer was among the
    nodes it saw, and it cites no edge it did not see."""
    decline = "insufficient information in the graph — I could not reach the answer."
    # the gold answer is not among the seen nodes: the decline is honest
    ok, why = brr.honest_abstention(_rec(
        answer=decline, kb_answer_set=["Zirconium"], cited_edges=[]))
    assert ok and why == "ok_abstain"
    # gold "C" is among the seen nodes: the answer was reachable
    bad, why = brr.honest_abstention(_rec(answer=decline, kb_answer_set=["C"]))
    assert not bad and why == "abstain:answer_was_reachable"
    # an answered episode is never an abstention
    no, why = brr.honest_abstention(_rec())
    assert not no and why == "abstain:did_not_abstain"
    # a decline that cites an edge it never saw is not honest either; the gate
    # classifies a decline before it reads any citation
    fab, why = brr.honest_abstention(_rec(
        answer=decline, kb_answer_set=["Zirconium"],
        cited_edges=[{"source": "A", "relation": "invented", "target": "Q"}]))
    assert not fab and why == "abstain:hallucinated_citation"


def test_abstention_ratio_defaults_to_zero():
    """`--abstain-ratio` defaults to 0. The argparse line is read as text, so
    this pins the declared default only, not the script's behaviour."""
    src = SCRIPT.read_text()
    assert '"--abstain-ratio", type=float, default=0.0' in src


def test_the_abstention_filter_is_the_deployed_rule_not_a_second_copy():
    """`abstained()` agrees with `eval.parse.detect_abstention`, the rule the
    gate and the metric use: on declines in several phrasings, on a
    near-decline the metric does not count, and on a real answer."""
    from graph_reasoner.eval.parse import detect_abstention

    for answer in [
        "The evidence is insufficient to name the etchant.",
        "The graph does not contain a path from A to B.",
        "This cannot be determined from the edges I saw.",
        "Unable to determine the answer from the evidence.",
        "No relevant information was returned.",
        "Not enough information.",
        "I could not reach the target node.",       # not an abstention under the metric
        "The etchant is Zirconium.",                # a real answer
    ]:
        assert brr.abstained({"answer": answer}) == bool(detect_abstention(answer)), \
            f"the curation filter and the deployed metric disagree on: {answer!r}"
