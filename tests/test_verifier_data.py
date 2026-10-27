"""The verifier data builder: `select/verifier_data.py`, its CLI
`scripts/select/build_verifier_data.py`, the shared prompt in
`select/prompts.py`, and `completion_logprob_span` in `train/tokenize.py`.

Only asserted, gate-passing rollouts become rows, labelled by `gated_acc`. The
prompt the verifier trains on is the one it scores (`render_prompt`), and the
scored log-probability span is exactly the SFT-weighted span, because
`completion_logprob_span` reuses `supervised_arrays`.
"""
import json
import subprocess
import sys

import pytest

from conftest import ROOT, script, unmasked_text, write_run
from graph_reasoner.select import verifier_data as vd
from graph_reasoner.select.prompts import ANSWER_CAP, EDGE_CAP, NO, YES, render_prompt
from graph_reasoner.train.tokenize import completion_logprob_span, supervised_arrays

EDGE = {"source": "A", "relation": "rel", "target": "B"}


def _write_run(tmp_path, name, rows):
    """A minimal eval run dir: the fields `select.load` consumes. A row is
    gate-passing and asserted unless it says otherwise."""
    results = [{"id": r["id"], "em_claim": r.get("em_claim", r.get("gated_acc", False)),
                "gated_acc": r.get("gated_acc", False), "gate_pass": r.get("gate_pass", True),
                "reached_answer": False, "abstained": r.get("abstained", False),
                "asserted": r.get("asserted", True), "n_calls": 3} for r in rows]
    trajectories = [{"id": r["id"], "tier": "easy",
                     "question": r.get("question", "Which node follows A?"),
                     "answer": r.get("answer", "B"),
                     "cited_edges": r.get("cited", [EDGE])} for r in rows]
    return write_run(tmp_path, f"{name}--train_bench", results, trajectories)


def test_labels_are_gated_acc_and_only_asserted_gate_passing_rollouts_count(tmp_path):
    """A gate-passing rollout whose claim is wrong (gated_acc false) is a NO,
    and the rollouts the selector is never offered (an abstention, a
    gate-fail) are not rows at all."""
    d = _write_run(tmp_path, "r1", [
        {"id": "i1", "gated_acc": True},                        # correct claim
        {"id": "i2", "gated_acc": False},                       # wrong claim
        {"id": "i3", "gate_pass": False},                      # gate fail
        {"id": "i4", "asserted": False, "abstained": True},    # abstained
    ])
    rows, stats = vd.build_rows([d], banned=set())
    assert {r["item_id"]: r["completion"][0]["content"] for r in rows} == \
        {"i1": YES, "i2": NO}
    assert [r["label"] for r in rows] == [True, False]
    assert stats["skip_gate_fail"] == 1 and stats["skip_abstained"] == 1
    assert stats["label_yes"] == 1 and stats["label_no"] == 1


def test_prompt_caps_edges_and_carries_question_and_answer(tmp_path):
    edges = [{"source": f"n{i}", "relation": "r", "target": f"n{i+1}"}
             for i in range(20)]
    d = _write_run(tmp_path, "r1", [
        {"id": "i1", "gated_acc": True, "question": "Q-text?",
         "answer": "A" * 400, "cited": edges}])
    rows, _ = vd.build_rows([d], banned=set())
    p = rows[0]["prompt"][0]["content"]
    assert "Q-text?" in p
    assert p.count("--[r]-->") == EDGE_CAP, "edge list capped at 12"
    assert "A" * ANSWER_CAP in p and "A" * (ANSWER_CAP + 1) not in p


def test_question_less_rollouts_are_refused_not_rendered_blank(tmp_path):
    d = _write_run(tmp_path, "r1",
                   [{"id": "i1", "gated_acc": True, "question": ""}])
    rows, stats = vd.build_rows([d], banned=set())
    assert rows == [] and stats["skip_no_question"] == 1


def test_exact_duplicates_collapse_and_conflicting_labels_survive(tmp_path):
    # Same rendered prompt three times: two with label yes (one collapses),
    # one with label no (a conflict — kept, because gate_pass depends on the
    # rollout's own seen set and the disagreement is real).
    spec = dict(question="Q?", answer="B", cited=[EDGE])
    d1 = _write_run(tmp_path, "r1", [dict(spec, id="i1", gated_acc=True)])
    d2 = _write_run(tmp_path, "r2", [dict(spec, id="i1", gated_acc=True)])
    d3 = _write_run(tmp_path, "r3", [dict(spec, id="i1", gated_acc=False)])
    rows, _ = vd.build_rows([d1, d2, d3], banned=set())
    deduped, dups, conflicts = vd.dedupe(rows)
    assert len(deduped) == 2 and dups == 1 and conflicts == 1
    assert {r["completion"][0]["content"] for r in deduped} == {YES, NO}


def test_leaked_eval_item_aborts(tmp_path):
    d = _write_run(tmp_path, "r1", [{"id": "core-1", "gated_acc": True}])
    with pytest.raises(SystemExit, match="ABORT"):
        vd.build_rows([d], banned={"core-1"})


def test_rows_round_trip_load_transcripts_one_to_one(tmp_path):
    from graph_reasoner.train.transcripts import load_transcripts

    # The same item across two runs (distinct answers), plus a second item —
    # the shape that stresses the per-item prefix-containment dedup.
    d1 = _write_run(tmp_path, "r1", [
        {"id": "i1", "gated_acc": True, "answer": "B"},
        {"id": "i2", "gated_acc": False},
    ])
    d2 = _write_run(tmp_path, "r2", [
        {"id": "i1", "gated_acc": False, "answer": "C"},
    ])
    rows, _ = vd.build_rows([d1, d2], banned=set())
    assert len(rows) == 3
    out = tmp_path / "v.jsonl"
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    transcripts = load_transcripts(str(out), verbose=False)
    # Prefix-containment dedup cannot merge 2-message rows unless identical:
    # every emitted row must survive as its own transcript.
    assert len(transcripts) == len(rows)
    assert all(len(t) == 2 for t in transcripts)


def test_cli_writes_the_rows_and_a_provenance_sidecar(tmp_path):
    """The script's output is the builder's rows, deduplicated, and a sidecar
    naming the arguments and the source runs, so a stored row file can say
    what made it."""
    d = _write_run(tmp_path, "r7", [{"id": "i1", "gated_acc": True},
                                   {"id": "i2", "gated_acc": False},
                                   {"id": "i3", "gate_pass": False}])
    bench = tmp_path / "core.jsonl"
    bench.write_text(json.dumps({"id": "core-1"}) + "\n")
    out = tmp_path / "rows.jsonl"
    proc = subprocess.run(
        [sys.executable, str(script("build_verifier_data.py")), "--rollouts", str(d),
         "--exclude", str(bench), "--out", str(out)],
        capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [(r["item_id"], r["label"]) for r in rows] == [("i1", True), ("i2", False)]
    side = json.loads((tmp_path / "rows.jsonl.provenance.json").read_text())
    assert side["tool"] == "scripts/select/build_verifier_data.py"
    assert side["argv"][:2] == ["--rollouts", str(d)]
    assert [r["run"] for r in side["source_runs"]] == [d.name]
    assert side["label"] == "gated_acc" and side["exclude"] == [str(bench)]
    assert side["rows"] == 2 and side["positive"] == 1
    assert side["counts"]["skip_gate_fail"] == 1


def test_scored_span_is_exactly_the_sft_weighted_span(shipped_tokenizer):
    tok = shipped_tokenizer
    msgs = [{"role": "user", "content": render_prompt("Q?", "B", [EDGE])},
            {"role": "assistant", "content": YES}]
    inp, tgt, w = supervised_arrays(msgs, tok)
    full, positions = completion_logprob_span(msgs, tok)
    # Same tokenization: full == the untruncated ids supervised_arrays saw.
    assert full == list(inp) + [tgt[-1]]
    # positions index the per-token log-probabilities of `full` at exactly the weighted targets.
    assert positions == [i + 1 for i, wi in enumerate(w) if wi]
    trained = unmasked_text(tgt, w, tok)
    assert YES in trained
    assert "Question" not in trained, "prompt tokens must carry no score"


def test_yes_and_no_transcripts_share_the_prompt_prefix(shipped_tokenizer):
    tok = shipped_tokenizer
    prompt = render_prompt("Q?", "B", [EDGE])
    fy, py = completion_logprob_span(
        [{"role": "user", "content": prompt},
         {"role": "assistant", "content": YES}], tok)
    fn, pn = completion_logprob_span(
        [{"role": "user", "content": prompt},
         {"role": "assistant", "content": NO}], tok)
    cut = min(py[0], pn[0])
    assert fy[:cut] == fn[:cut], "margins must compare like with like"


def test_completion_logprob_span_refuses_multi_assistant(shipped_tokenizer):
    tok = shipped_tokenizer
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": YES}]
    with pytest.raises(ValueError, match="exactly one assistant"):
        completion_logprob_span(msgs, tok)
