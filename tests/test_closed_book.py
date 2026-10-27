"""The closed-book arm is the evaluation's questions with the evidence removed.

`scripts/arms/build_closed_book.py` builds the arm's file from a graph's
evaluation file; the file is built, not tracked. These tests build the arm and
check what a model is shown on it, and the variants of the static prompts in
`eval/static_prompts.py`.
"""
import importlib.util
import json
import shutil
import subprocess
import sys

import pytest

from conftest import ROOT, script
from graph_reasoner.benchmarks import load_benchmark

# The evaluation file: the arm's source, and an item that carries edges for the
# retrieved-subgraph prompt's tests.
EVAL = ROOT / "data/benchmarks/tsmc/tsmc_core_agentic.jsonl"


def _builder():
    spec = importlib.util.spec_from_file_location(
        "build_closed_book", script("build_closed_book.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def arm():
    """The closed-book items, built from the evaluation file."""
    closed_book = _builder().closed_book
    return [closed_book(i) for i in load_benchmark(EVAL)]


def test_the_arm_is_the_evaluations_questions_under_their_ids(arm):
    """Same questions in the same order, each id tagged `#closed_book`: the
    scorers join an answer to its question's answer set by the part before
    `#`, so a renumbered or reordered arm would score against the wrong set."""
    source = load_benchmark(EVAL)
    assert len(arm) == len(source) == 195
    for got, src in zip(arm, source):
        assert got.id == f"{src.id}#closed_book"
        assert (got.question, got.tier, got.gold_path) == (src.question, src.tier, src.gold_path)
        assert got.meta["perturbation"]["variant"] == "closed_book"


def test_the_arm_has_no_evidence_at_all(arm):
    """No item carries an edge or any subgraph text."""
    assert len(arm) > 100          # an empty arm would satisfy the loop below
    for item in arm:
        assert item.subgraph_edges == []
        assert not item.subgraph_text.strip()


def test_the_script_writes_the_file_the_runs_are_named_for(tmp_path):
    """The script writes `<graph>_core__closed_book.jsonl` beside its source,
    with a provenance sidecar; `benchmarks.answers_for` maps that name back to
    the evaluation file."""
    src = tmp_path / EVAL.name
    shutil.copy(EVAL, src)
    p = subprocess.run([sys.executable, str(script("build_closed_book.py")), str(src)],
                       capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr
    out = tmp_path / "tsmc_core__closed_book.jsonl"
    assert len(load_benchmark(out)) == 195
    side = json.loads((tmp_path / (out.name + ".provenance.json")).read_text())
    assert side["n_items"] == 195


def test_a_closed_book_item_is_never_asked_to_read_a_graph(arm):
    """A closed-book item renders the question-only prompt. The graph prompt
    on it would say "Answer using ONLY the information in the graph edges
    below" and show no edges."""
    from graph_reasoner.eval.static_prompts import build_eval_prompt

    for it in arm[:5]:
        prompt = build_eval_prompt(it)
        assert "your own knowledge" in prompt, f"{it.id}: not the closed-book prompt"
        assert "ONLY the information in the graph" not in prompt, \
            f"{it.id}: asked to read a graph the arm deliberately withholds"
        assert it.question in prompt


def test_the_graph_arms_still_get_the_graph_prompt():
    """An item that carries edges still renders the graph prompt."""
    from graph_reasoner.eval.static_prompts import build_eval_prompt

    it = load_benchmark(EVAL)[0]
    prompt = build_eval_prompt(it)
    assert "ONLY the information in the graph" in prompt, "an item with edges lost its graph"
    assert it.subgraph_text in prompt


def test_static_prompt_nothink_is_the_same_contract_without_the_reasoning_line():
    """EVAL_PROMPT_NOTHINK is the retrieved-subgraph prompt without the
    reasoning instruction, renumbered; the citation block and the ANSWER line
    stay, and the default rendering is EVAL_PROMPT."""
    from graph_reasoner.eval.static_prompts import (EVAL_PROMPT, EVAL_PROMPT_NOTHINK,
                                                    build_eval_prompt)
    assert "think" not in EVAL_PROMPT_NOTHINK
    for must in ("<graph_json>", "ANSWER: <your concise answer>", "1. Output a <graph_json>",
                 "2. End with a final line"):
        assert must in EVAL_PROMPT_NOTHINK
    item = load_benchmark(EVAL)[0]
    assert build_eval_prompt(item) == EVAL_PROMPT.replace("{EDGES}", item.subgraph_text).replace("{QUESTION}", item.question)
    nothink = build_eval_prompt(item, think=False)
    assert "think" not in nothink and item.question in nothink and item.subgraph_text in nothink


def test_static_prompt_label_rule_is_one_added_instruction_and_closed_book_takes_none(arm):
    """EVAL_PROMPT_LABEL and EVAL_PROMPT_NOTHINK_LABEL are the retrieval
    prompts plus one numbered instruction after the ANSWER line, identical
    otherwise. The closed-book prompt shows no labels and renders the same
    with or without `label`."""
    from graph_reasoner.eval.static_prompts import (EVAL_PROMPT, EVAL_PROMPT_LABEL, EVAL_PROMPT_NOTHINK,
                                                    EVAL_PROMPT_NOTHINK_LABEL, LABEL_RULE,
                                                    build_eval_prompt)
    assert EVAL_PROMPT_LABEL.replace("\n4. " + LABEL_RULE, "") == EVAL_PROMPT
    assert EVAL_PROMPT_NOTHINK_LABEL.replace("\n3. " + LABEL_RULE, "") == EVAL_PROMPT_NOTHINK
    assert "think" not in EVAL_PROMPT_NOTHINK_LABEL
    assert EVAL_PROMPT_LABEL.count(LABEL_RULE) == 1
    item = load_benchmark(EVAL)[0]
    assert build_eval_prompt(item, label=True) == build_eval_prompt(item) .replace(
        "ANSWER: <your concise answer>.", "ANSWER: <your concise answer>.\n4. " + LABEL_RULE)
    assert build_eval_prompt(item, think=False, label=True) == build_eval_prompt(item, think=False).replace(
        "ANSWER: <your concise answer>.", "ANSWER: <your concise answer>.\n3. " + LABEL_RULE)
    closed = arm[0]
    assert build_eval_prompt(closed, label=True) == build_eval_prompt(closed)
    assert LABEL_RULE not in build_eval_prompt(closed, label=True)
