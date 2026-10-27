"""The evaluation items whose question names an answer-set member other than
the anchor, pinned by id.

`bench.agentic.augment` drops such an item ("an answer-set member is named in
the question"), since a model that restates the question is credited for it.
The released files predate the drop and keep theirs: the lists below are those
items, so a re-curation that removes them updates the lists in the same
commit, and no new one slips in.
"""
import json

import pytest

from conftest import ROOT
from graph_reasoner.bench.agentic import names_an_answer

NAMED = {
    "tsmc": ["easy-0011", "medium-0005", "medium-0044", "medium-0054", "medium-0077", "medium-0084",
             "hard-0013", "hard-0081", "hard-0098"],
    "biograph": ["easy-0040"],
    "pfas": ["easy-0015", "easy-0062", "easy-0075", "easy-0087", "medium-0031", "medium-0034", "medium-0049",
             "hard-0008", "hard-0029", "hard-0047", "hard-0092", "hard-0098"],
    "matprop": ["medium-0003", "hard-0004", "hard-0035"],
}


@pytest.mark.parametrize("graph", sorted(NAMED))
def test_the_questions_naming_an_answer_are_the_pinned_ones(graph):
    path = ROOT / "data/benchmarks" / graph / f"{graph}_core_agentic.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows, path
    hits = [r["id"] for r in rows
            if names_an_answer(r["question"], r["meta"].get("anchor") or "", r["meta"].get("kb_answer_set") or [])]
    assert hits == NAMED[graph]


def test_the_rule_spares_the_anchor_and_catches_another_spelling():
    assert names_an_answer("How does plasma etching affect the wafer?", "plasma etching", ["plasma etching"]) == []
    assert names_an_answer("Which gate-first step follows?", "x", ["gate first", "gate-first"]) == ["gate-first"]
