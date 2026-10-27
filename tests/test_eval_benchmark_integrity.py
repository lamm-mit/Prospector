"""The evaluation file is curated by hand; these tests pin its contents.

`data/benchmarks/tsmc/tsmc_core_agentic.jsonl` is the output of
`scripts/bench/augment_agentic.py` amended: it has 195 items, without
`hard-0023`; 18 items carry answer sets widened with another spelling of the
gold answer or with another node of the graph that answers the question, 11
of them recording each amendment as `{date, added, reason}` under
`meta.amendments` and seven carrying no record (`UNRECORDED`, pinned by
content); and `hard-0053`'s gold answer is corrected from the far end of its
gold path, which the script would make the answer, to the component the
question asks for. Re-running the script does not reproduce the file, so the
script refuses to overwrite a tracked benchmark without `--force`.

A deliberate re-curation updates the constants here in the same commit.
"""
import json

import pytest

from conftest import ROOT

EVAL = ROOT / "data/benchmarks/tsmc/tsmc_core_agentic.jsonl"


def _rows():
    return [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def test_the_eval_is_the_amended_195_not_the_raw_196():
    rows = _rows()
    assert len(rows) == 195, (
        f"{len(rows)} items — the shipped eval holds 195. 196 means this is raw "
        f"augment output: it still carries hard-0023 and lacks the amendments.")
    assert not [r for r in rows if r["id"] == "hard-0023"]


def test_every_amendment_is_self_describing():
    """Every amendment carries `date`, `added` and a non-empty `reason`."""
    amended = [r for r in _rows() if (r.get("meta") or {}).get("amendments")]
    assert len(amended) == 11, (
        f"{len(amended)} amended items, expected 11")
    for r in amended:
        for a in r["meta"]["amendments"]:
            assert set(a) == {"date", "added", "reason"}, (r["id"], a)
            assert a["added"] and a["reason"].strip(), r["id"]


#: The widenings without a record: id -> the entries the script's relation walk
#: does not produce: a spelling of the gold answer the graph uses, or another
#: node that answers the question.
UNRECORDED = {
    "easy-0050": ["low temperatures"],
    "medium-0018": ["Ar/C4F8/O2", "CF4 plasma"],
    "medium-0054": ["gate-first"],
    "medium-0068": ["ALE", "atomic layer etching"],
    "medium-0089": ["Metallic Contaminants", "Organic Contaminants"],
    "hard-0013": ["room-temperature DI water rinse"],
    "hard-0081": ["airborne particle levels"],
}


def test_the_unrecorded_amendments_are_pinned():
    """The seven widenings that carry no record, and the corrected gold answer."""
    rows = {r["id"]: r for r in _rows()}
    for i, added in UNRECORDED.items():
        meta = rows[i]["meta"]
        assert not meta.get("amendments"), (
            f"{i} now carries a record: move it out of UNRECORDED")
        missing = [a for a in added if a not in meta["kb_answer_set"]]
        assert not missing, (i, missing)
    r = rows["hard-0053"]
    assert r["gold_path"]["nodes"][0] == "cost reduction"      # the far end
    assert r["gold_answer"] == "gas chromatograph"
    assert r["meta"]["kb_answer_set"] == ["gas chromatograph"]
    assert not r["meta"].get("amendments")


def test_the_agentic_scoring_fields_are_present():
    """Every item carries `kb_answer_set`: every answer the gold relation
    sequence reaches over the full graph."""
    rows = _rows()
    assert rows, "no rows read — the check below would pass on an empty file"
    missing = [r["id"] for r in rows
               if not (r.get("meta") or {}).get("kb_answer_set")]
    assert not missing, f"{len(missing)} items lack kb_answer_set, e.g. {missing[:3]}"


def test_answer_set_of_prefers_the_full_graph_set():
    """The scored set is `kb_answer_set`, then `answer_set`, then the single
    gold answer; swapping the first two would change the scored set on 94 of
    the 195 semiconductor items."""
    from graph_reasoner.schema import answer_set_of
    assert answer_set_of({"kb_answer_set": ["a", "b"], "answer_set": ["a"]}, "a") == ["a", "b"]
    assert answer_set_of({"answer_set": ["x"]}, "g") == ["x"]
    assert answer_set_of({}, "g") == ["g"]
    assert answer_set_of(None, None) == []


def test_every_item_is_scored_against_its_kb_answer_set():
    from graph_reasoner.schema import BenchmarkItem
    for r in _rows():
        assert BenchmarkItem(**r).answer_set() == r["meta"]["kb_answer_set"], r["id"]


@pytest.mark.parametrize("path", sorted((ROOT / "data/benchmarks").glob("*/*_core_agentic.jsonl")),
                         ids=lambda p: p.parent.name)
def test_every_graph_s_evaluation_items_carry_kb_answer_set(path):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows, path
    missing = [r["id"] for r in rows if not (r.get("meta") or {}).get("kb_answer_set")]
    assert not missing, f"{path.name}: {len(missing)} items lack kb_answer_set, e.g. {missing[:3]}"
