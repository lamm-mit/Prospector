"""Item ids are joined exactly: the eval and train id spaces share roots.

The evaluation file and the train pool use the same `<tier>-<number>` ids, so
a train id such as `easy-0002a` differs from the unrelated eval id `easy-0002`
only by a trailing letter. Joins split on the `#` variant separator, which
keeps the letter.

These tests fail if code strips the suffix, or if the id spaces stop
colliding; in the second case, update them to assert the new invariant.
"""
import json
import pathlib
import re

import pytest

from conftest import ROOT

EVAL = ROOT / "data/benchmarks/tsmc/tsmc_core_agentic.jsonl"
TRAIN_POOLS = [ROOT / "data/benchmarks/tsmc/tsmc_train_agentic_4x.jsonl"]

# The guards run per substrate; only tsmc has a tracked train pool.
SUBSTRATES = {
    "tsmc": (EVAL, TRAIN_POOLS),
}

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _ids(path: pathlib.Path) -> set[str]:
    return {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _root(item_id: str) -> str:
    """The id with its trailing letters stripped."""
    return item_id.rstrip(LETTERS)


@pytest.mark.parametrize("substrate", SUBSTRATES)
def test_eval_and_train_ids_are_exactly_disjoint(substrate):
    """The property every join relies on: an exact id belongs to one set."""
    eval_path, pools = SUBSTRATES[substrate]
    ev = _ids(eval_path)
    assert ev and pools, f"eval ids {len(ev)}, pools {len(pools)}"
    for pool in pools:
        assert not (ev & _ids(pool)), f"{pool.name} shares exact ids with the eval set"


@pytest.mark.parametrize("substrate", SUBSTRATES)
def test_eval_and_train_gold_pathsets_are_disjoint(substrate):
    """No train item shares its gold node-set with an eval item, checked on
    the tracked files; the merge and augment scripts enforce the same at build
    time."""
    eval_path, pools = SUBSTRATES[substrate]
    ev = {frozenset(json.loads(l)["gold_path"]["nodes"])
          for l in eval_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    assert ev and pools
    for pool in pools:
        for line in pool.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            assert frozenset(row["gold_path"]["nodes"]) not in ev, \
                f"{pool.name}: {row['id']} shares a gold node-set with an eval item"


def test_the_id_spaces_collide_once_the_suffix_is_stripped():
    """Stripping the letter suffix maps some train id onto an eval id, which
    is why joins must be exact. If the id spaces are made distinct, update
    these tests to assert that instead."""
    ev = _ids(EVAL)
    collided = any(_root(i) in ev for pool in TRAIN_POOLS for i in _ids(pool))
    assert collided, (
        "the eval and train id roots do not collide — if the pools were "
        "renamed, update these tests to assert that")


def test_colliding_ids_are_genuinely_different_questions():
    """A train item whose root matches an eval id is a different question
    with a different gold node-set."""
    ev = {json.loads(l)["id"]: json.loads(l)
          for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()}
    checked = 0
    for line in TRAIN_POOLS[0].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        twin = ev.get(_root(item["id"]))
        if twin is None:
            continue
        checked += 1
        assert item["question"].strip().lower() != twin["question"].strip().lower(), item["id"]
        assert set((item.get("gold_path") or {}).get("nodes") or []) != \
               set((twin.get("gold_path") or {}).get("nodes") or []), item["id"]
    assert checked > 100, f"expected many root collisions to check, saw {checked}"


def test_no_code_strips_a_letter_suffix_from_an_item_id():
    """No code under `src/` or `scripts/` strips the letter suffix.

    `id.rstrip("abc...")`, or a regex that captures `(easy|medium|hard)-\\d+`
    without its suffix, would join `easy-0002a` to the unrelated `easy-0002`.
    Splitting on `#` keeps the suffix.
    """
    offenders = []
    scanned = list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))
    # The count guards against a glob that matches nothing, which would pass
    # without reading any code.
    assert len(scanned) > 50, f"scanned {len(scanned)} files under {ROOT}"
    for path in scanned:
        for n, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            stripping = re.search(r"\.rstrip\(\s*[\"'][a-z]{4,}", line)
            tierre = re.search(r"\((?:easy\|medium\|hard|hard\|medium\|easy)\)-\\d\+(?![a-z\\])", line)
            if stripping or tierre:
                offenders.append(f"{path}:{n}: {line.strip()[:90]}")
    assert not offenders, (
        "code appears to strip an item id's letter suffix, which silently joins "
        "unrelated train and eval items:\n  " + "\n  ".join(offenders))


def test_the_eval_leak_key_has_one_definition():
    """`BenchmarkItem.gold_pathset` and the merge script's `key`, which reads
    raw JSONL rows, compute the same leak key: the set of gold-path nodes."""
    import importlib.util

    from conftest import script
    from graph_reasoner.schema import BenchmarkItem

    spec = importlib.util.spec_from_file_location(
        "merge_train", script("merge_train.py"))
    merge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merge)

    row = {"id": "x", "tier": "hard", "question": "q", "gold_answer": "a",
           "gold_path": {"nodes": ["N1", "N2", "N3"],
                         "edges": [{"source": "N1", "relation": "r",
                                    "target": "N2"}]},
           "subgraph_edges": [], "subgraph_text": "", "meta": {}}
    item = BenchmarkItem(**row)

    assert item.gold_pathset() == merge.key(row), \
        "the merge script and the training-time guard disagree about the leak key"
    assert item.gold_pathset() == frozenset({"N1", "N2", "N3"})
