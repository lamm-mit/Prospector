"""The informative band: keep the items the policy sometimes solves.

A GRPO group yields gradient only when its rollouts disagree, so an item the
policy solves never or always adds nothing. The rule is a pure function of an
item's stored outcomes and (lo, hi), `band_keep` in `train/rollout.py`, which
both loaders call. These tests pin the rule, and that the two loaders read
`tiers` the same way.
"""
from __future__ import annotations

import json

import pytest

from graph_reasoner.train.rollout import band_keep


def _band_keep(rates, lo=0.05, hi=0.95):
    """The rule over a table of items: keep sometimes-but-not-always."""
    return [rid for rid, v in rates.items() if band_keep(v, lo, hi)]


RATES = {
    "never":     [False, False, False, False],   # 0/4  -> dropped
    "quarter":   [True, False, False, False],    # 1/4  -> kept
    "half":      [True, True, False, False],     # 2/4  -> kept
    "three":     [True, True, True, False],      # 3/4  -> kept
    "always":    [True, True, True, True],       # 4/4  -> dropped
    "unmeasured": [],                            # absent -> kept
}


def test_band_keeps_only_the_disagreeing_items():
    assert sorted(_band_keep(RATES)) == ["half", "quarter", "three", "unmeasured"]


def test_unmeasured_is_kept_not_dropped():
    """An item with no stored outcome is kept: unmeasured is not the same as
    uninformative."""
    assert "unmeasured" in _band_keep({"unmeasured": []})


def test_the_trainer_runs_this_band(tmp_path, monkeypatch):
    """`load_train_items` applies the band: it keeps the items with mixed
    outcomes and the unmeasured one."""
    import graph_reasoner.train.rollout as rollout

    items = [{"id": rid, "tier": "hard", "question": "q", "gold_answer": "a",
              "gold_path": {"nodes": [rid], "edges": []},
              "subgraph_edges": [], "subgraph_text": "", "meta": {}}
             for rid in RATES]
    bench = tmp_path / "b.jsonl"
    bench.write_text("\n".join(json.dumps(i) for i in items))

    # Stand in for the stored rollout passes. Pass i reports each item's i-th
    # outcome; "unmeasured" has none and so appears in no pass.
    passes = [f"pass{i}" for i in range(4)]

    def fake_load(d, recompute=True):
        i = passes.index(str(d))
        return {rid: {"gated_acc": vs[i]} for rid, vs in RATES.items() if vs}

    import graph_reasoner.select as sel
    monkeypatch.setattr(sel, "load", fake_load, raising=True)

    kept = {i.id for i in rollout.load_train_items({
        "benchmark": str(bench), "tiers": ["hard"],
        "informative_band": {"rollouts": passes, "lo": 0.05, "hi": 0.95}})}
    assert kept == {"quarter", "half", "three", "unmeasured"}, kept
    assert "never" not in kept and "always" not in kept


def _item(rid, tier):
    return {"id": rid, "tier": tier, "question": "q", "gold_answer": "a",
            "gold_path": {"nodes": [rid], "edges": []},
            "subgraph_edges": [], "subgraph_text": "", "meta": {}}


@pytest.mark.parametrize("tiers, want", [
    (None, {"e", "m", "h"}),          # absent: every tier, in both modes
    ([], {"e", "m", "h"}),
    (["medium", "hard"], {"m", "h"}),
    (["hard"], {"h"}),
])
def test_both_loaders_read_tiers_the_same_way(tmp_path, tiers, want):
    """An absent or empty `tiers` selects every tier in both the full-episode
    and the prefix loader, and a listed set selects the same items in either
    mode."""
    from graph_reasoner.train.prefixes import PREFIX_SCHEMA
    from graph_reasoner.train.rollout import load_train_items, load_train_prefixes

    items = [_item("e", "easy"), _item("m", "medium"), _item("h", "hard")]
    bench = tmp_path / "b.jsonl"
    bench.write_text("\n".join(json.dumps(i) for i in items))
    prefixes = tmp_path / "p.jsonl"
    prefixes.write_text("\n".join(json.dumps({
        "schema": PREFIX_SCHEMA, "id": f"{i['id']}#p", "item_id": i["id"],
        "messages": [], "seen_edges": [], "n_prefix_turns": 0}) for i in items))

    config = {"benchmark": str(bench), "prefixes": str(prefixes)}
    if tiers is not None:
        config["tiers"] = tiers
    from_items = {i.id for i in load_train_items(config)}
    from_prefixes = {it.id for _p, it in load_train_prefixes(config)}
    assert from_items == from_prefixes == want
