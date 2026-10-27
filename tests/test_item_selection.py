"""Which benchmark items a run draws: `select_items` and `ids_file` in
`agent/eval.py`, tier-balanced shards (`shard="k/n"`) and a run restricted to
a list of item ids."""
from types import SimpleNamespace

import pytest

from graph_reasoner.agent.eval import select_items


def _items():
    # three tiers of unequal size
    tiers = ["easy"] * 7 + ["medium"] * 6 + ["hard"] * 5
    return [SimpleNamespace(id=f"{t}-{i:04d}", tier=t) for i, t in enumerate(tiers)]


def test_shards_partition_the_benchmark_tier_by_tier():
    items = _items()
    shards = [select_items(items, None, f"{k}/3") for k in range(3)]
    ids = [x.id for s in shards for x in s]
    assert sorted(ids) == sorted(x.id for x in items)          # a partition
    assert len(set(ids)) == len(items)                           # disjoint
    for s in shards:                                             # balanced per tier
        for t, n in (("easy", 7), ("medium", 6), ("hard", 5)):
            got = sum(1 for x in s if x.tier == t)
            assert n // 3 <= got <= n // 3 + 1


def test_shard_applies_after_the_per_tier_limit():
    items = _items()
    s = select_items(items, 4, "1/2")
    assert all(x.tier in ("easy", "medium", "hard") for x in s)
    assert len(s) == 6                                           # 4 per tier -> every second one


@pytest.mark.parametrize("bad", ["3/3", "-1/2", "a/b", "2"])
def test_malformed_shard_is_refused(bad):
    with pytest.raises(ValueError):
        select_items(_items(), None, bad)


def test_ids_file_restricts_a_run_to_the_listed_items(tmp_path, monkeypatch):
    """`ids_file` restricts a run to the listed ids, and an id that is not in
    the benchmark is an error."""
    from graph_reasoner.agent import eval as ev
    items = _items()
    ids = tmp_path / "ids.txt"
    ids.write_text("easy-0001\nhard-0015\n")
    monkeypatch.setattr("graph_reasoner.benchmarks.load_benchmark", lambda p: items)
    captured = {}
    def fake_run_episodes(sel, *a, **k):
        captured["ids"] = [x.id for x in sel]
        raise RuntimeError("stop here")
    monkeypatch.setattr(ev, "run_episodes", fake_run_episodes)
    monkeypatch.setattr(ev, "load_backend", lambda *a, **k: type("B", (), {
        "name": "toy", "fingerprint": lambda self: "x"})())
    monkeypatch.setattr(ev, "infer_backend_kind", lambda *a, **k: "toy")
    with pytest.raises(RuntimeError, match="stop here"):
        ev.run_agentic_eval("bench.jsonl", "gpt-x", kb_path="kb", ids_file=str(ids))
    assert captured["ids"] == ["easy-0001", "hard-0015"]
    ids.write_text("easy-0001\nnot-an-item\n")
    with pytest.raises(SystemExit, match="not in"):
        ev.run_agentic_eval("bench.jsonl", "gpt-x", kb_path="kb", ids_file=str(ids))
