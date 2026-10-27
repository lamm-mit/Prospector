"""Resuming an episode from a stored prefix: `seed_episode` and
`resume_episodes_batched` in `agent/loop.py`, `train/prefixes.py`, and
`load_train_prefixes` in `train/rollout.py`. Stage 4 of the recipe trains on
episodes resumed this way.

A resumed episode must be the episode the loop would have produced had the
model emitted those turns live: same seen set, same call count, same budgets
from that point on. A scripted generator stands in for the model, on the film
graph from conftest.
"""
import json

import pytest

from conftest import FILM_GOLD_EDGES, final_answer, tool_call
from graph_reasoner.agent.loop import (EpisodeConfig, resume_episodes_batched,
                                       run_episodes_batched, seed_episode,
                                       trajectory_record)
from graph_reasoner.agent.metrics import agentic_metrics
from graph_reasoner.agent.prompts import FORCED_ANSWER
from graph_reasoner.train.prefixes import (PREFIX_SCHEMA, load_prefixes,
                                                zero_prefix_turns)


def _scripted(turns):
    """A generate seam that replays `turns` per conversation, in order."""
    state = {"i": 0}

    def generate(conversations, max_new_tokens):
        out = []
        for _ in conversations:
            out.append(turns[min(state["i"], len(turns) - 1)])
            state["i"] += 1
        return out
    return generate


def _run(kb, item, cfg, turns):
    (ep,) = run_episodes_batched([item], kb, _scripted(turns), cfg)
    return ep


def test_seed_reproduces_the_state_before_the_final_turn(film_kb, film_item):
    cfg = EpisodeConfig(max_turns=5)
    ep = _run(film_kb, film_item, cfg, [
        tool_call("expand", {"node": "Film A", "relation": "written_by"}),
        tool_call("expand", {"node": "Jane Doe"}),
        final_answer("Film A", FILM_GOLD_EDGES),          # the source episode picks a rival
    ])
    rec = json.loads(json.dumps(trajectory_record(ep, film_item, film_kb, cfg, model="scripted")))
    assert not agentic_metrics(ep, film_item)["gated_acc"]

    seeded = seed_episode(rec, film_item, cfg, episode_id="p1")
    assert seeded.item_id == "p1" and seeded.status == "running" and not seeded.forced
    assert seeded.resumed_turns == 2 and seeded.n_turns == 2 and seeded.n_calls == 2
    assert seeded.messages == rec["messages"][:-1]
    assert set(seeded.seen) == set(ep.seen)
    assert seeded.gold_complete_at_call == ep.gold_complete_at_call
    # the repeat index survives: re-issuing a prefix call is a repeat, not a fetch
    assert len(seeded._call_index) == 2
    # the observation text was read back for the repeat echo
    assert all(c.result_text for c in seeded.calls)


def test_resumed_final_turn_is_scored_like_a_live_one(film_kb, film_item):
    cfg = EpisodeConfig(max_turns=5)
    ep = _run(film_kb, film_item, cfg, [
        tool_call("expand", {"node": "Film A", "relation": "written_by"}),
        tool_call("expand", {"node": "Jane Doe"}),
        final_answer("Film A", FILM_GOLD_EDGES),
    ])
    rec = json.loads(json.dumps(trajectory_record(ep, film_item, film_kb, cfg, model="scripted")))
    seeded = seed_episode(rec, film_item, cfg, episode_id="p1")
    (out,) = resume_episodes_batched([seeded], film_kb, _scripted([final_answer("Film B", FILM_GOLD_EDGES)]), cfg)
    m = agentic_metrics(out, film_item)
    assert m["gated_acc"] and out.status == "answered"
    assert out.n_turns == 3 and out.n_calls == 2 and out.resumed_turns == 2
    # the corrected episode is byte-identical to a live one up to the final turn
    assert out.messages[:-1] == rec["messages"][:-1]


def test_resumed_continuation_may_search_and_still_meets_the_cap(film_kb, film_item):
    cfg = EpisodeConfig(max_turns=3)
    ep = _run(film_kb, film_item, cfg, [
        tool_call("expand", {"node": "Film A", "relation": "written_by"}),
        final_answer("Jane Doe", FILM_GOLD_EDGES[:1]),
    ])
    rec = json.loads(json.dumps(trajectory_record(ep, film_item, film_kb, cfg, model="scripted")))
    seeded = seed_episode(rec, film_item, cfg, episode_id="p2")
    assert seeded.n_calls == 1
    (out,) = resume_episodes_batched([seeded], film_kb, _scripted([
        tool_call("expand", {"node": "Jane Doe"}),
        tool_call("expand", {"node": "Film B"}),          # third call hits the cap -> forced final
        final_answer("Film B", FILM_GOLD_EDGES),
    ]), cfg)
    assert out.n_calls == 3 and out.status == "budget"
    assert agentic_metrics(out, film_item)["gated_acc"]
    assert any(m.get("content") == FORCED_ANSWER for m in out.messages)


def test_forced_prefix_gets_one_final_and_keeps_its_status(film_kb, film_item):
    cfg = EpisodeConfig(max_turns=1)
    ep = _run(film_kb, film_item, cfg, [
        tool_call("expand", {"node": "Film A", "relation": "written_by"}),   # cap -> forced
        final_answer("Jane Doe", FILM_GOLD_EDGES[:1]),
    ])
    assert ep.status == "budget"
    rec = json.loads(json.dumps(trajectory_record(ep, film_item, film_kb, cfg, model="scripted")))
    assert rec["messages"][-2]["content"] == FORCED_ANSWER
    seeded = seed_episode(rec, film_item, cfg, episode_id="p3")
    assert seeded.forced and seeded.forced_pending and seeded.status == "budget"
    assert not seeded.active            # inactive, like a live episode at this point
    calls = {"n": 0}

    def generate(conversations, max_new_tokens):
        calls["n"] += 1
        assert max_new_tokens == cfg.final_max_tokens     # the final budget, as live
        return [final_answer("Film A", FILM_GOLD_EDGES[:1]) for _ in conversations]
    (out,) = resume_episodes_batched([seeded], film_kb, generate, cfg)
    assert calls["n"] == 1 and out.status == "budget" and out.final_text
    assert out.n_calls == 1 and out.n_turns == 2


def test_seed_refuses_a_budget_mismatch(film_kb, film_item):
    cfg = EpisodeConfig(max_turns=5)
    ep = _run(film_kb, film_item, cfg, [tool_call("expand", {"node": "Film A"}), final_answer("Film B", FILM_GOLD_EDGES)])
    rec = json.loads(json.dumps(trajectory_record(ep, film_item, film_kb, cfg, model="scripted")))
    with pytest.raises(ValueError, match="carries its budget"):
        seed_episode(rec, film_item, EpisodeConfig(max_turns=20))


def test_mid_episode_cut_truncates_seen_and_calls(film_kb, film_item):
    cfg = EpisodeConfig(max_turns=5)
    ep = _run(film_kb, film_item, cfg, [
        tool_call("expand", {"node": "Film A", "relation": "written_by"}),
        tool_call("expand", {"node": "Jane Doe"}),
        final_answer("Film B", FILM_GOLD_EDGES),
    ])
    rec = json.loads(json.dumps(trajectory_record(ep, film_item, film_kb, cfg, model="scripted")))
    seeded = seed_episode(rec, film_item, cfg, cut=2)       # before the second assistant turn
    assert seeded.n_calls == 1 and seeded.resumed_turns == 1
    assert len(seeded.seen) == rec["calls"][0]["cum_edges_seen"]
    assert seeded.gold_complete_at_call is None        # completed at call 2, after the cut


def test_prefix_record_is_resumed_where_it_ends(film_kb, film_item):
    """A record whose messages already end in a user turn (build_prefixes.py
    output) is resumed where it ends, not cut again."""
    cfg = EpisodeConfig(max_turns=1)
    ep = _run(film_kb, film_item, cfg, [
        tool_call("expand", {"node": "Film A", "relation": "written_by"}),   # cap -> forced
        final_answer("Jane Doe", FILM_GOLD_EDGES[:1]),
    ])
    rec = json.loads(json.dumps(trajectory_record(ep, film_item, film_kb, cfg, model="scripted")))
    positions = [i for i, m in enumerate(rec["messages"]) if m["role"] == "assistant"]
    pre = {**rec, "messages": rec["messages"][:positions[-1]], "n_prefix_turns": len(positions) - 1}
    seeded = seed_episode(pre, film_item, cfg, episode_id="p4")
    assert seeded.messages == pre["messages"]
    assert seeded.forced and seeded.forced_pending and seeded.n_calls == 1
    assert seeded.resumed_turns == 1
    # and a full record still cuts before its final turn — same state
    full = seed_episode(rec, film_item, cfg, episode_id="p5")
    assert full.messages == seeded.messages and full.n_calls == seeded.n_calls
    with pytest.raises(ValueError, match="n_prefix_turns"):
        seed_episode({**pre, "n_prefix_turns": 0}, film_item, cfg)


def test_zero_prefix_turns_masks_only_the_prefix():
    mask = [0, 0, 1, 1, 0, 1, 1, 0, 1]
    turn = [-1, -1, 0, 0, -1, 1, 1, -1, 2]
    assert zero_prefix_turns(mask, turn, 2) == [0, 0, 0, 0, 0, 0, 0, 0, 1]
    assert zero_prefix_turns(mask, turn, 0) == mask
    with pytest.raises(ValueError):
        zero_prefix_turns([1], [0, 0], 1)


def test_load_train_prefixes_filters_leaks_and_bands(film_kb, film_item, tmp_path):
    cfg = EpisodeConfig(max_turns=5)
    ep = _run(film_kb, film_item, cfg, [tool_call("expand", {"node": "Film A"}), final_answer("Film A", FILM_GOLD_EDGES)])
    rec = json.loads(json.dumps(trajectory_record(ep, film_item, film_kb, cfg, model="scripted")))
    positions = [i for i, m in enumerate(rec["messages"]) if m["role"] == "assistant"]
    base = {**rec, "schema": PREFIX_SCHEMA, "item_id": film_item.id, "class": "fixable",
            "n_prefix_turns": len(positions) - 1, "messages": rec["messages"][:positions[-1]]}
    prefixes = tmp_path / "p.jsonl"
    with prefixes.open("w") as f:
        for pid in ("a#medium-0000", "b#medium-0000", "c#medium-0000"):
            f.write(json.dumps({**base, "id": pid}) + "\n")
    assert len(load_prefixes(prefixes)) == 3

    bench = tmp_path / "bench.jsonl"
    bench.write_text(json.dumps(film_item.model_dump()) + "\n")
    run = tmp_path / "pass--p"
    run.mkdir()
    with (run / "results.jsonl").open("w") as f:
        for pid, hits in (("a#medium-0000", [1, 0, 1, 0]), ("b#medium-0000", [1, 1, 1, 1]),
                          ("c#medium-0000", [0, 0, 0, 0])):
            for h in hits:
                f.write(json.dumps({"id": pid, "gated_acc": bool(h)}) + "\n")

    from graph_reasoner.train.rollout import load_train_prefixes
    pairs = load_train_prefixes({"prefixes": str(prefixes), "benchmark": str(bench),
                                 "prefix_band": {"rollouts": [str(run)], "lo": 0.05, "hi": 0.95}})
    assert [p["id"] for p, _ in pairs] == ["a#medium-0000"]
    assert pairs[0][1].id == film_item.id
    # the leak filter is the item's gold-path set, like load_train_items
    with pytest.raises(SystemExit):
        load_train_prefixes({"prefixes": str(prefixes), "benchmark": str(bench),
                             "exclude": [str(bench)]})
