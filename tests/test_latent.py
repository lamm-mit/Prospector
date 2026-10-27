"""The latent-state package: the moves read off stored episodes, the readings'
statistics, the directions' identities and the figure's aggregation, on
synthetic reads and on episodes the real loop produced from scripted turns
(conftest's film graph). The read-position tests need the shipped tokenizer
and skip without it; the hook tests need torch and skip without it.
"""
import json

import numpy as np
import pytest

from conftest import final_answer, tool_call
from graph_reasoner.agent.loop import is_answer
from graph_reasoner.agent.prompts import FORCED_ANSWER, SCHEMA_REMINDER
from graph_reasoner.latent import directions, figure, moves, readings


# ------------------------------------------------------------ moves

def _page(n, *edges):
    return (f"OBSERVATION (call {n}/20, expand):\n" + "\n".join(f"{a} --[r]--> {b}" for a, b in edges)
            + "\n[1 new edges; 1 edges seen; 18 tool calls left]")


def test_is_answer_is_an_answer_line_without_a_tool_call():
    assert is_answer("<graph_json>{}</graph_json>\nANSWER: Node B")
    assert not is_answer(tool_call("expand", {"node": "x"}))
    assert not is_answer("I would say ANSWER: x <tool_call>{}</tool_call>")
    assert not is_answer("The answer: x")


def test_page_nodes_reads_edge_lines_and_find_hits():
    page = _page(1, ("Anchor Node", "Node B")) + "\n  Node C  [material, 3 connections]\nNo edges returned"
    assert moves.page_nodes(page) == {"anchor node", "node b", "node c"}


def test_intent_of_names_the_seven_moves():
    last, earlier = {"node b"}, {"anchor node", "node a"}
    assert moves.intent_of(tool_call("expand", {"node": "Node B"}), last, earlier) == "follow"
    assert moves.intent_of(tool_call("expand", {"node": "Node A"}), last, earlier) == "backtrack"
    assert moves.intent_of(tool_call("expand", {"node": "Node Z"}), last, earlier) == "fresh"
    assert moves.intent_of(tool_call("find", {"name": "x"}), last, earlier) == "find"
    assert moves.intent_of(tool_call("connect", {"a": "x", "b": "y"}), last, earlier) == "connect"
    assert moves.intent_of(final_answer("Node B", []), last, earlier) == "answer"
    assert moves.intent_of("nothing parseable", last, earlier) == "other"


def _episode_messages():
    return [{"role": "system", "content": "s"}, {"role": "user", "content": "Question: q"},
            {"role": "assistant", "content": tool_call("expand", {"node": "Anchor Node"})},
            {"role": "user", "content": _page(1, ("Anchor Node", "Node B"))},
            {"role": "assistant", "content": tool_call("expand", {"node": "Node B"})},       # follow
            {"role": "user", "content": _page(2, ("Node B", "Node C"))},
            {"role": "assistant", "content": tool_call("expand", {"node": "Anchor Node"})},  # backtrack
            {"role": "user", "content": SCHEMA_REMINDER},
            {"role": "assistant", "content": tool_call("expand", {"node": "Node C"})},       # still follows page 2
            {"role": "user", "content": FORCED_ANSWER},
            {"role": "assistant", "content": tool_call("expand", {"node": "Node B"})}]       # the forced final: not a call


def test_label_rows_reads_the_turn_after_each_observation():
    rec = {"id": "x-1", "anchor": "Anchor Node", "messages": _episode_messages()}
    rows = [{"obs_msg": 3, "msg": 4}, {"obs_msg": 5, "msg": 6}]
    moves.label_rows(rec, rows)
    assert [r["intent"] for r in rows] == ["follow", "backtrack"]


def test_mix_counts_moves_against_the_page_just_seen(tmp_path):
    """A follow expands a node of the last page, a backtrack one from an
    earlier page; a schema reminder leaves the last page in place; the forced
    final is not a call; the lag is over self-answered episodes."""
    msgs = _episode_messages()
    rec = {"id": "x-1", "anchor": "Anchor Node", "status": "budget", "n_calls": 4, "gold_complete_at_call": 2, "messages": msgs}
    rec2 = dict(rec, id="x-2", status="answered", n_calls=3,
                messages=msgs[:4] + [{"role": "assistant", "content": final_answer("Node B", [])}])
    d = tmp_path / "knob-dose0--fixture"
    d.mkdir()
    (d / "trajectories.jsonl").write_text("\n".join(json.dumps(r) for r in (rec, rec2)) + "\n")
    (d / "results.jsonl").write_text("\n".join(json.dumps({"id": r["id"], "gated_acc": r["status"] == "answered", "gate_pass": True})
                                               for r in (rec, rec2)) + "\n")
    row = moves.analyse(d)
    # five calls: the first expand of the anchor is a backtrack (the anchor is always "earlier"), then
    # follow, backtrack, follow in the first episode and the anchor expand of the second
    assert row["calls_total"] == 5 and row["follow"] == 0.4 and row["backtrack"] == 0.6 and row["fresh"] == 0.0
    assert row["lag"] == 1 and row["budget"] == 0.5 and row["premature"] == 0.0 and row["complete"] == 1.0
    # the root: an arm counts once it has a summary, and carries its knob.json's kind and setting
    (d / "knob.json").write_text(json.dumps({"arm": {"kind": "fol", "value": 1.0}, "layer": 20, "direction": "follow"}))
    assert moves.mix(tmp_path) == {}
    (d / "summary.json").write_text("{}")
    table = moves.mix(tmp_path)
    assert list(table) == ["fixture"] and table["fixture"]["dose0"]["kind"] == "fol"
    assert table["fixture"]["dose0"]["setting"] == 1.0 and table["fixture"]["dose0"]["calls_total"] == 5


# ------------------------------------------------------------ readings

def test_auroc_is_the_rank_statistic():
    assert readings.auroc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == 0.75
    assert readings.auroc(np.array([3.0, 2.0, 1.0]), np.array([1, 1, 0])) == 1.0
    assert readings.auroc(np.array([1.0, 1.0, 1.0]), np.array([1, 0, 1])) == 0.5
    assert np.isnan(readings.auroc(np.array([1.0, 2.0]), np.array([1, 1])))


def test_grouped_folds_keep_every_item_in_one_fold_and_repeat_by_seed():
    items = np.array([f"item{i % 40}" for i in range(400)])
    folds = readings.grouped_folds(items, 5, seed=0)
    assert sorted(np.concatenate(folds)) == list(range(400))
    for f in folds:
        others = np.setdiff1d(np.arange(400), f)
        assert not set(items[f]) & set(items[others])
    again = readings.grouped_folds(items, 5, seed=0)
    assert all(np.array_equal(a, b) for a, b in zip(folds, again))
    assert not all(np.array_equal(a, b) for a, b in zip(folds, readings.grouped_folds(items, 5, seed=1)))


def test_shrunk_inv_interpolates_between_the_covariance_and_its_scaled_identity():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(500, 4)) @ np.diag([3.0, 1.0, 0.5, 0.1])
    S = np.cov(Z, rowvar=False)
    assert np.allclose(readings.shrunk_inv(Z, lam=0.0), np.linalg.inv(S))
    assert np.allclose(readings.shrunk_inv(Z, lam=1.0), np.eye(4) * 4 / np.trace(S))
    assert readings.shrunk_inv(np.zeros((5, 3))) is None


def _planted(rng, n, d, classes, shifts):
    """Rows with a class-mean shift along a planted direction per class."""
    lab = rng.choice(classes, size=n, p=[0.45, 0.45, 0.10])
    X = rng.normal(size=(n, d))
    for c, s in shifts.items():
        X[lab == c] += s
    return X, lab


def test_lda_plane_separates_three_planted_classes():
    rng = np.random.default_rng(1)
    d = 12
    e0, e1 = np.eye(d)[0], np.eye(d)[1]
    X, lab = _planted(rng, 600, d, ("follow", "backtrack", "answer"),
                      {"follow": 2.5 * e0, "backtrack": -2.5 * e0, "answer": 4.0 * e1})
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6)
    A = readings.lda_plane(Z, lab, ("follow", "backtrack", "answer"))
    assert A.shape == (d, 2)
    C = Z @ A
    cen = {c: C[lab == c].mean(0) for c in ("follow", "backtrack", "answer")}
    within = max(C[lab == c].std(0).max() for c in cen)
    for a in cen:
        for b in cen:
            if a < b:
                assert np.linalg.norm(cen[a] - cen[b]) > 3 * within


def _reads(tmp_path, name, rng, n_items=30, calls=4, layers=(0, 1), d=16, sep=3.0):
    """A synthetic extract and labels: items with `calls` observations each,
    every observation an expand page, the moves follow / backtrack / answer
    planted along two axes at layer 1 and unseparated at layer 0."""
    reads = tmp_path / name
    reads.mkdir()
    rows, labs = [], []
    for i in range(n_items):
        for c in range(1, calls + 1):
            lab = "answer" if c == calls else ("follow" if (i + c) % 2 else "backtrack")
            rows.append({"id": f"it-{i}", "tier": "easy", "turn": c + 1, "msg": 2 * c + 2, "obs_msg": 2 * c + 1,
                         "kind": "observation", "call": c, "tool": "expand", "G": 0, "hop": 0, "row": len(rows)})
            labs.append(lab)
    labs = np.array(labs)
    (reads / "index.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for layer in layers:
        X = rng.normal(size=(len(rows), d))
        if layer:
            X[labs == "follow", 0] += sep
            X[labs == "backtrack", 0] -= sep
            X[labs == "answer", 1] += 2 * sep
        np.save(reads / f"L{layer}.npy", X.astype(np.float32))
    labels = tmp_path / f"labels_{name}.npz"
    np.savez(labels, keep=np.arange(len(rows)), lab=labs, G=np.zeros(len(rows), int), hop=np.zeros(len(rows), int),
             call=np.array([r["call"] for r in rows]), tool=np.array([r["tool"] for r in rows]),
             tier=np.array([r["tier"] for r in rows]), item=np.array([r["id"] for r in rows]), run=name)
    return reads, labels


def test_labels_join_maps_every_labelled_observation_to_its_read_row(tmp_path):
    reads, labels = _reads(tmp_path, "home", np.random.default_rng(0))
    L = readings.labels_join(reads / "index.jsonl", labels)
    assert np.array_equal(L["idx"], np.arange(120)) and set(L) >= {"idx", "lab", "tool", "item", "G", "hop"}


def test_profile_reads_the_planted_moves_at_the_separated_layer_only(tmp_path):
    rng = np.random.default_rng(2)
    home = _reads(tmp_path, "home", rng)
    bio = _reads(tmp_path, "bio", rng, n_items=20)
    prof = readings.profile(home[0], home[1], bio[0], bio[1], [0, 1])
    keys = [k for k, *_ in readings.READINGS]
    assert sorted(prof) == sorted(keys)
    for k in keys:
        assert 0.3 < prof[k]["home"][0] < 0.7 and 0.3 < prof[k]["bio"][0] < 0.7   # layer 0: no signal
        assert prof[k]["home"][1] > 0.9 and prof[k]["bio"][1] > 0.9                 # layer 1: planted


# ------------------------------------------------------------ directions

def test_perp_direction_removes_the_other_axis_and_keeps_the_gap_identity():
    rng = np.random.default_rng(0)
    a = rng.normal(size=32)
    a /= np.linalg.norm(a)
    b = rng.normal(size=32)
    b -= (b @ a) * a
    b /= np.linalg.norm(b)
    cos = -0.4
    f = cos * a + np.sqrt(1 - cos**2) * b            # a unit "follow" axis at cosine -0.4 to the "answer" axis a
    p = directions.perp_direction(f, a)
    assert abs(p @ a) < 1e-12 and abs(np.linalg.norm(p) - 1) < 1e-12 and np.allclose(p, b)
    X1 = rng.normal(size=(50, 32))
    X0 = X1 - 3.0 * f                                 # class means 3.0 apart along f
    assert np.isclose((X1.mean(0) - X0.mean(0)) @ p, 3.0 * np.sqrt(1 - cos**2))


def _episode_run(tmp_path, reads, n_items, calls):
    """A stored run whose observations and turns agree with `_reads`' labels:
    a follow expands a node of the page just seen, a backtrack the anchor, which
    no page names."""
    run = tmp_path / "run"
    run.mkdir()
    recs = []
    for i in range(n_items):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "Question: q"},
                {"role": "assistant", "content": tool_call("expand", {"node": "Anchor"})}]
        for c in range(1, calls + 1):
            msgs.append({"role": "user", "content": _page(c, (f"P{c}", f"Q{c}"), (f"P{c}", f"R{c}"))})   # no anchor on the page
            if c == calls:
                msgs.append({"role": "assistant", "content": final_answer(f"P{c}", [])})
            elif (i + c) % 2:
                msgs.append({"role": "assistant", "content": tool_call("expand", {"node": f"P{c}"})})   # follow
            else:
                msgs.append({"role": "assistant", "content": tool_call("expand", {"node": "Anchor"})})  # backtrack
        recs.append({"id": f"it-{i}", "anchor": "Anchor", "messages": msgs})
    (run / "trajectories.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return run


def test_build_gives_unit_axes_whose_gaps_are_the_class_mean_distances(tmp_path):
    rng = np.random.default_rng(3)
    reads, _labels = _reads(tmp_path, "home", rng, n_items=30, calls=4, layers=(1,))
    run = _episode_run(tmp_path, reads, 30, 4)
    vec, meta = directions.build(reads, run, [1], n_random=2)
    assert sorted(vec) == ["answer_L1", "answer_perp_L1", "follow_L1", "follow_perp_L1", "random2_L1", "random_L1"]
    for v in vec.values():
        assert v.dtype == np.float32 and abs(np.linalg.norm(v) - 1) < 1e-5
    m = meta["layers"]["1"]
    assert m["follow"]["n"] == [45, 45] and m["answer"]["n"] == [30, 90]
    assert m["follow"]["gap"] > 5 and m["answer"]["gap"] > 5 and m["follow"]["auroc"] > 0.95
    assert abs(float(vec["follow_perp_L1"] @ vec["answer_L1"])) < 1e-5
    assert np.isclose(m["follow_perp"]["gap_over_parent"], np.sqrt(1 - m["follow_perp"]["cos_removed"] ** 2), atol=0.05)
    # the random draws are seeded by draw and layer
    again, _ = directions.build(reads, run, [1], n_random=2)
    assert np.array_equal(vec["random2_L1"], again["random2_L1"])
    files, meta1 = directions.per_layer_files(vec, meta, 1)
    assert sorted(files) == ["answer", "answer_perp", "follow", "follow_perp", "random", "random2"]
    assert meta1["random"]["gap"] == m["answer"]["gap"] and meta1["layer"] == 1 and "follow_perp" in meta1


# ------------------------------------------------------------ figure

def _mix(rows_by_arm):
    return {"bench": {arm: [dict(r, kind=kind, setting=setting) for r in rows]
                      for arm, (kind, setting, rows) in rows_by_arm.items()}}


def test_agg_sorts_by_setting_and_spreads_by_read_count():
    mix = _mix({"dose0": ("dose0", 0.0, [{"calls": 12.0}, {"calls": 12.4}, {"calls": 12.2}]),
                "fol+1": ("fol", 1.0, [{"calls": 10.0}, {"calls": 11.0}]),
                "fol-1": ("fol", -1.0, [{"calls": 14.0}]),
                "ans+1": ("ans", 1.0, [{"calls": 9.0}, {"calls": 9.0}])})
    x, m, s, n = figure.agg(mix, "bench", "fol", "calls", min_reads=1)
    assert list(x) == [-1.0, 0.0, 1.0] and list(n) == [1, 3, 2]
    assert np.isclose(m[1], 12.2) and np.isclose(s[1], np.std([12.0, 12.4, 12.2], ddof=1))
    assert s[0] == 0.0 and s[2] == 0.5
    x, *_ = figure.agg(mix, "bench", "fol", "calls")          # the default drops single reads
    assert list(x) == [0.0, 1.0]


def test_the_null_pools_draws_mean_first():
    mix = _mix({"rndf+1": ("rndf", 1.0, [{"calls": 12.0}, {"calls": 14.0}, {"calls": 13.0}]),
                "rndf2+1": ("rndf2", 1.0, [{"calls": 11.0}]),
                "rndf3+1": ("rndf3", 1.0, [{"calls": 10.0}]),
                "rndf2-1": ("rndf2", -1.0, [{"calls": 12.5}]),
                "rndf3-1": ("rndf3", -1.0, [{"calls": 12.0}])})
    x, m, s, n = figure.agg_draws(mix, "bench", ("rndf", "rndf2", "rndf3"), "calls")
    assert list(x) == [1.0] and np.isclose(m[0], (13.0 + 11.0 + 10.0) / 3) and list(n) == [3]   # each draw's mean counts once
    assert np.isclose(s[0], np.std([13.0, 11.0, 10.0], ddof=1))
    null = figure.null_of(mix, "bench", ("rndf", "rndf2", "rndf3"), "fol", gaps={})
    assert list(null("calls")[0]) == [1.0]
    # one draw alone, and the answer-unit draw rescaled when none ran at this axis's unit
    one = figure.null_of(_mix({"rndf+1": ("rndf", 1.0, [{"calls": 12.0}, {"calls": 14.0}])}), "bench", ("rndf",), "fol", gaps={})
    assert list(one("calls")[0]) == [1.0]
    scaled = figure.null_of(_mix({"rnd+1": ("rnd", 1.0, [{"calls": 12.0}, {"calls": 14.0}])}), "bench", ("rndf",), "fol",
                            gaps={"random": {"gap": 5.0}, "follow": {"gap": 2.5}})
    assert list(scaled("calls")[0]) == [2.0]


def test_panel_a_data_projects_the_unseen_graph_onto_the_fitted_plane(tmp_path):
    rng = np.random.default_rng(4)
    home = _reads(tmp_path, "home", rng, layers=(1,))
    bio = _reads(tmp_path, "bio", rng, n_items=20, layers=(1,))
    out = figure.panel_a_data(home[0], home[1], bio[0], bio[1], [1])
    d = out[1]
    Ch, lh = d["fit"]
    Cb, lb = d["test"]
    assert Ch.shape == (120, 2) and Cb.shape == (80, 2) and set(lh) == set(figure.CLASSES)
    (xlo, xhi), (ylo, yhi) = d["lim"]
    assert xlo < Ch[:, 0].min() and xhi > Ch[:, 0].max() and ylo < Ch[:, 1].min() and yhi > Ch[:, 1].max()
    assert np.isclose(xhi - xlo, yhi - ylo)                      # a square box
    src, dst = d["arrows"]["follow"]
    assert np.allclose(src, Ch[lh == "backtrack"].mean(0)) and np.allclose(dst, Ch[lh == "follow"].mean(0))
    assert list(figure.load_mix([tmp_path])) == []


# ------------------------------------------------------------ states

from conftest import FILM_GOLD_EDGES, run_turns  # noqa: E402
from graph_reasoner.agent.loop import EpisodeConfig, trajectory_record  # noqa: E402
from graph_reasoner.agent.prompts import EMPTY_TURN_NUDGE  # noqa: E402
from graph_reasoner.latent import states, steer  # noqa: E402


@pytest.fixture
def film_record(film_kb, film_item):
    """find, expand the anchor (hop 1 arrives), an empty turn and its nudge,
    expand the intermediate (hop 2 arrives), the final answer; produced by
    the real loop on scripted turns."""
    cfg = EpisodeConfig()
    ep = run_turns(film_kb, film_item, [
        tool_call("find", {"name": "Film A"}),
        tool_call("expand", {"node": "Film A"}),
        "",
        tool_call("expand", {"node": "Jane Doe"}),
        final_answer("Film B", FILM_GOLD_EDGES),
    ], cfg)
    return json.loads(json.dumps(trajectory_record(ep, film_item, film_kb, cfg, model="scripted")))


def test_decision_rows_label_each_turn_with_the_message_before_it(film_record):
    rows = states.decision_rows(film_record)
    assert [r["turn"] for r in rows] == [1, 2, 3, 4]
    assert [r["kind"] for r in rows] == ["observation", "observation", "nudge", "observation"]
    obs = [r for r in rows if r["kind"] == "observation"]
    assert [r["call"] for r in obs] == [1, 2, 3]
    assert [r["tool"] for r in obs] == ["find", "expand", "expand"]
    assert [r["G"] for r in obs] == [0, 1, 1]
    assert [r["hop"] for r in obs] == [0, 1, 2]
    nudge = rows[2]
    assert "G" not in nudge and film_record["messages"][nudge["obs_msg"]]["content"] == EMPTY_TURN_NUDGE
    assert all(r["msg"] == r["obs_msg"] + 1 for r in rows)


def test_hop_index_walks_the_chain_from_the_anchor():
    rec = {"anchor": "Film A",
           "gold_path_edges": [{"source": "Film B", "relation": "written_by", "target": "Jane Doe"},
                               {"source": "Film A", "relation": "written_by", "target": "Jane Doe"}]}
    assert states.gold_hops(rec) == [2, 1]
    rec["anchor"] = "nobody"
    assert states.gold_hops(rec) == [None, None]


def test_a_stored_n_new_gold_that_disagrees_with_the_seen_edges_is_refused(film_record):
    film_record["calls"][1]["n_new_gold"] = 0
    with pytest.raises(ValueError, match="n_new_gold"):
        states.decision_rows(film_record)


def test_read_position_is_the_header_end_of_every_later_assistant_turn(film_record, shipped_tokenizer):
    tok = shipped_tokenizer
    ids, positions = states.read_positions(film_record["messages"], tok)
    assert sorted(positions) == [1, 2, 3, 4]
    text = tok.apply_chat_template(film_record["messages"], tokenize=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    assert list(enc["input_ids"]) == ids
    # Independently: the k-th occurrence of the header in the rendered text
    # (the first is the prompt's) ends at the character the position's token ends at.
    ends = {}
    start = 0
    for k in range(5):
        i = text.index(states.HEADER, start)
        ends[k] = i + len(states.HEADER)
        start = i + 1
    by_end = {e: n for n, (_s, e) in enumerate(enc["offset_mapping"])}
    for t, pos in positions.items():
        assert pos == by_end[ends[t]]
        assert tok.decode(ids[pos - 2: pos + 1]) == states.HEADER
        assert tok.decode([ids[pos]]) == "\n"
    assert tok.decode([ids[positions[2] + 1]]) == "<|im_end|>"      # the empty turn ends at once
    assert tok.decode([ids[positions[3] + 1]]) == "<tool_call>"


def test_the_engine_prompt_ends_on_the_read_position(film_record, shipped_tokenizer):
    """What the steered engine feeds vLLM for a prefix, the rendered prompt
    with the generation header, ends on the token the full transcript is
    read at for that turn."""
    from graph_reasoner.train.generate import encode_prompts, render_prompts
    tok = shipped_tokenizer
    ids, positions = states.read_positions(film_record["messages"], tok)
    for r in states.decision_rows(film_record):
        prefix = film_record["messages"][: r["obs_msg"] + 1]
        p_ids = encode_prompts(render_prompts([prefix], tok), tok)[0]
        assert list(p_ids) == ids[: positions[r["turn"]] + 1]


def test_the_tokenizer_must_carry_a_named_concatenative_template(tmp_path, shipped_tokenizer):
    shipped_tokenizer.save_pretrained(tmp_path / "tok")
    assert states.rendering_tokenizer(str(tmp_path / "tok"), None).chat_template
    plain = states.rendering_tokenizer(str(tmp_path / "tok"), "plain_chatml")
    assert plain.chat_template == shipped_tokenizer.chat_template
    cfg = tmp_path / "tok" / "tokenizer_config.json"
    d = json.loads(cfg.read_text(encoding="utf-8"))
    d.pop("chat_template", None)
    cfg.write_text(json.dumps(d), encoding="utf-8")
    for p in (tmp_path / "tok").glob("chat_template*"):
        p.unlink()
    with pytest.raises(SystemExit, match="named concatenative"):
        states.rendering_tokenizer(str(tmp_path / "tok"), None)


def test_compare_reports_the_agreement_of_two_extractions(tmp_path):
    rng = np.random.default_rng(5)
    rows = [{"id": "a", "turn": t, "row": t - 1} for t in (1, 2, 3)]
    for name, noise in (("x", 0.0), ("y", 0.01)):
        d = tmp_path / name
        d.mkdir()
        (d / "index.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        base = np.random.default_rng(0).normal(size=(3, 8))
        np.save(d / "L1.npy", (base + noise * rng.normal(size=base.shape)).astype(np.float32))
    rep = states.compare(tmp_path / "x", tmp_path / "y", log=lambda s: None)
    assert rep["rows"] == 3 and rep["layers"][1]["cos_min"] > 0.99 and rep["layers"][1]["rel_max"] < 0.05
    with pytest.raises(SystemExit, match="share no rows"):
        (tmp_path / "y" / "index.jsonl").write_text(json.dumps({"id": "b", "turn": 1, "row": 0}) + "\n")
        states.compare(tmp_path / "x", tmp_path / "y", log=lambda s: None)


# ------------------------------------------------------------ steer

def test_arms_name_their_kind_and_the_nulls_borrow_a_real_unit():
    assert steer.parse_arm("dose0") == {"name": "dose0", "kind": "dose0"}
    assert steer.parse_arm("fol:+2") == {"name": "fol+2", "kind": "fol", "value": 2.0}
    assert steer.parse_arm("ans:-0.5") == {"name": "ans-0.5", "kind": "ans", "value": -0.5}
    assert steer.parse_arm("rndf2:-2") == {"name": "rndf2-2", "kind": "rndf2", "value": -2.0}
    assert steer.null_direction("rnd") == ("random", "answer") and steer.null_direction("rndf") == ("random", "follow")
    assert steer.null_direction("rnd2") == ("random2", "answer") and steer.null_direction("rndf3") == ("random3", "follow")
    assert steer.null_direction("fol") is None and steer.null_direction("rndx") is None
    for bad in ("bud:12", "lb:-1", "folp:+1", "ev:1", "fol"):
        with pytest.raises(SystemExit):
            steer.parse_arm(bad)


def test_the_seed_comes_from_the_flag_or_the_root_suffix():
    assert steer.read_seed(None, "data/runs/_probe/knob_vllm") == 0
    assert steer.read_seed(None, "data/runs/_probe/knob_vllm_r3/") == 3
    assert steer.read_seed(7, "data/runs/_probe/knob_vllm_r3") == 7


def test_arm_vector_is_dose_times_gap_times_the_unit_direction():
    rng = np.random.default_rng(6)
    u = rng.normal(size=8).astype(np.float32)
    u /= np.linalg.norm(u)
    dirs = {"follow": u, "random2": -u}
    meta = {"follow": {"gap": 3.0}, "answer": {"gap": 5.0}, "resid_norm_median": 10.0}
    vec, rec = steer.arm_vector({"name": "fol+2", "kind": "fol", "value": 2.0}, dirs, meta)
    assert np.allclose(vec, 6.0 * u) and rec["direction"] == "follow" and np.isclose(rec["vector_norm_over_resid"], 0.6)
    vec, rec = steer.arm_vector({"name": "rnd2-1", "kind": "rnd2", "value": -1.0}, dirs, meta)
    assert np.allclose(vec, 5.0 * u) and rec["direction"] == "random2" and rec["gap"] == 5.0
    assert steer.arm_vector({"name": "dose0", "kind": "dose0"}, dirs, meta) == (None, {})
    with pytest.raises(SystemExit, match="random3"):
        steer.arm_vector({"name": "rndf3+1", "kind": "rndf3", "value": 1.0}, dirs, meta)


class _Loc:
    def __init__(self, loc):
        self.query_start_loc = loc


def test_last_rows_reads_each_request_s_last_token_from_the_query_offsets():
    assert steer.last_rows(_Loc([0, 5, 9]), 9) == [4, 8]
    assert steer.last_rows([_Loc([0, 3])], 3) == [2]
    assert steer.last_rows({"attn": object(), "gdn": _Loc([0, 2, 4])}, 4) == [1, 3]
    assert steer.last_rows(_Loc(np.array([0, 2, 6])), 4) == [1]          # a row past the step is dropped
    assert steer.last_rows(_Loc([0]), 4) == []
    assert steer.last_rows(object(), 4) is None and steer.last_rows([], 4) is None and steer.last_rows({}, 4) is None


def test_the_hook_adds_the_vector_on_the_named_rows_and_records_the_stream():
    torch = pytest.importorskip("torch")

    class DecoderLayer(torch.nn.Module):
        def forward(self, h, r):
            return h + 1.0, r * 2.0

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))
            self.layers = torch.nn.ModuleList([DecoderLayer(), DecoderLayer()])

        def forward(self, h, r):
            for layer in self.layers:
                h, r = layer(h, r)
            return h, r

    m = Model()
    rows = {"pos": [0, 2]}
    assert steer.install_hook(m, 2, positions=lambda residual: rows["pos"]) == "DecoderLayer"
    h0, r0 = torch.zeros(3, 4), torch.ones(3, 4)
    h, r = m(h0.clone(), r0.clone())
    assert torch.equal(r, torch.full((3, 4), 4.0))                         # no vector: untouched
    steer.set_vec(m, np.full(4, 0.5, dtype=np.float32))
    h, r = m(h0.clone(), r0.clone())
    assert torch.allclose(r[[0, 2]], torch.full((2, 4), 4.5)) and torch.allclose(r[1], torch.full((4,), 4.0))
    steer.set_vec(m, None)
    steer.set_record(m, True)
    h, r = m(h0.clone(), r0.clone())
    rec = steer.take_record(m)
    assert len(rec) == 1 and rec[0].shape == (2, 4) and np.allclose(rec[0], 6.0)   # hidden 2 + residual 4 on those rows
    assert steer.take_record(m) == []
    rows["pos"] = []
    h, r = m(h0.clone(), r0.clone())                                       # no rows: a no-op
    with pytest.raises(ValueError, match="layer 3"):
        steer.install_hook(m, 3, positions=lambda residual: [])


class _Engine:
    def __init__(self):
        self.vec_log = []
        self.max_model_len = 16384

    def set_vec(self, vec):
        self.vec_log.append(None if vec is None else float(np.asarray(vec).sum()))

    def generate_chat(self, conversations, max_new_tokens=800, temperature=None, stop=None, desc=""):
        return ["<tool_call>{}</tool_call>"] * len(conversations)


def test_the_seam_switches_the_vector_off_for_a_batch_of_forced_finals():
    from graph_reasoner.agent.tools import available_tools
    from graph_reasoner.agent.prompts import nudge
    cfg = EpisodeConfig()
    forced = nudge("FORCED_ANSWER", cfg.prompt_version, available_tools(cfg.tool_cfg))
    engine = _Engine()
    knob = {"vec": np.ones(4, dtype=np.float32)}
    generate = steer.steered_seam(engine, cfg, knob)
    ordinary = [[{"role": "user", "content": "q"}, {"role": "assistant", "content": "x"}, {"role": "user", "content": "OBSERVATION (call 1/20, expand):\n..."}]]
    assert generate(ordinary, 1024) == ["<tool_call>{}</tool_call>"]
    assert engine.vec_log == [4.0, None]                                    # set for the call, cleared after
    finals = [[{"role": "user", "content": "q"}, {"role": "assistant", "content": "x"}, {"role": "user", "content": forced}]] * 2
    generate(finals, 900)
    assert engine.vec_log == [4.0, None, None, None]                       # off for the forced finals
    assert knob["n_generate_calls"] == 2 and knob["n_forced_calls"] == 1
    with pytest.raises(RuntimeError, match="mixes"):
        generate(ordinary + finals[:1], 1024)
    assert generate.max_model_len == 16384
