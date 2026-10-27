"""The pool a selector is offered, the verifier's margins and pick, and the join
the paper reads them back through: `select/pool.py` (`load_pair`, `load`,
`live_indices`, `offered`, `pick_by_margin`) and `select/verifier.py`
(`score_pool`, `score_rows`).

The paper does not rerun the verifier: its figures/extract_data.py reads the
stored `scores.jsonl` back, rebuilds each item's offered pool from the run
dirs in `runs_in_order`, and takes the largest margin. A `cand_idx` is a
position in `offered(...)` of the runs in that order, not a run index, so
these tests write runs where the two differ.
"""
import json

import pytest

from conftest import write_run
from graph_reasoner import benchmarks
from graph_reasoner.select import live_indices, load, load_pair, offered, pick_by_margin, select
from graph_reasoner.select.prompts import NO, YES, render_prompt
from graph_reasoner.select.verifier import score_pool, score_rows

BENCH = "fakepoolbench"


def _cand(name, *, gate_pass=True, abstained=False):
    return {"id": "x", "tier": "easy", "answer": f"the answer is {name}",
            "cited": [], "gate_pass": gate_pass, "abstained": abstained,
            "em_claim": False, "gated_acc": False, "reached": False, "name": name}


def _write_run(root, tag, items):
    """A run dir as the eval writes it: `items` maps id -> (gate_pass, answer)."""
    results = [{"id": iid, "gate_pass": gate_pass, "reached_answer": True,
                "abstained": False, "asserted": True, "n_calls": 2,
                "em_claim": False, "gated_acc": False}
               for iid, (gate_pass, _answer) in items.items()]
    trajectories = [{"id": iid, "tier": "easy", "question": f"{iid}?",
                     "answer": f"the answer is {answer}", "cited_edges": []}
                    for iid, (_gate_pass, answer) in items.items()]
    return write_run(root, f"{tag}--{BENCH}", results, trajectories)


# ---------------------------------------------------------------- the offered pool

def test_offered_is_the_live_candidates_in_run_order():
    cands = [_cand("a", gate_pass=False), _cand("b"), _cand("c", abstained=True),
             _cand("d")]
    assert live_indices(cands) == [1, 3]
    assert [c["name"] for c in offered(cands)] == ["b", "d"]


def test_offered_falls_back_to_every_candidate_and_the_vote_does_not():
    """With nothing live, a scorer is offered the whole pool while the vote
    returns the reference; `live_indices` itself has no fallback."""
    cands = [_cand("a", gate_pass=False), _cand("b", abstained=True),
             _cand("c", gate_pass=False)]
    assert live_indices(cands) == []
    assert [c["name"] for c in offered(cands)] == ["a", "b", "c"]
    assert select(cands)["name"] == "a"


# ---------------------------------------------------------------- the pick

def test_pick_by_margin_takes_the_largest_and_the_earliest_of_a_tie():
    assert pick_by_margin([0.1, 2.0, -1.0]) == 1
    assert pick_by_margin([1.0, 3.0, 3.0, 2.0]) == 1
    assert pick_by_margin([-2.0, -2.0]) == 0
    assert pick_by_margin([5.0]) == 0
    with pytest.raises(ValueError):
        pick_by_margin([])


# ---------------------------------------------------------------- the margins

def test_score_pool_is_yes_minus_no_over_the_trained_span(monkeypatch):
    """A fake span and a fake scorer, so the arithmetic is checked without
    weights: each label's log-probability is summed over exactly the positions
    `completion_logprob_span` names, the margin is yes minus no, and the
    prompt is the shared render of the question, answer and cited chain."""
    from graph_reasoner.train import tokenize

    pool = [_cand("oxide"), _cand("etch rate")]
    prompts = [render_prompt("Q?", c["answer"], c["cited"]) for c in pool]
    calls = []

    def fake_span(messages, tok):
        assert tok == "TOK"
        prompt, label = messages[0]["content"], messages[1]["content"]
        calls.append((prompt, label))
        # [prompt marker, header, label token, end]: the last two are scored
        return [prompts.index(prompt), 7, {YES: 1, NO: 2}[label], 9], [2, 3]

    table = {(0, 1): -0.25, (0, 2): -2.0, (1, 1): -3.0, (1, 2): -0.5}

    def score_ids(ids):
        # None at 0 must never be summed; position 3 (the end token) adds -0.1
        return [None, -50.0, table[(ids[0], ids[2])], -0.1]

    monkeypatch.setattr(tokenize, "completion_logprob_span", fake_span)
    margins = score_pool(score_ids, "TOK", "Q?", pool)
    assert margins == pytest.approx([(-0.35) - (-2.1), (-3.1) - (-0.6)])
    assert calls == [(prompts[0], YES), (prompts[0], NO),
                     (prompts[1], YES), (prompts[1], NO)]
    assert pick_by_margin(margins) == 0


def test_score_rows_mark_the_pick_and_key_by_pool_position():
    pool = [dict(_cand("a"), gated_acc=True, em_claim=True), _cand("b"), _cand("c")]
    rows = score_rows("i9", pool, [0.5, 2.0, 2.0])
    assert [(r["id"], r["cand_idx"], r["picked"]) for r in rows] == \
        [("i9", 0, False), ("i9", 1, True), ("i9", 2, False)]
    assert rows[0]["gated_acc"] is True and rows[1]["gated_acc"] is False
    assert set(rows[0]) == {"id", "cand_idx", "margin", "picked", "gated_acc",
                            "em_claim", "gate_pass"}


# ---------------------------------------------------------------- the join

def test_load_pair_refuses_a_repeated_id(tmp_path):
    d = _write_run(tmp_path, "w", {"i1": (True, "a")})
    with (d / "results.jsonl").open("a") as f:
        f.write(json.dumps({"id": "i1", "gate_pass": False}) + "\n")
    with pytest.raises(ValueError, match="repeats id 'i1'.*one run directory per seed"):
        load_pair(d)
    with pytest.raises(ValueError, match="repeats id"):
        load(d)


def test_load_pair_refuses_a_one_sided_id(tmp_path):
    d = _write_run(tmp_path, "p", {"i1": (True, "a"), "i2": (True, "b")})
    lines = (d / "trajectories.jsonl").read_text().splitlines()
    (d / "trajectories.jsonl").write_text(lines[0] + "\n")
    with pytest.raises(ValueError, match="1 id\\(s\\) only in results.jsonl"):
        load_pair(d)


def test_load_pair_joins_by_id_not_by_line(tmp_path):
    d = _write_run(tmp_path, "o", {"i1": (True, "a"), "i2": (False, "b")})
    lines = (d / "results.jsonl").read_text().splitlines()
    (d / "results.jsonl").write_text("\n".join(reversed(lines)) + "\n")
    res, traj = load_pair(d)
    assert list(traj) == ["i1", "i2"]
    assert res["i2"]["gate_pass"] is False
    assert load(d)["i2"]["gate_pass"] is False


# ------------------------------------------------- the paper's read of the pick

def _figure_picks(rows, scores_path):
    """figures/extract_data.py::pool_verifier's reconstruction, as the paper
    repository writes it: margins keyed by (id, cand_idx), the offered pool
    rebuilt from the runs in order, the largest margin with the earlier
    candidate on a tie."""
    ids = sorted(set.intersection(*[set(r) for r in rows]))
    margins: dict[str, dict[int, float]] = {}
    with scores_path.open() as fh:
        for line in fh:
            d = json.loads(line)
            margins.setdefault(d["id"], {})[d["cand_idx"]] = d["margin"]
    picks = {}
    for i in ids:
        pool = offered([r[i] for r in rows])
        m = margins.get(i)
        if not m:
            continue
        best = max(range(len(pool)), key=lambda n: (m.get(n, float("-inf")), -n))
        picks[i] = pool[best]
    return picks


def test_stored_margins_read_back_to_the_scorers_pick(tmp_path, monkeypatch):
    """Three runs, four items: a tie (i1), a gate-fail ahead of the pick so
    cand_idx and run index differ (i2), nothing live so the whole pool is
    offered (i3), and a pool of one, which is not scored (i4). Scores are
    written by `score_rows`, as verifier_select.py writes them; the figure's
    reconstruction must land on the very candidate `pick_by_margin` chose."""
    gold = {"i1": ["b1"], "i2": ["c2"], "i3": ["b3"], "i4": ["c4"]}
    monkeypatch.setitem(benchmarks._GOLD_CACHE, BENCH, gold)
    runs = [
        _write_run(tmp_path, "s101", {"i1": (True, "a1"), "i2": (False, "a2"),
                                      "i3": (False, "a3"), "i4": (False, "a4")}),
        _write_run(tmp_path, "s102", {"i1": (True, "b1"), "i2": (True, "b2"),
                                      "i3": (False, "b3"), "i4": (False, "b4")}),
        _write_run(tmp_path, "s103", {"i1": (True, "c1"), "i2": (True, "c2"),
                                      "i3": (False, "c3"), "i4": (True, "c4")}),
    ]
    margins = {"i1": [1.0, 3.0, 3.0], "i2": [0.5, 2.0], "i3": [-1.0, -0.5, -2.0]}

    rows = [load(r, recompute=True) for r in runs]
    chosen = {}
    scores = tmp_path / "scores.jsonl"
    with scores.open("w") as fh:
        for i in sorted(rows[0]):
            pool = offered([r[i] for r in rows])
            if len(pool) < 2:          # verifier_select scores no pool of one
                continue
            chosen[i] = pool[pick_by_margin(margins[i])]
            for row in score_rows(i, pool, margins[i]):
                fh.write(json.dumps(row) + "\n")

    assert {i: c["answer"] for i, c in chosen.items()} == {
        "i1": "the answer is b1", "i2": "the answer is c2", "i3": "the answer is b3"}
    picks = _figure_picks([load(r, recompute=True) for r in runs], scores)
    assert {i: c["answer"] for i, c in picks.items()} == \
        {i: c["answer"] for i, c in chosen.items()}
    # the labels the figure counts are the recomputed gated_acc of that candidate
    assert {i: c["gated_acc"] for i, c in picks.items()} == \
        {"i1": True, "i2": True, "i3": False}


def _verifier_select(monkeypatch, tmp_path, score_pool):
    """scripts/select/verifier_select.py over two stored runs, its scorer replaced."""
    import importlib.util
    import sys

    from conftest import script
    spec = importlib.util.spec_from_file_location("verifier_select", script("verifier_select.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_local_scorer", lambda ref, template: (None, None))
    monkeypatch.setattr("graph_reasoner.select.verifier.score_pool", score_pool)
    runs = [str(_write_run(tmp_path / "runs", f"p-s{k}", {"a": (True, "x"), "b": (True, "y")})) for k in (0, 1)]
    monkeypatch.setattr(sys, "argv", ["verifier_select.py", "--verifier", "hf:models/v", "--runs", *runs,
                                      "--name", "vsel", "--out-root", str(tmp_path / "out")])
    return mod.main, tmp_path / "out" / f"vsel--{BENCH}"


def test_a_scoring_that_dies_leaves_no_summary_and_one_that_ends_writes_it(tmp_path, monkeypatch):
    """The runner takes the scoring run's summary.json as the mark of a
    finished scoring, so the file is written last."""
    def dies(*args):
        raise RuntimeError("out of memory")

    main, out = _verifier_select(monkeypatch, tmp_path / "dies", dies)
    with pytest.raises(RuntimeError):
        main()
    assert out.is_dir() and not (out / "summary.json").exists()

    main, out = _verifier_select(monkeypatch, tmp_path / "ends", lambda score_ids, tok, question, pool: [1.0, 0.0])
    assert main() == 0
    assert json.loads((out / "summary.json").read_text())["runs_in_order"][0].endswith(f"p-s0--{BENCH}")
    assert (out / "scores.jsonl").exists() and not (out / "summary.json.tmp").exists()
