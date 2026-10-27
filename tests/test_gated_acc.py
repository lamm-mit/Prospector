"""The claim-scoped metric: an answer is scored on what it asserts.

An answer may restate its cited chain before the claim ("Because A rel B;
B rel C, the answer is X"), so a match over the whole text would credit a
chain that merely passes through the gold entity. `em_claim` matches only
the text after the last "the answer is" (the whole answer when the marker is
absent), and the headline is `gated_acc = em_claim and gate_pass`.

Covers `extract_claim` (eval/parse.py), `em_claim_of` and `agentic_metrics`
(agent/metrics.py), the pool loader and the vote (select/pool.py and
select/vote.py, re-exported as `graph_reasoner.select.load` and `.select`),
and the training reward (train/reward.py).
"""
import pytest

from conftest import FILM_GOLD_EDGES, final_answer, run_turns, tool_call, write_run
from graph_reasoner import benchmarks
from graph_reasoner import select as best_of_n
from graph_reasoner.agent.metrics import agentic_metrics, em_claim_of
from graph_reasoner.eval.parse import extract_claim


# ---------------------------------------------------------------- extract_claim

def test_claim_is_the_tail_after_the_last_marker():
    ans = "Because A rel B; B rel C, the answer is Removal."
    assert extract_claim(ans) == "Removal."


def test_no_marker_falls_back_to_the_full_string():
    """Without the marker the claim is the whole answer, so an answer that
    never restates a chain scores the same under both rules."""
    assert extract_claim("Electron-beam microscopy (SEM).") == \
        "Electron-beam microscopy (SEM)."
    assert extract_claim("") == ""


def test_marker_is_case_insensitive_and_last_wins():
    ans = "The Answer Is X. On reflection, the ANSWER is Y."
    assert extract_claim(ans) == "Y."


# ---------------------------------------------------------------- em_claim_of

def test_chain_mention_no_longer_scores():
    """Gold 'Desorption' appears twice in the chain prose but the claim is
    'Removal': a match over the whole text credits it, `em_claim` does not."""
    ans = ("Because Reactive species weakens Surface Modification; Surface "
           "Modification followed by Desorption; and Removal is achieved "
           "through Desorption, the answer is Removal.")
    from graph_reasoner.agent.metrics import score_answer
    assert score_answer(ans, ["Desorption"]) is True             # whole text
    assert em_claim_of(ans, ["Desorption"]) is False             # claim only


def test_correct_claim_still_scores():
    ans = "Because A rel B, the answer is Desorption."
    assert em_claim_of(ans, ["Desorption"]) is True


# ------------------------------------------------- agentic_metrics integration

def test_gated_acc_rejects_a_chain_that_only_mentions_gold(film_kb, film_item):
    """Gold 'Film B' sits in the because-clause and the claim is 'Jane Doe':
    the gate passes, the claim rule does not, so the headline is 0."""
    ep = run_turns(film_kb, film_item, [
        tool_call("expand", {"node": "Film A"}),
        tool_call("expand", {"node": "Film B"}),
        final_answer("Because Film B written_by Jane Doe and Film A written_by "
                "Jane Doe, the answer is Jane Doe.", FILM_GOLD_EDGES),
    ])
    m = agentic_metrics(ep, film_item)
    assert m["gate_pass"] is True
    assert m["em_claim"] is False
    assert m["gated_acc"] is False              # the headline


def test_gated_acc_credits_a_correct_claim(film_kb, film_item):
    ep = run_turns(film_kb, film_item, [
        tool_call("expand", {"node": "Film A"}),
        tool_call("expand", {"node": "Film B"}),
        final_answer("Because Film A written_by Jane Doe and Film B written_by "
                "Jane Doe, the answer is Film B.", FILM_GOLD_EDGES),
    ])
    m = agentic_metrics(ep, film_item)
    assert m["em_claim"] is True and m["gated_acc"] is True


# -------------------------------------------------------- best_of_n: the vote

def _cand(name, answer, cited):
    return {"id": "x", "tier": "hard", "answer": answer,
            "cited": [{"source": s, "relation": "r", "target": t} for s, t in cited],
            "gate_pass": True, "abstained": False,
            "em_claim": False, "gated_acc": False,
            "reached": False, "n_calls": 5, "name": name}


def test_vote_counts_claims_not_chain_nodes():
    """The reference restates 'oxide' in its chain but claims 'etch rate'. The
    vote counts it for 'etch rate' only, so a rollout that asserts oxide wins;
    a vote over the full text would return the reference."""
    ref = _cand("greedy",
                "Because oxide rel step, the answer is etch rate.",
                [("oxide", "etch rate")])
    others = [_cand(f"s{i}", "Because step rel x, the answer is oxide.",
                    [("oxide", "step")]) for i in range(2)]
    assert best_of_n.select([ref] + others)["name"] == "s0"


# -------------------------------------------------------- best_of_n: the load

BASE_RES = {"gate_pass": True,
            "reached_answer": False, "abstained": False, "n_calls": 3,
            "asserted": True}


def test_load_recomputes_the_claim_rule_for_a_run_without_it(tmp_path, monkeypatch):
    """A stored run without `em_claim` gets it recomputed from the answer
    text, against the benchmark named in the directory suffix."""
    d = write_run(tmp_path, "tag--fakebench",
                   [{"id": "i1", **BASE_RES}],
                   [{"id": "i1", "tier": "easy", "question": "q",
                     "answer": "Because gold rel mid, the answer is decoy.",
                     "cited_edges": []}])
    monkeypatch.setitem(benchmarks._GOLD_CACHE, "fakebench", {"i1": ["gold"]})
    row = best_of_n.load(d)["i1"]
    assert row["em_claim"] is False and row["gated_acc"] is False


def test_load_trusts_stored_claim_fields(tmp_path):
    """A run that carries the fields keeps them: `load` recomputes over
    stored labels only when asked."""
    d = write_run(tmp_path, "tag2--fakebench2",
                   [{"id": "i1", **BASE_RES, "em_claim": True, "gated_acc": True}],
                   [{"id": "i1", "tier": "easy", "question": "q",
                     "answer": "Because gold rel mid, the answer is decoy.",
                     "cited_edges": []}])
    row = best_of_n.load(d)["i1"]
    assert row["em_claim"] is True and row["gated_acc"] is True


def test_load_recompute_overrides_stored_fields_and_drops_retired(tmp_path, monkeypatch):
    """`recompute=True` scores every row against the benchmark's current
    answer sets and drops rows whose item is absent from it; the default
    keeps the stored labels."""
    d = write_run(tmp_path, "tag3--fakebench3",
                   [{"id": "i1", **BASE_RES, "em_claim": False, "gated_acc": False},
                    {"id": "gone", **BASE_RES, "em_claim": True, "gated_acc": True}],
                   [{"id": "i1", "tier": "easy", "question": "q",
                     "answer": "Because x rel y, the answer is ALE.",
                     "cited_edges": []},
                    {"id": "gone", "tier": "easy", "question": "q",
                     "answer": "x", "cited_edges": []}])
    # The current answer set includes "ALE"; item "gone" is absent from the benchmark.
    monkeypatch.setitem(benchmarks._GOLD_CACHE, "fakebench3", {"i1": ["Cyclic RIE", "ALE"]})
    rows = best_of_n.load(d, recompute=True)
    assert rows["i1"]["em_claim"] is True and rows["i1"]["gated_acc"] is True
    assert "gone" not in rows
    # The default keeps the stored labels.
    assert best_of_n.load(d)["i1"]["em_claim"] is False


def test_load_refuses_a_run_without_claim_scores_it_cannot_resolve(tmp_path):
    d = write_run(tmp_path, "nodashes",
                   [{"id": "i1", **BASE_RES}],
                   [{"id": "i1", "tier": "easy", "question": "q",
                     "answer": "x", "cited_edges": []}])
    with pytest.raises(RuntimeError, match="no claim-scoped scores"):
        best_of_n.load(d)


# ---------------------------------------------------------- the training reward

def test_score_episodes_scores_the_claim_rule(film_kb, film_item):
    """The training reward's correctness is the claim rule: a chain that
    restates gold and then asserts the wrong node earns nothing, and the
    `gated_acc` column agrees with the reward bit."""
    from graph_reasoner.train.reward import score_episodes

    prefix_credit = run_turns(film_kb, film_item, [
        tool_call("expand", {"node": "Film A"}),
        tool_call("expand", {"node": "Film B"}),
        final_answer("Because Film B written_by Jane Doe, the answer is Jane Doe.",
                FILM_GOLD_EDGES),
    ])
    correct = run_turns(film_kb, film_item, [
        tool_call("expand", {"node": "Film A"}),
        tool_call("expand", {"node": "Film B"}),
        final_answer("Because Film A written_by Jane Doe, the answer is Film B.",
                FILM_GOLD_EDGES),
    ])
    em_col = lambda **cols: [float(e) for e in cols["em"]]
    r, c = score_episodes([prefix_credit, correct], [film_item, film_item], em_col)
    assert r == [0.0, 1.0] and c["gated_acc"] == [False, True]
    assert c["em"] == [False, True] and c["gate_pass"] == [True, True]


# ------------------------------------------------------------ the normalization

@pytest.mark.parametrize("answer, gold, step", [
    ("the answer is HAR Processing", "HAR processing", "case folding"),
    ("the answer is center  gas\nflow", "center gas flow", "whitespace collapsed inside a label"),
    ("the answer is ﬁn structure", "fin structure", "NFKC (the ligature ﬁ is 'fi')"),
])
def test_correctness_matches_under_the_normalization(answer, gold, step):
    """`score_answer` compares under `kg.serialize.normalize`, and each case
    here fails without one of its three steps. Case folding alone carries
    0.100 of the 4B's stored headline: 0.839, against 0.739 under a raw
    substring test over the same runs."""
    assert em_claim_of(answer, [gold]) is True, step
