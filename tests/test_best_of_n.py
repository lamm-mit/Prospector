"""The gold-free best-of-n vote, `graph_reasoner.select.select`, which
`scripts/select/best_of_n.py` runs over a pool of stored runs. Its loader and
the claim-scoped vote are covered in `test_gated_acc.py`, the script's own
surface in `test_cli_surface.py`.

The selector reads no gold-derived field, which is what makes best-of-n usable
where no gold answer exists. These tests pin its choices and that property.
"""
from graph_reasoner.select import select


def _cand(name, answer, cited, *, gate_pass=True, abstained=False):
    return {"id": "x", "tier": "hard", "answer": answer,
            "cited": [{"source": s, "relation": "r", "target": t} for s, t in cited],
            "gate_pass": gate_pass, "abstained": abstained,
            "em_claim": False, "gated_acc": False, "reached": False, "n_calls": 5,
            "name": name}


def test_k_of_one_returns_the_reference_unchanged():
    """A pool of one returns the reference rollout itself, so best-of-1
    reproduces the reference run exactly."""
    only = _cand("greedy", "the answer is etch rate", [("etch rate", "step")])
    assert select([only]) is only


def test_majority_vote_beats_the_reference_when_the_samples_agree():
    """Three rollouts naming `oxide` outvote the reference naming `etch rate`,
    and the earliest of the winning bloc is returned."""
    greedy = _cand("greedy", "the answer is etch rate", [("etch rate", "step")])
    others = [_cand(f"s{i}", "the answer is oxide", [("oxide", "step")])
              for i in range(3)]
    assert select([greedy] + others)["name"] == "s0", "earliest of the winning bloc"


def test_ties_resolve_to_the_reference():
    """One vote each: the tie goes to the reference rollout."""
    greedy = _cand("greedy", "the answer is etch rate", [("etch rate", "step")])
    other = _cand("s0", "the answer is oxide", [("oxide", "step")])
    assert select([greedy, other])["name"] == "greedy"


def test_gate_failures_and_abstentions_are_excluded_from_the_pool():
    """A rollout that abstained or failed the gate is not a candidate, however
    many of them agree."""
    greedy = _cand("greedy", "the answer is etch rate", [("etch rate", "step")])
    bad = [_cand("h", "the answer is oxide", [("oxide", "step")], gate_pass=False),
           _cand("a", "insufficient information", [("oxide", "step")], abstained=True),
           _cand("h2", "the answer is oxide", [("oxide", "step")], gate_pass=False)]
    assert select([greedy] + bad)["name"] == "greedy"


def test_an_empty_pool_falls_back_to_the_reference_rather_than_erroring():
    """When every rollout failed the gate the reference is returned, so the
    item stays in the evaluated set."""
    cands = [_cand("greedy", "x", [("a", "b")], gate_pass=False),
             _cand("s0", "y", [("c", "d")], gate_pass=False)]
    assert select(cands)["name"] == "greedy"


def test_a_gate_passing_rollout_whose_answer_names_no_cited_node_does_not_win():
    """`_supporting_nodes` is empty when the answer names nothing it cited, so
    such a rollout casts no vote — it cannot win a majority on its own."""
    greedy = _cand("greedy", "the answer is etch rate", [("etch rate", "step")])
    mute = [_cand(f"m{i}", "some unrelated prose", [("oxide", "step")])
            for i in range(3)]
    assert select([greedy] + mute)["name"] == "greedy"


def test_selector_reads_no_gold_field():
    """Flipping every gold-derived field leaves the choice unchanged."""
    pool = [_cand("greedy", "the answer is etch rate", [("etch rate", "step")]),
            _cand("s0", "the answer is oxide", [("oxide", "step")]),
            _cand("s1", "the answer is oxide", [("oxide", "step")])]
    before = select([dict(c) for c in pool])["name"]
    for c in pool:
        c["em_claim"] = c["gated_acc"] = c["reached"] = True   # gold-derived, all flipped
    assert select(pool)["name"] == before


class _GoldFree(dict):
    """A candidate that raises when a gold-derived field is read."""
    GOLD = ("em_claim", "gated_acc", "reached", "label")

    def __getitem__(self, k):
        if k in self.GOLD:
            raise AssertionError(f"the selector read the gold field {k!r}")
        return super().__getitem__(k)

    def get(self, k, default=None):
        if k in self.GOLD:
            raise AssertionError(f"the selector read the gold field {k!r}")
        return super().get(k, default)


def test_the_verifier_path_reads_no_gold_field(monkeypatch):
    """The offered pool, the margins and the pick never touch a gold field: a
    verifier that read `gated_acc` would score the paper's pool with the
    answer key. The span and the scorer are stubbed, so the margin follows the
    rendered prompt alone (question, answer, cited chain)."""
    import graph_reasoner.train.tokenize as tokenize
    from graph_reasoner.select.pool import offered, pick_by_margin
    from graph_reasoner.select.verifier import score_pool

    def fake_span(messages, tok):          # "ids": the prompt and the reply; the reply is the scored span
        return [messages[0]["content"], messages[1]["content"]], [1]

    def score_ids(ids):                    # log P(yes) is high when the prompt names oxide
        prompt, label = ids
        return [None, 1.0 if (label == "yes") == ("oxide" in prompt) else 0.0]

    monkeypatch.setattr(tokenize, "completion_logprob_span", fake_span)
    pool = [_GoldFree(_cand("greedy", "the answer is etch rate", [("etch rate", "step")])),
            _GoldFree(_cand("s0", "the answer is oxide", [("oxide", "step")])),
            _GoldFree(_cand("s1", "the answer is oxide", [("oxide", "step")], gate_pass=False))]
    live = offered(pool)
    assert [c["name"] for c in live] == ["greedy", "s0"]
    margins = score_pool(score_ids, None, "q", live)
    assert margins == [-1.0, 1.0] and pick_by_margin(margins) == 1
