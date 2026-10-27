"""The SFT rows train exactly the tokens the shared mask weights: `hf_row` in
`train/sft.py` against `supervised_arrays` in `train/tokenize.py`.

`supervised_arrays` scores position i as "predict target[i] from
input[:i+1]" and HF shifts labels internally, so `hf_row` must offset its
labels by exactly one.
"""
from __future__ import annotations

from graph_reasoner.train.sft import hf_row
from graph_reasoner.train.tokenize import supervised_arrays

MESSAGES = [
    {"role": "system", "content": "You are a graph agent."},
    {"role": "user", "content": "Which gas etches silicon nitride?"},
    {"role": "assistant",
     "content": '<think>Find it.</think>\n<tool_call>{"tool": "find"}</tool_call>'},
    {"role": "user", "content": "OBSERVATION (call 1/8): FOUND 1 node"},
    {"role": "assistant", "content": "<graph_json>{}</graph_json>\nANSWER: CF4"},
]


def test_labels_select_the_same_tokens_the_weights_do(shipped_tokenizer):
    tok = shipped_tokenizer
    inp, tgt, w = supervised_arrays(MESSAGES, tok)
    ids, labels = hf_row(inp, tgt, w)

    core_targets = [tgt[i] for i in range(len(w)) if w[i]]
    hf_targets = [x for x in labels if x != -100]
    assert hf_targets == core_targets
    assert len(labels) == len(ids) == len(inp) + 1


def test_the_shift_is_off_by_exactly_one_in_the_right_direction():
    """labels[i+1] pairs with weights[i], over the reassembled ids."""
    inp, tgt = [10, 11, 12, 13], [11, 12, 13, 14]     # ids == [10,11,12,13,14]
    assert hf_row(inp, tgt, [0, 1, 1, 0])[1] == [-100, -100, 12, 13, -100]
    assert hf_row(inp, tgt, [1, 0, 0, 0])[1] == [-100, 11, -100, -100, -100]
    # The last weight still trains a token: labels built on input_tokens
    # rather than the reassembled ids would drop it.
    assert hf_row(inp, tgt, [0, 0, 0, 1])[1] == [-100, -100, -100, -100, 14]


def test_no_observation_token_is_ever_a_target(shipped_tokenizer):
    """No tool-result token is a training target; the model's own turns are.
    A model trained on observations learns to emit them itself."""
    tok = shipped_tokenizer
    inp, tgt, w = supervised_arrays(MESSAGES, tok)
    _ids, labels = hf_row(inp, tgt, w)
    trained = tok.decode([x for x in labels if x != -100])
    assert "OBSERVATION" not in trained
    assert "ANSWER: CF4" in trained
    assert "tool_call" in trained


def test_weights_are_binary_so_minus_100_is_lossless(shipped_tokenizer):
    """`hf_row` expresses a weight as include/exclude, which is exact only
    because `supervised_arrays` emits weights of 0 or 1."""
    tok = shipped_tokenizer
    _inp, _tgt, w = supervised_arrays(MESSAGES, tok)
    assert set(w) <= {0, 1}
