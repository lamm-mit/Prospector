"""The `plain_chatml` override renders plain ChatML byte for byte.

Covers `override_chat_template`, `template_name` and `turn_stop_tokens` in
`train/tokenize.py`, and `_effective_chat_template` in `agent/eval.py`, which
reads the effective template back from a saved tokenizer. Qwen3.5's own
template rewrites earlier turns, so no loss mask is computable under it; every
training config selects the concatenative `plain_chatml` instead.

The tests use cached tokenizers only and skip when one is unavailable.
"""
from __future__ import annotations

import pytest

from conftest import unmasked_text
from graph_reasoner.train.tokenize import (
    PLAIN_CHATML,
    override_chat_template,
    supervised_arrays,
    token_spans,
)

HYBRID = "Qwen/Qwen3.5-0.8B"      # its own template rewrites earlier turns

MESSAGES = [
    {"role": "system", "content": "You are a graph agent."},
    {"role": "user", "content": "Which gas etches silicon nitride?"},
    {"role": "assistant",
     "content": '<think>Find the anchor.</think>\n<tool_call>{"tool": "find"}</tool_call>'},
    {"role": "user", "content": "OBSERVATION (call 1/8): FOUND 1 node"},
    {"role": "assistant",
     "content": '<think>Expand it.</think>\n<tool_call>{"tool": "expand"}</tool_call>'},
    {"role": "user", "content": "OBSERVATION (call 2/8): a --[etched by]--> CF4"},
    {"role": "assistant", "content": "<graph_json>{}</graph_json>\nANSWER: CF4"},
]

#: MESSAGES rendered as plain ChatML: the string the override must produce.
REFERENCE_RENDERING = (
    "<|im_start|>system\nYou are a graph agent.<|im_end|>\n"
    "<|im_start|>user\nWhich gas etches silicon nitride?<|im_end|>\n"
    "<|im_start|>assistant\n<think>Find the anchor.</think>\n"
    '<tool_call>{"tool": "find"}</tool_call><|im_end|>\n'
    "<|im_start|>user\nOBSERVATION (call 1/8): FOUND 1 node<|im_end|>\n"
    "<|im_start|>assistant\n<think>Expand it.</think>\n"
    '<tool_call>{"tool": "expand"}</tool_call><|im_end|>\n'
    "<|im_start|>user\nOBSERVATION (call 2/8): a --[etched by]--> CF4<|im_end|>\n"
    "<|im_start|>assistant\n<graph_json>{}</graph_json>\nANSWER: CF4<|im_end|>\n"
)


def _tok(repo):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(repo, local_files_only=True)
    except OSError:                     # not in the cache; any other failure is an error
        pytest.skip(f"{repo} tokenizer not cached")


def test_plain_chatml_reproduces_the_reference_rendering_on_the_shipped_family():
    """The model's own template is refused as not concatenative, and the
    override renders MESSAGES as the reference string."""
    alt = _tok(HYBRID)
    with pytest.raises(ValueError, match="not concatenative"):
        supervised_arrays(MESSAGES, alt)

    override_chat_template(alt, "plain_chatml", model=HYBRID)
    assert alt.apply_chat_template(MESSAGES, tokenize=False) == REFERENCE_RENDERING


def test_the_override_trains_exactly_the_assistant_turns():
    """Under the override, the SFT weights cover each assistant turn's content
    and its turn end, and nothing else; the GRPO span split's prompt is the
    reference rendering up to the first generation header."""
    alt = override_chat_template(_tok(HYBRID), "plain_chatml", model=HYBRID)
    _inp, tgt, w = supervised_arrays(MESSAGES, alt)
    assert unmasked_text(tgt, w, alt) == "".join(
        m["content"] + "<|im_end|>\n" for m in MESSAGES if m["role"] == "assistant")
    # token_spans uses a different masking convention from supervised_arrays
    # (see tokenize.py), so it is checked separately.
    p_ids, _c_ids, _mask, _turns = token_spans(MESSAGES, alt)
    head = REFERENCE_RENDERING.split("<|im_start|>assistant\n")[0]
    assert alt.decode(p_ids) == head + "<|im_start|>assistant\n"


def test_none_is_a_no_op():
    alt = _tok(HYBRID)
    before = alt.apply_chat_template(MESSAGES, tokenize=False)
    override_chat_template(alt, None, model=HYBRID)
    assert alt.apply_chat_template(MESSAGES, tokenize=False) == before


def test_unknown_name_raises():
    alt = _tok(HYBRID)
    with pytest.raises(SystemExit, match="unknown chat_template"):
        override_chat_template(alt, "no_such_template", model=HYBRID)


def test_non_chatml_vocabulary_is_refused_rather_than_silently_wrong():
    """A vocabulary on which a ChatML delimiter is not one token is refused."""
    alt = _tok("openai/gpt-oss-20b")
    with pytest.raises(SystemExit, match="not ChatML-native"):
        override_chat_template(alt, "plain_chatml", model="openai/gpt-oss-20b")


def test_template_is_a_name_not_a_raw_string():
    """A raw Jinja string is refused: a template is selected by name."""
    alt = _tok(HYBRID)
    with pytest.raises(SystemExit):
        override_chat_template(alt, PLAIN_CHATML, model=HYBRID)



# --------------------------------------------------------- the disclosure seam
# sft.py installs the override and saves the tokenizer beside the adapter; the
# eval summary reads that directory back and names the template.

def test_template_name_reverse_lookup():
    from graph_reasoner.train.tokenize import template_name

    assert template_name(PLAIN_CHATML) == "plain_chatml"
    assert template_name("{% for m in messages %}nope{% endfor %}") is None
    assert template_name(None) is None


def test_saved_tokenizer_carries_the_override(tmp_path):
    alt = _tok(HYBRID)
    override_chat_template(alt, "plain_chatml", model=HYBRID)
    alt.save_pretrained(str(tmp_path))

    import transformers

    back = transformers.AutoTokenizer.from_pretrained(
        str(tmp_path), local_files_only=True)
    assert back.apply_chat_template(MESSAGES, tokenize=False) == REFERENCE_RENDERING

    from graph_reasoner.agent.eval import _effective_chat_template

    # The template is named from the saved directory alone.
    assert _effective_chat_template(f"hf:{tmp_path}", None) == "plain_chatml"


def test_effective_chat_template_hf_resolution(tmp_path):
    import json as _json

    from graph_reasoner.agent.eval import _effective_chat_template

    # Explicit request wins without touching disk.
    assert _effective_chat_template("hf:/nowhere", "plain_chatml") == "plain_chatml"

    # A dir with an unrecognized (native) template stays None, not a guess.
    native = tmp_path / "native"
    native.mkdir()
    (native / "tokenizer_config.json").write_text(
        _json.dumps({"chat_template": "{{ bespoke }}"}), encoding="utf-8")
    assert _effective_chat_template(f"hf:{native}", None) is None

    # A named template in tokenizer_config.json is recovered by name, and the
    # last adapter in a chain is the directory that is read.
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "tokenizer_config.json").write_text(
        _json.dumps({"chat_template": PLAIN_CHATML}), encoding="utf-8")
    assert _effective_chat_template(f"hf:{native}", None,
                                    adapter=f"{native},{plain}") == "plain_chatml"

    # Missing dir / no tokenizer files: None rather than an exception.
    assert _effective_chat_template("hf:/nowhere", None) is None


# ---------------------------------------------------------------- the guard
def test_a_foreign_vocabulary_is_refused():
    """A template's delimiters must each be one token on the target
    vocabulary; a tokenizer that splits them is refused."""
    class Shattering:
        chat_template = None

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2, 3]}

    with pytest.raises(SystemExit, match="not ChatML-native"):
        override_chat_template(Shattering(), "plain_chatml", model="foreign")


def test_turn_stop_tokens_follow_the_effective_template():
    """`<|im_end|>` is the Qwen tokenizer's eos, so neither `plain_chatml` nor
    the native template adds a stop token."""
    from graph_reasoner.train.tokenize import turn_stop_tokens

    alt = _tok(HYBRID)
    assert turn_stop_tokens(alt) == []                # native/None template
    override_chat_template(alt, "plain_chatml", model=HYBRID)
    assert turn_stop_tokens(alt) == []
