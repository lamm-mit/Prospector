"""The sampling seams share one render, one stop configuration and one turn split.

`train/generate.py` defines them for the three local engines in
`eval/local_model.py`, the GRPO rollouts in `train/grpo.py` and the steered
engine in `latent/steer.py`; these tests hold each seam to that definition.

Nothing here imports torch, trl, peft or vllm. The two `*_kwargs` builders
return plain dicts, so they are checked without an engine.
"""
import ast
import inspect
import pathlib

import pytest

from graph_reasoner.train import generate as G
from graph_reasoner.train.tokenize import token_spans

MESSAGES = [
    {"role": "system", "content": "You explore a graph with tools."},
    {"role": "user", "content": "Which etchant is used for the nitride step?"},
    {"role": "assistant", "content": "<tool_call>{\"name\":\"find\"}</tool_call>"},
    {"role": "user", "content": "OBSERVATION: 2 edges"},
    {"role": "assistant", "content": "ANSWER: CF4"},
]


class Recorder:
    """A tokenizer stand-in that records how it was called, so the kwarg
    assertions need no cached tokenizer."""

    def __init__(self, template=None):
        self.calls = []
        self.chat_template = template
        self.eos_token_id = 7

    def apply_chat_template(self, conv, **kw):
        self.calls.append(kw)
        return f"<rendered {len(conv)}>"

    def __call__(self, text, **kw):
        self.calls.append(kw)
        return {"input_ids": [1, 2, 3]}

    def convert_tokens_to_ids(self, tok):
        return 106


# ------------------------------------------------------- one definition

REWIRED = ("graph_reasoner.eval.local_model", "graph_reasoner.train.grpo", "graph_reasoner.latent.steer")


@pytest.mark.parametrize("modname", REWIRED)
def test_the_generate_contract_has_one_definition(modname):
    """Neither module renders or encodes prompts itself, or spells the
    tool-call stop string. Checked on the AST, so a mention in a comment or a
    docstring does not count."""
    import importlib

    mod = importlib.import_module(modname)
    src = inspect.getsource(mod)
    assert "render_prompts(" in src, f"{modname} does not go through the owner"
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr != "apply_chat_template", \
                f"{modname} renders its own prompts"
        if isinstance(node, ast.keyword):
            assert node.arg != "add_special_tokens", \
                f"{modname} encodes its own prompts"
        if isinstance(node, ast.Constant):
            assert node.value != "</tool_call>", \
                f"{modname} spells the stop string instead of importing it"


# ------------------------------------------------------- the sampling/training tie

def test_the_sampling_render_is_the_training_render(shipped_tokenizer):
    """The prompt the sampler conditions on at turn 1 (`generate.render_prompts`)
    is the prompt the trainer masks (`tokenize.token_spans`)."""
    tok = shipped_tokenizer
    prompt_ids, _completion, _mask, _turns = token_spans(MESSAGES, tok)
    sampled = G.encode_prompts(G.render_prompts([MESSAGES[:2]], tok), tok)[0]
    assert sampled == list(prompt_ids)


# ------------------------------------------------------- the Qwen no-op reads

def test_the_qwen_path_is_a_byte_identical_no_op(shipped_tokenizer):
    """`plain_chatml` on a Qwen tokenizer has no turn delimiter to add (the eos
    ends the turn), so `generate.py` adds no stop string, eos id or stop token
    id."""
    tok = shipped_tokenizer
    assert G.stop_strings(tok, None) == []
    assert G.stop_strings(tok, [G.TOOL_CALL_STOP]) == ["</tool_call>"]
    assert G.hf_generate_kwargs(tok, None) == {}
    kw = G.vllm_sampling_kwargs(tok, max_new_tokens=64, temperature=0.0,
                                top_p=1.0)
    assert kw["stop_token_ids"] is None, "[] is a shape no vLLM path has passed"
    assert G.turn_text("<tool_call>{}</tool_call>", tok) == "<tool_call>{}</tool_call>"


def test_a_qwen_generate_call_does_not_pin_eos_token_id(shipped_tokenizer):
    """With no turn delimiter, `eos_token_id` is left unset, so the eos ids
    the model's generation config lists stay in force."""
    tok = shipped_tokenizer
    assert "eos_token_id" not in G.hf_generate_kwargs(tok, [G.TOOL_CALL_STOP])


def test_no_stop_means_no_stop(shipped_tokenizer):
    """A caller that passes no stop gets none: `None` does not default to the
    tool-call stop."""
    tok = shipped_tokenizer
    assert G.stop_strings(tok, None) == []
    assert "stop_strings" not in G.hf_generate_kwargs(tok, None)
    assert G.vllm_sampling_kwargs(tok, max_new_tokens=8, temperature=0.7,
                                  top_p=0.95)["stop"] is None


# ------------------------------------------------- the engine must stop on time

def test_stop_id_set_normalises_every_shape_an_engine_reports():
    """An int, a list, a set and `None` (no stop declared) all normalise to
    a set of ids."""
    assert G.stop_id_set(None) == set()
    assert G.stop_id_set(248046) == {248046}
    assert G.stop_id_set([1, 106]) == {1, 106}
    assert G.stop_id_set({248044}, 248046, None) == {248044, 248046}


def test_a_stop_set_that_covers_the_policy_is_silent():
    G.assert_stop_covers_policy([248044, 248046], 248046, engine="e",
                                source="s", remedy="r")


def test_the_qwen35_config_tokenizer_disagreement_is_refused():
    """An engine stop set that misses the policy's eos is refused.

    The ids are Qwen3.5-4B's: config.json says 248044 (`<|endoftext|>`) and
    tokenizer_config.json says 248046 (`<|im_end|>`), so an engine that reads
    its stop from the config would run every `plain_chatml` answer turn to
    max_tokens without an error.
    """
    with pytest.raises(SystemExit) as e:
        G.assert_stop_covers_policy(248044, 248046, engine="mlx-lm",
                                    source="config.json", remedy="Rebuild it.")
    msg = str(e.value)
    assert "248044" in msg and "248046" in msg
    # The message names where the stop set was read and the remedy.
    assert "config.json" in msg and "Rebuild it." in msg


def test_an_engine_with_no_stop_at_all_is_refused():
    with pytest.raises(SystemExit):
        G.assert_stop_covers_policy(None, 248046, engine="e", source="s",
                                    remedy="r")


def test_a_tokenizer_with_no_eos_is_not_something_to_check():
    """A tokenizer that declares no eos has nothing to cover."""
    G.assert_stop_covers_policy(None, None, engine="e", source="s", remedy="r")


def test_every_local_engine_asserts_its_stop_set_at_load():
    """All three local engine constructors call `assert_stop_covers_policy`,
    checked on the AST of `eval/local_model.py`."""
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src/graph_reasoner/eval/local_model.py").read_text()
    tree = ast.parse(src)
    seen = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        init = next((f for f in node.body if isinstance(f, ast.FunctionDef)
                     and f.name == "__init__"), None)
        seen[node.name] = bool(init) and any(
            isinstance(n, ast.Name) and n.id == "assert_stop_covers_policy"
            for n in ast.walk(init))
    assert seen == {"LocalHFModel": True, "LocalVLLMModel": True,
                    "LocalMLXModel": True}, seen


# ------------------------------------------------------- render and encode

def test_render_returns_strings_not_a_pre_tokenized_batch():
    """`render_prompts` renders to text with a generation prompt, and
    `encode_prompts` encodes separately without adding special tokens."""
    rec = Recorder()
    out = G.render_prompts([MESSAGES[:2], MESSAGES], rec)
    assert all(isinstance(t, str) for t in out)
    assert rec.calls[0] == {"tokenize": False, "add_generation_prompt": True}
    rec2 = Recorder()
    ids = G.encode_prompts(["abc"], rec2)
    assert ids == [[1, 2, 3]]
    assert rec2.calls[0] == {"add_special_tokens": False}


def test_include_stop_str_in_output_is_not_negotiable():
    """HF's `stop_strings` leaves the stop in the text and the turn parser
    expects to see `</tool_call>`; vLLM strips it unless told otherwise."""
    kw = G.vllm_sampling_kwargs(Recorder(), max_new_tokens=8, temperature=0.9,
                                top_p=0.95)
    assert kw["include_stop_str_in_output"] is True


# ------------------------------------------------------- pure-python contract

def test_a_prompt_at_the_context_limit_is_dropped_not_raised():
    """vLLM raises on a prompt at its context window, so such a prompt gets an
    empty turn instead, and the episode loop ends an episode after `max_empty`
    of them. A limit of `None` keeps every prompt."""
    assert G.fits_context([[0] * 10, [0] * 100], 100) == [0]
    assert G.fits_context([[0] * 10, [0] * 100], None) == [0, 1]
    assert G.fits_context([], 100) == []
    assert G.CONTEXT_MARGIN == 8
    # exactly at the margin is out: limit - MARGIN is a strict upper bound
    assert G.fits_context([[0] * 92], 100) == []
    assert G.fits_context([[0] * 91], 100) == [0]


def test_length_order_is_stable():
    """Short to long by string length, ties in input order: the order decides
    which prompts share a padded batch, and therefore the sampling stream."""
    assert G.length_order(["bb", "a", "cc"]) == [1, 0, 2]
    assert G.length_order([]) == []


def test_the_shared_owner_reaches_no_engine_state():
    """`generate.py` imports only `__future__` and `tokenize`, and reads
    neither caller's context limit: the limit arrives as an argument."""
    tree = ast.parse(inspect.getsource(G))
    # The AST, not a substring scan, so a name in a docstring does not count.
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "max_seq_len" not in names | attrs
    assert "max_model_len" not in names | attrs
    imported = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    for node in imported:
        assert isinstance(node, ast.ImportFrom), f"bare import in core: {ast.dump(node)}"
        assert node.module in ("__future__", "tokenize"), node.module
