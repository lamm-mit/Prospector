"""The arithmetic the served adapters trained under, pinned on CPU.

The GRPO optimiser settings (`grpo._grpo_args`), the rollout environment a
config leaves to the defaults (`rollout.episode_config`), the objective the
trainer echoes at launch (`rollout.build_objective`) and the two masks the
trainers compute (`tokenize.token_spans` for GRPO,
`tokenize.supervised_arrays` for SFT) are each read by one function, and each
is pinned here.

Torch-free. The mask tests need the shipped tokenizer from the local cache
and skip without it.
"""
from __future__ import annotations

import math
import pathlib

import pytest
import yaml

from graph_reasoner.train.grpo import _bessel, _grpo_args
from graph_reasoner.train.rollout import build_objective, episode_config

CONFIGS = pathlib.Path(__file__).resolve().parents[1] / "configs"
GRPO_FILES = sorted(CONFIGS.glob("grpo_*.yaml"))
#: The served rounds' hyperparameters: both decision rounds of every size.
SERVED_ROUND = CONFIGS / "grpo_decision.yaml"


# ------------------------------------------------------------ the optimiser

def test_grpo_args_pin_the_reference_arithmetic():
    """The optimiser settings `_grpo_args` fixes or defaults: the per-sequence
    loss type, the cosine schedule with warmup, no KL unless `kl_beta` says
    so, Adam's beta2 and epsilon, and group z-scoring."""
    args = _grpo_args({"output_dir": "x", "temperature": 0.9})
    assert args["loss_type"] == "grpo"
    assert args["lr_scheduler_type"] == "cosine"
    assert args["warmup_steps"] == 0.03, "below 1: a ratio of the run's steps"
    assert args["beta"] == 0.0
    assert args["adam_beta2"] == 0.95
    assert args["adam_epsilon"] == 1e-12
    assert args["scale_rewards"] == "group"
    assert args["num_generations"] == 16, "the recipe's group size"
    assert args["bf16"] is False and args["fp16"] is False, "fp32 unless asked"


def test_grpo_args_pass_the_config_through():
    args = _grpo_args({"output_dir": "x", "temperature": 0.9, "kl_beta": 0.015,
                       "scale_rewards": "none", "model_dtype": "bfloat16",
                       "max_seq_len": 16384, "save_total_limit": 0})
    assert args["beta"] == 0.015
    assert args["scale_rewards"] == "none", "a config saying none is not z-scored"
    assert args["bf16"] is True
    assert args["max_completion_length"] == 16384
    assert args["save_total_limit"] is None, "0 keeps every checkpoint"


def test_a_batch_scaled_config_is_refused():
    """`_bessel` corrects TRL's group z-scoring and leaves the unscaled case
    alone; TRL's `batch` divides by another sample standard deviation that it
    does not correct, so the value is refused rather than mis-scaled."""
    with pytest.raises(SystemExit, match="scale_rewards"):
        _grpo_args({"output_dir": "x", "temperature": 0.9, "scale_rewards": "batch"})


def test_a_null_checkpoint_limit_keeps_every_checkpoint():
    args = _grpo_args({"output_dir": "x", "temperature": 0.9, "save_total_limit": None})
    assert args["save_total_limit"] is None
    assert _grpo_args({"output_dir": "x", "temperature": 0.9})["save_total_limit"] == 2


def test_the_served_round_trained_under_these_settings():
    """A golden read of the decision rounds' file, so a change to a default
    or to the file is visible as a change to what the served rounds ran."""
    config = {**yaml.safe_load(SERVED_ROUND.read_text()), "output_dir": "x"}
    args = _grpo_args(config)
    got = {k: args[k] for k in (
        "per_device_train_batch_size", "gradient_accumulation_steps",
        "learning_rate", "num_generations", "beta", "max_steps", "save_steps",
        "save_total_limit", "bf16", "max_completion_length", "temperature",
        "top_p", "scale_rewards")}
    assert got == {
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": 64,
        "learning_rate": 1.5e-5, "num_generations": 16, "beta": 0.015,
        "max_steps": 120, "save_steps": 20, "save_total_limit": None,
        "bf16": True, "max_completion_length": 16384, "temperature": 0.9,
        "top_p": 0.95, "scale_rewards": "group"}


def test_bessel_turns_trl_sample_std_into_population_std():
    """TRL divides each group's centred rewards by the sample standard
    deviation. `_bessel` is the factor the parity loss multiplies by to make
    that the population one: sqrt(n/(n-1)) when TRL z-scores, 1 when it only
    centres."""
    assert _bessel(16, "group") == pytest.approx(math.sqrt(16 / 15))
    assert _bessel(16, "none") == 1.0
    assert _bessel(1, "group") == 1.0


# ------------------------------------------------------- the rollout environment

def test_episode_config_defaults_are_the_served_protocol():
    """A key a GRPO config leaves out takes the eval's default, and the
    defaults are the protocol the served policies are evaluated under."""
    cfg = episode_config({"temperature": 0.9})
    assert cfg.max_turns == 20
    assert cfg.turn_max_tokens == 1024
    assert cfg.seed == 0
    assert cfg.tool_cfg.page_size == 40
    assert cfg.tool_cfg.hub_threshold == 150
    assert cfg.tool_cfg.connect_max_hops == 3


# ------------------------------------------------------------ the objective

def test_the_objective_echo_states_the_recipe(capsys):
    """The launch line is the record of what a run trained. An empty config
    trains the recipe's objective, and says so."""
    reward_fn = build_objective({})
    assert capsys.readouterr().out.strip() == (
        "objective: 1 for a gated success, else 0.25 x evidence_recall")
    assert reward_fn(em=[True, False], gate_pass=[True, True],
                     evidence_recall=[1.0, 0.4]) == [1.0, pytest.approx(0.1)]


@pytest.mark.parametrize("path", GRPO_FILES, ids=lambda p: p.stem)
def test_every_shipped_round_trains_the_default_objective(path, capsys):
    """Both GRPO files state the objective explicitly and it is the default
    one, so the defaults and the files cannot drift apart."""
    assert len(GRPO_FILES) == 2, GRPO_FILES
    config = yaml.safe_load(path.read_text())
    assert {"partial_on", "partial_credit"} <= set(config), \
        "the objective is stated, not inherited from the defaults"
    build_objective(config)
    echoed = capsys.readouterr().out
    build_objective({})
    assert echoed == capsys.readouterr().out


# ------------------------------------------------------------- the two masks

MESSAGES = [
    {"role": "system", "content": "You are a graph agent."},
    {"role": "user", "content": "Which gas etches silicon nitride?"},
    {"role": "assistant",
     "content": '<think>Find it.</think>\n<tool_call>{"tool": "find"}</tool_call>'},
    {"role": "user", "content": "OBSERVATION (call 1/8): FOUND 1 node"},
    {"role": "assistant",
     "content": '<think>Expand it.</think>\n<tool_call>{"tool": "expand"}</tool_call>'},
    {"role": "user", "content": "OBSERVATION (call 2/8): a --[etched by]--> CF4"},
    {"role": "assistant", "content": "<graph_json>{}</graph_json>\nANSWER: CF4"},
]


def test_supervised_arrays_is_fenceposted_against_the_tokenizers_own_ids(shipped_tokenizer):
    tok = shipped_tokenizer
    from graph_reasoner.train.tokenize import supervised_arrays

    ids = tok(tok.apply_chat_template(MESSAGES, tokenize=False),
              add_special_tokens=False)["input_ids"]
    inp, tgt, w = supervised_arrays(MESSAGES, tok)
    assert inp == ids[:-1] and tgt == ids[1:] and len(w) == len(tgt)


def test_an_over_length_transcript_is_dropped_not_truncated(shipped_tokenizer):
    """The answer is the last thing in a trajectory, so an over-length
    transcript is dropped: right-truncation would cut the answer."""
    tok = shipped_tokenizer
    from graph_reasoner.train.tokenize import supervised_arrays

    assert supervised_arrays(MESSAGES, tok, max_length=8) is None
    assert supervised_arrays(MESSAGES, tok, max_length=100_000) is not None


def test_the_grpo_and_sft_masks_differ_by_exactly_the_generation_headers(shipped_tokenizer):
    """GRPO's span mask also trains the `<|im_start|>assistant\\n` header of
    every assistant turn after the first (the first one ends the prompt);
    SFT's weights never do. Everything else is the same token set, and
    neither touches an observation."""
    tok = shipped_tokenizer
    from graph_reasoner.train.tokenize import supervised_arrays, token_spans

    p_ids, c_ids, mask, _turns = token_spans(MESSAGES, tok)
    inp, tgt, w = supervised_arrays(MESSAGES, tok)
    ids = list(inp) + [tgt[-1]]
    assert list(p_ids) + list(c_ids) == ids, "one tokenization of one rendering"

    grpo = {len(p_ids) + i for i, m in enumerate(mask) if m}
    sft = {j + 1 for j, x in enumerate(w) if x}
    assert sft <= grpo
    header = tok("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    n_assistant = sum(m["role"] == "assistant" for m in MESSAGES)
    assert len(grpo - sft) == (n_assistant - 1) * len(header)
    extra = sorted(grpo - sft)
    assert [ids[i] for i in extra] == header * (n_assistant - 1)

    for trained in (grpo, sft):
        text = tok.decode([ids[i] for i in sorted(trained)])
        assert "OBSERVATION" not in text and "etched by" not in text
        assert "ANSWER: CF4" in text and '"expand"' in text
