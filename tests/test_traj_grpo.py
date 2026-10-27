"""GRPO components that run without a GPU: the reward and the token bookkeeping.

Guards `train/reward.py` (the reward's shape), `train/tokenize.py`
(`boundaries_to_mask`, `token_spans`), and the config validation of both
trainers, `train/grpo.py` and `train/sft.py`. The loss itself is tested in
`test_parity_loss.py`, which needs torch and is skipped without it; the
trainer's arithmetic is pinned in `test_trainer_arithmetic.py`.

The mask must give no observation token a policy gradient. The boundary
mapper is pure, so it is tested exactly; the tokenizer round trip runs only
where the shipped tokenizer is cached.
"""
import pytest

from graph_reasoner.train.reward import make_gated_reward
from graph_reasoner.train.tokenize import boundaries_to_mask, token_spans

# ---------------------------------------------------------------- reward

def test_a_correct_gated_answer_earns_one_and_everything_else_the_dense_term():
    """The default reward: 1 when the answer is correct and the gate passed,
    otherwise a quarter of the evidence recall. A correct answer whose chain
    fails the gate earns only the dense term, like any other failure."""
    reward = make_gated_reward()
    out = reward(em=[True, True, False, False],
                 gate_pass=[True, False, True, False],
                 evidence_recall=[1.0, 0.8, 0.4, 0.0],
                 path_recall=[1.0, 0.0, 0.0, 0.0])
    assert out == [1.0, pytest.approx(0.20), pytest.approx(0.10), 0.0]


def test_no_failure_out_earns_a_success():
    """The dense term only orders the failures: with every gold edge seen a
    failure still earns `partial` < 1, and a `partial` that could reach 1 is
    refused."""
    reward = make_gated_reward()
    assert reward(em=[False], gate_pass=[True], evidence_recall=[1.0]) == [0.25]
    for bad in (1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="partial"):
            make_gated_reward(partial=bad)


# ---------------------------------------------------------------- mask mapper

def test_boundary_mapper_refuses_misaligned_boundaries():
    offsets = [(0, 4), (4, 8), (8, 12)]
    with pytest.raises(ValueError, match="inside a token"):
        boundaries_to_mask(offsets, prompt_char_end=6, segments=[(12, "assistant")])


def test_boundary_mapper_handles_end_of_string():
    offsets = [(0, 3), (3, 6)]
    prompt_len, mask, turns = boundaries_to_mask(offsets, 3, [(6, "assistant")])
    assert prompt_len == 1 and mask == [1]


# ---------------------------------------------------------------- round trip

def test_token_spans_round_trip_on_real_template(shipped_tokenizer):
    tok = shipped_tokenizer
    messages = [
        {"role": "system", "content": "You are a graph agent."},
        {"role": "user", "content": "who directed Kotch?"},
        {"role": "assistant",
         "content": '<think>find</think>\n<tool_call>{"tool": "find", "args": {"name": "Kotch"}}</tool_call>'},
        {"role": "user", "content": 'OBSERVATION (call 1/8): FOUND 1 node(s) matching "Kotch"'},
        {"role": "assistant", "content": "<graph_json>{}</graph_json>\nANSWER: X"},
    ]
    p_ids, c_ids, mask, turns = token_spans(messages, tok)
    assert len(mask) == len(c_ids)
    # Decoding the masked-out span must reproduce the observation and nothing else.
    obs = tok.decode([t for t, m in zip(c_ids, mask) if m == 0])
    assert "OBSERVATION (call 1/8)" in obs
    assert "graph_json" not in obs and "tool_call" not in obs
    # And every model-authored string sits in the mask-1 span.
    kept = tok.decode([t for t, m in zip(c_ids, mask) if m == 1])
    assert "tool_call" in kept and "ANSWER: X" in kept
    # Prompt must end at the generation prompt, before any assistant content.
    prompt = tok.decode(p_ids)
    assert prompt.rstrip().endswith("assistant") and "tool_call" not in prompt


# ------------------------------------------------- per-turn credit assignment

def test_turn_index_agrees_with_the_mask_and_numbers_assistant_turns():
    """`turn_idx[i] >= 0` iff `mask[i] == 1`. The observation's token is
    masked 0, both assistant turns' tokens 1, and the prompt is excluded."""
    # tokens: [0,4) [4,8) | [8,12) [12,16) | [16,20) | [20,24)
    # prompt ends at 8; assistant turn ends at 16; observation ends at 20;
    # final assistant ends at 24.
    offsets = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
    prompt_len, mask, turns = boundaries_to_mask(
        offsets, prompt_char_end=8,
        segments=[(16, "assistant"), (20, "user"), (24, "assistant")])
    assert prompt_len == 2
    assert mask == [1, 1, 0, 1]
    assert turns == [0, 0, -1, 1], "two assistant turns, numbered in order"
    assert [t >= 0 for t in turns] == [bool(m) for m in mask]


def test_partial_credit_reads_the_recall_it_is_told_to():
    """The dense term reads the recall `partial_on` names: `evidence_recall`
    (gold edges seen), the default, or `path_recall` (gold edges cited)."""
    kw = dict(em=[False], gate_pass=[False],
              path_recall=[0.2], evidence_recall=[0.8])
    cited = make_gated_reward(partial=0.25, partial_on="path_recall")(**kw)
    seen = make_gated_reward(partial=0.25, partial_on="evidence_recall")(**kw)
    assert cited == [pytest.approx(0.05)], "0.25 x path_recall"
    assert seen == [pytest.approx(0.20)], "0.25 x evidence_recall"
    assert make_gated_reward()(**kw) == seen, "the default reads what the search saw"


def test_partial_on_rejects_an_unknown_recall_rather_than_silently_zeroing():
    """An unknown `partial_on` raises rather than zeroing the dense term."""
    with pytest.raises(ValueError, match="partial_on"):
        make_gated_reward(partial=0.25, partial_on="evidence")


# ---------------------------------------------------------- rollout environment

def test_the_rollout_environment_reads_every_tool_field_by_its_name():
    """Every `ToolConfig` field is a config key under its own name and the
    trainer accepts it, so training rollouts can follow any tool setting the
    eval can change."""
    from dataclasses import fields

    from graph_reasoner.agent.loop import EpisodeConfig
    from graph_reasoner.agent.tools import ToolConfig
    from graph_reasoner.train.grpo import KNOWN_CONFIG_KEYS
    from graph_reasoner.train.rollout import EPISODE_KEYS, TOOL_KEYS

    assert {f.name for f in fields(ToolConfig)} <= set(TOOL_KEYS)
    assert set(TOOL_KEYS) <= {f.name for f in fields(ToolConfig)}
    assert set(EPISODE_KEYS) <= {f.name for f in fields(EpisodeConfig)}
    assert set(TOOL_KEYS) | set(EPISODE_KEYS) | {"env_seed"} <= KNOWN_CONFIG_KEYS


def test_a_key_the_config_omits_takes_the_eval_default():
    """Training rollouts and evaluation read one default per knob, the
    `EpisodeConfig` dataclass's."""
    from graph_reasoner.agent.loop import SERVED_TURN_MAX_TOKENS, EpisodeConfig
    from graph_reasoner.train.rollout import episode_config

    cfg = episode_config({"temperature": 0.7})
    assert cfg == EpisodeConfig(temperature=0.7)
    assert cfg.turn_max_tokens == SERVED_TURN_MAX_TOKENS == 1024

    cfg = episode_config({"temperature": 0.7, "turn_max_tokens": 512, "env_seed": 3,
                          "tools": ["find", "expand"], "hub_summary_limit": 50})
    assert cfg.turn_max_tokens == 512 and cfg.seed == 3
    assert cfg.tool_cfg.tools == ("find", "expand")
    assert cfg.tool_cfg.hub_summary_limit == 50


def test_grpo_refuses_unknown_config_key():
    """A config key the trainer does not read is refused
    (`config.check_config_keys`). tests/test_module_health.py checks the other
    direction, that every declared key is read."""
    from graph_reasoner.train.grpo import validate_config_keys

    with pytest.raises(SystemExit, match="unknown config key"):
        validate_config_keys({"lr": 1e-5, "retuned_discount": 0.9})


@pytest.mark.parametrize("key, value", [
    ("turn_lambda", 0.0), ("turn_td", 10), ("turn_floor", 0.45),
    ("loss_impl", "parity"), ("rollout_engine", "vllm"),
    ("group_notes", ["a"]), ("stop_bonus", 0.1), ("abstain_credit", 0.3),
    ("reward_mode", "expansion"), ("prefix_max_new_turns", 1), ("beta", 0.015),
])
def test_retired_dials_are_refused_not_ignored(key, value):
    """A key for a dial the trainer does not have is refused before anything
    loads, and the message names it. `beta` is one of them: the KL weight's
    key is `kl_beta`."""
    from graph_reasoner.train.grpo import validate_config_keys

    with pytest.raises(SystemExit, match=key):
        validate_config_keys({key: value})


def test_grpo_refuses_an_empty_substrate():
    """Without `substrate_adapters` the trainer refuses to start: it would
    otherwise train the raw base model, which has not learned the tool
    protocol."""
    from graph_reasoner.train.grpo import run_grpo

    with pytest.raises(SystemExit, match="substrate_adapters is empty"):
        run_grpo({"lr": 1e-5, "output_dir": "/tmp/x"})


@pytest.mark.parametrize("size", ["0.8b", "2b", "4b"])
def test_every_stage_of_every_size_resolves_and_validates(tmp_path, size):
    """Each stage's hyperparameter file under configs/, merged with the paths
    the recipe derives for a size, passes its trainer's key validation (a
    verifier is an SFT run); round 2 reads round 1's pick from peak.txt."""
    from conftest import ROOT
    from graph_reasoner.recipe import TRAINING, Layout, Spec, training_config
    from graph_reasoner.train.grpo import validate_config_keys
    from graph_reasoner.train.sft import validate_sft_config_keys

    (tmp_path / "configs").mkdir()
    for st in TRAINING.values():
        (tmp_path / st.config).write_text((ROOT / st.config).read_text(encoding="utf-8"))
    lay = Layout.of(Spec("tsmc", size), root=tmp_path)
    (tmp_path / lay.d1_dir / "checkpoint-100").mkdir(parents=True)
    (tmp_path / lay.d1_dir / "peak.txt").write_text("100\n")
    for stage, st in TRAINING.items():
        config, manifest = training_config(lay, stage, argv=[])
        (validate_sft_config_keys if st.trainer == "sft" else validate_config_keys)(config)
        assert config["base_model"] == f"Qwen/Qwen3.5-{size.upper()}"
        assert manifest["config_file"]["path"] == st.config
    d2, _ = training_config(lay, "d2", argv=[])
    assert d2["substrate_adapters"] == [lay.sft2, f"{lay.d1_dir}/checkpoint-100"]

