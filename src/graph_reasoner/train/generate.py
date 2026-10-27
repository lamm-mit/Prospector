"""Conversations -> prompts -> turn text, defined once for every sampling site.

`tokenize.py` owns which tokens carry loss; this module owns what the model is
conditioned on and where a turn ends. Five call sites share its render, stop
configuration and decode split, so a policy is never sampled under one
rendering and scored under another (`tests/test_generate_contract.py`):

    eval/local_model.py    LocalHFModel.generate_chat     transformers
    eval/local_model.py    LocalVLLMModel.generate_chat   vLLM
    eval/local_model.py    LocalMLXModel.generate_chat    mlx-lm
    train/grpo.py          _vllm_generate                 GRPO rollouts, vLLM
    latent/steer.py        SteeredVLLM.generate_chat      vLLM, with the steering hook

No torch, trl or vllm is imported here, at module scope or inside a function.
The two `*_kwargs` builders return plain dicts; `SamplingParams(**kw)` and
`model.generate(**kw)` stay at the call sites.

Three things stay with the call site. It resolves greedy decoding: evaluation
maps `temperature is None` to greedy, and GRPO always samples, because a
greedy group has no variance and gives no gradient. It supplies its engine's
context limit. And it returns one completion per conversation: best-of-n
forks at the episode, as separate sampled runs (the 32-seed pool the recipe's `final` stage draws),
where the draws can call different tools.
"""
from __future__ import annotations

from .tokenize import turn_stop_tokens

__all__ = [
    "TOOL_CALL_STOP",
    "CONTEXT_MARGIN",
    "render_prompts",
    "encode_prompts",
    "stop_strings",
    "stop_id_set",
    "assert_stop_covers_policy",
    "split_stops",
    "cut_at_stop",
    "turn_text",
    "fits_context",
    "length_order",
    "hf_generate_kwargs",
    "vllm_sampling_kwargs",
]

#: The harness parses `<tool_call>...</tool_call>`, so the closing tag ends an
#: agentic turn on every backend. It is a stop string that stays in the text:
#: HF's `stop_strings` leaves it in and vLLM is given
#: `include_stop_str_in_output`, because the turn parser expects it. It is not
#: a default: `stop_strings(tok, None)` adds no caller stop.
TOOL_CALL_STOP = "</tool_call>"

#: Prompt headroom, in tokens, below the engine's context window. vLLM raises
#: on a prompt at the window, so a conversation too long for it gets an empty
#: turn instead; the episode loop forces the final answer after `max_empty`
#: of them.
CONTEXT_MARGIN = 8


def render_prompts(conversations: list[list[dict]], tokenizer) -> list[str]:
    """Conversations -> prompt strings, each opening an assistant turn.
    `encode_prompts` turns them into ids."""
    return [tokenizer.apply_chat_template(c, tokenize=False,
                                          add_generation_prompt=True)
            for c in conversations]


def encode_prompts(texts: list[str], tokenizer) -> list[list[int]]:
    """Rendered prompts -> token ids, without re-adding special tokens.

    The template already emitted them; a tokenizer adding its own would
    prepend a second BOS on the families that have one.
    """
    return [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]


def stop_strings(tokenizer, stop: list[str] | None = None) -> list[str]:
    """Every string that ends a turn: the caller's stops, then the template's.

    The agentic sites pass `[TOOL_CALL_STOP]`. The template's turn-end
    delimiter is read off the tokenizer's installed template and is absent on
    every Qwen path, where the family eos already ends the turn.
    """
    return list(stop or []) + turn_stop_tokens(tokenizer)


def split_stops(tokenizer, stops: list[str]) -> tuple[dict[int, str], list[str]]:
    """The stops an engine can watch as single tokens (id -> string), and the
    rest.

    vLLM and HF take stop strings and need none of this. mlx-lm's batch
    generator takes stop token sequences, so a stop that is not one token
    falls back to a substring check over the decoded text. `</tool_call>` is
    one token on the Qwen3.5 tokenizer.
    """
    single: dict[int, str] = {}
    multi: list[str] = []
    for s in stops:
        enc = tokenizer.encode(s, add_special_tokens=False)
        if len(enc) == 1:
            single[enc[0]] = s
        else:
            multi.append(s)
    return single, multi


def stop_id_set(*sources) -> set[int]:
    """Token ids from any mix of `None`, a bare int, and iterables of ints,
    the shapes in which engines and tokenizers report their stop ids."""
    out: set[int] = set()
    for src in sources:
        if src is None:
            continue
        if isinstance(src, int):
            out.add(src)
        else:
            out.update(int(t) for t in src)
    return out


def assert_stop_covers_policy(engine_stop_ids, policy_eos_id, *, engine: str,
                              source: str, remedy: str) -> None:
    """Raise SystemExit unless the engine's stop ids include the id this
    policy ends its turns with.

    Tool-call turns stop on `</tool_call>` whatever the eos is, so with a
    wrong stop a rollout still runs and scores; only the final answer turn
    decodes to `max_tokens`, without an error. A checkpoint can state
    different eos ids in config.json, generation_config.json and its
    tokenizer, and the engines resolve their stop from different ones.
    `source` and `remedy` go into the message. `policy_eos_id=None` means the
    tokenizer declares no eos, and nothing is checked.
    """
    if policy_eos_id is None:
        return
    stops = stop_id_set(engine_stop_ids)
    if int(policy_eos_id) in stops:
        return
    raise SystemExit(
        f"{engine} will stop on {sorted(stops) or 'nothing'} (from {source}), "
        f"but this policy ends its turns with {int(policy_eos_id)}. Every "
        f"answer turn would decode to max_tokens, and nothing would error. "
        f"{remedy}")


def cut_at_stop(text: str, stops: list[str]) -> str:
    """Everything up to and including the earliest stop present, or all of it.

    The stop is kept because the tool-call parser reads `</tool_call>` as the
    closing tag. The earliest position wins, whatever the order of `stops`.
    """
    hits = [text.index(s) + len(s) for s in stops if s in text]
    return text[: min(hits)] if hits else text


def turn_text(text: str, tokenizer) -> str:
    """A generated string, cut before the template's turn-end delimiter.

    A no-op where `turn_stop_tokens` is empty, which is every current
    template.
    """
    stops = turn_stop_tokens(tokenizer)
    return text.split(stops[0], 1)[0] if stops else text


def fits_context(prompt_ids: list[list[int]], context_limit: int | None) -> list[int]:
    """Indices of the prompts shorter than `context_limit - CONTEXT_MARGIN`
    tokens. `None` keeps every prompt. Callers give the rest an empty turn.
    """
    if context_limit is None:
        return list(range(len(prompt_ids)))
    limit = int(context_limit) - CONTEXT_MARGIN
    return [i for i, ids in enumerate(prompt_ids) if len(ids) < limit]


def length_order(texts: list[str]) -> list[int]:
    """Indices sorted short-to-long, so a padded batch is not mostly padding.

    The sort is stable, and its tie order decides which prompts share a batch
    and therefore the sampling stream.
    """
    return sorted(range(len(texts)), key=lambda i: len(texts[i]))


def hf_generate_kwargs(tokenizer, stop: list[str] | None = None) -> dict:
    """Stop configuration for `model.generate`.

    `stop_strings` needs the tokenizer passed with it. The template's turn-end
    delimiter is added as an extra eos id only when the template has one:
    passing `eos_token_id` otherwise would override the eos ids of the
    model's generation_config.
    """
    kwargs: dict = {}
    if stop:
        kwargs["stop_strings"] = list(stop)
        kwargs["tokenizer"] = tokenizer
    stops = turn_stop_tokens(tokenizer)
    if stops:
        kwargs["eos_token_id"] = [tokenizer.eos_token_id,
                                  tokenizer.convert_tokens_to_ids(stops[0])]
    return kwargs


def vllm_sampling_kwargs(tokenizer, *, max_new_tokens: int,
                         temperature: float, top_p: float,
                         stop: list[str] | None = None) -> dict:
    """`SamplingParams(**...)` arguments matching the HF path's behaviour.

    `temperature` and `top_p` arrive resolved; the caller decides greedy.
    `include_stop_str_in_output` matches HF's `stop_strings`, which leaves the
    stop in the text the turn parser reads. `stop_token_ids` is `None`, not
    `[]`, when the template has no delimiter.
    """
    stops = turn_stop_tokens(tokenizer)
    return {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "stop": list(stop) if stop else None,
        "stop_token_ids": [tokenizer.convert_tokens_to_ids(s)
                           for s in stops] or None,
        "include_stop_str_in_output": True,
    }
