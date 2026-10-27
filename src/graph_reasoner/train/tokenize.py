"""Transcript -> (tokens, per-token loss weights), defined once.

The SFT trainer, the GRPO trainer and the verifier scorer read the mask
computed here, so they agree on which tokens carry loss. Observation tokens
never do: a model trained on them learns to emit its own tool results.

No torch or trl is imported here, at module scope or inside a function. The
tokenizer arrives as an argument and is only called.

The two masks differ by the 3-token generation header
`<|im_start|>assistant\\n` of every assistant turn after the first, and
tests/test_trainer_arithmetic.py pins the difference:

  `token_spans`        GRPO. An assistant segment includes its own header.
  `supervised_arrays`  SFT. The header is masked.

`supervised_arrays` shifts its inputs against its targets by one token:

    input_tokens  = ids[:-1]
    target_tokens = ids[1:]
    weights       = w[1:]       aligned to the targets

so `weights[i]` gates the loss on predicting `target_tokens[i]`.
"""
from __future__ import annotations

__all__ = [
    "boundaries_to_mask",
    "token_spans",
    "supervised_arrays",
    "completion_logprob_span",
    "override_chat_template",
    "CHAT_TEMPLATES",
    "PLAIN_CHATML",
    "TURN_STOP",
    "turn_stop_tokens",
    "NOT_CONCATENATIVE",
    "template_name",
]

#: Some chat templates rewrite earlier turns when a later one is appended:
#: Qwen3.5's shipped template strips `<think>` from every assistant message
#: but the last. render(msgs[:k]) is then not a prefix of render(msgs), and no
#: per-token mask is computable.
NOT_CONCATENATIVE = (
    "chat template is not concatenative: rendering message {k} changed an "
    "earlier turn, so no per-token mask is computable. Install a "
    "concatenative template with `chat_template: plain_chatml`, as every "
    "training config does"
)


# ------------------------------------------------------------ template override

#: Plain ChatML, the concatenative template every training config installs,
#: and so the rendering the trained adapters are served under.
#: tests/test_chat_template_override.py holds its rendering as a frozen
#: string. It leaves `<think>` blocks in earlier turns, which Qwen3.5's own
#: template strips.
PLAIN_CHATML = (
    "{% for m in messages %}"
    "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

#: Named templates a config may select. A config gives a name, never a raw
#: Jinja string, so every rendering a run can use is defined here and tested.
CHAT_TEMPLATES = {"plain_chatml": PLAIN_CHATML}

#: What override_chat_template checks per template: the family name its
#: refusal prints, and the delimiters that must each be one token on the
#: target vocabulary. Add an entry when adding a template.
TEMPLATE_NATIVE = {
    "plain_chatml": ("ChatML", ("<|im_start|>", "<|im_end|>")),
}

#: On a foreign vocabulary a template's delimiters split into several tokens.
#: The rendering and the concatenative check still pass, and the model would
#: train against delimiters it has never seen, so the count is asserted.
NOT_DELIMITER_NATIVE = (
    "{model} is not {family}-native: {tok!r} tokenizes to {n} tokens, not 1. "
    "The {name} template would train it against delimiters outside its "
    "vocabulary. Write a family-appropriate template and add it to "
    "CHAT_TEMPLATES with a test, rather than reusing this one."
)


def override_chat_template(tokenizer, name: str | None, *, model: str = "?"):
    """Install a named concatenative chat template. Returns the tokenizer.

    Every training config names `plain_chatml`. `name=None` is a no-op that
    keeps the tokenizer's current template.
    """
    if name is None:
        return tokenizer
    if name not in CHAT_TEMPLATES:
        raise SystemExit(
            f"unknown chat_template {name!r}; known: "
            f"{', '.join(sorted(CHAT_TEMPLATES))}")
    family, delimiters = TEMPLATE_NATIVE[name]
    for tok_str in delimiters:
        n = len(tokenizer(tok_str, add_special_tokens=False)["input_ids"])
        if n != 1:
            raise SystemExit(NOT_DELIMITER_NATIVE.format(
                model=model, family=family, name=name, tok=tok_str, n=n))
    tokenizer.chat_template = CHAT_TEMPLATES[name]
    return tokenizer


#: Per-template turn-end token, for a template whose turn end the tokenizer's
#: eos does not cover; without an entry, such a template's final answer turns
#: run to max_tokens. Empty: plain_chatml ends a turn at `<|im_end|>`, the
#: Qwen tokenizer's eos, and each local engine checks at load that it stops
#: there (`generate.assert_stop_covers_policy`).
TURN_STOP: dict[str, str] = {}


def turn_stop_tokens(tokenizer) -> list[str]:
    """Turn-end stop strings for the tokenizer's installed template, read back
    through `template_name` rather than taken from a config. [] where the
    family eos already ends the turn, which is every current template.
    Callers convert the strings to ids for token-level stops and strip them
    from decoded text."""
    name = template_name(getattr(tokenizer, "chat_template", None))
    return [TURN_STOP[name]] if name in TURN_STOP else []


def template_name(template: str | None) -> str | None:
    """The configured name for a template string, else None.

    A run record uses it to state the template a saved tokenizer carries.
    """
    for name, tpl in CHAT_TEMPLATES.items():
        if template == tpl:
            return name
    return None



# ---------------------------------------------------------------- pure mapper

def boundaries_to_mask(offsets: list[tuple[int, int]], prompt_char_end: int,
                       segments: list[tuple[int, str]],
                       ) -> tuple[int, list[int], list[int]]:
    """(token offsets, prompt char end, [(segment char end, role)])
    -> (prompt token count, env mask, per-token assistant-turn index).

    Pure, so it is testable without a tokenizer. Every char boundary must
    fall on a token boundary, else ValueError. `turn_idx` is the 0-based
    index of the assistant turn a completion token belongs to and -1 for
    environment tokens, so `turn_idx[i] >= 0` iff `mask[i] == 1`.
    """
    starts = {s: i for i, (s, _e) in enumerate(offsets)}

    def token_at(char_pos: int) -> int:
        if char_pos == (offsets[-1][1] if offsets else 0):
            return len(offsets)
        if char_pos not in starts:
            raise ValueError(f"char boundary {char_pos} is inside a token — "
                             f"template/tokenizer drift, refusing to guess")
        return starts[char_pos]

    prompt_len = token_at(prompt_char_end)
    mask: list[int] = []
    turn_idx: list[int] = []
    prev_tok = prompt_len
    turn = 0
    for char_end, role in segments:
        tok_end = token_at(char_end)
        n = tok_end - prev_tok
        if role == "assistant":
            mask.extend([1] * n)
            turn_idx.extend([turn] * n)
            turn += 1
        else:
            mask.extend([0] * n)
            turn_idx.extend([-1] * n)
        prev_tok = tok_end
    return prompt_len, mask, turn_idx


# ---------------------------------------------------------------- GRPO spans

def token_spans(messages: list[dict], tokenizer):
    """(prompt_ids, completion_ids, env_mask, turn_idx) for one episode transcript.

    Used by the GRPO trainer. The prompt is the first two messages and the
    generation header. The chat template must be concatenative, each
    render(msgs[:k]) a string prefix of render(msgs[:k+1]); otherwise
    ValueError.

    `turn_idx` is the per-token assistant-turn index (-1 for environment
    tokens); see `boundaries_to_mask`. `prefixes.zero_prefix_turns` uses it to
    mask a resumed episode's prefix turns out of the loss.
    """
    n = len(messages)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_text = tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True)
    if not full_text.startswith(prompt_text):
        raise ValueError(NOT_CONCATENATIVE.format(k=2))
    segments: list[tuple[int, str]] = []
    prev = prompt_text
    for k in range(3, n + 1):
        cur = tokenizer.apply_chat_template(messages[:k], tokenize=False)
        if not cur.startswith(prev):
            raise ValueError(NOT_CONCATENATIVE.format(k=k))
        segments.append((len(cur), messages[k - 1]["role"]))
        prev = cur
    if segments[-1][0] != len(full_text):
        raise ValueError("final segment does not close the transcript")

    enc = tokenizer(full_text, add_special_tokens=False,
                    return_offsets_mapping=True)
    prompt_len, mask, turn_idx = boundaries_to_mask(
        enc["offset_mapping"], len(prompt_text), segments)
    ids = enc["input_ids"]
    return ids[:prompt_len], ids[prompt_len:], mask, turn_idx


# ---------------------------------------------------------------- SFT arrays

def supervised_arrays(messages: list[dict], tokenizer,
                      max_length: int | None = None):
    """One transcript -> (input_tokens, target_tokens, weights), shifted by one
    token as the module docstring shows.

    Returns None when the transcript exceeds `max_length` or carries no
    assistant tokens. An over-length transcript is dropped, never truncated:
    the answer comes last in a trajectory, so a right-truncated row would
    teach searching without answering.

    Weight 1 covers each assistant message's content and its turn end; the
    generation header `<|im_start|>assistant\\n` is masked.
    """
    render = tokenizer.apply_chat_template
    full_text = render(messages, tokenize=False)
    enc = tokenizer(full_text, add_special_tokens=False,
                    return_offsets_mapping=True)
    ids = enc["input_ids"]
    if max_length is not None and len(ids) > max_length:
        return None

    starts = {s: i for i, (s, _e) in enumerate(enc["offset_mapping"])}

    def token_at(char_pos: int) -> int:
        if char_pos == len(full_text):
            return len(ids)
        if char_pos not in starts:
            raise ValueError(f"char boundary {char_pos} is inside a token — "
                             f"template/tokenizer drift, refusing to guess")
        return starts[char_pos]

    weights = [0] * len(ids)
    for k in range(1, len(messages) + 1):
        if messages[k - 1].get("role") != "assistant":
            continue
        head = render(messages[:k - 1], tokenize=False,
                      add_generation_prompt=True)
        body = render(messages[:k], tokenize=False)
        if not body.startswith(head) or not full_text.startswith(body):
            raise ValueError(NOT_CONCATENATIVE.format(k=k))
        for i in range(token_at(len(head)), token_at(len(body))):
            weights[i] = 1

    if not any(weights):
        return None
    # The fencepost: inputs lose the last token, targets and weights the first.
    return ids[:-1], ids[1:], weights[1:]


def completion_logprob_span(messages: list[dict], tokenizer
                            ) -> tuple[list[int], list[int]]:
    """(full_ids, positions) for scoring log P(final assistant msg | prefix).

    For the yes/no verifier (`select/verifier.py`). `full_ids` is the
    transcript tokenized as `supervised_arrays` tokenizes it for SFT, and
    `positions` index a scorer's output (lp[j] = log P(ids[j] | ids[:j]),
    None at 0) at the tokens SFT put weight on, so summing lp over
    `positions` scores the trained span. Tokenizing the prompt and the reply
    separately could yield other ids, because BPE can merge across the
    generation-header seam.
    """
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("last message must be the assistant turn to score")
    if sum(m.get("role") == "assistant" for m in messages) != 1:
        raise ValueError("exactly one assistant message — a multi-turn "
                         "transcript would sum logprobs over earlier turns")
    got = supervised_arrays(messages, tokenizer)  # no max_length: never drop
    if got is None:
        raise ValueError("no weighted assistant tokens to score")
    inp, tgt, w = got
    # Undo the shift: ids == inp + [tgt[-1]], and weights[i] gates ids[i+1],
    # which the scorer returns at index i+1. Index 0 is never a position.
    full = list(inp) + [tgt[-1]]
    return full, [i + 1 for i, wi in enumerate(w) if wi]

