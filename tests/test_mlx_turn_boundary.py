"""Where the MLX driver ends an assistant turn, and what its prefix cache holds.

vLLM and HF are given stop strings and return a finished turn. MLX takes stop
tokens, so the driver handles a stop that is not one token and decides what
stays in the turn: the stop string is kept, because the tool-call parser needs
the closing `</tool_call>`, the eos is dropped, and decoding ends at the stop.

The driver decodes the whole batch through mlx-lm's `BatchGenerator` and keeps
one prefix-cache entry per sequence of the last batch.

`_many` takes its generator class as an argument, so these tests need no mlx
and no weights.
"""
from types import SimpleNamespace

import pytest

from graph_reasoner.eval.local_model import LocalMLXModel
from graph_reasoner.train.generate import cut_at_stop, split_stops, turn_text

STOP = "</tool_call>"
EOS, TOOL, IMEND = 0, 90, 91
_SPECIAL = {STOP: TOOL, "<|im_end|>": IMEND, "<eos>": EOS}
_BACK = {v: k for k, v in _SPECIAL.items()}


class _Tok:
    """The specials are one token each and everything else is one token per
    character, so the single-token and the substring stop paths are both
    reachable."""
    chat_template = None
    eos_token_id = EOS

    def encode(self, s, add_special_tokens=False):
        return [_SPECIAL[s]] if s in _SPECIAL else [1000 + ord(c) for c in s]

    def decode(self, ids):
        return "".join(_BACK[i] if i in _BACK else chr(i - 1000) for i in ids)


def _toks(s):
    return _Tok().encode(s) if s in _SPECIAL else [1000 + ord(c) for c in s]


def _script(*parts):
    """A token stream from a mix of plain strings and specials."""
    out = []
    for p in parts:
        out.extend(_toks(p))
    return out


def _stub():
    """A real instance with `__init__` skipped, so `_many` runs the real
    `_reuse` and `_hit_text_stop`. Only the attributes they read are set."""
    obj = LocalMLXModel.__new__(LocalMLXModel)
    obj.model = object()
    obj._caches = []
    obj.tokenizer = _Tok()
    obj._mlx_tok = SimpleNamespace(eos_token_ids={EOS})
    obj.prefill_batch_size = 2
    obj.prefix_cache = True
    return obj


class _FakeGen:
    """A BatchGenerator that replays a canned token stream per sequence.
    `fed` records the tokens each sequence was given and `emitted` every token
    produced; a sequence emits nothing after its stop token."""
    made = []

    def __init__(self, model, sampler=None, stop_tokens=None,
                 completion_batch_size=None, prefill_batch_size=None):
        self.stop = {t[0] for t in (stop_tokens or [])}
        self.completion_batch_size = completion_batch_size
        self.prefill_batch_size = prefill_batch_size
        self.scripts, self.fed, self.caches = [], [], []
        self.pos, self.live, self.emitted, self.cap = {}, set(), [], {}
        _FakeGen.made.append(self)

    def insert(self, prompts, max_tokens=None, caches=None):
        uids = list(range(len(prompts)))
        for u, p in zip(uids, prompts):
            self.fed.append(list(p))
            self.caches.append((caches or [None] * len(prompts))[u])
            self.pos[u] = 0
            self.cap[u] = (max_tokens or [None] * len(prompts))[u]
            self.live.add(u)
        return uids

    def next_generated(self):
        if not self.live:
            return []
        out = []
        for u in sorted(self.live):
            script = self.scripts[u]
            if self.pos[u] >= len(script):
                self.live.discard(u)
                out.append(SimpleNamespace(
                    uid=u, token=EOS, finish_reason="length", match_sequence=None,
                    prompt_cache=("cache", u), all_tokens=list(script)))
                continue
            t = script[self.pos[u]]
            self.pos[u] += 1
            self.emitted.append((u, t))
            if t in self.stop:
                self.live.discard(u)
                out.append(SimpleNamespace(
                    uid=u, token=t, finish_reason="stop", match_sequence=[t],
                    prompt_cache=("cache", u),
                    all_tokens=self.fed[u] + script[: self.pos[u]]))
            elif self.cap[u] is not None and self.pos[u] >= self.cap[u]:
                # mlx-lm delivers the token that hits max_tokens with
                # finish_reason "length": generated text, already in the cache.
                self.live.discard(u)
                out.append(SimpleNamespace(
                    uid=u, token=t, finish_reason="length", match_sequence=None,
                    prompt_cache=("cache", u),
                    all_tokens=self.fed[u] + script[: self.pos[u]]))
            else:
                out.append(SimpleNamespace(
                    uid=u, token=t, finish_reason=None, match_sequence=None,
                    prompt_cache=None, all_tokens=None))
        return out

    def remove(self, uids, return_prompt_caches=False):
        got = {}
        for u in uids:
            self.live.discard(u)
            got[u] = (("cache", u), self.fed[u] + self.scripts[u][: self.pos[u]])
        return got if return_prompt_caches else None

    def close(self):
        pass


def _run(scripts, prompts=None, stops=(STOP,), obj=None, max_tokens=64):
    prompts = prompts if prompts is not None else [[1, 2, 3]] * len(scripts)
    _FakeGen.made = []

    def factory(model, **kw):
        g = _FakeGen(model, **kw)
        g.scripts = [list(s) for s in scripts]
        return g

    out = LocalMLXModel._many(obj or _stub(), prompts, max_tokens, None, list(stops),
                              factory, lambda m: object(), turn_text,
                              split_stops, cut_at_stop)
    return out, _FakeGen.made[-1]


# ------------------------------------------------------------- the stop rule

def test_the_stop_string_stays_in_the_turn():
    """The stop string stays in the turn, as vLLM's
    `include_stop_str_in_output=True` keeps it."""
    out, _ = _run([_script('{"name": "find", "arguments": {}}', STOP)])
    assert out == ['{"name": "find", "arguments": {}}' + STOP]


def test_generation_stops_reading_at_the_stop():
    """Decoding ends at the stop. Draining the generator and truncating
    afterwards would return the same string and take longer."""
    out, gen = _run([_script("a", STOP, "wasted" * 20)])
    assert out == ["a" + STOP]
    assert len(gen.emitted) == 2, f"read past the stop: {gen.emitted[:5]}"


def test_eos_ends_the_turn_without_appearing_in_it():
    """eos delimits the turn, it is not part of it. The stop string is."""
    out, _ = _run([_script("done", "<eos>")])
    assert out == ["done"]


def test_a_multi_token_stop_is_still_found():
    """A stop the engine cannot watch for, because it is not one token. The
    driver falls back to a substring check over the decoded tail."""
    out, gen = _run([_script("call", "HALT", "junk")], stops=("HALT",))
    assert out == ["callHALT"]
    assert gen.stop == {EOS}, "a multi-token stop must not be handed to the engine"


def test_nothing_after_a_multi_token_stop_survives():
    """The token that completes a multi-token stop can carry text past it, so
    the decoded text is cut at the stop as well."""
    out, _ = _run([_script("x", "HALTtail")], stops=("HALT",))
    assert out == ["xHALT"]


def test_no_stop_means_the_whole_completion():
    out, _ = _run([_script("The answer is BRCA1.")])
    assert out == ["The answer is BRCA1."]


def test_the_token_that_hits_the_cap_is_kept():
    """A turn cut at max_new_tokens keeps its last token, as the other two
    drivers return max_new_tokens tokens: mlx-lm delivers that token with
    finish_reason "length", and the cache has already consumed it."""
    out, _ = _run([_script("abcdef")], max_tokens=4)
    assert out == ["abcd"]


@pytest.mark.parametrize("stops", [(), (STOP, "<|im_end|>")])
def test_the_caller_owns_the_stop_set(stops):
    """No stop is defaulted in when the caller passes none, and every stop
    the caller passes is honoured."""
    out, _ = _run([_script("x", "<|im_end|>", "y")], stops=stops)
    assert out == (["x<|im_end|>y"] if not stops else ["x<|im_end|>"])


# ------------------------------------------------------------- the batching

def test_the_whole_batch_is_one_decode_group():
    """The completion batch is the caller's whole batch; a smaller one would
    decode the draws in series."""
    _, gen = _run([_script("a", STOP)] * 8, prompts=[[1, 2, i] for i in range(8)])
    assert gen.completion_batch_size == 8


# ---------------------------------------------------------------- the cache

def test_a_growing_transcript_feeds_only_the_new_tokens():
    """Turn n+1's prompt is turn n's prompt, the turn just generated and the
    observation, so the second call feeds the observation only."""
    obj = _stub()
    first = [1, 2, 3]
    _, g1 = _run([_script("ok", STOP)], prompts=[first], obj=obj)
    consumed = g1.fed[0] + g1.scripts[0][: g1.pos[0]]
    second = list(consumed) + [7, 8]
    _, g2 = _run([_script("ok", STOP)], prompts=[second], obj=obj)
    assert g2.fed == [[7, 8]]


def test_every_sequence_in_the_batch_keeps_its_own_cache():
    """Two episodes advance together, and on the next turn each hits its own
    prefix, not only the one that finished last."""
    obj = _stub()
    a, b = [1, 2, 3], [4, 5, 6]
    _, g1 = _run([_script("ok", STOP)] * 2, prompts=[a, b], obj=obj)
    grown = [list(g1.fed[i] + g1.scripts[i][: g1.pos[i]]) + [70 + i]
             for i in range(2)]
    _, g2 = _run([_script("ok", STOP)] * 2, prompts=grown, obj=obj)
    assert g2.fed == [[70], [71]], f"re-prefilled: {[len(f) for f in g2.fed]}"


def test_a_diverging_prompt_rebuilds_rather_than_reusing():
    """A truncated turn or a new episode breaks the prefix, and a cache reused
    across a break would condition the answer on another episode's transcript."""
    obj = _stub()
    _, _ = _run([_script("ok", STOP)], prompts=[[1, 2, 3]], obj=obj)
    _, g2 = _run([_script("ok", STOP)], prompts=[[1, 2, 99, 4]], obj=obj)
    assert g2.fed == [[1, 2, 99, 4]]


def test_an_unextended_prompt_rebuilds():
    """Equal-length or shorter leaves nothing to feed, and a sequence cannot be
    inserted with a zero-token prompt."""
    obj = _stub()
    _, _ = _run([_script("ok", STOP)], prompts=[[1, 2, 3]], obj=obj)
    _, g2 = _run([_script("ok", STOP)], prompts=[[1, 2, 3]], obj=obj)
    assert g2.fed == [[1, 2, 3]]


def test_the_store_holds_one_batch_and_not_a_history():
    """Entries are replaced wholesale on every call, so the caches of
    finished episodes are dropped instead of accumulating."""
    obj = _stub()
    _run([_script("ok", STOP)] * 4, prompts=[[1, 2, i] for i in range(4)], obj=obj)
    assert len(obj._caches) == 4
    _run([_script("ok", STOP)], prompts=[[9, 9, 9]], obj=obj)
    assert len(obj._caches) == 1


def test_prefill_width_is_separate_from_decode_width():
    """Prefill is the memory peak, so its batch size is set separately from
    the decode batch."""
    obj = _stub()
    obj.prefill_batch_size = 2
    _, gen = _run([_script("a", STOP)] * 8, prompts=[[1, 2, i] for i in range(8)],
                  obj=obj)
    assert gen.completion_batch_size == 8 and gen.prefill_batch_size == 2


def test_reuse_can_be_switched_off_to_buy_memory():
    """Reuse holds this turn's caches while the generator builds the next
    turn's. With it off every sequence re-prefills and nothing is retained
    between calls."""
    obj = _stub()
    obj.prefix_cache = False
    _, g1 = _run([_script("ok", STOP)], prompts=[[1, 2, 3]], obj=obj)
    grown = list(g1.fed[0] + g1.scripts[0][: g1.pos[0]]) + [7]
    _, g2 = _run([_script("ok", STOP)], prompts=[grown], obj=obj)
    assert g2.fed == [grown], "re-used a prefix with the cache switched off"
    assert obj._caches == []
