"""`llm/client.py`: one item's soft failure does not abort a pass.

A truncated or refused call returns an empty answer and is counted. An
unknown bad request, or a transient error that outlasts the retries,
propagates. Also covered: the token tallies, the cache key, and the key and
endpoint a model id is routed to.
"""
import asyncio

import pytest

from graph_reasoner.llm.client import LLMClient


def _client(tmp_path):
    return LLMClient(model="test-model", api_key="test-key",
                     cache_dir=tmp_path / "cache")


def _patch_create(client, exc):
    async def raiser(kwargs):
        raise exc
    client._create = raiser


def test_content_filter_refusal_returns_empty_and_counts(tmp_path):
    """A content-filter refusal (400 invalid_prompt) returns an empty answer
    and is counted instead of raising. TypeError shares the except tuple with
    BadRequestError, so the test needs no httpx response."""
    c = _client(tmp_path)
    _patch_create(c, TypeError(
        "Error code: 400 - invalid_prompt: flagged as potentially violating "
        "our usage policy"))
    out = asyncio.run(c.chat([{"role": "user", "content": "q"}], cache=False))
    assert out == ""
    assert c.usage["refused"] == 1


def test_unrecognized_bad_request_still_raises(tmp_path):
    """Only the named recoverable classes are swallowed; an unknown 400
    propagates instead of being scored as an empty answer."""
    c = _client(tmp_path)
    _patch_create(c, TypeError("Error code: 400 - some_other_error"))
    with pytest.raises(TypeError, match="some_other_error"):
        asyncio.run(c.chat([{"role": "user", "content": "q"}], cache=False))


class _Usage:
    def __init__(self, prompt, completion, reasoning, cached=0):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.completion_tokens_details = type("D", (), {"reasoning_tokens": reasoning})()
        self.prompt_tokens_details = type("P", (), {"cached_tokens": cached})()


class _Resp:
    def __init__(self, text, usage):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]
        self.usage = usage


def test_reasoning_and_cached_tokens_are_tallied(tmp_path):
    """Reasoning tokens are billed as completion tokens that never appear in
    the text, and a provider's prefix cache bills prompt tokens at a discount;
    both are tallied so a run's cost can be computed."""
    c = _client(tmp_path)
    async def fake_create(kwargs):
        return _Resp("OK", _Usage(13, 4, 3, cached=8))
    c._create = fake_create
    out = asyncio.run(c.chat([{"role": "user", "content": "q"}], cache=False))
    assert out == "OK"
    assert c.usage["reasoning_tokens"] == 3
    assert c.usage["completion_tokens"] == 4
    assert c.usage["cached_prompt_tokens"] == 8
    assert "3 of them reasoning" in c.usage_summary()
    assert "8 prompt tok served from the provider cache" in c.usage_summary()


def test_the_cache_key_is_the_stored_payload(tmp_path):
    """A text call is cached under the hash of exactly this payload; a new
    key in it would orphan every stored call."""
    c = _client(tmp_path)
    async def fake_create(kwargs):
        return _Resp("OK", None)
    c._create = fake_create
    asyncio.run(c.chat([{"role": "user", "content": "q"}]))
    path = c._cache_path({"model": "test-model", "messages": [{"role": "user", "content": "q"}],
                          "json_mode": False, "temperature": None, "max_tokens": 2048})
    assert path.exists()


def test_a_timeout_after_the_retries_propagates(tmp_path):
    """A transient error that survives the retry budget fails the call: only
    the named soft failures (truncated, refused) return an empty answer."""
    from openai import APITimeoutError
    c = _client(tmp_path)
    async def fake_create(kwargs):
        raise APITimeoutError(request=None)
    c._create = fake_create
    with pytest.raises(APITimeoutError):
        asyncio.run(c.chat([{"role": "user", "content": "q"}], cache=False))


def test_a_model_id_routes_to_its_provider(monkeypatch):
    """A vendor-prefixed id goes to OpenRouter on its key, a bare id to OpenAI
    on its own, and an exported `OPENAI_BASE_URL` wins."""
    from graph_reasoner.llm.providers import OPENROUTER_BASE_URL, route
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert route("google/gemini-3.7-flash") == ("OPENROUTER_API_KEY", OPENROUTER_BASE_URL)
    assert route("gpt-6-astra") == ("OPENAI_API_KEY", None)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    assert route("google/gemini-3.7-flash") == ("OPENAI_API_KEY", "http://localhost:8000/v1")


def test_an_openai_model_is_refused_on_openrouter(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    for model in ("gpt-5-mini", "o3", "openai/gpt-6-astra"):
        with pytest.raises(RuntimeError, match="OpenAI model"):
            LLMClient(model=model, api_key="test-key", cache_dir=tmp_path / "cache")
    LLMClient(model="google/gemini-3.7-flash", api_key="test-key", cache_dir=tmp_path / "cache")


def test_the_provider_pin_reaches_openrouter_only(tmp_path, monkeypatch):
    """`OPENROUTER_PROVIDER` is OpenRouter's field: a call routed to OpenAI's
    endpoint carries no `extra_body`, a call routed to OpenRouter does."""
    monkeypatch.setenv("OPENROUTER_PROVIDER", "SomeHost")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    seen = {}

    async def fake_create(kwargs):
        seen.clear()
        seen.update(kwargs)
        return _Resp("OK", None)

    c = _client(tmp_path)                   # a bare id: OpenAI's own endpoint
    c._create = fake_create
    asyncio.run(c.chat([{"role": "user", "content": "q"}], cache=False))
    assert "extra_body" not in seen
    r = LLMClient(model="vendor/model", api_key="test-key", cache_dir=tmp_path / "c2")
    r._create = fake_create
    asyncio.run(r.chat([{"role": "user", "content": "q"}], cache=False))
    assert seen["extra_body"] == {"provider": {"order": ["SomeHost"], "allow_fallbacks": False}}


def test_the_native_channel_retries_transient_errors(tmp_path, monkeypatch):
    """`chat_responses` carries the same retry policy as `chat`: a timeout
    that clears within the attempts does not fail the call."""
    from types import SimpleNamespace

    from openai import APITimeoutError
    from tenacity import wait_none

    monkeypatch.setattr(LLMClient._create_responses.retry, "wait", wait_none())
    c = _client(tmp_path)
    attempts = []

    async def create(**kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise APITimeoutError(request=None)
        return SimpleNamespace(output=[], output_text="fine", id="r1",
                               status="completed", usage=None)

    c._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    out = asyncio.run(c.chat_responses(
        instructions=None, input_items=[{"role": "user", "content": "q"}],
        previous_response_id=None, tools=[], cache=False))
    assert out["content"] == "fine" and len(attempts) == 3


def test_a_cache_file_cut_short_is_a_miss_and_is_replaced(tmp_path):
    """A write interrupted mid-file leaves JSON that does not parse; the next
    call treats it as a miss, asks the model, and stores the whole reply."""
    import json
    c = _client(tmp_path)
    calls = []
    async def fake_create(kwargs):
        calls.append(1)
        return _Resp("OK", None)
    c._create = fake_create
    path = c._cache_path({"model": "test-model", "messages": [{"role": "user", "content": "q"}],
                          "json_mode": False, "temperature": None, "max_tokens": 2048})
    path.write_text('{"text": "O')
    assert asyncio.run(c.chat([{"role": "user", "content": "q"}])) == "OK"
    assert calls == [1] and json.loads(path.read_text()) == {"text": "OK"}
    assert asyncio.run(c.chat([{"role": "user", "content": "q"}])) == "OK" and calls == [1]


def test_the_provider_pin_and_another_endpoint_enter_the_cache_key_only_when_set(tmp_path, monkeypatch):
    """A pinned OpenRouter host or an explicit endpoint can change a reply, so
    each is part of the key when set; without them the stored keys stand."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_PROVIDER", raising=False)
    c = LLMClient(model="vendor/test", api_key="k", cache_dir=tmp_path / "a")       # routed to OpenRouter
    assert c._key_extras({"m": 1}, pin=True) == {"m": 1}
    monkeypatch.setenv("OPENROUTER_PROVIDER", "host-a")
    assert c._key_extras({"m": 1}, pin=True) == {"m": 1, "provider": "host-a"}
    assert c._key_extras({"m": 1}, pin=False) == {"m": 1}                            # the native channel has no pin
    d = LLMClient(model="test-model", api_key="k", base_url="http://localhost:1/v1", cache_dir=tmp_path / "b")
    assert d._key_extras({"m": 1}, pin=True) == {"m": 1, "base_url": "http://localhost:1/v1"}
