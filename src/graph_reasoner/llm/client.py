"""Async OpenAI-compatible client with disk caching, retries, and usage tally.

The cache is keyed on (model, messages, params) and lives under
`<cache_dir>/<model>/<sha>.json`. `cache_dir` defaults to the relative
`data/cache`, so a process started outside the repo root starts an empty
cache and pays for every call again.

`providers.route` picks the key and the endpoint from the model id, and an
OpenAI model aimed at OpenRouter is refused. Keys come from the environment
or `.env`, never from the command line.

`chat` and `chat_responses` retry rate limits, timeouts, connection and 5xx
errors, and `chat` a 200 response with no `choices` (`EmptyChoices`). In
`chat`, a request the provider rejects for its output limit or its content
policy is counted (`truncated`, `refused`), returns "" instead of raising and
is not cached, so a later run retries it. `chat_responses` counts an
`incomplete` reply as truncated and caches it like any other reply; a
rejection raises.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ..files import write_text_atomic
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tqdm.asyncio import tqdm_asyncio

from .providers import refuse_openai_through_openrouter, route


class EmptyChoices(Exception):
    """A 200 response carrying no `choices`, which the SDK does not type as
    an error.

    OpenRouter returns upstream rate limits and provider faults this way, with
    the cause in a top-level `error` object. Raising it puts the call on the
    retry path and keeps the provider's message.
    """


_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError,
              InternalServerError, EmptyChoices)


async def gather_with_progress(coros, desc: str = ""):
    return await tqdm_asyncio.gather(*coros, desc=desc)


class LLMClient:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: str | Path = "data/cache",
        concurrency: int = 8,
        timeout: float = 180.0,
        cache: bool = True,
    ):
        if api_key is None:
            # Only when no key was handed in: a caller (a test) that passes its
            # own key must not pull the real ones from .env into the process.
            load_dotenv()
        key_env, routed_base = route(model)
        key = api_key or os.getenv(key_env)
        if not key:
            raise RuntimeError(
                f"No API key for {model!r}: set {key_env} in .env or pass api_key explicitly."
            )
        base = base_url or routed_base
        refuse_openai_through_openrouter(model, base)
        self.model = model
        self._base = base
        self._routed_base = routed_base
        self._client = AsyncOpenAI(api_key=key, base_url=base, timeout=timeout)
        self._sem = asyncio.Semaphore(concurrency)
        safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model)
        self._cache = Path(cache_dir) / safe_model
        self._cache.mkdir(parents=True, exist_ok=True)
        # A run-wide off switch on top of the per-call one. A latency probe
        # needs it: a cached replay would be timed as a model call.
        self._cache_enabled = bool(cache)
        self.usage = {"calls": 0, "cached": 0, "truncated": 0, "refused": 0,
                      "prompt_tokens": 0, "completion_tokens": 0,
                      # Billed as completion tokens but absent from the text.
                      "reasoning_tokens": 0,
                      # Prompt tokens the provider served from its prefix
                      # cache, billed at a discount.
                      "cached_prompt_tokens": 0}

    # ---- caching ----

    def _cache_path(self, payload: dict) -> Path:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return self._cache / (hashlib.sha256(blob).hexdigest() + ".json")

    def _key_extras(self, payload: dict, *, pin: bool) -> dict:
        """The payload with what else decides a reply: an endpoint other than
        the one the model id routes to, and, for a text call through
        OpenRouter, the provider pin. Each enters the key only when set, so
        the keys of the calls stored without them stand."""
        payload = dict(payload)
        if self._base != self._routed_base:
            payload["base_url"] = self._base
        provider = os.getenv("OPENROUTER_PROVIDER") if pin else None
        if provider and self._base and "openrouter" in self._base.lower():
            payload["provider"] = provider
        return payload

    @staticmethod
    def _cached(cpath: Path) -> dict | None:
        """The stored reply, or None when there is none or the file is not a
        whole JSON object (a write cut short): that is a miss, and the reply
        overwrites it."""
        try:
            out = json.loads(cpath.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return out if isinstance(out, dict) else None

    @staticmethod
    def _store(cpath: Path, obj: dict) -> None:
        write_text_atomic(cpath, json.dumps(obj, ensure_ascii=False))

    # ---- core call ----

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential_jitter(multiplier=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    async def _create(self, kwargs: dict) -> Any:
        resp = await self._client.chat.completions.create(**kwargs)
        if not getattr(resp, "choices", None):
            err = (getattr(resp, "model_extra", None) or {}).get("error") or {}
            raise EmptyChoices(
                f"{self.model}: response carried no choices "
                f"({err.get('message') or err or 'no error field'})")
        return resp

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential_jitter(multiplier=2, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    async def _create_responses(self, kwargs: dict) -> Any:
        return await self._client.responses.create(**kwargs)

    async def chat(
        self,
        messages: list[dict],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int = 2048,
        cache: bool = True,
    ) -> str:
        """One completion for `messages`, replayed from the response cache when
        the same payload was answered before and `cache` is on."""
        payload = {
            "model": self.model,
            "messages": messages,
            "json_mode": json_mode,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        cpath = self._cache_path(self._key_extras(payload, pin=True))
        hit = self._cached(cpath) if cache and self._cache_enabled else None
        if hit is not None and "text" in hit:
            self.usage["cached"] += 1
            return hit["text"]

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if temperature is not None:
            kwargs["temperature"] = temperature
        provider = os.getenv("OPENROUTER_PROVIDER")
        if provider and self._base and "openrouter" in self._base.lower():
            # OpenRouter's provider pin, sent to OpenRouter only: a
            # comma-separated host order with no fallback, since a model can
            # behave differently by host. Part of the cache key when set.
            kwargs["extra_body"] = {"provider": {"order": [p.strip() for p in provider.split(",")],
                                                 "allow_fallbacks": False}}

        async with self._sem:
            try:
                resp = await self._create(kwargs)
            except (BadRequestError, TypeError) as e:
                msg = str(e)
                # Older/self-hosted servers want max_tokens; some models reject temperature.
                if "max_completion_tokens" in msg:
                    kwargs.pop("max_completion_tokens", None)
                    kwargs["max_tokens"] = max_tokens
                    resp = await self._create(kwargs)
                elif "temperature" in msg and "temperature" in kwargs:
                    kwargs.pop("temperature")
                    resp = await self._create(kwargs)
                elif "max_tokens" in msg or "output limit" in msg:
                    # A reasoning model can exhaust the budget before emitting
                    # any text. Count the truncation and return an empty
                    # answer rather than abort the caller's pass.
                    self.usage["truncated"] += 1
                    return ""
                elif "invalid_prompt" in msg or "usage policy" in msg:
                    # The provider's content filter refused the prompt. As
                    # with a truncation: count it and return an empty answer.
                    self.usage["refused"] += 1
                    return ""
                else:
                    raise

        text = resp.choices[0].message.content or ""
        self.usage["calls"] += 1
        if getattr(resp, "usage", None):
            self.usage["prompt_tokens"] += resp.usage.prompt_tokens or 0
            self.usage["completion_tokens"] += resp.usage.completion_tokens or 0
            details = getattr(resp.usage, "completion_tokens_details", None)
            self.usage["reasoning_tokens"] += (
                getattr(details, "reasoning_tokens", None) or 0) if details else 0
            pdetails = getattr(resp.usage, "prompt_tokens_details", None)
            self.usage["cached_prompt_tokens"] += (
                getattr(pdetails, "cached_tokens", None) or 0) if pdetails else 0
        if cache and self._cache_enabled:
            self._store(cpath, {"text": text})
        return text

    async def chat_responses(
        self,
        *,
        instructions: str | None,
        input_items: list[dict],
        previous_response_id: str | None,
        tools: list[dict],
        max_tokens: int = 2048,
        cache: bool = True,
        cache_key: list | None = None,
    ) -> dict:
        """One turn through `/v1/responses` with native function tools
        (agent/native_tools.py). Returns {"content", "tool_calls": [{id, name,
        arguments}], "response_id", "status"}; usage goes into the same
        counters as `chat`. The cache key is the transcript the caller passes
        (`cache_key`, else `input_items`), not the server's response id."""
        payload = {"model": self.model, "api": "responses", "tools": tools,
                   "transcript": cache_key if cache_key is not None else input_items,
                   "max_tokens": max_tokens}
        cpath = self._cache_path(self._key_extras(payload, pin=False))
        hit = self._cached(cpath) if cache and self._cache_enabled else None
        if hit is not None and "content" in hit:
            self.usage["cached"] += 1
            return hit
        kwargs: dict = {"model": self.model, "input": input_items, "tools": tools,
                        "parallel_tool_calls": False, "max_output_tokens": max_tokens}
        if instructions:
            kwargs["instructions"] = instructions
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
        async with self._sem:
            resp = await self._create_responses(kwargs)
        calls = [{"id": o.call_id, "name": o.name, "arguments": o.arguments or "{}"}
                 for o in resp.output if getattr(o, "type", None) == "function_call"]
        out = {"content": getattr(resp, "output_text", "") or "", "tool_calls": calls,
               "response_id": resp.id, "status": getattr(resp, "status", None)}
        self.usage["calls"] += 1
        u = getattr(resp, "usage", None)
        if u:
            self.usage["prompt_tokens"] += getattr(u, "input_tokens", 0) or 0
            self.usage["completion_tokens"] += getattr(u, "output_tokens", 0) or 0
            od = getattr(u, "output_tokens_details", None)
            self.usage["reasoning_tokens"] += (getattr(od, "reasoning_tokens", None) or 0) if od else 0
            idd = getattr(u, "input_tokens_details", None)
            self.usage["cached_prompt_tokens"] += (getattr(idd, "cached_tokens", None) or 0) if idd else 0
        if out["status"] == "incomplete":
            self.usage["truncated"] += 1
        if cache and self._cache_enabled:
            self._store(cpath, out)
        return out

    def usage_summary(self) -> str:
        u = self.usage
        return (
            f"{self.model}: {u['calls']} api calls ({u['cached']} cache hits), "
            f"{u['prompt_tokens']:,} prompt tok, {u['completion_tokens']:,} completion tok"
            + (f" ({u['reasoning_tokens']:,} of them reasoning)" if u.get("reasoning_tokens") else "")
            + (f", {u['cached_prompt_tokens']:,} prompt tok served from the provider cache"
               if u.get("cached_prompt_tokens") else "")
            + (f", {u['truncated']} truncated (no output)" if u["truncated"] else "")
        )
