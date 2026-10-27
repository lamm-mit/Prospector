"""Which key and which endpoint an API model id is served through.

An OpenAI model runs on `OPENAI_API_KEY` against OpenAI itself and never
through OpenRouter; every other API model runs through OpenRouter on
`OPENROUTER_API_KEY`. `route` applies the rule and `LLMClient` refuses the one
combination it forbids.

A vendor-prefixed id (`google/gemini-3.7-flash`, `anthropic/...`) is the
OpenRouter form; a bare id (`gpt-5-mini`) is OpenAI's own. An exported
`OPENAI_BASE_URL` overrides the endpoint and is used with `OPENAI_API_KEY`.
"""
from __future__ import annotations

import os
import re

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_OPENAI_ID_RE = re.compile(r"^(?:openai/)?(?:gpt-|o\d)", re.IGNORECASE)


def route(model: str) -> tuple[str, str | None]:
    """`(the env var holding the key, the base URL)` for this model id; a
    None base URL is OpenAI's own endpoint."""
    explicit = os.getenv("OPENAI_BASE_URL")
    if explicit:
        return "OPENAI_API_KEY", explicit
    if "/" in model:
        return "OPENROUTER_API_KEY", OPENROUTER_BASE_URL
    return "OPENAI_API_KEY", None


def is_openai_model(model: str) -> bool:
    """An OpenAI model id, bare (`gpt-5-mini`, `o3`) or vendor-prefixed."""
    return bool(_OPENAI_ID_RE.match(model))


def refuse_openai_through_openrouter(model: str, base_url: str | None) -> None:
    """Raise if an OpenAI model would be sent to OpenRouter."""
    if base_url and "openrouter" in base_url.lower() and is_openai_model(model):
        raise RuntimeError(
            f"{model!r} is an OpenAI model and the base URL is OpenRouter "
            f"({base_url}); OpenAI models run on OPENAI_API_KEY against OpenAI "
            f"itself. Unset OPENAI_BASE_URL and pass the bare id.")


__all__ = ["OPENROUTER_BASE_URL", "route", "is_openai_model",
           "refuse_openai_through_openrouter"]
