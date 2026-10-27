"""The sampling backend over HTTP: `client.py`, for anything OpenAI-compatible.
The local drivers live in `eval/local_model.py`.
"""
from .client import LLMClient, gather_with_progress

__all__ = ["LLMClient", "gather_with_progress"]
