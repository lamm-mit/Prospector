"""The first errors a stranger meets name the fix, before anything heavy runs:
a graph file that is not there, a local driver whose package is not installed,
an adapter path that does not exist.
"""
import importlib.util

import pytest

from graph_reasoner.agent.backend import load_backend
from graph_reasoner.agent.eval import build_generate_seam
from graph_reasoner.agent.loop import EpisodeConfig


def test_a_missing_graph_file_is_refused_by_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/raw/README.md"):
        load_backend("biograph", str(tmp_path / "nope.graphml"))


def test_a_driver_without_its_package_is_refused_before_anything_loads(monkeypatch):
    """Without the check the vllm: driver fails inside its constructor, after
    the adapters were merged (minutes, gigabytes)."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)
    with pytest.raises(SystemExit, match="vllm"):
        build_generate_seam("vllm:Qwen/Qwen3.5-4B", EpisodeConfig())


def test_a_missing_adapter_is_refused_by_path(tmp_path):
    """Live mode with the adapters not yet downloaded names the path, not a
    Hub repository-id error."""
    from graph_reasoner.eval.local_model import _resolve_for_serving
    with pytest.raises(FileNotFoundError, match="no adapter at"):
        _resolve_for_serving("Qwen/Qwen3.5-0.8B", str(tmp_path / "models" / "nope"), None, engine="hf")
