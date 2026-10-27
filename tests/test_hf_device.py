"""Where `LocalHFModel` puts the weights, and in what precision.

`_pick_device` chooses CUDA, then MPS, then CPU, so an Apple Silicon machine
runs on its GPU and not on CPU in fp32. It takes the torch module as an
argument, so the decision is tested here without torch.
"""
from types import SimpleNamespace

import pytest

from graph_reasoner.eval.local_model import _pick_device


@pytest.fixture(autouse=True)
def _no_device_overrides(monkeypatch):
    """The two overrides `_pick_device` reads are cleared, so a developer's
    exported GR_HF_DEVICE or GR_HF_DTYPE does not decide these tests."""
    monkeypatch.delenv("GR_HF_DEVICE", raising=False)
    monkeypatch.delenv("GR_HF_DTYPE", raising=False)


def _torch(*, cuda=False, bf16=True, mps=False):
    """The attributes `_pick_device` reads."""
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda,
                             is_bf16_supported=lambda: bf16),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        bfloat16="bfloat16", float16="float16", float32="float32")


@pytest.mark.parametrize("kw,device,dtype", [
    (dict(cuda=True, bf16=True), "cuda:0", "bfloat16"),
    (dict(cuda=True, bf16=False), "cuda:0", "float16"),
    # Apple Silicon: no CUDA, but MPS.
    (dict(mps=True), "mps", "bfloat16"),
    # CUDA wins when both are present.
    (dict(cuda=True, mps=True), "cuda:0", "bfloat16"),
    (dict(), "cpu", "float32"),
])
def test_device_and_precision(kw, device, dtype):
    assert _pick_device(_torch(**kw)) == (device, dtype)


def test_mps_keeps_bf16_because_the_checkpoints_are_bf16(monkeypatch):
    """On MPS the dtype is bf16, as on a CUDA device that supports it, and
    not fp16."""
    monkeypatch.delenv("GR_HF_DTYPE", raising=False)
    assert _pick_device(_torch(mps=True))[1] == "bfloat16"


def test_env_overrides_exist_for_the_numerics_control(monkeypatch):
    """`GR_HF_DEVICE` and `GR_HF_DTYPE` override both choices, so an
    fp32-on-CPU control runs through the same code path."""
    monkeypatch.setenv("GR_HF_DEVICE", "cpu")
    monkeypatch.setenv("GR_HF_DTYPE", "float32")
    assert _pick_device(_torch(cuda=True)) == ("cpu", "float32")
