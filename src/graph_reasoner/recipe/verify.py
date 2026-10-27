"""Verify before spending (launch/README.md, section 5): five checks of one
size on the GPU machine, stopping at the first failure.

    prospector verify --graph tsmc --size 4b

  1. the environment imports torch, transformers, vLLM and the kernels, and sees the card
  2. a probe adapter for the base (`scripts/infra/make_probe_lora.py`, CPU)
  3. `probe_vllm_rollout.py`: vLLM serves the base with the adapter and generates
  4. `probe_vllm_sleep.py`: three wake, generate and sleep cycles at `--gpu-frac`,
     a large allocation while the engine sleeps, outputs against a no-sleep control
  5. a three-step fp32 SFT smoke of stage 1 under a memory watermark; its peak
     says whether SFT fits at that size
"""
from __future__ import annotations

from ..train.adapters import ADAPTER_CONFIG
from .plan import Context
from .run import Failed, Runner, log_lines, say, sft_smoke, shown

IMPORTS = (
    'import importlib, torch\n'
    'mods = {m: getattr(importlib.import_module(m), "__version__", "ok") '
    'for m in ("torch", "transformers", "trl", "peft", "vllm", "fla", "causal_conv1d")}\n'
    'assert torch.cuda.is_available(), "no CUDA"\n'
    'print(mods, torch.cuda.get_device_name(0))\n'
)


def verify(ctx: Context, runner: Runner, gpu_frac: float = 0.35) -> int:
    L = ctx.layout
    size, base = L.spec.size, L.spec.base_model
    probe = f"models/probe_lora_{size}"
    step = lambda text: say(f"=== VERIFY {size} {text} {runner.now()}")    # noqa: E731

    def fail(text: str) -> int:
        say(f"!!! VERIFY {size} FAIL: {text} {runner.now()}")
        return 1

    step("1 imports")
    rc, out = runner.capture([runner.python(), "-c", IMPORTS])
    for line in out.splitlines():
        if "PyTorch was not found" not in line and "outdated" not in line:
            say(line)
    if rc != 0:
        return fail("imports")

    step("2 probe adapter")
    if not all((L.root / probe / f).exists() for f in (ADAPTER_CONFIG, "tokenizer_config.json")):
        cmd = [runner.python(), "scripts/infra/make_probe_lora.py", base, probe]
        log = L.root / "logs" / f"verify_{size}_probe_lora.log"
        say(f"    {shown(cmd)}")
        if runner.run(cmd, log) != 0:
            return fail(f"make_probe_lora ({(log_lines(log, '.', last=1) or ['?'])[0]})")
        for line in log_lines(log, ".", last=2):
            say(line)
    say(" ".join(p.name for p in sorted((L.root / probe).iterdir())) if (L.root / probe).is_dir() else f"no {probe}")

    step("3 vllm rollout probe")
    runner.reap()
    cmd = [runner.python(), "scripts/infra/probe_vllm_rollout.py", base, probe]
    log = L.root / "logs" / f"verify_{size}_rollout.log"
    say(f"    {shown(cmd)}")
    if runner.run(cmd, log) != 0:
        return fail(f"probe_vllm_rollout ({(log_lines(log, 'Error|error', last=1) or ['?'])[0][:120]})")
    for line in log_lines(log, r"PASS|FAIL|tok/s|tokens/s", last=3):
        say(line)

    step(f"4 vllm sleep probe at gpu_frac {gpu_frac}")
    runner.reap()
    cmd = [runner.python(), "scripts/infra/probe_vllm_sleep.py", base, probe, str(gpu_frac)]
    log = L.root / "logs" / f"verify_{size}_sleep.log"
    say(f"    {shown(cmd)}")
    if runner.run(cmd, log) != 0:
        return fail(f"probe_vllm_sleep ({(log_lines(log, 'Error|error|FAIL', last=1) or ['?'])[0][:120]})")
    for line in log_lines(log, r"PASS|FAIL|asleep|GiB|cycle", last=6):
        say(line)

    step("5 sft fit smoke (3 steps, fp32)")
    try:
        sft_smoke("sft1", L.sft1_dir, ctx, runner, remove=True)
    except Failed as e:
        return fail(e.message)
    say(f"=== VERIFY {size} COMPLETE {runner.now()}")
    return 0
