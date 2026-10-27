"""Where an adapter chain's weights and tokenizer come from, and whether the
load kept every tensor.

`adapter_chain`, `resolve_base` and `tokenizer_source` are pure path rules,
shared by the engines in eval/local_model.py and by agent/eval.py.

`assert_adapter_resolves` is the integrity check. PEFT's `from_pretrained`
wraps the modules of the current model that match `target_modules` and
silently drops checkpoint tensors naming modules the model does not have, so
`merge_and_unload` can return a model missing trained deltas. Call the check
after every `PeftModel.from_pretrained`, before `merge_and_unload`;
`eval/local_model.py::merge_chain_into` does. It compares names and reads no
weights.

The transformers versions supported here name Qwen3.5's linear-attention
projection fused (`in_proj_qkv`). An adapter that names `in_proj_q/k/v` fails
the check and must be converted first.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..recipe import MANIFEST, read_manifest

__all__ = ["adapter_chain", "resolve_base", "tokenizer_source", "is_adapter_dir",
           "recorded_chat_template", "chat_template_for",
           "assert_adapter_resolves", "write_substrate_marker"]


def adapter_chain(adapter: str | None) -> list[str]:
    """`"a, b ,"` -> `["a", "b"]`. Order is load order, which is training
    order: a later adapter is a delta on the merge of the earlier ones."""
    return [a.strip() for a in (adapter or "").split(",") if a.strip()]


def resolve_base(model_name: str, adapter: str | None = None
                 ) -> tuple[str, str | None]:
    """An adapter directory given as the model -> (its base, itself), so
    `--model hf:models/<run>/final` works with no `--adapter`. When `adapter`
    is set, both arguments are returned unchanged."""
    cfg = Path(model_name, "adapter_config.json")
    if adapter is None and cfg.exists():
        base = json.loads(cfg.read_text(encoding="utf-8"))["base_model_name_or_path"]
        return base, model_name
    return model_name, adapter


#: The file that makes a directory a saved adapter; the one name spelled here.
ADAPTER_CONFIG = "adapter_config.json"


def is_adapter_dir(path) -> bool:
    """Whether `path` holds a saved adapter (its `adapter_config.json`)."""
    return (Path(path) / ADAPTER_CONFIG).exists()


def tokenizer_source(model_name: str, adapter: str | None = None) -> str:
    """Where the tokenizer is loaded from: the last adapter in the chain,
    whose directory carries the chat template its run installed, or the base
    model when there is no adapter or that directory has no tokenizer.

    A directory without a tokenizer carries no template either; the template
    then comes from the run's record (`chat_template_for`).
    """
    src = adapter_chain(adapter)[-1] if adapter_chain(adapter) else model_name
    if Path(src).is_dir() and not (Path(src) / "tokenizer_config.json").exists():
        return model_name
    return src


def recorded_chat_template(adapter: str | None) -> str | None:
    """The chat template the last adapter's run recorded, from the `run.json`
    beside it or in its run directory (`hyperparameters.chat_template`), or
    None when there is no adapter or no record."""
    chain = adapter_chain(adapter)
    if not chain:
        return None
    manifest = read_manifest(chain[-1])
    value = ((manifest or {}).get("hyperparameters") or {}).get("chat_template")
    return value if isinstance(value, str) and value else None


def chat_template_for(model_name: str, adapter: str | None, named: str | None) -> str | None:
    """The named template to install before rendering, under one rule.

    Without an adapter it is `named`: None keeps the base model's own template.
    With an adapter, the run's record is the truth: `named` must agree with
    it or the load is refused; when nothing is named the record is installed;
    an adapter with no record keeps the template its directory's tokenizer
    carries (None), and one with neither record nor tokenizer is refused
    until `--chat-template` says which template it was trained under. A hub
    adapter is not a directory and is left to `named`.
    """
    chain = adapter_chain(adapter)
    if not chain:
        return named
    last = chain[-1]
    record = recorded_chat_template(adapter)
    if named and record and named != record:
        raise SystemExit(f"{last} was trained under the chat template {record!r} (its run.json), "
                         f"but --chat-template names {named!r}: drop the flag or name the recorded one")
    if named:
        return named
    if record:
        return record
    if not Path(last).is_dir() or (Path(last) / "tokenizer_config.json").exists():
        return None
    raise SystemExit(f"{last} carries neither a run.json nor a tokenizer, so nothing says which chat "
                     f"template it was trained under: pass --chat-template (the recipe's is plain_chatml)")


def write_substrate_marker(final_dir, chain: list[str]) -> None:
    """Write `substrate.json` beside a trained adapter: `merged_adapters`, the
    chain it is a delta on, in load order.

    A GRPO adapter is trained on its substrate chain merged, but its
    adapter_config names the raw base, so loading it alone applies the delta
    to the wrong weights. `eval/local_model.py` warns when an adapter is
    loaded without the recorded chain before it. An SFT stage trains from
    base, so its chain is empty.
    """
    (Path(final_dir) / "substrate.json").write_text(
        json.dumps({"merged_adapters": [str(a) for a in chain]}), encoding="utf-8")


def copy_run_manifest(run_dir, final_dir) -> None:
    """Copy the run's manifest (`run.json`, written by the trainer scripts
    from `graph_reasoner.recipe`) beside the saved adapter, so the adapter
    carries what it was trained from. A run started by calling `run_sft` or
    `run_grpo` directly, rather than through scripts/train_sft.py or
    scripts/train_traj_grpo.py, has no manifest and saves nothing extra."""
    src = Path(run_dir) / MANIFEST
    if src.exists():
        shutil.copyfile(src, Path(final_dir) / MANIFEST)


def assert_adapter_resolves(peft_model, adapter_dir: str) -> None:
    """Every lora tensor in `adapter_dir`'s checkpoint must land on a wrapped
    module of `peft_model`, else SystemExit naming the orphans. A directory
    without `adapter_model.safetensors` is not checked."""
    from pathlib import Path

    from safetensors import safe_open

    f = Path(adapter_dir) / "adapter_model.safetensors"
    if not f.exists():  # sharded or .bin checkpoints are not checked
        return
    with safe_open(str(f), framework="pt") as sf:
        modules = {k.rsplit(".lora_", 1)[0] for k in sf.keys() if ".lora_" in k}
    orphans = []
    for m in sorted(modules):
        try:
            sub = peft_model.get_submodule(m)
        except AttributeError:
            orphans.append(m)
            continue
        if not hasattr(sub, "lora_A"):
            orphans.append(m)
    if orphans:
        leaves = sorted({m.split(".")[-1] for m in orphans})
        raise SystemExit(
            f"adapter {adapter_dir}: {len(orphans)} trained lora modules "
            f"(leaf names {leaves}) name modules the current model does not "
            f"have — PEFT would DROP them silently and the merged model "
            f"would be missing trained deltas. The release's adapters all "
            f"carry the fused naming; an adapter from another source must "
            f"be converted to it first.")
