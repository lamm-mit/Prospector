"""The writer of vLLM-servable merged checkpoints, shared by the GRPO trainer
and the `vllm:` eval engine.

The checkpoint is written directly, remapped keys through safetensors with the
config and tokenizer beside them, not with `save_pretrained`: `state_dict()`
emits hub-compatible aliased keys (`model.language_model.*`) and
`save_pretrained` re-applies that aliasing, while the text-only config vLLM
loads expects plain `model.layers.*`. When the state dict carries aliased
keys, only the language tower and lm_head are kept.

Every tensor is cloned: on a model already in bf16, `.to(bfloat16)` returns
the same storage, the tied embed_tokens/lm_head pair stays shared, and
safetensors refuses tensors that share memory.

`get_text_config()` can name an architecture vLLM's registry rejects, so
SERVABLE_ARCH pins the servable name per model family; a family it does not
list keeps its config's own names.
"""
from __future__ import annotations

from pathlib import Path

#: text-config model_type -> the architecture name vLLM's registry serves.
SERVABLE_ARCH = {
    "qwen3_5_text": "Qwen3_5ForCausalLM",
}


def write_vllm_checkpoint(model, tokenizer, out_dir) -> str:
    """Write `model` (HF, adapters already merged) as a bf16 checkpoint vLLM
    can serve, and return its directory. An `out_dir` that already holds a
    config.json is reused as it is."""
    import copy

    import torch
    from safetensors.torch import save_file

    out = Path(out_dir)
    if (out / "config.json").exists():
        return str(out)
    sd = model.state_dict()
    if any(k.startswith("model.language_model.") for k in sd):
        sd = {k: v for k, v in sd.items()
              if k.startswith("model.language_model.")
              or k.startswith("lm_head")}
    bstate = {k.replace("model.language_model.", "model."):
              v.detach().to(torch.bfloat16).contiguous().clone()
              for k, v in sd.items()}
    assert bstate, "state dict filtered to nothing; aliasing changed upstream"
    out.mkdir(parents=True, exist_ok=True)
    save_file(bstate, str(out / "model.safetensors"), metadata={"format": "pt"})
    vcfg = copy.deepcopy(
        model.config.get_text_config()
        if hasattr(model.config, "get_text_config")
        else model.config)
    arch = SERVABLE_ARCH.get(getattr(vcfg, "model_type", None))
    if arch:
        vcfg.architectures = [arch]
    vcfg.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"vllm checkpoint (merged, bf16) -> {out}")
    return str(out)
