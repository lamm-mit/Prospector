#!/usr/bin/env python
"""Build a fresh LoRA adapter for the infra probes.

Usage: make_probe_lora.py <base_model> <out_dir>

Loads the base on CPU, wraps it as the GRPO trainer does (all-linear, rank
16), perturbs every lora_B so the adapter is a nonzero delta (PEFT
zero-initialises B, and an all-zero adapter cannot reveal slot corruption),
and saves the adapter with the base tokenizer, under the `plain_chatml`
template, beside it.

The target modules are resolved against the installed transformers
architecture, the naming vLLM serves: it refuses an adapter that names the
linear-attention projections unfused (in_proj_q/k/v, where current
transformers has in_proj_qkv).
"""
import sys


def main() -> int:
    if len(sys.argv) < 3 or "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return 0 if ("--help" in sys.argv or "-h" in sys.argv) else 2

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base, out = sys.argv[1], sys.argv[2]
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        task_type="CAUSAL_LM", target_modules="all-linear"))
    torch.manual_seed(0)
    n = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.add_(torch.randn_like(p) * 0.002)  # small, nonzero, benign
                n += 1
    model.save_pretrained(out)
    # Save the tokenizer, with the plain_chatml template, beside the adapter:
    # probe_vllm_rollout.py loads its tokenizer from this directory.
    from transformers import AutoTokenizer
    from graph_reasoner.train.tokenize import override_chat_template
    tok = AutoTokenizer.from_pretrained(base)
    override_chat_template(tok, "plain_chatml", model=base)
    tok.save_pretrained(out)
    print(f"perturbed {n} lora_B tensors; adapter + tokenizer (plain_chatml) saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
