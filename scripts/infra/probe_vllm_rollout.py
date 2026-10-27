#!/usr/bin/env python
"""Probe: can vLLM serve the rollout policy, a base model plus a LoRA adapter?

Usage: probe_vllm_rollout.py <base_model> <adapter_dir>

Loads the base under vLLM with prefix caching, attaches the adapter as a
LoRARequest, renders four prompts with the adapter directory's tokenizer
(given the `plain_chatml` template when it carries none) and generates
without and with the adapter, printing tokens per second and a sample for
each. PROBE PASS means both generations completed; read the adapter's sample
to judge whether it is coherent.

Needs vllm and a CUDA GPU. `--help` or a missing positional prints this text.
"""
import sys
import time


def main() -> int:
    if len(sys.argv) < 3 or "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return 0 if ("--help" in sys.argv or "-h" in sys.argv) else 2

    from transformers import AutoTokenizer

    from graph_reasoner.train.tokenize import override_chat_template

    base, adapter = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(adapter)  # the adapter directory's own tokenizer
    if tok.chat_template is None:
        override_chat_template(tok, "plain_chatml", model=base)

    CONVS = [
        [{"role": "system", "content": "You are a graph agent. Use tools."},
         {"role": "user", "content": f"Which process step follows {x}? Think, then call a tool."}]
        for x in ("etch", "lithography", "deposition", "CMP")
    ]

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=base, enable_lora=True, max_lora_rank=16, dtype="bfloat16",
              enable_prefix_caching=True, gpu_memory_utilization=0.85,
              max_model_len=16384)
    # Render to text, then encode: train/generate.py's rendering contract.
    texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
             for c in CONVS]
    ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    sp = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=256)
    lora = LoRARequest("sft", 1, adapter)

    for name, req in (("base", None), ("adapter", lora)):
        t0 = time.time()
        outs = llm.generate([{"prompt_token_ids": i} for i in ids], sp, lora_request=req)
        dt = time.time() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"--- {name}: {ntok} tokens in {dt:.1f}s = {ntok/dt:.0f} tok/s")
        print("sample:", outs[0].outputs[0].text[:200].replace("\n", " | "))
    print("PROBE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
