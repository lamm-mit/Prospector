"""Training: LoRA SFT (`sft.run_sft`) and GRPO (`grpo.run_grpo`) on one CUDA
GPU, and the modules the trainers and the samplers share.

The shared modules are `tokenize` (which tokens carry loss), `generate` (what
a sampled turn is), `reward` (the objective), `rollout` (config -> the rollout
environment), `transcripts` (the training rows), `defaults`, `config` (the
config checks), `prefixes` (the prefix records) and `adapters` (the adapter
rules). They import no torch, trl or peft and reach `agent/`, `eval/` and
`select/` only inside functions, so the CPU-only tests exercise the objective
in seconds (`tests/test_module_health.py`). The two trainers and `vllm_ckpt`,
the vLLM checkpoint writer, import torch inside their entry points.

Nothing is re-exported here: a caller imports each name from the module that
owns it.
"""
