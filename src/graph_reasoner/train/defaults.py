"""Training defaults that more than one file reads.

`SHARED` holds the values that must be the same in every stage: the LoRA
geometry, the seed, the sampling nucleus and the group size. Every key is read
with `config.get(k, D)`, so a default written in two places can diverge
without an error; `tests/test_module_health.py` requires every `config.get`
in the training modules whose key is in SHARED to take its default from here.

Not here: the values that differ by stage (`max_steps`, `batch_size`), which
each trainer states where it reads them, and the run settings (`lr`,
`grad_accum`, `save_steps`, `logging_steps`, `max_seq_len`), which the
configs set explicitly.
"""
from __future__ import annotations

# One value in every stage. Changing one of these changes the recipe.
SHARED = {
    # LoRA geometry, alpha/r = 2 at rank 16.
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    # Seed and warmup.
    "seed": 0,
    "warmup_ratio": 0.03,
    # Rollout sampling and the GRPO group. num_generations is the group size
    # the advantage centres over; top_p is the nucleus every sampler draws at.
    "num_generations": 16,
    "top_p": 0.95,
    "scale_rewards": "group",
    "gradient_checkpointing": True,
}

#: The two stages adapt different LoRA leaf sets on Qwen3.5. SFT adapts these
#: ten, the default of `lora_target_modules`; GRPO adapts `all-linear`
#: (grpo.py), which adds the linear-attention mixer gates in_proj_a and
#: in_proj_b.
SFT_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "out_proj",
)
