"""LoRA SFT on agent trajectories, with transformers' Trainer. Config-driven.

The dataset stores one prompt/completion row per assistant turn.
`transcripts.load_transcripts` reassembles each trajectory and
`tokenize.supervised_arrays` masks it, so a trajectory trains as one row in
which only the assistant content carries loss. Every stage trains from the
base model; no prior adapter is merged.

`run_sft` checks its config keys first (`config.py`) and imports torch inside
the function, so importing this module does not need torch.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def hf_row(input_tokens: list[int], target_tokens: list[int],
           weights: list[int]) -> tuple[list[int], list[int]]:
    """The shifted triple of `tokenize.supervised_arrays` -> HF (input_ids,
    labels).

    `supervised_arrays` returns

        input_tokens  = ids[:-1]     target_tokens = ids[1:]     weights = w[1:]

    so weights[i] gates predicting ids[i+1]. HF shifts internally and wants
    the unshifted sequence, `input_tokens + [target_tokens[-1]]`; labels built
    on `input_tokens` alone would lose the last trainable token. The weights
    are 0 or 1, so a label of -100 expresses them exactly.
    """
    ids = [*input_tokens, target_tokens[-1]]
    labels = [-100] * len(ids)
    for i, w in enumerate(weights):
        if w:
            labels[i + 1] = ids[i + 1]
    return ids, labels


def _transcript_rows(data_path: str, tokenizer, max_length: int):
    """One {input_ids, labels} row per transcript, labels -100 wherever no
    loss is taken. A transcript over `max_length` tokens is dropped, not
    truncated (`tokenize.supervised_arrays`), and the count is printed."""
    from .tokenize import supervised_arrays
    from .transcripts import load_transcripts

    rows, dropped = [], 0
    for msgs in load_transcripts(data_path):
        got = supervised_arrays(msgs, tokenizer, max_length=max_length)
        if got is None:
            dropped += 1
            continue
        ids, labels = hf_row(*got)
        rows.append({"input_ids": ids, "labels": labels})
    if dropped:
        print(f"dropped {dropped} transcripts over max_length={max_length}")
    return rows


# Every key `run_sft` reads. A config key outside this set is refused
# (`config.check_config_keys`); tests/test_module_health.py checks that every
# key here is read. `SFT_RESOLVED_KEYS` are the ones the recipe derives from
# --graph/--size/--tag/--stage (graph_reasoner.recipe); a stage's
# hyperparameter file carries the others only.
SFT_CONFIG_KEYS = frozenset({
    "base_model", "batch_size", "chat_template", "dataset", "epochs",
    "grad_accum", "gradient_checkpointing", "logging_steps",
    "lora_alpha", "lora_dropout", "lora_r", "lora_target_modules", "lr",
    "matmul_precision", "max_length", "max_steps", "model_dtype", "output_dir",
    "save_steps", "save_strategy", "seed",
})
SFT_RESOLVED_KEYS = frozenset({"base_model", "dataset", "output_dir"})


def validate_sft_config_keys(config: dict) -> None:
    """Refuse a config with a key `run_sft` does not read."""
    from .config import check_config_keys

    check_config_keys(config, SFT_CONFIG_KEYS, known_name="SFT_CONFIG_KEYS")


def run_sft(config: dict) -> Path:
    """Config -> a LoRA adapter trained from base, saved with its tokenizer
    into `output_dir/final`."""
    from .defaults import SFT_TARGET_MODULES, SHARED

    validate_sft_config_keys(config)
    if not config.get("output_dir"):
        raise SystemExit("config must set output_dir: a run resumes from the "
                         "checkpoints it finds there, so no default is safe")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)
    from transformers.trainer_utils import get_last_checkpoint

    from .adapters import copy_run_manifest, is_adapter_dir, write_substrate_marker
    from .config import refuse_broken_cuda
    from .tokenize import override_chat_template

    refuse_broken_cuda(torch)

    # The default is torch's own, "highest". Any other value also enables
    # TF32 for CUDA matmuls, which is faster at reduced matmul precision.
    # Every shipped config asks for "high".
    prec = str(config.get("matmul_precision", "highest"))
    torch.set_float32_matmul_precision(prec)
    if prec != "highest":
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"matmul precision: {prec} (TF32 tensor cores enabled)")

    base = config["base_model"]
    out_dir = Path(config["output_dir"])
    # A run resumes from the last checkpoint in output_dir. `final/` with no
    # checkpoint beside it is a finished run whose checkpoints are gone, or a
    # released adapter unpacked at this path: refused rather than overwritten.
    last_ckpt = get_last_checkpoint(str(out_dir)) if out_dir.exists() else None
    if last_ckpt is None and is_adapter_dir(out_dir / "final"):
        raise SystemExit(
            f"{out_dir}/final holds a finished adapter and there is no checkpoint "
            f"to resume from. Move that directory aside, or set another output_dir.")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    # fp32 master weights by default: trajectory SFT puts loss on a small
    # fraction of the tokens, and bf16 weights can give NaN gradients there.
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[str(config.get("model_dtype", "float32"))]
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=dtype, attn_implementation="sdpa")
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(base)
    # Qwen3.5's own template rewrites earlier turns, so training needs a
    # concatenative one, and every config installs `plain_chatml`. It is
    # installed before the rows are built and saved with the tokenizer into
    # final/, so evaluation renders with it when it loads the adapter.
    override_chat_template(tokenizer, config.get("chat_template"), model=base)

    data_path = config["dataset"]
    rows = _transcript_rows(data_path, tokenizer, int(config.get("max_length", 2048)))
    if not rows:
        raise SystemExit("every transcript exceeds max_length; nothing to train on")
    ds = Dataset.from_list(rows)
    tot = sum(len(r["input_ids"]) for r in rows)
    trained = sum(sum(1 for x in r["labels"] if x != -100) for r in rows)
    print(f"Dataset: {len(ds)} TRANSCRIPTS from {data_path} — "
          f"{tot:,} tokens, {trained:,} trainable ({trained/max(1,tot):.1%})")

    lora = LoraConfig(
        r=int(config.get("lora_r", SHARED["lora_r"])),
        lora_alpha=int(config.get("lora_alpha", SHARED["lora_alpha"])),
        lora_dropout=float(config.get("lora_dropout", SHARED["lora_dropout"])),
        target_modules=list(config.get("lora_target_modules") or SFT_TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=float(config.get("epochs", 2)),
        per_device_train_batch_size=int(config.get("batch_size", 2)),
        gradient_accumulation_steps=int(config.get("grad_accum", 8)),
        learning_rate=float(config.get("lr", 1e-4)),
        lr_scheduler_type="cosine",
        warmup_steps=SHARED["warmup_ratio"],   # below 1: a ratio of the run's steps
        bf16=use_bf16 and dtype is torch.bfloat16,
        fp16=(not use_bf16) and dtype is torch.float16,
        gradient_checkpointing=config.get("gradient_checkpointing", SHARED["gradient_checkpointing"]),
        # PEFT + checkpointing wants the non-reentrant path; the reentrant one
        # can drop grads on inputs that never required them.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=int(config.get("max_steps", -1)),
        logging_steps=int(config.get("logging_steps", 10)),
        save_strategy=config.get("save_strategy", "epoch"),
        save_steps=int(config.get("save_steps", 500)),
        report_to=[],
        seed=int(config.get("seed", SHARED["seed"])),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, label_pad_token_id=-100, padding=True),
    )
    # Resume from the last checkpoint in output_dir, if there is one.
    if last_ckpt:
        print(f"Resuming from checkpoint: {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)

    # The adapter is saved under a temporary name and renamed once whole, so a
    # kill mid-save leaves no `final/` for the recipe to take as the trained
    # adapter (`is_adapter_dir` sees the config file before the weights).
    final = out_dir / "final"
    staging = out_dir / "final.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    trainer.save_model(str(staging))
    tokenizer.save_pretrained(str(staging))
    write_substrate_marker(staging, [])
    copy_run_manifest(out_dir, staging)
    if final.exists():
        shutil.rmtree(final)
    os.replace(staging, final)
    print(f"Adapter saved -> {final}")
    return final
