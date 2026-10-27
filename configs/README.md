# configs/: the recipe's hyperparameters and the benchmark builds

This folder holds nine files. Five hold the hyperparameters of the recipe's
four training stages and its verifier, one file each. The four under
`benchmark/` configure the graphs' question builds.

| file | stage | trainer |
|---|---|---|
| `sft1.yaml` | 1, LoRA SFT from base on the teacher's demonstrations | `scripts/train_sft.py --stage sft1` |
| `grpo_full.yaml` | 2, GRPO over whole episodes (learning rate 3e-5) | `scripts/train_traj_grpo.py --stage full` |
| `sft2.yaml` | 3, LoRA SFT from base on the demonstrations plus stage 2's verified episodes | `scripts/train_sft.py --stage sft2` |
| `grpo_decision.yaml` | 4, both decision rounds (learning rate 1.5e-5) | `scripts/train_traj_grpo.py --stage d1`, `--stage d2` |
| `verifier.yaml` | the yes/no verifier on the served policy's own passes | `scripts/train_sft.py --stage verifier` |

A hyperparameter file names no path and no base model.
`graph_reasoner.recipe` derives the base model and every path a stage reads
or writes from `--graph`, `--size`, `--tag` and `--stage`. The same files
serve every size. Before anything loads, `check_hyperparameter_keys` in
`train/config.py` refuses a file that sets the base model, a path, or a key
the trainer does not read.

A key the file leaves out takes the trainer's default. The shipped files
also set the reward and the group size of the GRPO stages and the LoRA
modules of the SFT stages, at the trainers' default values. Each file opens
with a comment that states what it trains and what a user needs to know to
run it. `--config FILE` on a trainer replaces the stage's file for an
experiment.

## What a training run records

Each trainer writes `run.json` into its adapter directory and copies it
beside the adapter in `final/`. The file records the spec (graph, size, tag
and model), the stage, the base model, every resolved input path, the
hyperparameters, and the hyperparameter file with its hash. The file also
records the launch, with its arguments, git state and environment, and every
relaunch. A trainer refuses an adapter directory whose `run.json` records
other inputs, so two runs never share a directory.

The released adapters' `run.json` files carry `source: backfill`. A
backfilled file is reconstructed from the files its adapter was trained from
and records no training launch.

## The served stacks

Each size's policy is `Qwen/Qwen3.5-<size>` with three LoRA adapters merged
in the order listed: the stage-3 adapter, then one checkpoint of each
decision round. The policy is served with the `plain_chatml` chat template,
20 tool calls and 1,024 tokens a turn.

| size | policy (`--adapter`, load order) | verifier |
|---|---|---|
| 4B | `models/traj_sft_4b_flash20g2/final`, `models/traj_grpo_4b_flash20d1/checkpoint-120`, `models/traj_grpo_4b_flash20d2/checkpoint-120` | `models/verifier_sft_4b_flash20v/final` |
| 2B | `models/traj_sft_2b_flash20g2/final`, `models/traj_grpo_2b_flash20d1/checkpoint-100`, `models/traj_grpo_2b_flash20d2/checkpoint-120` | `models/verifier_sft_2b_flash20v/final` |
| 0.8B | `models/traj_sft_0.8b_flash20g2/final`, `models/traj_grpo_0.8b_flash20d1/checkpoint-100`, `models/traj_grpo_0.8b_flash20d2/checkpoint-120` | `models/verifier_sft_0.8b_flash20v/final` |

`prospector names --graph tsmc --size <size> stack` prints a size's stack
from the checkpoint numbers recorded in `models/<round>/peak.txt`.
Stage 2's adapter is not in the stack. The episodes that stage 3 adds to the
demonstrations are sampled from the checkpoint recorded in stage 2's
`peak.txt`: `checkpoint-100` of `models/traj_grpo_<size>_flash20full` at 4B
and 2B, and `checkpoint-40` at 0.8B.

## The benchmark builds

`benchmark/tsmc_v0.yaml` configures the build of the semiconductor graph's
raw question file, which curation reduces to its evaluation questions.
`benchmark/{biograph,pfas,matprop}_v0.yaml` configure the builds of the
other three graphs. The main README's section
[Training on your own graph](../README.md#training-on-your-own-graph) gives
the commands.
