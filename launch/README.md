# launch/: the GPU machine

Stages 1 to 4 of the recipe and its selection run on a machine with one
CUDA GPU. This guide prepares and checks that machine, runs each stage by
hand, and lists what to copy off a rented machine before it is terminated.
The recipe itself runs through `prospector`, the package's command line,
which [the recipe's README](../src/graph_reasoner/recipe/README.md)
describes. The folder holds three files besides this guide:

- `setup.sh` installs uv, the virtual environment on the pinned stack, nvcc
  and the fast-path kernels, then checks the imports and caches the tests'
  tokenizer. It takes no arguments and stops on any machine other than
  Linux with an NVIDIA card.
- `constraints-gpu.txt` pins that stack. `constraints-mac.txt` pins the
  Apple-silicon environment of the main README's setup.

`home` is your own machine, which holds the API keys. In a command, `<box>`
is the GPU machine's ssh name. Memory figures are for the 4B on a 94.5 GiB
card, an NVIDIA GH200.

## 1. The machine

The recipe needs one CUDA GPU with 80 to 96 GB of memory, on Ubuntu 22.04.
Section 3 installs the CUDA compiler from NVIDIA's repository for that
release. `setup.sh` handles both aarch64 and x86_64. `prospector run` prints
the card's memory first and warns under 74 GiB; it does not refuse.

Provision about 300 GB of disk for a 4B run. Every chain the recipe serves
is merged to bf16 before vLLM loads it (9.7 GB each at 4B, written inside
the directory of the chain's last adapter under `models/`), and a run merges about twenty of
them, beside the base
model, the Hugging Face cache, the round directories with their checkpoints
and the stored reads.

`setup.sh` and `prospector` find the repository root themselves, so the
clone can live anywhere. Run the scripts by hand from the repository root.
The commands below assume `~/Prospector`.

**A rented machine's disk is destroyed when the instance terminates**, with
the models, the Hugging Face cache and the logs on it. Copy home what you
need before terminating it (section 8).

## 2. The repository and the data

Clone the repository on the GPU machine, or mirror the tracked files of a
working copy from home:

```bash
git clone https://github.com/lamm-mit/Prospector ~/Prospector          # on the machine
git ls-files -z | rsync -az --from0 --files-from=- ./ <box>:Prospector/   # or from home
```

The mirror copies tracked files only, so `.env` and its API keys stay home.
No stage on the GPU machine calls an API, and the teacher pass runs at home
(`prospector teach`). When you mirror, edit only at home, because the next
mirror overwrites an edit made on the GPU machine.

The questions (`data/benchmarks/`) and the provenance sidecars
(`data/traj/*.provenance.json`) are tracked. The recipe needs three files
that are not:

| what | where | used by |
|---|---|---|
| the semiconductor graph, 8 MB | `data/raw/GRAPHDATA_TSMC_OUTPUT/tsmc_5b10p.graphml` | the GRPO rounds and every evaluation on that graph |
| the bio-inspired materials graph, 15 MB | `data/raw/biograph/large_graph_simple_giant.graphml` | the transfer evaluation after every stage (`--transfer-graph biograph`) |
| the demonstrations | `data/traj/tsmc_sft_mix_flash20.jsonl` | stages 1 and 3 (`prospector teach` builds it at home) |

A machine that will not train still needs the demonstrations file, because
the last step of `prospector verify` (section 5) trains three steps on it.
Fetch the graphs and check their checksums as
[data/raw/README.md](../data/raw/README.md) shows, or copy all three files
from home:

```bash
rsync -az --relative data/raw/GRAPHDATA_TSMC_OUTPUT/tsmc_5b10p.graphml \
    data/raw/biograph/large_graph_simple_giant.graphml \
    data/traj/tsmc_sft_mix_flash20.jsonl <box>:Prospector/
```

After section 3, check the copy from the active environment on the GPU
machine with `python -c "import graph_reasoner"`.

## 3. The virtual environment and the kernels

`.venv` is installed with the `train`, `rl` and `app` extras and holds
everything the GPU machine runs: the two trainers, the `vllm:` evaluation
driver, the selection stage, the app and the tests.

`setup.sh` runs this whole section on a fresh machine and can be rerun. Run
it in the background and follow `logs/bootstrap.log`:

```bash
mkdir -p logs
(nohup launch/setup.sh > logs/bootstrap.log 2>&1 < /dev/null &)
```

The steps by hand use `uv` from its standard installer (`~/.local/bin/uv`):

```bash
cd ~/Prospector
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,train,rl,app]" -c launch/constraints-gpu.txt
source .venv/bin/activate                  # every command in this document runs in it
```

**Install against `launch/constraints-gpu.txt`**, which pins the versions
the recipe ran on. An unpinned install resolves newer releases, and the
kernels below compile against the installed torch. `pyproject.toml` explains
the bounds on transformers and TRL.

**Then install Qwen3.5's two fast-path kernels**, `flash-linear-attention`
and `causal-conv1d`. Without both, transformers falls back to a pure-torch
implementation about ten times slower. A training log that says "Falling
back to torch implementation" means this step is missing. `causal-conv1d`
has no wheel for the pinned torch and CUDA on either architecture. The
package compiles here against CUDA 13.0, the version torch was built with,
using `nvcc` from NVIDIA's apt repository:

```bash
ARCH=$([ "$(uname -m)" = aarch64 ] && echo sbsa || echo x86_64)   # NVIDIA's repo for this arch
curl -fsSL -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/$ARCH/cuda-keyring_1.1-1_all.deb
sudo dpkg -i /tmp/cuda-keyring.deb && sudo apt-get update -qq && sudo apt-get install -y -qq cuda-nvcc-13-0 cuda-cudart-dev-13-0
uv pip install --python .venv/bin/python setuptools wheel ninja packaging      # the build's own deps
CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH \
  uv pip install --python .venv/bin/python -c launch/constraints-gpu.txt flash-linear-attention
CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH \
  uv pip install --python .venv/bin/python -c launch/constraints-gpu.txt --no-build-isolation \
    --no-binary causal-conv1d causal-conv1d                                    # --no-binary: PyPI's aarch64 wheels are for other torch versions
```

The compile takes about eight minutes. When a wheel kept under
`launch/wheels/` matches the stack, `setup.sh` installs that wheel instead
(section 8).

**Keep `--no-build-isolation`.** Without the flag, uv builds the source
distribution in a sandbox with the package's own build requirements. Those
requirements name `torch` unpinned, so the extension compiles against the
newest torch on PyPI instead of the environment's. The symptom is an
installed package whose `import causal_conv1d` fails with an undefined
`c10::cuda::...` symbol. `setup.sh` detects that case and rebuilds.

## 4. Environment variables

`prospector` changes to the repository root and gives every job it launches
what the stack needs: `HF_HOME` (`~/.cache/huggingface` unless set),
`PYTHONUNBUFFERED`, `VLLM_USE_FLASHINFER_SAMPLER=0`, expandable CUDA
segments, and `.venv/bin` first on `PATH`. A job run by hand needs the same:

```bash
cd ~/Prospector && source .venv/bin/activate && export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
```

**Activate the environment before any vLLM command by hand.** vLLM's
flashinfer JIT calls `ninja`, which is in `.venv/bin`, so that directory
must be on `PATH`. Naming the interpreter does not put it there. Without
`ninja` on `PATH`, the engine fails to start with
`Engine core initialization failed`, wrapping `FileNotFoundError: 'ninja'`.
Whether the JIT runs depends on its cache, so a launch without activation
can work several times and then fail.

## 5. Verify before spending

The test suite and `prospector verify` check the machine before any
training:

```bash
python -m pytest            # the full suite; with torch installed the torch-gated tests run too
nohup prospector verify --graph tsmc --size 4b > logs/verify_4b.log 2>&1 < /dev/null &
```

The suite's tokenizer tests skip until the `Qwen/Qwen3.5-0.8B` tokenizer is
in the Hugging Face cache, which `setup.sh` fills.

`prospector verify` checks one size in five steps, and `--help` lists its
flags. The check stops at the first failure with a `FAIL:` line, and
otherwise ends with `VERIFY <size> COMPLETE`. The steps run in this order:

1. The environment imports torch, transformers, vLLM and the kernels, and
   sees the card.
2. `scripts/infra/make_probe_lora.py` builds a probe adapter, saved with the
   base tokenizer under the `plain_chatml` template.
3. `scripts/infra/probe_vllm_rollout.py` serves the base model with the
   adapter under vLLM and generates.
4. `scripts/infra/probe_vllm_sleep.py` runs three cycles of wake, generate
   and sleep at the engine's share of the card (`--gpu-frac`, default 0.35),
   with a large allocation while the engine sleeps.
5. `scripts/train_sft.py --smoke` trains 3 steps in fp32 on stage 1's rows,
   into `models/traj_sft_<size>_flash20_smoke`, which is removed afterwards.
   GPU memory is logged every 5 s, and the printed peak shows whether SFT
   fits at that size.

Greedy decoding on this stack is not bit-reproducible, so the sleep probe
compares outputs within a tolerance. The probe fails when a call deviates
from the no-sleep control by more than 0.5 nats, or when every call after a
sleep deviates and no control call does. A `NOTE:` line about sporadic
deviations is expected, and a `PROBE FAIL` line is a failure.

## 6. The stages by hand

`prospector run` runs the whole recipe for one size, the selection included,
as [the recipe's README](../src/graph_reasoner/recipe/README.md) describes.
Its section "The builders by hand" lists the data builders between the
stages. The commands below run the two trainers and an evaluation by hand.

### SFT

```bash
nohup python scripts/train_sft.py \
  --graph tsmc --size 4b --stage sft1 > logs/sft.log 2>&1 &      # sft1, sft2 or verifier
```

- **Keep `batch_size: 1` with `grad_accum: 32`.** At 12,288-token rows, the
  fp32 logits over Qwen3.5's vocabulary of about 248,000 tokens bound memory
  before the weights do, and a batch of 4 does not fit. Any other split must
  keep the effective batch of 32. The verifier's rows are short, so
  `configs/verifier.yaml` runs batch 8 with accumulation 4, the same
  effective batch.
- **Remove the output directory before a fresh run.** Both trainers resume
  from the last checkpoint in `output_dir`, and a directory already at its
  last step trains nothing. `prospector` removes the directory itself.

### GRPO

```bash
nohup python scripts/train_traj_grpo.py \
  --graph tsmc --size 4b --stage d1 > logs/grpo.log 2>&1 &       # full, d1 or d2; --smoke for three steps
```

- **Keep the memory settings of the GRPO files.** A round holds the
  trainer's copy of the model and the sampling engine's copy on one card, and
  the two take turns. Without these settings, the 4B runs out of memory on a
  94.5 GiB card:
  - `vllm_sleep: true` puts the engine to sleep (level 1) while the trainer
    runs, and about 1.9 GiB of the engine stays resident.
  - `logps_chunk: 512` computes log-probabilities 512 positions at a time.
    The full-vocabulary logits allocation of 9 to 11 GiB becomes a transient
    of about half a GiB.
  - `model_dtype: bfloat16` and `gradient_checkpointing: true` reduce the
    trainer's memory.
  - `vllm_gpu_frac: 0.35` sets the engine's share of the card.
  - `vllm_max_num_seqs: 64` bounds the engine's concurrent sequences. vLLM
    reserves state for every decode sequence of a linear-attention hybrid,
    and its default of 1,024 can fill the card.
- **Check what the round trains on.** An absent `tiers` key means every
  tier. The trainer prints the tiers and the count it loaded (the `tiers=`
  line), then the count that the pass-rate filter keeps (`informative_band`
  in the full round, `prefix_band` in the decision rounds). Read both counts
  in the smoke run.
- **Adapters use transformers' fused naming.** transformers names Qwen3.5's
  linear-attention projection `in_proj_qkv`. PEFT silently drops the tensors
  of an adapter that names it unfused (`in_proj_q/k/v`), so the load guard
  (`train/adapters.py`) refuses such an adapter until it is converted. The
  released adapters and every adapter trained here use the fused name.
- **Copy checkpoints home while a round runs.** The recipe evaluates every
  checkpoint of a round and selects one, the round's *pick*. The trainer
  keeps the last two checkpoints unless the file sets `save_total_limit`,
  and the shipped files set `0`, which keeps all. On a rented machine, run
  this from home at intervals:
  `rsync -az --exclude vllm_base --exclude vllm_lora_live --exclude vllm_merged --exclude hf_merged <box>:Prospector/models/<run>/ models/<run>/`.
  The excluded directories are working copies that the trainer and the
  drivers rebuild.
- **Each step's log line carries `t_*` time sums.** `t_wake`, `t_adapter`,
  `t_gen`, `t_score`, `t_echo` and `t_sleep` time the rollout, and
  `t_fwd_grad` and `t_fwd_nograd` time the trainer's forward passes.
  `t_microbatch` times the whole training step, and `t_train_resid` is the
  remainder after the others.
- **Read the memory log of a round or a smoke.** Every 5 s, `prospector`
  appends the output of `nvidia-smi --query-gpu=memory.used,utilization.gpu`
  to a `logs/*_watermarks.log` file. A healthy 4B round alternates between
  about 44 GiB while generating and 16 GiB while training.

### Evaluation with the `vllm:` driver

```bash
python scripts/run_agent_eval.py \
  --graph tsmc --model vllm:Qwen/Qwen3.5-4B \
  --adapter models/traj_sft_4b_flash20g2/final,models/traj_grpo_4b_flash20d1/checkpoint-120,models/traj_grpo_4b_flash20d2/checkpoint-120 \
  --turn-max-tokens 1024 --seed 0 --name <name>
```

Run each evaluation at `--seed 0`, `1` and `2` and compare means, since
greedy decoding on this model differs from run to run.
[Serving Prospector](../README.md#serving-prospector) in the main README
describes the flags and the drivers.

## 7. Operational traps

- **Reap orphaned engines before starting one.** A vLLM engine whose parent
  dies survives as a child of PID 1, holding most of the card, and every
  later engine start fails with `Engine core initialization failed`.
  `prospector` reaps them before every engine start. To reap them by hand,
  run
  `for p in $(pgrep -f "[E]ngineCore"); do [ "$(ps -o ppid= -p $p | tr -d ' ')" = 1 ] && kill -9 $p; done`.
- **Kill over ssh by a bracketed pattern, in a command of its own.**
  `ssh <box> 'pkill -f train_sft'` matches the remote shell's own command
  line and kills the session. The bracketed `pkill -f "train_[s]ft"` avoids
  that only when the unbracketed string appears nowhere else in the command.
  Never put a kill and a relaunch in one line. Killing by PID avoids the
  problem altogether.
- **Wait on a file or a log line instead of on `pgrep -f`.** A script
  started from a heredoc has its own text on the launching shell's command
  line, so a `pgrep -f <pattern>` wait inside it matches the launcher and
  never ends. `prospector` waits on the files and log lines a job writes.
- **A sleeping engine's teardown message is harmless.** A process that exits
  while its engine is asleep prints
  `CUDA Error: invalid argument (cumem_allocator)` after its final saves.
  The outputs are intact. Check afterwards that no orphaned engine is left.
- **Find what holds memory that no process claims.** When `nvidia-smi` shows
  memory in use and no compute process, look for leftover `nvidia-smi -l`
  logging loops. `sudo lsof /dev/nvidia0` names whatever holds the device.
  Make every memory-logging loop end with its job.
- **Watch logs through a reconnecting ssh.** A `tail -F` over plain ssh ends
  in `Broken pipe` after some hours. Loop over
  `ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 <box> 'tail -n 0 -F ...'`
  instead. When a watch dies, check with `pgrep` on the machine whether the
  run died too.
- **Mirror only between runs.** Every job `prospector` launches imports the
  package afresh, so a mirror during a run changes the code of the next
  stage. Stop the run, mirror and relaunch. The relaunch resumes at the
  first stage whose outputs are missing, and `--start-at` narrows where it
  starts.
- **Memory near the card's limit means a failing run.** Before the
  out-of-memory error, the allocator thrashes and steps slow several-fold. A
  healthy run stays well under the limit.

## 8. Before terminating a rented machine

While the machine runs, copy the picked adapters, every evaluation run and
the logs home as they land. Before terminating it, copy everything else:

```bash
# 1. Every adapter of every run: all checkpoints, the finals, the verifier. One run per call.
rsync -az --exclude vllm_merged --exclude hf_merged --exclude vllm_base --exclude vllm_lora_live \
    --exclude '*.pt' --exclude 'optimizer*' --exclude 'rng_state*' --exclude 'scheduler*' \
    '<box>:Prospector/models/<run>' models/
# 2. The data built on the machine: keeps, mixes, decision states, verifier rows. Copy every
#    provenance sidecar (they are committed), not only those a size pattern matches, then the files.
rsync -az '<box>:Prospector/data/traj/*.provenance.json' data/traj/
rsync -az '<box>:Prospector/data/traj/*<size>*' data/traj/
# 3. Every run directory. Keep the trajectories of the band passes, the rejection and verifier
#    rolls and the 32-sample pool: rebuilding rows or repeating the selection needs them.
rsync -az '<box>:Prospector/data/runs/<prefix>*' data/runs/
# 4. All logs, into a folder per machine. Each round's pick, models/<round>/peak.txt, came
#    with step 1.
rsync -az <box>:Prospector/logs/ logs/<box>/
# 5. The latent-state probe's outputs, when the machine ran `prospector latent`: the labels,
#    the depth profile, the directions, the arms' run directories with their knob.json and
#    each root's mix; the hidden states themselves (the per-episode shards and the per-layer
#    arrays, about 1.7 GB a run) regenerate from the chain's first stage.
rsync -az --exclude '*.npy' --exclude 'shards/' '<box>:Prospector/data/runs/_probe/' data/runs/_probe/
# 6. The compiled causal-conv1d wheel from uv's cache. setup.sh installs it on the next
#    machine with the same torch, CUDA, Python and architecture instead of building for
#    eight minutes; launch/wheels/ is ignored by git.
rsync -az '<box>:~/.cache/uv/sdists-v10/pypi/causal-conv1d/*/*/causal_conv1d-*.whl' launch/wheels/
```

Then compare the two machines, and terminate only when nothing is missing.
On the GPU machine, run `find models -name adapter_model.safetensors`,
`ls data/traj`, `ls data/runs` and `ls logs`, and check each listing against
home. The only expected misses are the probe adapter, the `_smoke`
directories and the `vllm_lora_live` copies.

Commit at home, then terminate from the provider's console. The instance
bills until it is terminated.
