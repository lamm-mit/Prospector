# recipe/: the names, the stages and the runner

The package installs one command, `prospector`, which runs the paper's
training recipe for one graph and one model size. The teacher pass runs on
your own machine. Stages 1 to 4 and the selection run on one CUDA GPU with
80 to 96 GB of memory, which [launch/README.md](../../../launch/README.md)
prepares. This folder is the package behind the command:

| module | what it holds |
|---|---|
| `names.py` | every name a run reads or writes, derived from (graph, size, tag) by `Layout`; `TRAINING`, the six training runs and their files under `configs/`; an adapter's `run.json`; `prospector names` |
| `plan.py` | a run's stages as steps with done tests, computed from the disk; `prospector plan` |
| `run.py` | the jobs a step launches, a round's attempts and watcher, a smoke, and the stage loop; `prospector run` and `smoke` |
| `teach.py`, `verify.py`, `latent.py` | `prospector teach`, `verify` and `latent` |
| `pick.py` | the rule that picks a round's checkpoint |
| `cli.py` | the command line |

## Terms

The recipe uses seven terms of its own:

- A round's *chain* is the adapter, or stack of adapters, it continues
  from, recorded as `substrate_adapters` in its `run.json`.
- A *roll* is four sampled passes of a policy over the training questions,
  at temperature 0.7.
- The *band* is a GRPO round's filter on measured pass rate. The full round
  trains on a question that 5% to 95% of the stage-1 roll's passes solve. A
  decision round trains on a decision state when 5% to 95% of its eight
  resampled continuations pass.
- A *gate* is the evaluation of the adapter after stage 1 or stage 3: three
  greedy seeds on the graph's evaluation questions and, with
  `--transfer-graph`, three on the transfer graph. The run stops unless the
  evaluation questions' mean `gated_acc` reaches the floor (`--floor`) and
  their mean number of calls reaches 11. A stage's gate differs from the
  citation gate in `agent/gate.py`.
- The *grid* is three greedy seeds on the evaluation questions at every
  checkpoint a round saves.
- The *pick* is the checkpoint that `graph_reasoner.recipe.pick` selects
  from a round's evaluations, with `scripts/analysis/ladder_peak.py` as the
  rules' command line. The full round picks the best mean `gated_acc` over
  the grid's three seeds, the later checkpoint on a tie. A decision round
  picks over the grid plus eight seeds on its two leaders: a checkpoint
  whose mean is within 0.03 of the best mean ties with it, and a tie goes to
  more seeds, then to the later checkpoint. These are the rules the released
  rounds were picked by.
- A *coda* is an evaluation of a round's pick beyond the grid: three greedy
  seeds on the transfer graph, and four samples at temperature 0.9 for
  pass@5.

## The commands

```
prospector teach  --graph G [--tag T] [--teacher M] [--shards N] [--dry-run] <shard>... | ids | p2 | mix
prospector plan   --graph G --size S [--tag T] [--model M] [--transfer-graph X] [--kb F] [--start-at STAGE] [--stop-after STAGE] [--floor F] [--force-gate] [--no-names]
prospector run    the same flags but --no-names, plus --dry-run
prospector smoke  --graph G --size S --stage {sft1,sft2,verifier,full,d1,d2} [--tag T] [--model M] [--kb F]
prospector verify --graph G --size S [--model M] [--tag T] [--gpu-frac 0.35]
prospector latent --graph G --size S --transfer-graph X --home-run TAG --bio-run TAG [--out DIR] [--reads-tag T] [--layer N] [--layers A-B] [--seeds ...] [--limit N] [--arms-home ...] [--arms-bio ...] [--start-at] [--stop-after]
prospector names  --graph G [--size S] [--tag T] [--model M] [--kb F] [--transfer-graph X] (KEY | --env | --json)
```

`--graph` names a registered graph, a folder under
`src/graph_reasoner/substrates/` that declares the graph file and the
evaluation and training questions. `--size` picks the base model: `0.8b`,
`2b` and `4b` imply `Qwen/Qwen3.5-<size>`, and another size needs `--model`.
`--tag` names the teacher pass the stages descend from, and defaults to the
paper's, `flash20`. A tag holds one teacher's pass: `prospector teach` takes
the teacher its stored runs and rows record, `google/gemini-3.7-flash` for a
new tag, and refuses a `--teacher` that contradicts the record, and any
teacher but `google/gemini-3.7-flash` under `flash20`, which is named after
it. `--transfer-graph` adds a second graph, evaluated at
every gate and coda. The paper's transfer graph is `biograph`.

`prospector` reads two settings from the environment: `TURN_CAP`, the tokens
a turn may generate in every evaluation (1,024 unless set, the budget of the
released policies), and `TARGET`, a round's last step (120 unless set). Every
evaluation the recipe runs uses the `vllm:` driver, the `plain_chatml`
template, 20 tool calls and `TURN_CAP` tokens a turn. A run name carries no
budget, so a stored read made at another budget is refused rather than
reused. `TARGET` must be a checkpoint both GRPO files save, a multiple of 20
up to 120.
`python -m graph_reasoner.recipe` is the same program.

## Names

`graph_reasoner.recipe` derives every run name, data file, adapter directory
and log of a run from its graph, size and tag.
`prospector names --graph tsmc --size 2b --env` prints the names a run uses.
The naming rule has two tokens: the graph's name unless the graph is `tsmc`, and
the tag unless the tag is `flash20`. Adapter directories carry the tag and
the graph token (`models/traj_sft_2b_flash20g2`,
`models/traj_sft_2b_pfas_flash20g2`), and row files carry the graph
(`data/traj/tsmc_prefixes_q2bd1.jsonl`). A run name starts with `q`, the
size and both tokens, followed by a stage word and a kind
(`q2bd1eval-ck100-s3`, `q2bpfasd1eval-ck100-s3`). Two sizes, graphs or tags
on one machine therefore never share a training run, row file, adapter or
evaluation. The teacher pass's runs carry the tag alone: their directories
differ by graph through the questions' stem, but their logs under `logs/` do
not.

## The stages of a run

`prospector run` executes these stages in order:

| stage | what it does | skipped when |
|---|---|---|
| `sft1` | stage 1: LoRA SFT from the base model on the demonstrations, into `models/traj_sft_<size>_<tag>` | the adapter is saved |
| `gate1` | the gate of the stage-1 adapter (`q<size>sft1eval`, `q<size>sft1bio`); under the floor, exit 3 unless `--force-gate` | every run is scored; the floor is checked either way |
| `band` | the stage-1 adapter's roll (`q<size>fullroll-s831..834`), which measures the full round's band | the full round is on disk |
| `round1` | stage 2, the full-episode round: its three-step smoke, the supervised 120-step round, the grid, the pick, three more seeds and the codas on the pick, and `peak.txt` | the smoke and the round, when the round is on disk; each evaluation, when scored |
| `rftroll` | a roll of the stage-1 adapter plus round 1's pick (`q<size>g2roll-s861..864`) | the stage-3 adapter is saved, or the stage-3 mix exists |
| `rftbuild` | the kept episodes (`gated_acc`, cap 2 an item, grounded declines 0.1, the evaluation questions excluded), blended with the demonstrations into the stage-3 mix | as `rftroll` |
| `sft2` | stage 3: LoRA SFT from the base model on the mix | the adapter is saved |
| `gate2` | the gate of the stage-3 adapter, as `gate1` | as `gate1` |
| `d1` | stage 4's first decision round, on the stage-3 adapter: a roll, the decision states (rebuilt at every launch until the round is on disk), the band (eight continuations a state), the smoke, the round, the grid, eight seeds on the two best checkpoints, the pick and its codas, and `peak.txt` | the round on disk skips everything before the grid; an existing pick is final, so its leaders are not re-read and its codas are read on it; each evaluation, when scored |
| `d2` | the second decision round, whose chain is the stage-3 adapter and `d1`'s pick | as `d1` |
| `final` | the served stack's roll for the verifier, the 32-sample pool over the evaluation questions, the gold-free vote and the ceiling, and the stage-by-stage means | the roll, when the verifier rows exist; each evaluation, when scored |
| `rows` | the verifier's yes/no rows from the roll, labeled by `gated_acc`, with the evaluation questions excluded | the rows exist |
| `verifier` | the verifier: LoRA SFT from the base model on the rows | the adapter is saved |
| `score` | the pool scored by margin with the verifier, then the vote, the ceiling and the verifier's pick over the same 32 | the scoring run exists |

## Resuming and retraining

A stage recomputes its steps from the disk after each one, so a relaunch
resumes where the files stop. `--start-at` and `--stop-after` take a slice
of the stages. The runner does not retrain a round whose directory holds
its target checkpoint or its recorded pick. The runner retrains any other
round from the start, after removing the round's directory and its grid and
coda runs.

Within one launch, the runner starts a round's trainer up to eight times,
and each attempt after a death resumes from the last checkpoint. A watcher
looks for the target checkpoint every minute and stops the trainer a minute
after it appears. Stop a launch with SIGINT or SIGTERM (`kill <pid>` of the
`prospector` process), and it kills the trainer it is watching; `kill -9`
bypasses that, and the orphaned trainer keeps writing into a directory the
next launch removes. A trainer that refuses its run is not relaunched: an
input is missing, or its directory holds another run's manifest.

The full round's pick is made once, over the grid's three seeds. A decision
round picks after eight seeds on its two best checkpoints. Each pick is
recorded in `models/<round dir>/peak.txt` after its codas. Every later stage
reads that file and refuses a pick whose checkpoint is not on disk.

To retrain everything at a size, move aside everything its stages take as
done, not only the adapters. That is the released
`models/traj_sft_<size>_flash20{,g2}`,
`models/traj_grpo_<size>_flash20{full,d1,d2}` and
`models/verifier_sft_<size>_flash20v`, the size's stored runs
`data/runs/<P>*`, and its three row files, which
`prospector names --graph tsmc --size <size> MIX2`, `G2_KEEPS` and `ROWS`
print. `<P>` is the run prefix that `prospector names ... P` prints, such as
`q2b`. Otherwise the gates are judged on the stored reads, the third stage
trains on the released mix and the verifier on the released rows. The
decision states are rebuilt at every launch.

## Exit codes and logs

A failed stage ends the run with exit 1 and a `!!!` line. A gate under the
floor exits 3. A wrong argument or a missing prerequisite exits 2 before
anything runs: an unknown graph, size, stage, flag or value, a recorded pick
whose checkpoint is not on disk, a `TARGET` no round saves, or a stored read
made at another `TURN_CAP`.

The run's stdout is its log. The runner prints every command before running
it, so a step can be rerun by hand from the log. Each job writes its own log
under `logs/`. In the names below, `<P>` is the run prefix (`q2b`), `<round>` a
round's run prefix (`q2bd1`) and `<round dir>` its directory's name:

| log | written by |
|---|---|
| `<run name>.log` | an evaluation |
| `<P>sft1_train.log`, `<P>g2_train.log`, `<P>_verifier_train.log` | the three SFT runs |
| `<round>_r_a<n>.log` | a round's attempts |
| `<round>_r_watermarks.log` | a round's GPU memory, every five seconds |
| `<round dir>_smoke.log` | a smoke |
| `<round>_stage0.log` | a decision round's band |
| `<P>_rftbuild.log`, `<P>_verifier_rows.log`, `<P>_verifier_select.log` | the builders |

## The environment

Every job runs from the repository root, under the interpreter that runs
`prospector` (`.venv`'s on a CUDA machine), with the environment that
[launch/README.md](../../../launch/README.md) lists in section 4. Before every
engine start, the runner kills the vLLM engines orphaned to PID 1, which
hold most of the card (section 7).

## The released sizes

The released adapters sit under the names the recipe derives, and each
released round carries its recorded pick, such as
`models/traj_grpo_<size>_flash20d1/peak.txt`. With the released adapters in
place, a run trains nothing. The stored 0.8B and 2B runs carry the derived
names, so a run at those sizes skips the stored evaluations it finds. The
released 4B's runs under `data/runs/` carry older names (`gen1sft1k`,
`full1k`, `gen2sft1k`, `flash20cap1024`, `flash20pool34`,
`flash20-verifier32-yn1a3`), so at 4B every gate, grid, coda and pool runs
again under the `q4b` names:

```bash
nohup prospector run --graph tsmc --size 4b --transfer-graph biograph > logs/q4b_run.log 2>&1 < /dev/null &
```

## The builders by hand

The training runs and the data builders between them also run by hand,
with the arguments the stored files were built with. The names in angle
brackets are placeholders, and each script's docstring lists its flags:

```bash
python scripts/train_sft.py --graph tsmc --size 4b --stage sft1                  # sft1, sft2 or verifier; --dry-run prints the resolved run
python scripts/train_traj_grpo.py --graph tsmc --size 4b --stage full         # full, d1 or d2; --smoke trains three steps
python scripts/traj/build_sft_rows.py --rollouts data/runs/<pass> ... --exclude data/benchmarks/tsmc/tsmc_core_agentic.jsonl --cap 2 --abstain-ratio 0.1 --out data/traj/tsmc_<size>_flash20g2_keeps.jsonl
python scripts/traj/build_substrate_mix.py --demos data/traj/tsmc_sft_mix_flash20.jsonl --keeps data/traj/tsmc_<size>_flash20g2_keeps.jsonl --keeps-name <size>_g2_keeps --out data/traj/tsmc_sft_mix_<size>_flash20g2.jsonl
python scripts/traj/build_prefixes.py --rollouts data/runs/<roll> ... --exclude data/benchmarks/tsmc/tsmc_core_agentic.jsonl --kb data/raw/GRAPHDATA_TSMC_OUTPUT/tsmc_5b10p.graphml --success-share 0.25 --seed 7 --out data/traj/tsmc_prefixes_q<size>d1.jsonl
python scripts/traj/sample_prefixes.py --prefixes data/traj/tsmc_prefixes_q<size>d1.jsonl --benchmark data/benchmarks/tsmc/tsmc_train_agentic_4x.jsonl --model vllm:Qwen/Qwen3.5-<size> --adapter <stack> --temperature 0.7 --n 8 --seed 911 --name q<size>d1-s911
python scripts/select/build_verifier_data.py --rollouts data/runs/<pass> ... --exclude data/benchmarks/tsmc/tsmc_core_agentic.jsonl --out data/traj/tsmc_verifier_yesno_<size>_flash20v.jsonl
```

The trainers take `--stage` and the four flags `--graph`, `--size`, `--tag`
and `--model`. `--config FILE` replaces the stage's hyperparameter file.
`prospector names --graph tsmc --size 4b --env` prints the names the
builders' arguments take.

`prospector teach mix` builds the demonstrations with
`build_sft_rows.py --cap 1 --strip-user-note`, `build_error_pairs.py` and
`build_substrate_mix.py --keeps ... --detours ...`. The teacher runs under
the policy's prompt plus one instruction appended with `--user-note`: to
write one or two sentences of reasoning before each call. `--strip-user-note`
removes that instruction from every row, so the student trains on the prompt
it is evaluated under.

## The stored files

The rows a stage trains from are stage 1's `tsmc_sft_mix_flash20.jsonl`,
which the sizes share, and per size `tsmc_sft_mix_<size>_flash20g2.jsonl`,
`tsmc_prefixes_q<size>d1.jsonl`, `tsmc_prefixes_q<size>d2.jsonl` and
`tsmc_verifier_yesno_<size>_flash20v.jsonl`. `prospector names` prints their
paths. The builders above also write the intermediates these rows are
blended from, the kept episodes and the detours.

A `.provenance.json` sidecar beside a row file records how the file was
built. A sidecar carries `moved_from` when its file was renamed to the
recipe's name. The detour rows and the verifier rows have no sidecar.
The 0.8B's and 2B's kept episodes, mixes and decision states were rebuilt
from their stored passes by these commands, which rebuild the 4B's files
byte for byte. The 2B's mix and kept episodes keep their original sidecars,
whose counts the rebuild matches; the other sidecars record the rebuild.
The 4B's verifier rows come from the four passes `flash20rftroll1k-s841..844`.
The documented commands rebuild the same rows, apart from three details of
the stored files. Three keeps sidecars record a `--v2` flag and a
`v2` field that `build_sft_rows.py` does not take or write. The stage-1 mix's sidecar lists two empty
components. The verifier rows carry an `answer_chars` field that
`build_verifier_data.py` does not write.

Some stored runs under `data/runs/` carry no `turn_max_tokens` under
`policy` in `summary.json`. `cap1024` in a run's name marks the 1,024-token
budget, as in the served policy's runs `flash20cap1024-s0` to `-s7`.
`prospector teach mix` rebuilds the demonstrations from the nine teacher
runs, `flash20-teach-4x-g0` to `-g7` and `-p2`.

The latent-state probe writes its derived files under `data/runs/_probe/`:
the labels, the depth profile, the directions, the arms' run directories
with their `knob.json`, and each root's `mix.json`. The `states` stage
regenerates the latent-state arrays (`L<n>.npy`). The release's own manifest
lists which stored files it ships.
