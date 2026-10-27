"""Every name the training recipe uses, derived from (graph, size, tag).

A training run is named by three things: the graph it trains on (a
registered substrate), the model size (which picks the base model), and a
tag, the name of the teacher pass whose demonstrations every stage descends
from. From those, `Layout` gives the path of every artifact a stage reads or
writes, the prefix of every run name, and the hyperparameter file of every
stage, so no configuration and no command spells a path. The command
line prints the same names (`prospector names --graph G --size S --env`),
the trainers read them through `training_config`, and a trained adapter records the names it was built from in `run.json`
(`write_manifest`).

The rule, in two tokens. The release's defaults are spelled as the empty
string: `g` is the graph's name unless the graph is `tsmc`, `t` is the tag
unless the tag is `flash20`. Directories carry the tag always and the graph
token with an underscore (`models/traj_sft_2b_flash20g2`,
`models/traj_sft_2b_pfas_flash20g2`); row files carry the graph always
(`data/traj/tsmc_sft_mix_flash20.jsonl`); run prefixes are `q` + size + g + t
(`q2b`, `q4b`, `q2bpfas`, `q2bexp1`), followed by a stage word and a kind
(`q2bd1eval-ck100-s3`). This reproduces every released 0.8B and 2B name and
the 4B's adapters and rows (the 4B's stored runs carry older names, such as
`flash20cap1024` and `flash20pool34`), and gives every other graph, size and
tag names of its own.

No torch here, and the substrate registry is imported inside functions.
This module is the names; `plan.py` and `run.py` beside it are the runner.
Tests: tests/test_recipe.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NoReturn

import yaml

from ..benchmarks import ROOT, run_dir_name

class Refusal(SystemExit):
    """A wrong argument or a missing prerequisite, refused before anything
    runs: exit 2, with the message already printed to stderr by `refuse`."""
    def __init__(self, message: str):
        super().__init__(2)
        self.message = message

    def __str__(self) -> str:
        return self.message


def refuse(message: str) -> NoReturn:
    """Print `message` to stderr and exit 2: the names' refusals (a bad --graph,
    --size or --tag, a missing or unusable pick) and the plan's (a bad stage
    slice, a TARGET no round saves, a read at another TURN_CAP), whether the
    runner or a trainer script resolved them."""
    print(message, file=sys.stderr)
    raise Refusal(message)


DEFAULT_GRAPH = "tsmc"      # the released graph: its token in a name is empty
DEFAULT_TAG = "flash20"     # the released teacher pass: its token in a run prefix is empty
BASE_MODELS = {"0.8b": "Qwen/Qwen3.5-0.8B", "2b": "Qwen/Qwen3.5-2B", "4b": "Qwen/Qwen3.5-4B"}
SIZE_RE = re.compile(r"^[a-z0-9.]+$")
TAG_RE = re.compile(r"^[a-z0-9]+$")     # no '-' (the run separator) and no '_' (the file separator)

#: The seeds the recipe draws: the four sampled passes a band is measured on,
#: the four passes stage 3 keeps episodes from, the band of a decision round,
#: the gate reads, and the 32-sample pool.
BAND_SEEDS = (831, 832, 833, 834)
G2_ROLL_SEEDS = (861, 862, 863, 864)
PREFIX_BAND_SEED = 911
GATE_SEEDS = (0, 1, 2)
POOL_SEEDS = tuple(range(101, 133))
#: The token a transfer graph contributes to a run name.
TRANSFER_TOKENS = {"biograph": "bio"}

MANIFEST = "run.json"
MANIFEST_SCHEMA = "recipe_run_v1"


# ---------------------------------------------------------------- the spec

@dataclass(frozen=True)
class Spec:
    """What names a run: the graph, the size, the tag, and the base model
    when the size does not imply one."""
    graph: str
    size: str | None = None
    tag: str = DEFAULT_TAG
    model: str | None = None

    def __post_init__(self):
        if self.size is not None and not SIZE_RE.match(self.size):
            refuse(f"--size {self.size!r}: lower-case letters, digits and dots only")
        if not TAG_RE.match(self.tag):
            refuse(f"--tag {self.tag!r}: lower-case letters and digits only "
                             f"('-' separates a run from its benchmark, '_' the parts of a file name)")
        # The model a size implies, passed explicitly, is the same spec: the
        # runner (`run.trainer_args`) always passes --model, a user rarely does,
        # and both must write the same manifest.
        if self.model and self.model == BASE_MODELS.get(self.size or ""):
            object.__setattr__(self, "model", None)

    @property
    def base_model(self) -> str:
        if self.model:
            return self.model
        if self.size in BASE_MODELS:
            return BASE_MODELS[self.size]
        refuse(f"--size {self.size!r} implies no base model (the sizes that do: "
                         f"{', '.join(BASE_MODELS)}); pass --model <hf id>")

    @property
    def graph_token(self) -> str:
        return "" if self.graph == DEFAULT_GRAPH else self.graph

    @property
    def tag_token(self) -> str:
        return "" if self.tag == DEFAULT_TAG else self.tag

    @property
    def lineage(self) -> str:
        """The token after the size in a directory name: the tag, preceded by
        the graph token and an underscore for every graph but the released one."""
        return f"{self.graph_token}_{self.tag}" if self.graph_token else self.tag

    @property
    def run_prefix(self) -> str:
        """`q<size><graph token><tag token>`, the start of every run name."""
        if self.size is None:
            refuse("a run name needs --size")
        return f"q{self.size}{self.graph_token}{self.tag_token}"


@dataclass(frozen=True)
class Graph:
    """A registered graph and its three files."""
    name: str
    kb: str
    eval_benchmark: str
    train_benchmark: str

    @classmethod
    def of(cls, name: str, kb: str | None = None) -> "Graph":
        from ..agent.backend import available_backends, eval_benchmark, resolve_kb, train_benchmark

        if name not in available_backends():
            refuse(f"--graph {name!r} is not a registered substrate "
                             f"(registered: {', '.join(available_backends())}); a graph is a "
                             f"folder under src/graph_reasoner/substrates/")
        return cls(name, resolve_kb(name, kb), eval_benchmark(name), train_benchmark(name))

    @property
    def eval_stem(self) -> str:
        return Path(self.eval_benchmark).stem

    @property
    def train_stem(self) -> str:
        return Path(self.train_benchmark).stem

    @property
    def transfer_token(self) -> str:
        return TRANSFER_TOKENS.get(self.name, self.name)


# ---------------------------------------------------------------- the training runs

@dataclass(frozen=True)
class Training:
    """One of the recipe's six training runs: its trainer and its hyperparameter file."""
    name: str
    trainer: str        # "sft" or "grpo"
    config: str         # the hyperparameter file


TRAINING = {
    "sft1": Training("sft1", "sft", "configs/sft1.yaml"),
    "full": Training("full", "grpo", "configs/grpo_full.yaml"),
    "sft2": Training("sft2", "sft", "configs/sft2.yaml"),
    "d1": Training("d1", "grpo", "configs/grpo_decision.yaml"),
    "d2": Training("d2", "grpo", "configs/grpo_decision.yaml"),
    "verifier": Training("verifier", "sft", "configs/verifier.yaml"),
}
#: The band each GRPO stage's hyperparameter file carries as `{lo, hi}`; the
#: recipe adds its `rollouts`.
BAND_KEY = {"full": "informative_band", "d1": "prefix_band", "d2": "prefix_band"}


def _runs(prefix: str, seeds, benchmark: str) -> list[str]:
    return [f"data/runs/{run_dir_name(f'{prefix}-s{s}', benchmark)}" for s in seeds]


# ---------------------------------------------------------------- the layout

@dataclass(frozen=True)
class Layout:
    """The names of one (graph, size, tag), as repo-relative strings."""
    spec: Spec
    graph: Graph
    transfer: Graph | None = None
    root: Path = ROOT

    @classmethod
    def of(cls, spec: Spec, *, kb: str | None = None, transfer: str | None = None,
           root: Path = ROOT) -> "Layout":
        return cls(spec, Graph.of(spec.graph, kb), Graph.of(transfer) if transfer else None, root)

    # -- the parts of every name
    @property
    def P(self) -> str:            # noqa: N802 -- the run prefix, `P` in `prospector names`
        return self.spec.run_prefix

    @property
    def _S(self) -> str:
        if self.spec.size is None:
            refuse("this name needs --size")
        return self.spec.size

    @property
    def _lin(self) -> str:
        return self.spec.lineage

    @property
    def _G(self) -> str:
        return self.graph.name

    @property
    def _T(self) -> str:
        return self.spec.tag

    # -- stage 0: the teacher pass (size-free)
    @property
    def demos_keeps(self) -> str:
        return f"data/traj/{self._G}_{self._T}_keeps.jsonl"

    @property
    def demos_keeps_pass1(self) -> str:
        return f"data/traj/{self._G}_{self._T}_keeps_pass1.jsonl"

    @property
    def demos_failed_ids(self) -> str:
        return f"data/traj/{self._G}_{self._T}_pass1_failed_ids.txt"

    @property
    def demos_detours(self) -> str:
        return f"data/traj/{self._G}_error_pairs_{self._T}.jsonl"

    @property
    def demos_mix(self) -> str:
        return f"data/traj/{self._G}_sft_mix_{self._T}.jsonl"

    @property
    def teach_run_prefix(self) -> str:
        """`<tag>-teach`, plus `-<suffix>` when the training file's stem
        extends `<graph>_train_agentic_` (the released `flash20-teach-4x`)."""
        head = f"{self._G}_train_agentic_"
        stem = self.graph.train_stem
        suffix = stem[len(head):] if stem.startswith(head) else ""
        return f"{self._T}-teach" + (f"-{suffix}" if suffix else "")

    # -- stage 1
    @property
    def sft1_dir(self) -> str:
        return f"models/traj_sft_{self._S}_{self._lin}"

    @property
    def sft1(self) -> str:
        return f"{self.sft1_dir}/final"

    @property
    def gate1_prefix(self) -> str:
        return f"{self.P}sft1"

    # -- stage 2
    @property
    def full_prefix(self) -> str:
        return f"{self.P}full"

    @property
    def full_dir(self) -> str:
        return f"models/traj_grpo_{self._S}_{self._lin}full"

    @property
    def full_roll_prefix(self) -> str:
        return f"{self.P}fullroll"

    @property
    def full_band_runs(self) -> list[str]:
        return _runs(self.full_roll_prefix, BAND_SEEDS, self.graph.train_benchmark)

    # -- stage 3
    @property
    def g2_prefix(self) -> str:
        return f"{self.P}g2"

    @property
    def g2_roll_prefix(self) -> str:
        return f"{self.P}g2roll"

    @property
    def g2_roll_runs(self) -> list[str]:
        return _runs(self.g2_roll_prefix, G2_ROLL_SEEDS, self.graph.train_benchmark)

    @property
    def g2_keeps(self) -> str:
        return f"data/traj/{self._G}_{self._S}_{self._T}g2_keeps.jsonl"

    @property
    def g2_keeps_name(self) -> str:
        return f"{self._S}_g2_keeps"

    @property
    def sft2_mix(self) -> str:
        return f"data/traj/{self._G}_sft_mix_{self._S}_{self._T}g2.jsonl"

    @property
    def sft2_dir(self) -> str:
        return f"models/traj_sft_{self._S}_{self._lin}g2"

    @property
    def sft2(self) -> str:
        return f"{self.sft2_dir}/final"

    # -- stage 4
    def round(self, d: str) -> dict:
        """The names of decision round `d` ("d1" or "d2"): prefix, dir,
        roll_prefix, roll_runs, prefixes, band_name, band_run."""
        prefix = f"{self.P}{d}"
        prefixes = f"data/traj/{self._G}_prefixes_{prefix}.jsonl"
        band_name = f"{prefix}-s{PREFIX_BAND_SEED}"
        return {"prefix": prefix, "dir": f"models/traj_grpo_{self._S}_{self._lin}{d}",
                "roll_prefix": f"{prefix}roll",
                "roll_runs": _runs(f"{prefix}roll", BAND_SEEDS, self.graph.train_benchmark),
                "prefixes": prefixes, "band_name": band_name,
                "band_run": f"data/runs/{run_dir_name(band_name, prefixes)}"}

    @property
    def d1_prefix(self) -> str: return self.round("d1")["prefix"]
    @property
    def d1_dir(self) -> str: return self.round("d1")["dir"]
    @property
    def d1_roll_prefix(self) -> str: return self.round("d1")["roll_prefix"]
    @property
    def d1_roll_runs(self) -> list[str]: return self.round("d1")["roll_runs"]
    @property
    def d1_prefixes(self) -> str: return self.round("d1")["prefixes"]
    @property
    def d1_band_name(self) -> str: return self.round("d1")["band_name"]
    @property
    def d1_band_run(self) -> str: return self.round("d1")["band_run"]
    @property
    def d2_prefix(self) -> str: return self.round("d2")["prefix"]
    @property
    def d2_dir(self) -> str: return self.round("d2")["dir"]
    @property
    def d2_roll_prefix(self) -> str: return self.round("d2")["roll_prefix"]
    @property
    def d2_roll_runs(self) -> list[str]: return self.round("d2")["roll_runs"]
    @property
    def d2_prefixes(self) -> str: return self.round("d2")["prefixes"]
    @property
    def d2_band_name(self) -> str: return self.round("d2")["band_name"]
    @property
    def d2_band_run(self) -> str: return self.round("d2")["band_run"]

    # -- the selection stage
    @property
    def v_roll_prefix(self) -> str:
        return f"{self.P}vroll"

    @property
    def v_roll_runs(self) -> list[str]:
        return _runs(self.v_roll_prefix, BAND_SEEDS, self.graph.train_benchmark)

    @property
    def verifier_rows(self) -> str:
        return f"data/traj/{self._G}_verifier_yesno_{self._S}_{self._T}v.jsonl"

    @property
    def verifier_dir(self) -> str:
        return f"models/verifier_sft_{self._S}_{self._lin}v"

    @property
    def verifier(self) -> str:
        return f"{self.verifier_dir}/final"

    @property
    def pool_prefix(self) -> str:
        return f"{self.P}pool32"

    @property
    def pool_runs(self) -> list[str]:
        return _runs(self.pool_prefix, POOL_SEEDS, self.graph.eval_benchmark)

    @property
    def verifier_select_name(self) -> str:
        return f"{self.P}-verifier32-yn1"

    # -- logs
    @property
    def log_prefix(self) -> str:
        return f"logs/{self.P}"

    # -- the picks, read from the files the ladder writes
    def pick_problem(self, round_dir: str) -> str | None:
        """Why `round_dir/peak.txt` names no usable pick, or None when it does."""
        path = self.root / round_dir / "peak.txt"
        if not path.exists():
            return (f"no pick in {round_dir}/peak.txt: the round's grid and pick have "
                    f"not run (prospector run writes it)")
        text = path.read_text().strip()
        if not text.isdigit() or int(text) <= 0:
            return f"{round_dir}/peak.txt holds {text!r}, not a checkpoint number"
        if not (self.root / round_dir / f"checkpoint-{text}").is_dir():
            return f"{round_dir}/peak.txt names checkpoint-{text}, which is not on disk"
        return None

    def pick(self, round_dir: str) -> int:
        """The checkpoint `round_dir/peak.txt` names (the round's pick); a
        missing or unusable pick is refused."""
        problem = self.pick_problem(round_dir)
        if problem:
            refuse(problem)
        return int((self.root / round_dir / "peak.txt").read_text().strip())

    def has_pick(self, round_dir: str) -> bool:
        """Whether the round has a usable pick; prints nothing."""
        return self.pick_problem(round_dir) is None

    def full_chain(self) -> list[str]:
        return [self.sft1, f"{self.full_dir}/checkpoint-{self.pick(self.full_dir)}"]

    def d2_substrate(self) -> list[str]:
        return [self.sft2, f"{self.d1_dir}/checkpoint-{self.pick(self.d1_dir)}"]

    def served_stack(self) -> list[str]:
        return [self.sft2, f"{self.d1_dir}/checkpoint-{self.pick(self.d1_dir)}",
                f"{self.d2_dir}/checkpoint-{self.pick(self.d2_dir)}"]

    # -- what a stage trains from
    def resolved(self, stage: str) -> dict:
        """The path keys the trainer of `stage` reads, with the base model;
        for a GRPO stage the band's rollouts come as `band_rollouts`, which
        `training_config` folds into the file's band."""
        base = self.spec.base_model
        if stage == "sft1":
            return {"base_model": base, "dataset": self.demos_mix, "output_dir": self.sft1_dir}
        if stage == "sft2":
            return {"base_model": base, "dataset": self.sft2_mix, "output_dir": self.sft2_dir}
        if stage == "verifier":
            return {"base_model": base, "dataset": self.verifier_rows,
                    "output_dir": self.verifier_dir}
        common = {"base_model": base, "benchmark": self.graph.train_benchmark,
                  "exclude": [self.graph.eval_benchmark], "backend": self.graph.name,
                  "kb": self.graph.kb}
        if stage == "full":
            return {**common, "output_dir": self.full_dir, "substrate_adapters": [self.sft1],
                    "rollout_mode": "episode", "band_rollouts": self.full_band_runs}
        if stage == "d1":
            return {**common, "output_dir": self.d1_dir, "substrate_adapters": [self.sft2],
                    "rollout_mode": "prefix", "prefixes": self.d1_prefixes,
                    "band_rollouts": [self.d1_band_run]}
        if stage == "d2":
            return {**common, "output_dir": self.d2_dir, "substrate_adapters": self.d2_substrate(),
                    "rollout_mode": "prefix", "prefixes": self.d2_prefixes,
                    "band_rollouts": [self.d2_band_run]}
        refuse(f"unknown training run {stage!r} (the runs: {', '.join(TRAINING)})")

    def check_inputs(self, stage: str) -> list[str]:
        """The inputs of `stage` that are not on disk, as `what: path` lines."""
        from ..train.adapters import is_adapter_dir

        r = self.resolved(stage)
        missing = []

        def need(what, path, probe=None):
            p = self.root / path / probe if probe else self.root / path
            if not p.exists():
                missing.append(f"{what}: {path}")

        if "dataset" in r:
            need("dataset", r["dataset"])
            return missing
        for a in r["substrate_adapters"]:
            if not is_adapter_dir(self.root / a):
                missing.append(f"adapter: {a}")
        need("benchmark", r["benchmark"])
        for e in r["exclude"]:
            need("exclude", e)
        need("graph", r["kb"])
        if "prefixes" in r:
            need("prefixes", r["prefixes"])
        for b in r["band_rollouts"]:
            need("band run", b, "results.jsonl")
        return missing

    # -- the shell's view
    def env(self) -> dict[str, str]:
        """Every static name under its `prospector names` key, which `--env`
        prints as a shell export; the size-dependent ones only when the spec
        has a size; the picks only when on disk."""
        g, x = self.graph, self.transfer
        out = {
            "GRAPH": g.name, "TAG": self._T, "EVAL_BENCH": g.eval_benchmark,
            "EVAL_STEM": g.eval_stem, "TRAIN_BENCH": g.train_benchmark,
            "TRAIN_STEM": g.train_stem, "KB": g.kb,
            # TRANSFER is an input, exported with the others, so a shell that
            # evals --env derives the same names.
            "TRANSFER": x.name if x else "",
            "XFER": x.transfer_token if x else "", "XFER_GRAPH": x.name if x else "",
            "XFER_BENCH": x.eval_benchmark if x else "", "XFER_STEM": x.eval_stem if x else "",
            "XFER_KB": x.kb if x else "",
            "DEMOS": self.demos_mix, "DEMO_KEEPS": self.demos_keeps,
            "DEMO_KEEPS1": self.demos_keeps_pass1, "DEMO_FAILED": self.demos_failed_ids,
            "DETOURS": self.demos_detours, "TEACH": self.teach_run_prefix,
            "BAND_SEEDS": " ".join(map(str, BAND_SEEDS)),
            "G2_ROLL_SEEDS": " ".join(map(str, G2_ROLL_SEEDS)),
            "GATE_SEEDS": " ".join(map(str, GATE_SEEDS)),
            "POOL_SEEDS": " ".join(map(str, POOL_SEEDS)),
            "PREFIX_BAND_SEED": str(PREFIX_BAND_SEED),
        }
        if self.spec.size is None:
            return out
        out.update({
            "SIZE": self._S, "MODEL": self.spec.base_model, "P": self.P,
            "SFT1_DIR": self.sft1_dir, "SFT1": self.sft1, "SFT1_TAG": self.gate1_prefix,
            "FULL_TAG": self.full_prefix, "FULL_V": self.full_dir, "FULLROLL": self.full_roll_prefix,
            "G2_TAG": self.g2_prefix, "G2ROLL": self.g2_roll_prefix, "G2_KEEPS": self.g2_keeps,
            "G2_KEEPS_NAME": self.g2_keeps_name, "MIX2": self.sft2_mix,
            "SFT2_DIR": self.sft2_dir, "SFT2": self.sft2,
            "D1_TAG": self.d1_prefix, "D1_V": self.d1_dir, "D1ROLL": self.d1_roll_prefix,
            "PREFIXES_D1": self.d1_prefixes, "D1_BAND": self.d1_band_name, "D1_BAND_RUN": self.d1_band_run,
            "D2_TAG": self.d2_prefix, "D2_V": self.d2_dir, "D2ROLL": self.d2_roll_prefix,
            "PREFIXES_D2": self.d2_prefixes, "D2_BAND": self.d2_band_name, "D2_BAND_RUN": self.d2_band_run,
            "VROLL": self.v_roll_prefix, "ROWS": self.verifier_rows,
            "VER_DIR": self.verifier_dir, "VER": self.verifier,
            "POOL": self.pool_prefix, "VSEL": self.verifier_select_name,
            "LOG_PREFIX": self.log_prefix,
            "FULL_PICK": str(self.pick(self.full_dir)) if self.has_pick(self.full_dir) else "",
            "D1_PICK": str(self.pick(self.d1_dir)) if self.has_pick(self.d1_dir) else "",
            "D2_PICK": str(self.pick(self.d2_dir)) if self.has_pick(self.d2_dir) else "",
        })
        return out

    def value(self, key: str) -> str:
        """One name by its shell variable (any case) or attribute; lists are
        comma-joined. `stack`, `full_chain` and `d2_substrate` read the picks."""
        special = {"stack": self.served_stack, "served_stack": self.served_stack,
                   "full_chain": self.full_chain, "d2_substrate": self.d2_substrate}
        if key.lower() in special:
            return ",".join(special[key.lower()]())
        env = self.env()
        if key.upper() in env:
            return env[key.upper()]
        if not key.startswith("_") and hasattr(type(self), key) and isinstance(getattr(type(self), key), property):
            v = getattr(self, key)
            return ",".join(v) if isinstance(v, list) else str(v)
        refuse(f"unknown key {key!r}; the keys: {', '.join(sorted(env))}, "
                         f"stack, full_chain, d2_substrate")


# ---------------------------------------------------------------- the training config

def load_hyperparameters(path: str | Path, trainer: str) -> dict:
    """A stage's hyperparameter file, refused when it carries a key the
    recipe derives or one the trainer does not read."""
    from ..train.config import check_hyperparameter_keys

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if trainer == "sft":
        from ..train.sft import SFT_CONFIG_KEYS, SFT_RESOLVED_KEYS
        check_hyperparameter_keys(cfg, known=SFT_CONFIG_KEYS, resolved=SFT_RESOLVED_KEYS,
                                  known_name="SFT_CONFIG_KEYS", path=str(path))
    else:
        from ..train.grpo import GRPO_RESOLVED_KEYS, KNOWN_CONFIG_KEYS
        check_hyperparameter_keys(cfg, known=KNOWN_CONFIG_KEYS, resolved=GRPO_RESOLVED_KEYS,
                                  known_name="KNOWN_CONFIG_KEYS", path=str(path))
    return cfg


def smoke_config(config: dict) -> dict:
    """The three-step twin of a config: `max_steps 3`, `save_steps 100`, and
    the output directory suffixed `_smoke`, so a smoke never touches a run."""
    return {**config, "max_steps": 3, "save_steps": 100,
            "output_dir": config["output_dir"].rstrip("/") + "_smoke"}


def sha256_of(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_config(layout: Layout, stage: str, *, config_path: str | Path | None = None,
                    smoke: bool = False, argv: list[str] | None = None) -> tuple[dict, dict]:
    """(the dict the trainer consumes, the manifest to write beside the run).

    The dict is the stage's hyperparameter file plus the resolved paths, in
    the shape `run_sft` and `run_grpo` read; a GRPO stage's band gets its
    `rollouts` here. `config_path` replaces the stage's file for an experiment.
    """
    from ..provenance import env_state, git_state

    st = TRAINING[stage]
    path = Path(config_path) if config_path else layout.root / st.config
    hp = load_hyperparameters(path, st.trainer)
    resolved = layout.resolved(stage)
    config = dict(hp)
    for k, v in resolved.items():
        if k == "band_rollouts":
            band = BAND_KEY[stage]
            config[band] = {**dict(hp.get(band) or {}), "rollouts": list(v)}
        else:
            config[k] = v
    if smoke:
        config = smoke_config(config)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "spec": asdict(layout.spec),
        "graph": asdict(layout.graph),
        "stage": stage, "trainer": st.trainer,
        "base_model": layout.spec.base_model,
        "output_dir": config["output_dir"],
        "resolved": {k: v for k, v in resolved.items() if k != "base_model"},
        "hyperparameters": hp,
        "config_file": {"path": str(path.relative_to(layout.root)) if path.is_relative_to(layout.root)
                        else str(path), "sha256": sha256_of(path)},
        "smoke": bool(smoke),
        "source": "trainer",
        "launch": {"argv": list(argv) if argv is not None else sys.argv[1:],
                   "git": git_state(), "env": env_state()},
        "relaunches": [],
    }
    return config, manifest


_IDENTITY = ("spec", "graph", "stage", "base_model", "output_dir", "resolved", "hyperparameters", "smoke")


def write_manifest(out_dir: str | Path, manifest: dict) -> Path:
    """Write `run.json` into the run directory. A manifest already there with
    the same inputs records this launch under `relaunches` (a resume); one
    with other inputs is refused, since the directory holds another run."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / MANIFEST
    if path.exists():
        have = json.loads(path.read_text(encoding="utf-8"))
        for key in _IDENTITY:
            if have.get(key) != manifest.get(key):
                refuse(
                    f"{out} holds a run trained from other inputs ({key} differs: "
                    f"{json.dumps(have.get(key))[:120]} there, {json.dumps(manifest.get(key))[:120]} "
                    f"now); move that directory aside")
        have.setdefault("relaunches", []).append(manifest["launch"])
        manifest = have
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_manifest(adapter_path: str | Path) -> dict | None:
    """The manifest of the run that holds `adapter_path` (a run directory,
    its `final/` or one of its `checkpoint-N/`), or None."""
    p = Path(adapter_path)
    if p.name == "final" or p.name.startswith("checkpoint-"):
        p = p.parent
    path = p / MANIFEST
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- the CLI

def add_spec_arguments(ap: argparse.ArgumentParser, *, size_required: bool = True) -> None:
    """The four flags that name a run, for every recipe command but `teach`
    (which takes --graph and --tag alone) and for the two trainer scripts."""
    ap.add_argument("--graph", required=True, help="a registered substrate (the folder under "
                    "src/graph_reasoner/substrates/)")
    ap.add_argument("--size", required=size_required,
                    help="the model size: 0.8b, 2b or 4b imply the base model; another needs --model")
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help=f"the teacher pass the stages descend from (default {DEFAULT_TAG})")
    ap.add_argument("--model", default=None, help="the base model, when --size does not imply one")


def spec_from_args(args) -> Spec:
    return Spec(args.graph, args.size, args.tag, args.model)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="prospector names",
        description="Print the names the recipe derives from --graph, --size and --tag: one KEY, "
                    "every name as shell exports (--env), or everything as JSON (--json).")
    add_spec_arguments(ap, size_required=False)
    ap.add_argument("--kb", default=None, help="the graph file, when not the substrate's own")
    ap.add_argument("--transfer-graph", default=None, help="the transfer gate's graph, if any")
    ap.add_argument("--env", action="store_true", help="print `export NAME='value'` for every name")
    ap.add_argument("--json", action="store_true", help="print every name as JSON")
    ap.add_argument("key", nargs="?", help="one name (a shell variable, any case; or stack)")
    args = ap.parse_args(argv)
    layout = Layout.of(spec_from_args(args), kb=args.kb, transfer=args.transfer_graph)
    if args.env:
        for k, v in layout.env().items():
            print(f"export {k}={shlex.quote(v)}")
        return 0
    if args.json:
        print(json.dumps({"spec": asdict(layout.spec), "graph": asdict(layout.graph),
                          "transfer": asdict(layout.transfer) if layout.transfer else None,
                          **layout.env()}, indent=2))
        return 0
    if not args.key:
        ap.error("name a KEY, or pass --env or --json")
    print(layout.value(args.key))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
