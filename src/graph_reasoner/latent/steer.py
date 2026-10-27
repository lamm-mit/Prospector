"""A benchmark through the served policy with a vector added at one layer -> one arm per dose, through the ordinary eval.

Each arm is one `agent.eval.run_agentic_eval` of the benchmark under the
served contract (the policy's budgets, greedy decoding, the real tools, the
gate), written as the run directory `knob-<arm>--<benchmark>` under the root
with the usual results, trajectories, completions and summary, plus
`knob.json` saying what the knob was. The arms:

  dose0        nothing added: the reference point
  fol:<d>      the follow direction at dose d, in units of the gap between
               the follow and backtrack class means along it; positive pushes
               toward following the page just seen
  ans:<d>      the answer direction, positive toward answering
  rnd:<d>      a random unit direction at the answer direction's dose unit
  rndf:<d>     the same random direction at the follow direction's dose unit
  rnd<k>:<d>, rndf<k>:<d>   the k-th random draw (random2, random3, ...)

Every arm of one invocation runs at one episode seed, which seeds each item's
tool RNG and the engine's sampler as run_agent_eval.py's --seed does; a
repeat read is a different seed, never the same one. The seed defaults to N
when the root ends in `_r<N>` and to 0 otherwise.

The vector is added inside vLLM through a forward hook on the decoder layer:
at the generation header during each turn's prompt pass and at every token
generated in the turn, for every ordinary turn of the episode. The forced
final turn (the one the loop demands when a budget runs out, the loop's
rescue and not the policy's decision) is generated with the vector off; it is
recognised by its message, the forced-answer nudge the loop appends as the
last user message, which the loop generates in its own batch (`_drive_batched`).

What the hook rests on, in vLLM 0.27.1: a decoder layer's forward returns
`(hidden, residual)` and the stream after the layer is their sum, so the hook
adds to `residual` in place and records `hidden + residual`;
`get_forward_context().attn_metadata` is a per-layer dict (or a list of them)
whose values carry `query_start_loc`, so `query_start_loc[1:] - 1` are each
request's last rows in the step (the header on a prompt pass, the new token
on a decode step), and it raises outside a forward; no prompt is split and no
prefix is cached (`max_num_batched_tokens == max_model_len`,
`enable_chunked_prefill=False`, `enable_prefix_caching=False`), so a prompt
pass's last row is the header's last token; the Python hook runs every step
(`enforce_eager=True`, compilation mode 0); the layers are the
`*DecoderLayer` modules in `model.modules()` order, 1-based;
`LLM.apply_model` runs a function in every worker, with
`VLLM_ENABLE_V1_MULTIPROCESSING=0` set before the import so the engine core is
in-process. The arithmetic (`last_rows`, the add, the recorder) is tested on
a plain module; the conventions are checked on the GPU against stored reads.

`scripts/latent/steer.py` is the CLI; `scripts/latent/mix.py` tabulates the
arms. vLLM and torch are imported inside functions. Tests: tests/test_latent.py.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

from ..agent.loop import EpisodeConfig

__all__ = ["DIRECTION_OF", "null_direction", "parse_arm", "read_seed", "decoder_layers", "last_rows", "knob_positions",
           "install_hook", "set_vec", "set_record", "take_record", "SteeredVLLM", "steered_seam", "arm_vector",
           "run_arms"]

#: The direction each steered arm kind adds; the nulls borrow a real axis's unit (`null_direction`).
DIRECTION_OF = {"ans": "answer", "fol": "follow"}
NULL_KIND = re.compile(r"^rnd(f?)(\d*)$")


# ---------------------------------------------------------------- arms

def null_direction(kind: str) -> tuple[str, str] | None:
    """A null arm kind's (direction key, axis whose dose unit it borrows):
    rnd and rnd<k> at the answer unit, rndf and rndf<k> at the follow unit."""
    m = NULL_KIND.match(kind)
    if not m:
        return None
    draw = m.group(2)
    return ("random" if draw in ("", "1") else f"random{draw}", "follow" if m.group(1) else "answer")


def parse_arm(spec: str) -> dict:
    """`dose0`, or `<kind>:<dose>` -> {name, kind, value}; the name is the run
    directory's middle (`fol+1`, `rndf2-2`)."""
    if spec == "dose0":
        return {"name": "dose0", "kind": "dose0"}
    kind, _, val = spec.partition(":")
    if (kind not in DIRECTION_OF and null_direction(kind) is None) or not val:
        raise SystemExit(f"arm {spec!r}: dose0, ans:<d>, fol:<d>, rnd:<d>, rndf:<d>, rnd<k>:<d> or rndf<k>:<d>")
    v = float(val)
    return {"name": f"{kind}{v:+g}", "kind": kind, "value": v}


def read_seed(seed: int | None, out_root: str | Path) -> int:
    """The episode seed of an invocation: the flag, else N from a root ending
    in `_r<N>`, else 0."""
    if seed is not None:
        return int(seed)
    m = re.search(r"_r(\d+)$", str(out_root).rstrip("/"))
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------- the hook

def decoder_layers(model) -> list:
    """The decoder layers in module order (1-based in every argument here)."""
    return [m for m in model.modules() if type(m).__name__.endswith("DecoderLayer")]


def last_rows(attn_metadata, n_rows: int) -> list[int] | None:
    """The row of each request's last token in a step, from the attention
    metadata's `query_start_loc` (cumulative query offsets, one more entry
    than requests): the entries after the first, minus one, within the step's
    rows. The metadata may be the object itself, a list of them (the first is
    taken) or a per-layer dict (the first value with query offsets). None
    when there are no offsets to read."""
    md = attn_metadata
    if isinstance(md, list):
        md = md[0] if md else None
    if isinstance(md, dict):
        md = next((v for v in md.values() if hasattr(v, "query_start_loc")), None)
    if md is None or not hasattr(md, "query_start_loc"):
        return None
    loc = md.query_start_loc
    if hasattr(loc, "tolist"):
        loc = loc.tolist()
    pos = [int(x) - 1 for x in list(loc)[1:]]
    return [p for p in pos if 0 <= p < n_rows]


def knob_positions(residual) -> list[int] | None:
    """`last_rows` over vLLM's forward context; None outside a real forward
    (profiling and warm-up)."""
    from vllm.forward_context import get_forward_context

    try:
        md = get_forward_context().attn_metadata
    except (AssertionError, AttributeError):
        return None
    return last_rows(md, residual.shape[0])


def install_hook(model, layer: int, positions=knob_positions) -> str:
    """Run inside the engine's worker: a forward hook on decoder layer `layer`
    that adds the vector in `model._knob["vec"]` to the layer's returned
    residual at the rows `positions(residual)` names, and records the
    residual stream there (`hidden + residual`) while `model._knob["record"]`
    is a list. Returns the layer's class name."""
    import torch

    layers = decoder_layers(model)
    if not 1 <= layer <= len(layers):
        raise ValueError(f"layer {layer} is not one of 1..{len(layers)}")
    state = {"vec": None, "record": None}
    model._knob = state

    def hook(_module, _inputs, output):
        hidden, residual = output
        pos = positions(residual)
        if not pos:
            return output
        idx = torch.as_tensor(pos, dtype=torch.long, device=residual.device)
        if state["record"] is not None:
            state["record"].append((hidden[idx] + residual[idx]).float().cpu())
        if state["vec"] is not None:
            residual.index_add_(0, idx, state["vec"].to(residual.dtype).expand(len(pos), -1))
        return output

    layers[layer - 1].register_forward_hook(hook)
    return type(layers[layer - 1]).__name__


def set_vec(model, vec) -> None:
    import torch

    device = next(model.parameters()).device
    model._knob["vec"] = None if vec is None else torch.tensor(vec, dtype=torch.float32, device=device)


def set_record(model, on: bool) -> None:
    model._knob["record"] = [] if on else None


def take_record(model):
    rec = model._knob.get("record") or []
    model._knob["record"] = None
    return [r.numpy() for r in rec]


# ---------------------------------------------------------------- the engine

class SteeredVLLM:
    """The vLLM engine with the hook installed: the eval's generate contract
    (render, encode, fit to the window, greedy with the tool-call stop) plus
    `set_vec` and the recorder. Eager execution and whole-prompt prefill,
    since the hook must run every step and see each prompt's header as the
    last row of its pass; the prefix cache is off because on this hybrid
    architecture it needs chunked prefill, under which a prompt may be split.
    The served weights are the eval's own merge (`vllm_serving_path`)."""

    def __init__(self, model_name: str, layer: int, *, adapter: str | None = None, chat_template: str | None = None,
                 max_model_len: int = 16384, gpu_frac: float = 0.85, max_num_seqs: int = 64, seed: int = 0):
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from transformers import AutoTokenizer
        from vllm import LLM

        from ..eval.local_model import vllm_serving_path
        from ..train.generate import assert_stop_covers_policy

        serve, self.tokenizer, base = vllm_serving_path(model_name, adapter, chat_template)
        assert_stop_covers_policy(AutoTokenizer.from_pretrained(serve).eos_token_id, self.tokenizer.eos_token_id,
                                  engine="vllm", source=f"the tokenizer saved in {serve}",
                                  remedy=(f"Delete {serve} and let it rebuild." if serve != base else
                                          "pass --chat-template to say which template the policy uses"))
        self.serve, self.layer = serve, int(layer)
        self.max_model_len = int(max_model_len)
        self.llm = LLM(model=serve, dtype="bfloat16", enable_prefix_caching=False,
                       gpu_memory_utilization=float(gpu_frac), max_num_seqs=int(max_num_seqs),
                       max_model_len=self.max_model_len, max_num_batched_tokens=self.max_model_len,
                       enable_chunked_prefill=False, enforce_eager=True,
                       compilation_config={"mode": 0}, seed=int(seed))
        kinds = self.llm.apply_model(functools.partial(install_hook, layer=self.layer))
        self.layer_class = kinds[0]
        print(f"  steering hook on {kinds[0]} {self.layer} in {len(kinds)} worker(s)", flush=True)

    def set_vec(self, vec) -> None:
        self.llm.apply_model(functools.partial(set_vec, vec=None if vec is None else np.asarray(vec, dtype=np.float32)))

    def record(self, on: bool) -> None:
        self.llm.apply_model(functools.partial(set_record, on=on))

    def take_record(self):
        return self.llm.apply_model(take_record)[0]

    def generate_chat(self, conversations, max_new_tokens: int = 800, temperature=None,
                      stop=None, desc: str = "local gen") -> list[str]:
        from vllm import SamplingParams

        from ..train.generate import encode_prompts, fits_context, render_prompts, turn_text, vllm_sampling_kwargs

        ids = encode_prompts(render_prompts(conversations, self.tokenizer), self.tokenizer)
        kw = vllm_sampling_kwargs(self.tokenizer, max_new_tokens=max_new_tokens,
                                  temperature=0.0 if temperature is None else float(temperature),
                                  top_p=1.0 if temperature is None else 0.95, stop=stop)
        sp = SamplingParams(**kw)
        keep = fits_context(ids, self.max_model_len)
        results = [""] * len(ids)
        if keep:
            outs = self.llm.generate([{"prompt_token_ids": ids[i]} for i in keep], sp, use_tqdm=False)
            for i, o in zip(keep, outs):
                results[i] = turn_text(o.outputs[0].text, self.tokenizer)
        return results


def steered_seam(engine, cfg: EpisodeConfig, knob: dict):
    """The eval's generate seam over a steered engine. Before each call the
    vector `knob["vec"]` is set, or cleared when the call generates forced
    finals: every conversation then ends with the forced-answer nudge the loop
    appends, and the loop generates such a batch apart, so a mixed call is an
    error. The calls are counted into `knob` (`n_generate_calls`,
    `n_forced_calls`) for the arm's record."""
    from ..agent.eval import local_generate_seam
    from ..agent.prompts import nudge
    from ..agent.tools import available_tools

    base = local_generate_seam(engine, cfg)
    forced_text = nudge("FORCED_ANSWER", cfg.prompt_version, available_tools(cfg.tool_cfg))
    knob.setdefault("n_generate_calls", 0)
    knob.setdefault("n_forced_calls", 0)

    def generate(conversations, max_new_tokens):
        flags = [bool(c) and c[-1].get("role") == "user" and c[-1].get("content") == forced_text
                 for c in conversations]
        if any(flags) and not all(flags):
            raise RuntimeError("a generate call mixes forced finals with ordinary turns")
        forced = bool(flags) and all(flags)
        knob["n_generate_calls"] += 1
        knob["n_forced_calls"] += int(forced)
        engine.set_vec(None if forced else knob.get("vec"))
        try:
            return base(conversations, max_new_tokens)
        finally:
            engine.set_vec(None)

    generate.max_model_len = getattr(base, "max_model_len", None)
    return generate


# ---------------------------------------------------------------- the arms

def arm_vector(arm: dict, dirs, meta: dict) -> tuple[np.ndarray | None, dict]:
    """The vector an arm adds (dose x gap x unit direction) and its record
    fields; (None, {}) for dose0."""
    null = null_direction(arm["kind"])
    if arm["kind"] not in DIRECTION_OF and null is None:
        return None, {}
    dname, gap_axis = (DIRECTION_OF[arm["kind"]], DIRECTION_OF[arm["kind"]]) if null is None else null
    if dname not in dirs:
        raise SystemExit(f"arm {arm['name']}: the directions file has no {dname!r} vector (rebuild with --n-random)")
    gap = float(meta[gap_axis]["gap"])
    vec = arm["value"] * gap * np.asarray(dirs[dname], dtype=np.float32)
    return vec, {"direction": dname, "dose": arm["value"], "gap": gap, "vector_norm": float(np.linalg.norm(vec)),
                 "vector_norm_over_resid": float(np.linalg.norm(vec) / meta["resid_norm_median"])}


def _sha(path) -> str | None:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16] if path else None


def run_arms(engine: SteeredVLLM, arms: list[dict], *, benchmark: str, directions: Path, directions_json: Path,
             out_root: Path, seed: int, prefix: str = "knob-", limit: int | None = None,
             model: str = "", adapter: str | None = None, chat_template: str | None = None, gpu_frac: float | None = None,
             log=print) -> int:
    """One eval per arm under the vector the arm names, each written as
    `<out_root>/<prefix><arm>--<benchmark>/` plus knob.json; an arm whose run
    directory already holds a summary is skipped."""
    from .. import __version__
    from ..agent.eval import run_agentic_eval
    from ..benchmarks import run_dir_name
    from ..provenance import env_state, git_state

    dirs = np.load(directions)
    meta = json.loads(Path(directions_json).read_text(encoding="utf-8"))
    for arm in arms:
        cfg = EpisodeConfig(seed=seed)
        name = f"{prefix}{arm['name']}"
        out_dir = Path(out_root) / run_dir_name(name, benchmark)
        if (out_dir / "summary.json").exists():
            log(f"\n=== arm {name}: done already, skipped")
            continue
        vec, fields = arm_vector(arm, dirs, meta)
        knob = {"vec": vec}
        record = {"arm": arm, "layer": engine.layer, "layer_class": engine.layer_class, "directions": str(directions),
                  "model": model, "adapter": adapter, "served": engine.serve, "engine": "vllm", "seed": seed,
                  "package": {"graph_reasoner": __version__, **git_state()}, "env": env_state(),
                  "gpu_frac": gpu_frac, "max_model_len": engine.max_model_len,
                  "directions_sha256": _sha(directions), "directions_json_sha256": _sha(directions_json), **fields}
        log(f"\n=== arm {name}: {json.dumps({k: v for k, v in record.items() if k not in ('env', 'package')})}")
        run_agentic_eval(benchmark, f"vllm:{model}", kb_path=None, backend="auto", adapter=adapter, limit=limit,
                         run_name=name, cfg=cfg, chat_template=chat_template, out_root=out_root, latency=False,
                         max_model_len=engine.max_model_len, generate=steered_seam(engine, cfg, knob))
        record.update({k: knob[k] for k in ("n_generate_calls", "n_forced_calls")})
        (out_dir / "knob.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return 0
