"""Local weights behind `--model hf:<path>` (HF generate), `vllm:<path>` (a vLLM
engine) and `mlx:<path>` (MLX on Metal) — base, or base + a LoRA chain. The
callers are the agentic eval, agent/eval.py, which drives `generate_chat` one
rollout turn at a time, and the live app. `ENGINES` at the bottom is the
prefix registry.

torch/vllm/mlx imports are local so API-only usage stays torch-free.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path


def _warn_if_substrate_missing(adapter: str) -> None:
    """Warn when an adapter trained on a merged substrate is loaded without it.

    Such an adapter's adapter_config names the raw base, so loading it alone
    runs without error and scores like the base model. The trainers record
    the chain in `substrate.json`; this checks the caller's chain against it.
    """
    from ..train.adapters import adapter_chain

    chain = adapter_chain(adapter)
    for i, part in enumerate(chain):
        marker = Path(part) / "substrate.json"
        if not marker.exists() and Path(part).name.startswith("checkpoint-"):
            # A round's checkpoints are deltas on the chain its final/ records.
            marker = Path(part).parent / "final" / "substrate.json"
        if not marker.exists():
            continue
        try:
            needed = json.loads(marker.read_text(encoding="utf-8"))["merged_adapters"]
        except Exception:
            continue
        if needed and [str(Path(x)) for x in needed] != [str(Path(x)) for x in chain[:i]]:
            print(f"WARNING: {part} is a delta on a merged substrate "
                  f"({', '.join(needed)}), but was loaded after {chain[:i] or 'nothing'}. "
                  f"Results will silently look like the base model. "
                  f"Load it as: --adapter {','.join([*needed, part])}")


def _pick_device(torch) -> tuple[str, "object"]:
    """(device, dtype) for the hf: driver: CUDA, else Apple `mps`, else CPU.

    bf16 on CUDA (fp16 where the card lacks it) and fp32 on CPU. `mps` gets
    bf16, not fp16, to match the precision served on CUDA. `GR_HF_DEVICE` and
    `GR_HF_DTYPE` override the two.
    """
    dev = os.environ.get("GR_HF_DEVICE")
    if not dev:
        if torch.cuda.is_available():
            dev = "cuda:0"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"

    want = os.environ.get("GR_HF_DTYPE")
    if want:
        dtype = getattr(torch, want)
    elif dev.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif dev == "mps":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32          # CPU bf16 matmul is emulated and slower
    return dev, dtype


def _resolve_for_serving(model_name: str, adapter: str | None,
                        chat_template: str | None, *, engine: str,
                        padding_side: str | None = None):
    """(base model, adapter chain, rendering tokenizer) for any engine here.

    `padding_side` is passed by the hf: driver only, which left-pads a batch.
    `engine` names the driver in the printed load line. The path rules are in
    train/adapters.py.
    """
    from ..train.adapters import adapter_chain, chat_template_for, resolve_base, tokenizer_source

    model_name, adapter = resolve_base(model_name, adapter)
    # An adapter is a directory on disk or a Hub repository id; anything else
    # is refused by name here, before the tokenizer is fetched, not as a
    # repository-id error from the Hub.
    hub_id = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")
    for a in adapter_chain(adapter):
        if not Path(a).is_dir() and not hub_id.match(a):
            raise FileNotFoundError(
                f"no adapter at {a}: neither a directory nor a Hub repository id. The released "
                f"adapters land under models/ with the quick start's download (README, Quick start).")

    from transformers import AutoTokenizer

    from ..train.tokenize import override_chat_template

    print(f"Loading {engine} model: {model_name}"
          + (f" + adapter {adapter}" if adapter else ""))
    tok_src = tokenizer_source(model_name, adapter)
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if padding_side:
        tokenizer.padding_side = padding_side
    # The template comes from the adapter's record (train/adapters.py,
    # chat_template_for); `chat_template` names one for a base model or a
    # foreign adapter, and must agree with the record when there is one.
    override_chat_template(tokenizer, chat_template_for(model_name, adapter, chat_template),
                           model=tok_src)
    return model_name, adapter, tokenizer


def merge_chain_into(model, chain: list[str]):
    """`model` with each adapter of `chain` merged in, in load order, on the
    device and at the dtype it already has.

    Every merge goes through this loop. Each adapter passes
    `assert_adapter_resolves` before it is merged, which refuses an adapter
    whose tensors name modules the model lacks; PEFT would drop them silently.
    """
    from peft import PeftModel

    from ..train.adapters import assert_adapter_resolves

    for a in chain:
        model = PeftModel.from_pretrained(model, a)
        assert_adapter_resolves(model, a)
        model = model.merge_and_unload()
    return model


def _merge_chain(model_name: str, chain: list[str]):
    """The adapter chain merged into a fresh bf16 model on CPU.

    On CPU because vLLM sizes its memory pool at engine start and must find
    the GPU empty, and mlx-lm loads the result from disk. LocalHFModel merges
    onto the model it generates with instead, through the same loop.
    """
    import torch
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    return merge_chain_into(m, chain)


def _merge_key(model_name: str, chain: list[str]) -> dict:
    """What a merged checkpoint was built from: the base and, per adapter, its
    path and its weight file's size and mtime."""
    parts = []
    for a in chain:
        f = Path(a) / "adapter_model.safetensors"
        st = f.stat() if f.exists() else None
        parts.append({"path": str(Path(a)), "bytes": st.st_size if st else None,
                      "mtime": int(st.st_mtime) if st else None})
    return {"base": model_name, "chain": parts}


def _merged_checkpoint(model_name: str, chain: list[str], out, tokenizer,
                       write) -> str:
    """The merged checkpoint at `out`, built once. `write(model, tokenizer,
    out)` is the vLLM or the HF-layout writer; both write the same merge.

    `out/merge.json` records what the merge was built from (`_merge_key`),
    and a directory whose record is missing or differs is rebuilt: the
    directory's name carries only the last adapter, so two chains or two
    bases can share it. The merge is written beside `out` and moved into
    place once complete, so an interrupted build leaves no half-written
    checkpoint behind. The substrate warning fires on every call.
    """
    import shutil

    out = Path(out)
    _warn_if_substrate_missing(",".join(chain))
    key = _merge_key(model_name, chain)
    if (out / "config.json").exists():
        try:
            if json.loads((out / "merge.json").read_text(encoding="utf-8")) == key:
                return str(out)
            print(f"  {out} was merged from another chain or base; rebuilding")
        except (OSError, ValueError):
            print(f"  {out} carries no record of what it was merged from; rebuilding")
    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    m = _merge_chain(model_name, chain)
    write(m, tokenizer, tmp)
    del m
    (tmp / "merge.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    return str(out)


def merged_hf_checkpoint(model_name: str, adapter: str,
                         chat_template: str | None = None) -> str:
    """The bf16 HF-layout checkpoint the `mlx:` driver serves, built or reused.

    Public so that the latent-state probe (`scripts/latent/extract_states.py`)
    reads the hidden states of the same merge the driver serves.
    """
    model_name, adapter, tokenizer = _resolve_for_serving(
        model_name, adapter, chat_template, engine="merge")
    return _merged_hf(model_name, adapter, tokenizer)


def _merged_hf(model_name: str, adapter: str, tokenizer) -> str:
    """`<last adapter>/hf_merged`, written once; shared by the mlx driver and
    `merged_hf_checkpoint`."""
    import functools

    from ..train.adapters import adapter_chain

    chain = adapter_chain(adapter)
    return _merged_checkpoint(
        model_name, chain, Path(chain[-1]) / "hf_merged", tokenizer,
        functools.partial(_write_hf_layout, base=model_name))


class LocalHFModel:
    """The `--model hf:<path>` driver: `model.generate`, no engine.

    Runs on CUDA, Apple `mps` or CPU (`_pick_device`). `vllm:` on CUDA and
    `mlx:` on Apple Silicon are the batching engines; this is the driver for
    a machine that has neither. Sampling draws from torch's global generator,
    seeded at construction (`seed`). Unlike the two engines, this driver cuts
    no over-long conversation: one past the model's window fails in generate.
    """

    def __init__(self, model_name: str, adapter: str | None = None, batch_size: int = 8,
                 chat_template: str | None = None, seed: int = 0):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        torch.manual_seed(int(seed))
        self.batch_size = batch_size
        device, dtype = _pick_device(torch)
        # Only this driver pads: it left-pads a batch into model.generate.
        model_name, adapter, self.tokenizer = _resolve_for_serving(
            model_name, adapter, chat_template, engine="local",
            padding_side="left")

        # Merge on CPU when the target is Metal, then move once: loading
        # adapter weights onto an `mps` model can hang intermittently.
        merge_on_cpu = device == "mps" and adapter is not None
        print(f"  device={device} dtype={str(dtype).replace('torch.', '')}"
              + ("  (merging on cpu first)" if merge_on_cpu else ""))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, attn_implementation="sdpa",
            device_map="cpu" if merge_on_cpu else device,
        )
        # `generate` stops on `model.generation_config` only, and that can
        # lack the tokenizer's eos, the token the policy ends its turns with;
        # every non-tool turn would then decode to max_new_tokens. Add it as
        # a union: the config may list several eos ids.
        gc = getattr(self.model, "generation_config", None)
        eid = self.tokenizer.eos_token_id
        if gc is not None and eid is not None:
            have = gc.eos_token_id
            have = [] if have is None else (
                [have] if isinstance(have, int) else list(have))
            if eid not in have:
                gc.eos_token_id = have + [eid]
                print(f"  eos {have} + policy {eid} "
                      f"({self.tokenizer.eos_token!r})")
        # The post-condition of the union above, checked at load.
        from ..train.generate import assert_stop_covers_policy
        assert_stop_covers_policy(
            gc.eos_token_id if gc is not None else None, eid,
            engine="transformers.generate",
            source=f"{model_name} generation_config",
            remedy="This driver patches that set at load; reaching here means "
                   "the patch above did not apply.")

        if adapter:
            _warn_if_substrate_missing(adapter)

            from ..train.adapters import adapter_chain
            # Merged onto the model already loaded, at the serving dtype.
            self.model = merge_chain_into(self.model, adapter_chain(adapter))
        if merge_on_cpu:
            self.model = self.model.to(device)
        self.model.eval()

    def generate_chat(
        self,
        conversations: list[list[dict]],
        max_new_tokens: int = 800,
        temperature: float | None = None,
        stop: list[str] | None = None,
        desc: str = "local gen",
    ) -> list[str]:
        """One completion per conversation, in the order they were handed over.
        Best-of-n repeats the episode, as separate sampled runs
        (the 32-seed pool the recipe's `final` stage draws), not the turn.

        Conversations are batched in order of rendered length (`length_order`),
        so a left-padded batch is not mostly padding.
        """
        from tqdm import tqdm

        from ..train.generate import (hf_generate_kwargs, length_order,
                                           render_prompts, turn_text)

        sample = temperature is not None
        texts = render_prompts(conversations, self.tokenizer)
        order = length_order(texts)
        results: list[str] = [""] * len(texts)
        stop_kwargs = hf_generate_kwargs(self.tokenizer, stop)

        for i in tqdm(range(0, len(order), self.batch_size), desc=desc):
            idx = order[i: i + self.batch_size]
            enc = self.tokenizer([texts[j] for j in idx],
                                 return_tensors="pt", padding=True).to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=sample,
                    temperature=temperature if sample else None,
                    top_p=0.95 if sample else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **stop_kwargs,
                )
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            decoded = [
                turn_text(d, self.tokenizer) for d in
                self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]
            for k, j in enumerate(idx):
                results[j] = decoded[k]

        return results


def vllm_serving_path(model_name: str, adapter: str | None,
                      chat_template: str | None) -> tuple[str, object, str]:
    """(the directory vLLM serves, the rendering tokenizer, the resolved base):
    the base itself, or the adapter chain merged on CPU once and written as a
    bf16 checkpoint at `<last adapter>/vllm_merged`, which later loads reuse.
    Shared by `LocalVLLMModel` and the steered engine of
    `graph_reasoner.latent.steer`, so both serve the same merge.
    """
    model_name, adapter, tokenizer = _resolve_for_serving(
        model_name, adapter, chat_template, engine="vllm")
    if adapter:
        from ..train.adapters import adapter_chain
        from ..train.vllm_ckpt import write_vllm_checkpoint

        chain = adapter_chain(adapter)
        serve = _merged_checkpoint(model_name, chain, Path(chain[-1]) / "vllm_merged",
                                   tokenizer, write_vllm_checkpoint)
    else:
        serve = model_name
    return serve, tokenizer, model_name


class LocalVLLMModel:
    """LocalHFModel's generate contract on a vLLM engine, with continuous
    batching and prefix caching.

    Both drivers render, stop and split through `train/generate.py`. Decoding
    is greedy when `temperature` is None; greedy runs still differ from one
    another on this model family, so compare replicated runs. `seed` seeds
    the engine's sampler, so sampled runs that differ only in it draw
    different streams.

    An adapter chain is merged on CPU and written once as a bf16 checkpoint at
    `<last adapter>/vllm_merged`, which later loads of the same chain reuse.
    A conversation that no longer fits the context window gets an empty turn
    instead of an engine error; the episode loop counts empty turns and ends
    the search at its cap.
    """

    def __init__(self, model_name: str, adapter: str | None = None,
                 chat_template: str | None = None,
                 max_model_len: int = 16384, gpu_frac: float = 0.85,
                 seed: int = 0):
        serve, self.tokenizer, model_name = vllm_serving_path(model_name, adapter, chat_template)

        # vLLM takes its stop from the served tokenizer, and an existing
        # `vllm_merged` is reused as it is: one written under another chat
        # template would be served with that template's stop token. Checked
        # here so that it fails at load.
        from transformers import AutoTokenizer as _AutoTok

        from ..train.generate import assert_stop_covers_policy
        assert_stop_covers_policy(
            _AutoTok.from_pretrained(serve).eos_token_id,
            self.tokenizer.eos_token_id,
            engine="vllm",
            source=f"the tokenizer saved in {serve}",
            remedy=(f"Delete {serve} and let it rebuild."
                    if serve != model_name else
                    "The served weights and the rendering tokenizer disagree; "
                    "pass --chat-template to say which one the policy uses."))

        from vllm import LLM

        self.max_model_len = int(max_model_len)
        # max_num_seqs is capped: on this architecture vLLM reserves a state
        # block per decode sequence (see _ensure_vllm in train/grpo.py).
        self.llm = LLM(model=serve, dtype="bfloat16", enable_prefix_caching=True,
                       gpu_memory_utilization=float(gpu_frac), max_num_seqs=64,
                       max_model_len=self.max_model_len, seed=int(seed))

    def generate_chat(self, conversations: list[list[dict]],
                      max_new_tokens: int = 800,
                      temperature: float | None = None,
                      stop: list[str] | None = None,
                      desc: str = "local gen") -> list[str]:
        """One completion per conversation (see LocalHFModel.generate_chat)."""
        from vllm import SamplingParams

        from ..train.generate import (encode_prompts, fits_context,
                                           render_prompts, turn_text,
                                           vllm_sampling_kwargs)

        ids = encode_prompts(render_prompts(conversations, self.tokenizer),
                             self.tokenizer)
        # Greedy at temperature None is resolved here; the shared builder
        # takes temperature and top_p already resolved.
        sp = SamplingParams(**vllm_sampling_kwargs(
            self.tokenizer, max_new_tokens=max_new_tokens,
            temperature=0.0 if temperature is None else float(temperature),
            top_p=1.0 if temperature is None else 0.95, stop=stop))
        keep = fits_context(ids, self.max_model_len)
        results = [""] * len(ids)
        if keep:
            outs = self.llm.generate([{"prompt_token_ids": ids[i]} for i in keep],
                                     sp, use_tqdm=False)
            for i, o in zip(keep, outs):
                results[i] = turn_text(o.outputs[0].text, self.tokenizer)
        return results

    def close(self) -> None:
        """Stop the engine's process and free the card. A long-lived server
        calls this when it stops: an engine process left running holds the
        card's memory, and the next engine on it fails to start."""
        core = getattr(getattr(getattr(self, "llm", None), "llm_engine", None), "engine_core", None)
        if core is not None and hasattr(core, "shutdown"):
            core.shutdown()


def _write_hf_layout(model, tokenizer, out: Path, *, base: str) -> None:
    """Write a merged model as the plain HF checkpoint mlx-lm loads.

    `write_vllm_checkpoint` renames `model.language_model.*` keys to `model.*`
    and pins the architecture name vLLM serves; mlx-lm reads the upstream
    layout instead. `base` is the model the chain was merged onto, needed for
    the composite config below.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # mlx-lm reads eos from config.json, lets generation_config.json override
    # it, and applies the result over the tokenizer's own. Both files derive
    # from the base model, whose list may lack the policy's eos (the
    # tokenizer's), so both are set to the union: the base's ids and the
    # policy's, the stop set the hf: driver watches too (LocalHFModel).
    gc = getattr(model, "generation_config", None)
    have = gc.eos_token_id if gc is not None else None
    eos_ids = [have] if isinstance(have, int) else list(have or [])
    if tokenizer.eos_token_id not in eos_ids:
        eos_ids.append(tokenizer.eos_token_id)
    if gc is not None:
        gc.eos_token_id = eos_ids
    model.save_pretrained(out, safe_serialization=True)
    # Save the base's composite config over the one save_pretrained wrote.
    # AutoModelForCausalLM loads a multimodal base as its text tower, whose
    # saved model_type (`qwen3_5_text`) names no model class in mlx-lm; the
    # composite type does.
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(base)
    # The same union on the composite config (see above).
    cfg.eos_token_id = eos_ids
    if getattr(cfg, "text_config", None) is not None:
        cfg.text_config.eos_token_id = eos_ids
    cfg.save_pretrained(out)
    # mlx-lm loads the tokenizer from the same directory. Saving it also
    # carries the run's chat template into the served checkpoint.
    tokenizer.save_pretrained(out)


class LocalMLXModel:
    """LocalHFModel's generate contract on MLX, the batching engine for Apple
    Silicon (vLLM has no Metal backend).

    A rollout re-sends its whole transcript every turn, so the driver keeps
    each sequence's prompt cache and reuses it when the next prompt extends
    what that cache has consumed. Two environment variables trade speed for
    memory: `GR_MLX_PREFILL_BATCH` (prompts prefilled at once, default 2) and
    `GR_MLX_PREFIX_CACHE=0` (reuse off). `seed` seeds MLX's sampler.
    """

    def __init__(self, model_name: str, adapter: str | None = None,
                 chat_template: str | None = None,
                 max_model_len: int = 16384, seed: int = 0):
        model_name, adapter, self.tokenizer = _resolve_for_serving(
            model_name, adapter, chat_template, engine="mlx")

        if adapter:
            serve = _merged_hf(model_name, adapter, self.tokenizer)
        else:
            serve = model_name

        import mlx.core as mx
        from mlx_lm import load

        mx.random.seed(int(seed))
        self.max_model_len = int(max_model_len)
        self.model, self._mlx_tok = load(serve)
        # Append-only prefix cache: one entry per sequence of the last batch,
        # keyed by the tokens the entry has consumed (its prompt plus what was
        # generated from it). The next turn's prompt extends exactly that, so
        # reuse is a prefix check and never trims a cache; the linear-attention
        # layers' state cannot be trimmed. Replaced wholesale on every call.
        self._caches: list[tuple[list[int], object]] = []
        # How many prompts prefill at once. The right value depends on the
        # machine's memory, so the environment can override it.
        self.prefill_batch_size = int(os.environ.get("GR_MLX_PREFILL_BATCH", "2"))
        # Reuse keeps this turn's caches alive while the generator builds the
        # next turn's, which roughly doubles peak KV memory. Off, every turn
        # prefills its whole transcript.
        self.prefix_cache = os.environ.get("GR_MLX_PREFIX_CACHE", "1") != "0"
        # Print the stop set the engine watches, then check it covers the
        # policy's eos.
        eos = sorted(getattr(self._mlx_tok, "eos_token_ids", None)
                     or {self.tokenizer.eos_token_id})
        print(f"  mlx ready; eos {eos} = "
              f"{[self.tokenizer.decode([t]) for t in eos]}")
        from ..train.generate import assert_stop_covers_policy
        assert_stop_covers_policy(
            eos, self.tokenizer.eos_token_id, engine="mlx-lm",
            source=f"{serve} config.json, then generation_config.json",
            remedy=f"Delete {serve} and let it rebuild.")

    def generate_chat(self, conversations: list[list[dict]],
                      max_new_tokens: int = 800,
                      temperature: float | None = None,
                      stop: list[str] | None = None,
                      desc: str = "local gen") -> list[str]:
        """One completion per conversation (see LocalHFModel.generate_chat),
        the whole batch at once through `mlx_lm.generate.BatchGenerator`."""
        from mlx_lm.generate import BatchGenerator
        from mlx_lm.models.cache import make_prompt_cache

        from ..train.generate import (cut_at_stop, encode_prompts,
                                           fits_context, render_prompts,
                                           split_stops, stop_strings, turn_text)

        ids = encode_prompts(render_prompts(conversations, self.tokenizer),
                             self.tokenizer)
        stops = stop_strings(self.tokenizer, stop)
        sampler = self._sampler(temperature)
        # The same context rule as the vLLM driver: an over-budget
        # conversation gets an empty turn.
        keep = fits_context(ids, self.max_model_len)

        out = [""] * len(ids)
        if keep:
            texts = self._many([ids[i] for i in keep], max_new_tokens,
                               sampler, stops, BatchGenerator,
                               make_prompt_cache, turn_text,
                               split_stops, cut_at_stop)
            for i, text in zip(keep, texts):
                out[i] = text
        return out

    def _sampler(self, temperature: float | None):
        from mlx_lm.sample_utils import make_sampler

        # Greedy at temperature None, matching the other two drivers.
        return make_sampler(temp=0.0 if temperature is None else float(temperature),
                            top_p=0.0 if temperature is None else 0.95)

    def _reuse(self, prompt: list[int], make_prompt_cache):
        """(cache, residual tokens) for one prompt, against the last batch's
        caches; a fresh cache and the whole prompt when none matches or
        `prefix_cache` is off. The match is a strict prefix, because a
        sequence cannot be inserted with a zero-token prompt."""
        if self.prefix_cache:
            for seen, cache in self._caches:
                if len(prompt) > len(seen) and prompt[:len(seen)] == seen:
                    return cache, prompt[len(seen):]
        return make_prompt_cache(self.model), list(prompt)

    def _many(self, prompts, max_new_tokens, sampler, stops, batch_generator,
              make_prompt_cache, turn_text, split_stops, cut_at_stop) -> list[str]:
        eos = set(getattr(self._mlx_tok, "eos_token_ids", None)
                  or {self.tokenizer.eos_token_id})
        # The single-token/substring split and the cut rule belong to
        # train/generate.py and are passed in.
        single, multi = split_stops(self.tokenizer, stops)
        gen = batch_generator(
            self.model, sampler=sampler,
            stop_tokens=[[t] for t in sorted(eos | set(single))],
            # The caller's whole batch decodes as one group.
            completion_batch_size=max(1, len(prompts)),
            # Prefill is the memory peak and is batched separately from
            # decode (GR_MLX_PREFILL_BATCH).
            prefill_batch_size=self.prefill_batch_size)

        rests, caches = [], []
        for p in prompts:
            cache, rest = self._reuse(list(p), make_prompt_cache)
            caches.append(cache)
            rests.append(rest)
        # Optional observer: `on_token(i, tokens_so_far)` is called after
        # every generated token of prompt i, and may raise to abort the turn
        # (app/live.py sets it). It cannot change what is returned.
        on_token = getattr(self, "on_token", None)
        try:
            uids = gen.insert(rests, [max_new_tokens] * len(rests), caches=caches)
            index = {u: i for i, u in enumerate(uids)}

            toks: dict[int, list[int]] = {u: [] for u in uids}
            done: dict[int, tuple] = {}
            while len(done) < len(uids):
                batch = gen.next_generated()
                if not batch:
                    break
                for r in batch:
                    if r.uid in done:
                        continue
                    if r.finish_reason is None:
                        toks[r.uid].append(r.token)
                        if on_token is not None:
                            on_token(index[r.uid], toks[r.uid])
                        if multi and self._hit_text_stop(toks[r.uid], multi):
                            got = gen.remove([r.uid], return_prompt_caches=True)
                            done[r.uid] = got.get(r.uid, (None, None))
                        continue
                    # The token that hit the length cap is generated text too
                    # (mlx-lm's own batch_generate keeps it); the eos is not.
                    if (r.finish_reason == "length" and r.token is not None
                            and r.token not in eos):
                        toks[r.uid].append(r.token)
                    # Keep a matched stop token in the text, as the other two
                    # drivers keep the stop string: the tool-call parser needs
                    # the closing `</tool_call>`. The eos is not kept.
                    for t in (r.match_sequence or []):
                        if t in single:
                            toks[r.uid].append(t)
                    done[r.uid] = (r.prompt_cache, r.all_tokens)

            fresh: list[tuple[list[int], object]] = []
            out = []
            for uid, prompt, rest in zip(uids, prompts, rests):
                cache, consumed = done.get(uid, (None, None))
                if cache is not None and consumed is not None:
                    # What the cache has consumed: the already-cached part of
                    # this prompt plus everything the generator ran through it.
                    # That includes the eos, which `toks` omits and the next
                    # turn's prompt renders.
                    fresh.append((list(prompt[:len(prompt) - len(rest)])
                                  + list(consumed), cache))
                out.append(turn_text(cut_at_stop(self.tokenizer.decode(toks[uid]), stops),
                                     self.tokenizer))
            self._caches = fresh if self.prefix_cache else []
        finally:
            # Always close, so a turn the observer aborts leaves no sequences
            # in flight.
            gen.close()
        return out

    def _hit_text_stop(self, tokens: list[int], multi: list[str]) -> bool:
        """Whether a stop the engine cannot watch as one token has appeared.

        Decodes only the last `window` tokens, the longest stop's length in
        characters: a stop of L characters spans at most L tokens.
        """
        window = max(len(s) for s in multi)
        tail = self.tokenizer.decode(tokens[-window:])
        return any(s in tail for s in multi)


# ---------------------------------------------------------------- the registry

#: The local engines a model string can name, by prefix: `hf:<path>` (HF
#: generate), `vllm:<path>` (a vLLM engine), `mlx:<path>` (MLX on Metal); any
#: other string is an API model id. A class is looked up by name when an
#: engine is built (`engine_class`), so a test can substitute one.
#: `batch_size` reaches the hf: driver only.
ENGINES = {"hf": "LocalHFModel", "vllm": "LocalVLLMModel", "mlx": "LocalMLXModel"}
# What each driver imports, which agent/eval.build_generate_seam checks before
# anything is merged or loaded; a stand-in class without the attribute is not checked.
LocalHFModel.needs, LocalVLLMModel.needs, LocalMLXModel.needs = "transformers", "vllm", "mlx_lm"


def split_model(model: str) -> tuple[str | None, str]:
    """`(engine, path)` for a local model string, `(None, model)` for an API
    model id. Only a leading registered prefix counts, so an API id carrying
    a `:variant` suffix stays an API id."""
    engine, sep, path = model.partition(":")
    return (engine, path) if sep and engine in ENGINES else (None, model)


def engine_class(engine: str) -> type:
    """The driver class an engine prefix names."""
    return globals()[ENGINES[engine]]
