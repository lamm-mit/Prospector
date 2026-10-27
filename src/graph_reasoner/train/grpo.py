"""GRPO over live multi-turn rollouts, with the assertion gate in the reward.

The objective is `reward.py`'s: 1 for a correct answer whose cited chain
passes the gate (cited ⊆ seen), otherwise a quarter of the evidence recall.
`rollout.build_objective` builds it from the config and prints it at launch.

Contract with TRL's `grpo_trainer.py` (pyproject.toml bounds the versions and
launch/constraints-gpu.txt pins the ones the recipe ran on):
  * `rollout_func(prompts, trainer)` receives the batch after TRL's sampler
    has repeated each prompt `num_generations` times, consecutively. It
    returns exactly one episode per entry, in order, and adjacent repeats
    form the advantage groups.
  * It returns `prompt_ids`, `completion_ids` and `logprobs`; every other key
    becomes a per-row column handed to the reward functions, here the columns
    of `reward.score_episodes`.
  * The environment mask is returned as `env_mask`, which TRL pops into
    `tool_mask` only when the trainer's own `tools` is unset, so `tools` is
    never set. Observation tokens carry no gradient: `run_grpo` refuses a TRL
    whose source lacks the pop, and the loss refuses a batch without
    `tool_mask`.
  * `logprobs` are the behaviour logprobs: the vLLM engine that sampled the
    tokens scores them under the same weights, in an echo pass at temperature
    1 with no top-p (`_vllm_sequence_logprobs`); the trainer's logprobs are at
    the configured sampling temperature.

`_parity_loss` replaces TRL's loss. On this rollout path TRL's own loss takes
the importance ratio against a detached copy of the current logprobs, so its
ratio is identically 1.

`tokenize.token_spans` supplies the token ids and the mask (assistant segments
1, observation segments 0); it raises unless the chat template is
concatenative.

The trained adapter is a fresh LoRA on the `substrate_adapters` chain, merged
first. `substrate.json` in every saved checkpoint and in `final/` records the
chain, and evaluation must load the whole comma-joined chain, not the adapter
alone.
"""
from __future__ import annotations

import functools
from pathlib import Path


from ..agent.backend import load_backend
from ..agent.loop import EpisodeConfig, run_episodes_batched
from .config import check_config_keys, refuse_broken_cuda, require as _require
from .defaults import SHARED
from .prefixes import zero_prefix_turns
from .reward import find_rate_monitor, gate_pass_monitor, score_episodes
from .rollout import (build_objective, episode_config, load_train_items,
                      load_train_prefixes)
from .tokenize import token_spans

__all__ = ["run_grpo", "make_rollout_func", "parity_trainer_class"]


# ---------------------------------------------------------------- rollout

def _ensure_vllm(state: dict):
    """The in-process vLLM engine for rollouts, built on the first rollout
    call, after the trainer has claimed its GPU memory.

    `vllm_gpu_frac` is the engine's share of the card. `vllm_max_num_seqs`
    (default 64) must stay small on Qwen3.5: vLLM reserves state for every
    decode sequence of a linear-attention hybrid, and its own default of 1024
    reserves far more than a rollout batch uses. With `vllm_sleep` the
    engine's weights and cache are unmapped while the trainer runs.
    launch/README.md has the memory fit on one card.
    """
    if state.get("vllm_llm") is not None:
        return state["vllm_llm"]
    from vllm import LLM

    llm = LLM(model=state["vllm_base"], enable_lora=True, max_loras=1,
              max_lora_rank=state["lora_r"], dtype="bfloat16",
              enable_prefix_caching=True,
              gpu_memory_utilization=state["vllm_gpu_frac"],
              max_num_seqs=state["vllm_max_num_seqs"],
              # Lets sleep() unmap the weight and KV pools; allocates nothing
              # extra by itself.
              enable_sleep_mode=state["vllm_sleep"],
              max_model_len=state["max_seq_len"])
    state["vllm_llm"] = llm
    return llm


def _vllm_generate(llm, tokenizer, conversations, *, max_new_tokens,
                   temperature, top_p, lora_request, max_seq_len):
    """The shared sampling contract (generate.py) on the vLLM engine.

    The context limit is the run's `max_seq_len`. One completion is drawn per
    conversation: a group's size comes from TRL's repeat sampler.
    """
    from vllm import SamplingParams

    from .generate import (TOOL_CALL_STOP, encode_prompts, fits_context,
                           render_prompts, turn_text, vllm_sampling_kwargs)

    sp = SamplingParams(**vllm_sampling_kwargs(
        tokenizer, max_new_tokens=max_new_tokens, temperature=temperature,
        top_p=top_p, stop=[TOOL_CALL_STOP]))
    all_ids = encode_prompts(render_prompts(conversations, tokenizer), tokenizer)
    # vLLM raises on a prompt at the context window. A conversation that no
    # longer fits gets an empty turn instead, and the episode loop forces the
    # final answer after `max_empty` of them.
    keep = fits_context(all_ids, max_seq_len)
    results = [""] * len(all_ids)
    if keep:
        prompts = [{"prompt_token_ids": all_ids[i]} for i in keep]
        outs = llm.generate(prompts, sp, lora_request=lora_request, use_tqdm=False)
        for i, o in zip(keep, outs):
            results[i] = turn_text(o.outputs[0].text, tokenizer)
    return results


def _vllm_sequence_logprobs(llm, sequences, prompt_lens, *, lora_request,
                            max_seq_len):
    """Behaviour logprobs from the engine that sampled the tokens.

    vLLM's echo mode (`prompt_logprobs=0`) scores the given token ids under
    the kernels and adapter that generated them. The HF model is a second
    implementation of the same policy and its logprobs differ slightly, so
    scoring with it would bias the importance ratio.

    The echo runs at temperature 1 with no top-p, so these are the sampling
    weights' untempered logprobs, while the trainer divides its logits by the
    sampling temperature (`_chunked_logps_entropies`, and TRL's own path).
    The ratio `_parity_loss` forms is therefore the policy at the sampling
    temperature over the same weights at temperature 1. The shipped rounds
    trained under this arithmetic.
    """
    from vllm import SamplingParams

    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    # A max_seq_len sequence plus the one mandatory output token overflows
    # the context, so seq[:limit] is scored and any token beyond it gets 0.0:
    # at most the last token of an at-cap sequence.
    limit = int(max_seq_len) - 1
    prompts = [{"prompt_token_ids": seq[:limit]} for seq in sequences]
    outs = llm.generate(prompts, sp, lora_request=lora_request, use_tqdm=False)
    results = []
    for out, seq, plen in zip(outs, sequences, prompt_lens):
        pl = out.prompt_logprobs  # entry i covers token i; entry 0 is None
        toks = []
        for i in range(plen, len(seq)):
            if i < limit and i < len(pl) and pl[i] is not None and seq[i] in pl[i]:
                toks.append(float(pl[i][seq[i]].logprob))
            else:
                toks.append(0.0)
        results.append(toks)
    return results


def make_rollout_func(state: dict):
    """Build TRL's `rollout_func` over `state`, the dict `_rollout_state`
    builds: the items or prefixes, the backend, the episode config, the
    objective and the engine settings."""

    def rollout_func(prompts: list, trainer) -> dict:
        import gc
        import shutil
        import time as _time

        import torch
        from vllm.lora.request import LoRARequest

        # Phase timers. These stamps split the rollout half of a step and
        # ParityGRPOTrainer's buckets the trainer half; both are logged as
        # t_* sums once per optimizer step. The recipe's smoke runner counts a
        # smoke's steps by its `t_gen` lines.
        _tm: dict = {}
        _t0 = _time.perf_counter()

        def _stamp(key: str) -> None:
            nonlocal _t0
            now = _time.perf_counter()
            _tm[key] = _tm.get(key, 0.0) + (now - _t0)
            _t0 = now

        # One rollout per entry: the sampler has already repeated each prompt
        # num_generations times, and adjacent entries form a group.
        expanded = [state["items_by_id"][str(p)] for p in prompts]

        tok = trainer.processing_class
        model = trainer.accelerator.unwrap_model(trainer.model)
        was_training = model.training
        model.eval()
        try:
            cfg: EpisodeConfig = state["ep_cfg"]
            llm = _ensure_vllm(state)
            if state.get("vllm_asleep"):
                # wake_up() re-maps the engine's pages, so the trainer's
                # cached blocks from the last step are released first; the
                # wake runs out of memory otherwise.
                gc.collect()
                torch.cuda.empty_cache()
                llm.wake_up()
                state["vllm_asleep"] = False
            _stamp("wake")  # engine build lands here on step 1
            # This step's adapter: an adapter-only save and a new LoRARequest
            # id, which forces a reload; max_loras=1 evicts the previous one.
            step_id = int(state.get("vllm_step", 0)) + 1
            state["vllm_step"] = step_id
            live = Path(trainer.args.output_dir) / "vllm_lora_live"
            if live.exists():
                shutil.rmtree(live)
            model.save_pretrained(str(live))
            lora_req = LoRARequest("rl", step_id, str(live))
            _stamp("adapter")

            def generate(conversations, max_new_tokens):
                return _vllm_generate(
                    llm, tok, conversations,
                    max_new_tokens=max_new_tokens,
                    temperature=trainer.args.temperature,
                    top_p=trainer.args.top_p,
                    lora_request=lora_req,
                    max_seq_len=state["max_seq_len"],
                )

            if state["prefix_by_id"]:
                from ..agent.loop import resume_episodes_batched, seed_episode
                seeded = [seed_episode(state["prefix_by_id"][str(p)], it, cfg,
                                       episode_id=str(p))
                          for p, it in zip(prompts, expanded)]
                episodes = resume_episodes_batched(
                    seeded, state["backend"], generate, cfg)
            else:
                episodes = run_episodes_batched(expanded, state["backend"],
                                                generate, cfg)
            _stamp("gen")  # multi-turn decode + tool execution, one bucket

            # Columns from the shared scorer. TRL recomputes the reward from
            # them through the same reward_fn, so only the columns are kept.
            _, cols = score_episodes(episodes, expanded, state["reward_fn"])
            prompt_ids, completion_ids, masks = [], [], []
            sequences, plens = [], []
            n_truncated = n_prompt_over = 0
            for ep in episodes:
                p_ids, c_ids, mask, turns = token_spans(ep.messages, tok)
                if ep.resumed_turns:
                    # A resumed episode's prefix turns are conditioning, not
                    # behaviour this rollout sampled: the mask takes them out
                    # of the loss, the ratio and the KL.
                    mask = zero_prefix_turns(mask, turns, ep.resumed_turns)
                cap = state["max_seq_len"] - len(p_ids)
                # Both cases are counted and printed. cap <= 0 means the
                # prompt alone fills max_seq_len: the row keeps an empty
                # completion.
                if cap <= 0:
                    n_prompt_over += 1
                    c_ids, mask = [], []
                else:
                    if len(c_ids) > cap:
                        # The tail cut off is the final turn, which holds
                        # the answer.
                        n_truncated += 1
                    c_ids, mask = c_ids[:cap], mask[:cap]
                prompt_ids.append(p_ids)
                completion_ids.append(c_ids)
                masks.append(mask)
                sequences.append(p_ids + c_ids)
                plens.append(len(p_ids))

            if n_truncated or n_prompt_over:
                print(f"  max_seq_len={state['max_seq_len']}: "
                      f"{n_truncated}/{len(episodes)} completions truncated, "
                      f"{n_prompt_over} prompts already over budget "
                      f"(those train on an EMPTY completion)")

            _stamp("score")  # score_episodes + token_spans/cap loop
            logprobs = _vllm_sequence_logprobs(llm, sequences, plens,
                                               lora_request=lora_req,
                                               max_seq_len=state["max_seq_len"])
            _stamp("echo")  # the behaviour-logprobs pass
        finally:
            model.train(was_training)
            if state["vllm_sleep"] and state.get("vllm_llm") is not None:
                # Level 1 moves the engine's weights to CPU memory and drops
                # its cache, so TRL's reference forward, every loss
                # microbatch and the optimizer step run with the engine's
                # pages unmapped. In the finally so a crash during generation
                # still frees the card; the try/except keeps a sleep failure
                # from masking the try block's own exception.
                _ts = _time.perf_counter()
                try:
                    state["vllm_llm"].sleep(level=1)
                    state["vllm_asleep"] = True
                except Exception as e:
                    print(f"  vllm sleep(level=1) failed: {e}")
                _tm["sleep"] = _time.perf_counter() - _ts

        acc = getattr(trainer, "_t_acc", None)
        if acc is None:
            from collections import defaultdict
            acc = trainer._t_acc = defaultdict(float)
        for _k, _v in _tm.items():
            acc[f"t_{_k}"] += _v

        out = {"prompt_ids": prompt_ids, "completion_ids": completion_ids,
               "logprobs": logprobs, "env_mask": masks, **cols}
        # TRL indexes every returned list by batch position, so each must
        # hold exactly one entry per prompt.
        for key, val in out.items():
            assert len(val) == len(prompts), \
                f"rollout returned {len(val)} '{key}' rows for {len(prompts)} prompts"
        return out
    return rollout_func


# ---------------------------------------------------------------- the loss

def _chunked_logps_entropies(unwrapped, input_ids, attention_mask,
                             logits_to_keep, *, chunk: int, temperature: float,
                             compute_entropy: bool):
    """Sequence-chunked stand-in for TRL's `_get_per_token_logps_and_entropies`.

    TRL's helper allocates the full [B, logits_to_keep, vocab] logits tensor
    at once, several gigabytes per sequence at this vocabulary and these
    rollout lengths. This returns the same values without it: one trunk
    forward to the final hidden states, then the unembedding and TRL's
    `selective_log_softmax` on `chunk` positions at a time under
    torch.utils.checkpoint, so backward re-projects one chunk at a time. The
    vocabulary dimension is never split, so chunked and unchunked agree to
    floating-point tolerance (tests/test_chunked_logps.py).

    `unwrapped` is the accelerator-unwrapped, still PEFT-wrapped model: the
    forward goes through the LoRA layers and honours a disabled adapter,
    which is how TRL's reference-logps pass reaches this code.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    from trl.trainer.utils import entropy_from_logits, selective_log_softmax

    K = int(logits_to_keep)
    if K <= 0:  # a fully prompt-over microbatch; nothing to score
        z = torch.zeros(input_ids.size(0), 0, device=input_ids.device)
        return z, (z if compute_entropy else None)
    chunk = max(1, int(chunk))

    decoder = unwrapped.get_decoder()
    out_embed = unwrapped.get_output_embeddings()
    h = decoder(input_ids=input_ids, attention_mask=attention_mask,
                use_cache=False).last_hidden_state
    # TRL's window: keep K+1 logits, drop the last (it predicts past the
    # sequence end) — position -(K+1)+i predicts token ids[:, -K+i].
    h = h[:, -(K + 1):-1, :]
    ids = input_ids[:, -K:]

    def _logits(h_c):
        return out_embed(h_c) / temperature

    def _project_logps(h_c, ids_c):
        return selective_log_softmax(_logits(h_c), ids_c)

    # Under no_grad (TRL's reference pass) there is no graph to checkpoint.
    use_ckpt = torch.is_grad_enabled() and h.requires_grad
    lp_chunks = []
    for s in range(0, K, chunk):
        h_c, ids_c = h[:, s:s + chunk, :], ids[:, s:s + chunk]
        lp_chunks.append(checkpoint(_project_logps, h_c, ids_c,
                                    use_reentrant=False)
                         if use_ckpt else _project_logps(h_c, ids_c))
    logps = torch.cat(lp_chunks, dim=1)

    ents = None
    if compute_entropy:
        # Entropy is only logged here, so it is re-projected under no_grad
        # and keeps no graph.
        with torch.no_grad():
            ents = torch.cat(
                [entropy_from_logits(_logits(h[:, s:s + chunk, :]))
                 for s in range(0, K, chunk)], dim=1)
    return logps, ents


def _parity_loss(per_token_logps, sampling_logps, ref_logps, advantages,
                 mask, kl_beta: float, denom: float, bessel: float = 1.0,
                 ratio_clip: float | None = None):
    """The loss on one microbatch, as pure tensor arithmetic so that
    tests/test_parity_loss.py can pin it against `reward.token_advantages` on
    CPU.

        loss    = -sum( ratio * adv_tok ) / denom
        ratio   = exp(logp_trainer - logp_sampling), clamped to [1/c, c]
        adv_tok = (bessel * A_seq - kl_beta * k1_tok) * mask / ntok_seq

    * `ratio` is the per-token importance ratio against the behaviour
      logprobs the rollout supplied: the engine's echo at temperature 1
      (`_vllm_sequence_logprobs`) under the trainer's logprobs at the
      sampling temperature. `ratio_clip` (config `is_ratio_clip`) is `c`, and
      None leaves the ratio unclipped. Sampler (vLLM) and trainer (HF) are
      two implementations whose logprobs differ slightly; the clip bounds
      what that difference, and the temperature gap, do to the ratio.
    * `k1_tok = logp_trainer.detach() - logp_ref` is subtracted from the
      advantage, so the KL penalty is linear in the gap.
    * `mask / ntok_seq`, with `denom` the rows times the accumulation steps,
      gives every sequence the same gradient mass whatever its length.
    * `bessel` (`_bessel`) turns TRL's sample-std z-scores into
      population-std ones.
    """
    import torch

    ntok = mask.sum(-1, keepdim=True).clamp(min=1.0)
    if ref_logps is not None and kl_beta:
        kl_tok = per_token_logps.detach() - ref_logps
    else:
        kl_tok = torch.zeros_like(per_token_logps)
    adv = advantages.reshape(-1, 1) * float(bessel)
    adv_tok = (adv - float(kl_beta) * kl_tok) * mask / ntok
    ratio = torch.exp(per_token_logps - sampling_logps)
    if ratio_clip:
        c = float(ratio_clip)
        ratio = ratio.clamp(min=1.0 / c, max=c)
    return -(ratio * adv_tok).sum() / float(denom)


def _bessel(num_generations: int, scale_rewards: str) -> float:
    """The factor `_parity_loss` multiplies TRL's advantages by: sqrt(n/(n-1))
    when TRL z-scores within groups (`scale_rewards: group`, a division by
    the sample standard deviation), 1 when it only centres (`none`).
    `_grpo_args` refuses any other value: TRL's `batch` divides by a sample
    standard deviation over the whole generation batch, which this factor
    does not correct."""
    n = int(num_generations)
    return (n / (n - 1)) ** 0.5 if scale_rewards == "group" and n > 1 else 1.0


@functools.cache
def parity_trainer_class():
    """`ParityGRPOTrainer`, the trainer `run_grpo` builds.

    Built on first call because its base class is TRL's, and this module must
    import without torch.
    """
    import time
    from collections import defaultdict

    import torch
    from trl import GRPOTrainer

    class ParityGRPOTrainer(GRPOTrainer):
        """TRL's GRPOTrainer with `_parity_loss` in place of TRL's loss, the
        sequence-chunked logps path, and the step-phase timers.

        Beyond TRL's arguments: `logps_chunk` (positions per projection slice;
        0 = TRL's full-logits path), `kl_beta` (the k1 KL weight in the
        advantage, the same value as GRPOConfig's `beta`, which makes TRL
        compute the reference logps), `ratio_clip` (None = unclipped) and
        `bessel` (`_bessel`).
        """

        def __init__(self, *args, logps_chunk: int, kl_beta: float,
                     ratio_clip: float | None, bessel: float, **kwargs):
            # Set before TRL's __init__, which may already run methods below.
            self._logps_chunk = int(logps_chunk)
            self._kl_beta = float(kl_beta)
            self._ratio_clip = ratio_clip
            self._bessel = float(bessel)
            super().__init__(*args, **kwargs)

        # Phase timers, trainer side: t_fwd_grad (policy microbatch forwards),
        # t_fwd_nograd (TRL's reference and old-logps passes) and t_microbatch
        # (the whole training_step). No cuda synchronize is issued at the
        # boundaries, so per-step sums are sound and the split between
        # forward and backward is approximate.
        def _t_bucket(self, key: str, dt: float) -> None:
            acc = getattr(self, "_t_acc", None)
            if acc is None:
                acc = self._t_acc = defaultdict(float)
            acc[key] += dt

        def _get_per_token_logps_and_entropies(self, *a, **kw):
            _t0 = time.perf_counter()
            try:
                return self._ptl_impl(*a, **kw)
            finally:
                self._t_bucket(
                    "t_fwd_grad" if torch.is_grad_enabled() else "t_fwd_nograd",
                    time.perf_counter() - _t0)

        def training_step(self, *a, **kw):
            _t0 = time.perf_counter()
            try:
                return super().training_step(*a, **kw)
            finally:
                self._t_bucket("t_microbatch", time.perf_counter() - _t0)

        def log(self, logs, *args, **kw):
            # Flush the sums into TRL's log line, and so trainer_state.json,
            # once per optimizer step. TRL runs the rollout and the reference
            # pass inside training_step, so t_microbatch contains every other
            # bucket and approximates the step's wall time; t_train_resid is
            # t_microbatch minus the others.
            acc = getattr(self, "_t_acc", None)
            if acc:
                inner = sum(v for k, v in acc.items() if k != "t_microbatch")
                if "t_microbatch" in acc:
                    logs["t_train_resid"] = round(
                        float(acc["t_microbatch"] - inner), 2)
                logs.update({k: round(float(v), 2) for k, v in sorted(acc.items())})
                acc.clear()
            super().log(logs, *args, **kw)

        def _ptl_impl(self, model, input_ids, attention_mask, logits_to_keep,
                      batch_size=None, compute_entropy=False, **kw):
            # Every full-logits forward of a step goes through this method:
            # _compute_loss below and TRL's reference and old-logps passes.
            # logps_chunk 0, or any extra keyword that is set, falls back to
            # TRL's original; **kw is passed through as it came because TRL's
            # keyword set varies by version.
            if (self._logps_chunk <= 0
                    or any(v is not None and v is not False for v in kw.values())):
                return super()._get_per_token_logps_and_entropies(
                    model, input_ids, attention_mask, logits_to_keep,
                    batch_size=batch_size, compute_entropy=compute_entropy, **kw)
            assert not getattr(self, "_entropy_bonus_enabled", False), (
                "chunked entropies are computed under no_grad; an entropy "
                "BONUS needs them differentiable — remove logps_chunk to use it")
            unwrapped = self.accelerator.unwrap_model(model)
            # TRL's reference pass sends the whole generation batch at once,
            # and the per-chunk logits tensor scales with rows x chunk x
            # vocab, so at most four rows are projected at a time.
            bs = min(int(batch_size or input_ids.size(0)), 4)
            lps, ents = [], []
            for i in range(0, input_ids.size(0), bs):
                lp, en = _chunked_logps_entropies(
                    unwrapped, input_ids[i:i + bs], attention_mask[i:i + bs],
                    logits_to_keep, chunk=self._logps_chunk,
                    temperature=self.temperature,
                    compute_entropy=compute_entropy)
                lps.append(lp)
                ents.append(en)
            return (torch.cat(lps, dim=0),
                    torch.cat(ents, dim=0) if compute_entropy else None,
                    None)

        def _compute_loss(self, model, inputs):
            prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
            completion_ids = inputs["completion_ids"]
            completion_mask = inputs["completion_mask"]
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)
            # TRL treats tool_mask as optional. Here it is required: the
            # rollout always returns env_mask, and without it the loss would
            # train on observation tokens. run_grpo's source check covers the
            # pop, not this name.
            assert "tool_mask" in inputs, (
                "env_mask did not reach the loss as `tool_mask` -- the "
                "installed TRL renamed it. Refusing to train on observation "
                "tokens.")
            mask = (completion_mask * inputs["tool_mask"]).float()
            per_token_logps, entropies, _ = self._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep,
                compute_entropy=True)
            sampling = inputs["sampling_per_token_logps"]
            ref = inputs.get("ref_per_token_logps") if self._kl_beta else None
            adv = inputs["advantages"]
            mode = "train" if self.model.training else "eval"
            accum = (self.current_gradient_accumulation_steps
                     if mode == "train" else 1)
            denom = prompt_ids.size(0) * accum
            loss = _parity_loss(per_token_logps, sampling, ref, adv, mask,
                                self._kl_beta, denom, bessel=self._bessel,
                                ratio_clip=self._ratio_clip)
            with torch.no_grad():
                msum = float(mask.sum().clamp(min=1.0))
                if self._ratio_clip is not None:
                    # The fraction of trained tokens whose ratio the clip
                    # changed.
                    c = self._ratio_clip
                    r = torch.exp(per_token_logps.detach() - sampling)
                    clipped = ((r > c) | (r < 1.0 / c)).float() * mask
                    self._metrics[mode]["ratio_clipped_frac"].append(
                        float(clipped.sum()) / msum)
                if ref is not None:
                    self._metrics[mode]["kl"].append(
                        float(((per_token_logps.detach() - ref) * mask).sum()) / msum)
                self._metrics[mode]["entropy"].append(
                    float((entropies.detach() * mask).sum()) / msum)
                # Mean |trainer - sampler| logprob gap over trained tokens.
                self._metrics[mode]["sampling_drift"].append(
                    float(((per_token_logps.detach() - sampling).abs() * mask).sum()) / msum)
                ratio = torch.exp(per_token_logps.detach() - sampling) * mask
                self._metrics[mode]["is_ratio_mean"].append(float(ratio.sum()) / msum)
            return loss

    return ParityGRPOTrainer


# ---------------------------------------------------------------- the config

def _grpo_args(config: dict) -> dict:
    """GRPOConfig kwargs from the config, pinned by
    tests/test_trainer_arithmetic.py. `run_grpo` reads the resolved values
    back from this dict.

    Set against TRL's defaults: `loss_type="grpo"`, the per-sequence mean
    `_parity_loss` computes (TRL's default is a token-level mean), a cosine
    schedule, and Adam's beta2 0.95 and epsilon 1e-12. `beta` is `kl_beta`;
    when it is non-zero TRL computes the reference logprobs.
    """
    model_dtype = str(config.get("model_dtype", "float32"))
    if model_dtype not in ("float32", "bfloat16", "float16"):
        raise SystemExit(f"model_dtype must be float32, bfloat16 or float16, "
                         f"got {model_dtype!r}")
    scale_rewards = str(config.get("scale_rewards", SHARED["scale_rewards"]))
    if scale_rewards not in ("group", "none"):
        raise SystemExit(f"scale_rewards must be group or none (the two "
                         f"`_bessel` corrects for), got {scale_rewards!r}")
    return dict(
        output_dir=str(Path(config["output_dir"])),
        per_device_train_batch_size=config.get("batch_size", 8),
        gradient_accumulation_steps=config.get("grad_accum", 8),
        learning_rate=float(config.get("lr", 1e-5)),
        lr_scheduler_type="cosine",
        # The config's warmup_ratio, passed as transformers' `warmup_steps`: a
        # value below 1 is a ratio of the run's steps, the same schedule, and
        # the `warmup_ratio` argument is deprecated.
        warmup_steps=float(config.get("warmup_ratio", SHARED["warmup_ratio"])),
        adam_beta2=0.95,
        adam_epsilon=1e-12,
        num_generations=int(config.get("num_generations", SHARED["num_generations"])),
        temperature=config["temperature"],
        top_p=config.get("top_p", SHARED["top_p"]),
        beta=float(config.get("kl_beta", 0.0)),
        loss_type="grpo",
        scale_rewards=scale_rewards,
        max_steps=config.get("max_steps", 200),
        logging_steps=config.get("logging_steps", 1),
        save_strategy="steps",
        save_steps=config.get("save_steps", 25),
        # Default 2; 0 or null in the config keeps every checkpoint.
        save_total_limit=(int(config["save_total_limit"] or 0) or None
                          if "save_total_limit" in config else 2),
        seed=config.get("seed", SHARED["seed"]),
        # These two must agree with the dtype of the trainer's weights, and
        # run_grpo reads that dtype back from them.
        bf16=model_dtype == "bfloat16",
        fp16=model_dtype == "float16",
        gradient_checkpointing=config.get("gradient_checkpointing", SHARED["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # max_seq_len, the rollout's context limit; run_grpo reads it back.
        max_completion_length=int(config.get("max_seq_len", 6144)),
        report_to=[],
        reward_weights=[1.0, 0.0, 0.0],      # reward + two weight-0 monitors
    )


# Every key this trainer reads. `validate_config_keys` refuses a config with
# any other key before anything loads; tests/test_module_health.py checks that
# every key here is read. `GRPO_RESOLVED_KEYS` are the ones the recipe derives
# from --graph/--size/--tag/--stage (graph_reasoner.recipe); a stage's
# hyperparameter file carries the others only, and its band `{lo, hi}` without
# `rollouts`.
KNOWN_CONFIG_KEYS = {
    "backend", "base_model", "batch_size", "benchmark", "chat_template",
    "connect_limit", "connect_max_hops", "env_seed", "exclude",
    "find_k", "grad_accum", "gradient_checkpointing",
    "hub_summary_limit", "hub_threshold", "informative_band",
    "is_ratio_clip", "kb", "kl_beta", "logging_steps", "logps_chunk",
    "lora_alpha", "lora_dropout", "lora_r", "lr", "max_seq_len", "max_steps",
    "max_turns", "model_dtype", "num_generations", "output_dir", "page_size",
    "partial_credit", "partial_on", "prefix_band", "prefixes",
    "rollout_mode", "save_steps", "save_total_limit", "scale_rewards", "seed",
    "substrate_adapters", "temperature", "tiers", "tools", "top_p",
    "turn_max_tokens", "vllm_gpu_frac", "vllm_max_num_seqs", "vllm_sleep",
    "warmup_ratio",
}
GRPO_RESOLVED_KEYS = frozenset({
    "base_model", "output_dir", "substrate_adapters", "benchmark", "exclude",
    "kb", "backend", "prefixes", "rollout_mode",
})


def validate_config_keys(config: dict) -> None:
    """Refuse a config with a key this trainer does not read."""
    check_config_keys(config, KNOWN_CONFIG_KEYS)


def _rollout_state(config: dict, max_seq_len: int, lora_r: int
                   ) -> tuple[dict, list[str]]:
    """Config -> (the rollout's state, the dataset's prompt ids).

    `rollout_mode: episode` (the default): each prompt is a benchmark item and
    a rollout is a fresh search. `prefix`: each prompt is a stored episode cut
    before an assistant turn (prefixes.py) and a rollout resumes it; only the
    continuation's turns carry loss.
    """
    state: dict = {"prefix_by_id": None}
    if config.get("rollout_mode", "episode") == "prefix":
        pairs = load_train_prefixes(config)
        state["items_by_id"] = {pf["id"]: it for pf, it in pairs}
        state["prefix_by_id"] = {pf["id"]: pf for pf, _ in pairs}
        train_ids = [pf["id"] for pf, _ in pairs]
        print(f"rollout mode: prefix — {len(train_ids)} prefixes "
              f"(continuation turns only carry loss)")
    else:
        items = load_train_items(config)
        state["items_by_id"] = {i.id: i for i in items}
        train_ids = [i.id for i in items]
    state["backend"] = load_backend(_require(config, "backend"),
                                    _require(config, "kb"))
    print(f"rollout backend: {state['backend'].name}")
    state["ep_cfg"] = episode_config(config)
    state["reward_fn"] = build_objective(config)
    state["max_seq_len"] = max_seq_len
    state["lora_r"] = lora_r
    state["vllm_gpu_frac"] = float(config.get("vllm_gpu_frac", 0.35))
    state["vllm_max_num_seqs"] = int(config.get("vllm_max_num_seqs", 64))
    # vllm_sleep time-shares the card: the engine sleeps (level 1) after each
    # rollout and wakes before the next, so its memory peak and the trainer's
    # never coincide. Off by default: on a card that holds both, sleeping
    # only costs the engine its prefix cache.
    state["vllm_sleep"] = bool(config.get("vllm_sleep", False))
    if state["vllm_sleep"]:
        print("vllm_sleep: engine sleeps (level 1) between rollouts -- "
              "trainer and engine peaks never co-reside")
    return state, train_ids


# ---------------------------------------------------------------- the run

def run_grpo(config: dict) -> None:
    """Config -> rollout state -> ParityGRPOTrainer -> `output_dir/final`.

    `config` is the dict scripts/train_traj_grpo.py resolved: the round's
    hyperparameter file plus the paths the recipe derives.
    """
    # The config is checked before the torch stack is imported, so a refusal
    # is immediate and works on a machine without torch.
    validate_config_keys(config)
    if not config.get("output_dir"):
        raise SystemExit("config must set output_dir: a run resumes from the "
                         "checkpoints it finds there, so no default is safe")
    if not config.get("substrate_adapters"):
        raise SystemExit(
            "substrate_adapters is empty. Point it at the SFT run's adapter "
            "dir(s): RL starts from a policy that already speaks the tool "
            "protocol.")
    if config.get("rollout_mode", "episode") not in ("episode", "prefix"):
        raise SystemExit(f"rollout_mode must be episode or prefix, "
                         f"got {config['rollout_mode']!r}")
    ratio_clip = float(config["is_ratio_clip"]) if config.get("is_ratio_clip") else None
    if ratio_clip is not None and ratio_clip <= 1.0:
        raise SystemExit(f"is_ratio_clip must be > 1 (a bound on the per-token "
                         f"ratio), got {ratio_clip}")
    grpo_kwargs = _grpo_args(config)
    lora_r = int(config.get("lora_r", SHARED["lora_r"]))

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig

    from ..eval.local_model import merge_chain_into
    from .adapters import copy_run_manifest, write_substrate_marker
    from .tokenize import override_chat_template
    from .vllm_ckpt import write_vllm_checkpoint

    refuse_broken_cuda(torch)

    # Refuse a TRL whose source lacks the env_mask pop the loss mask needs.
    import inspect
    import trl.trainer.grpo_trainer as _G
    assert 'extra_fields.pop("env_mask"' in inspect.getsource(_G), (
        "installed TRL does not support env_mask via rollout_func — "
        "the env-token mask would be silently dropped; refusing to train")

    out_dir = Path(grpo_kwargs["output_dir"])
    state, train_ids = _rollout_state(
        config, max_seq_len=grpo_kwargs["max_completion_length"], lora_r=lora_r)
    dataset = Dataset.from_list([{"prompt": pid} for pid in train_ids])

    # ---- model: merge the substrate chain, then a fresh LoRA on top
    base = config["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base)
    override_chat_template(tokenizer, config.get("chat_template"), model=base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # fp32 master weights by default. `model_dtype: bfloat16` halves the
    # trainer's weights, for a card that cannot hold the fp32 trainer beside
    # the engine. It changes the numerics of the objective: check KL and
    # reward against an fp32 run.
    dtype = (torch.bfloat16 if grpo_kwargs["bf16"]
             else torch.float16 if grpo_kwargs["fp16"] else torch.float32)
    print(f"trainer dtype: {dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=dtype, attn_implementation="sdpa")
    substrate = [str(a) for a in config["substrate_adapters"]]
    # merge_chain_into refuses an adapter whose tensors name modules the
    # model does not have (adapters.assert_adapter_resolves).
    model = merge_chain_into(model, substrate)
    for adapter in substrate:
        print(f"merged substrate adapter: {adapter}")

    # The engine serves base+substrate, and the training LoRA rides on top as
    # a hot-swapped LoRARequest. The checkpoint is written once per
    # output_dir (vllm_ckpt.py) and reused on resume.
    state["vllm_base"] = write_vllm_checkpoint(model, tokenizer, out_dir / "vllm_base")

    # `all-linear` resolves to twelve leaves on Qwen3.5: the SFT stages' ten
    # (defaults.SFT_TARGET_MODULES) plus the linear-attention mixer gates
    # in_proj_a and in_proj_b.
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=config.get("lora_alpha", SHARED["lora_alpha"]),
        lora_dropout=config.get("lora_dropout", SHARED["lora_dropout"]),
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    logps_chunk = int(config.get("logps_chunk", 0))
    kl_beta = grpo_kwargs["beta"]
    bessel = _bessel(grpo_kwargs["num_generations"], grpo_kwargs["scale_rewards"])
    print(f"loss: parity — true IS ratio vs sampling logprobs "
          f"(clip {ratio_clip or 'off'}), k1-in-advantage KL (beta={kl_beta}), "
          f"per-sequence norm, bessel {bessel:.4f}, logps_chunk="
          f"{logps_chunk if logps_chunk else 'off (full logits)'}")
    class _MarkSubstrate(TrainerCallback):
        """`substrate.json` into every checkpoint the trainer saves, so a
        checkpoint served on its own carries the chain it is a delta on."""

        def on_save(self, args, state, control, **kwargs):
            ck = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if ck.is_dir():
                write_substrate_marker(ck, substrate)
            return control

    trainer = parity_trainer_class()(
        model=model,
        args=GRPOConfig(**grpo_kwargs),
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[state["reward_fn"], find_rate_monitor, gate_pass_monitor],
        rollout_func=make_rollout_func(state),
        peft_config=peft_config,
        callbacks=[_MarkSubstrate()],
        logps_chunk=logps_chunk, kl_beta=kl_beta, ratio_clip=ratio_clip,
        bessel=bessel,
    )
    assert not trainer.tools, "tools must be unset or TRL discards env_mask"

    last = get_last_checkpoint(str(out_dir)) if out_dir.exists() else None
    trainer.train(resume_from_checkpoint=last)

    final = out_dir / "final"
    trainer.save_model(str(final))
    write_substrate_marker(final, substrate)
    copy_run_manifest(out_dir, final)
    print(f"Adapter saved -> {final}")
