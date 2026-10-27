#!/usr/bin/env python
"""Probe: can vLLM sleep mode time-share one GPU with a trainer?

Usage: probe_vllm_sleep.py <base_model> <adapter_dir> [gpu_frac=0.35]

Builds the engine with sleep mode and LoRA and runs three cycles of the
per-step pattern train/grpo.py uses under `vllm_sleep`:

    wake -> fresh LoRARequest (a new id forces a reload) -> generate ->
    sleep(level=1) -> allocate 30 GiB with torch in the freed space ->
    free it -> wake ...

The sleep arm generates twice per cycle. A control arm makes the same greedy
calls without sleeping, before and after the sleep arm. A call deviates when
its text differs from the first control call's, and the deviation is measured
as the largest per-token logprob difference over the shared prefix. Each
cycle prints the memory in use awake and asleep.

The probe fails when the 30 GiB allocation fails while the engine sleeps, when
a call's output is degenerate or deviates from the control by more than 0.5
nats, or when every sleep-arm call deviates and no control call does. A
smaller deviation is reported and tolerated as nondeterminism.

<adapter_dir> must name its target modules as the installed transformers
does (the fused in_proj_qkv); vLLM refuses the unfused in_proj_q/k/v.
scripts/infra/make_probe_lora.py builds such an adapter.

Needs vllm and a CUDA GPU. `--help` or a missing positional prints this text.
"""
import subprocess
import sys
import time


def _smi_used_gib() -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    return float(out) / 1024.0


def main() -> int:
    if len(sys.argv) < 3 or "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        return 0 if ("--help" in sys.argv or "-h" in sys.argv) else 2

    import torch
    from transformers import AutoTokenizer

    from graph_reasoner.train.tokenize import override_chat_template

    base, adapter = sys.argv[1], sys.argv[2]
    frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
    # The base tokenizer with the template override, as the GRPO trainer loads it.
    tok = AutoTokenizer.from_pretrained(base)
    override_chat_template(tok, "plain_chatml", model=base)

    CONVS = [
        [{"role": "system", "content": "You are a graph agent. Use tools."},
         {"role": "user", "content": f"Which process step follows {x}? Think, then call a tool."}]
        for x in ("etch", "lithography", "deposition", "CMP")
    ]
    texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
             for c in CONVS]
    ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=base, enable_lora=True, max_loras=1, max_lora_rank=16,
              dtype="bfloat16", enable_prefix_caching=True,
              gpu_memory_utilization=frac, max_num_seqs=64,
              enable_sleep_mode=True, max_model_len=16384)
    sp = SamplingParams(temperature=0.0, max_tokens=192, logprobs=1)  # greedy: see docstring
    MAX_DLP = 0.5  # nats; the largest per-token logprob deviation tolerated as nondeterminism
    print(f"--- engine up (frac {frac}): {_smi_used_gib():.1f} GiB used")

    asleep_marks = []
    lora_id = [0]

    def cycles(tag, do_sleep):
        """Three cycles of the trainer's pattern; the sleep arm generates twice per
        cycle and sleeps, allocates and wakes after the second. Returns one
        (sub, texts, logprobs, token ids) per call."""
        out_texts = []
        for cycle in (1, 2, 3):
          for sub in ("a", "b") if do_sleep else ("a",):  # sleep arm: two calls per cycle
              lora_id[0] += 1  # fresh id every call, as the trainer does
              lora = LoRARequest("rl", lora_id[0], adapter)
              # Both arms recompute the prompt's KV on every call: sleep(1)
              # discards the KV pool, so a control arm that reused its prefix
              # cache would not be comparable with the sleep arm.
              llm.reset_prefix_cache()
              t0 = time.time()
              outs = llm.generate([{"prompt_token_ids": i} for i in ids], sp,
                                  lora_request=lora, use_tqdm=False)
              dt = time.time() - t0
              ntok = sum(len(o.outputs[0].token_ids) for o in outs)
              texts_out = [o.outputs[0].text for o in outs]
              assert len("".join(texts_out).strip()) >= 20, (
                  f"[{tag}{cycle}] DEGENERATE OUTPUT: {texts_out!r}")
              out_texts.append((sub, texts_out, [[d[t].logprob for d, t in zip(o.outputs[0].logprobs or [], o.outputs[0].token_ids)] for o in outs],
                                [list(o.outputs[0].token_ids) for o in outs]))
              print(f"[{tag}{cycle}] {ntok} tok in {dt:.1f}s ({ntok / dt:.0f} "
                    f"tok/s), awake {_smi_used_gib():.1f} GiB | "
                    + texts_out[0][:110].replace("\n", " | "))
              if do_sleep and sub == "b":
                  t0 = time.time()
                  llm.sleep(level=1)
                  used = _smi_used_gib()
                  asleep_marks.append(used)
                  print(f"[{tag}{cycle}] sleep(1) in {time.time() - t0:.1f}s -> "
                        f"{used:.1f} GiB resident asleep")
                  # The trainer's role: claim a big block of the freed memory.
                  big = torch.empty(int(30 * 2**30 // 2), dtype=torch.float16,
                                    device="cuda")
                  big.zero_()
                  print(f"[{tag}{cycle}] trainer-side 30 GiB allocation OK "
                        f"while asleep ({_smi_used_gib():.1f} GiB used)")
                  del big
                  torch.cuda.empty_cache()
                  t0 = time.time()
                  llm.wake_up()
                  print(f"[{tag}{cycle}] wake_up in {time.time() - t0:.1f}s "
                        f"-> {_smi_used_gib():.1f} GiB")
        return out_texts

    # A control arm before and after the sleep arm: identical calls, no sleep,
    # so a deviation can be told apart from sleep and from call order.
    control = cycles("ctl", do_sleep=False)
    slept = cycles("slp", do_sleep=True)
    control2 = cycles("ct2", do_sleep=False)

    ref_texts, ref_lps, ref_ids = control[0][1], control[0][2], control[0][3]
    def max_dlp(lps, ids):  # max |logprob delta| against the control over each sequence's shared prefix
        m = 0.0  # (after the first differing token the two runs score different tokens, so deltas there mean nothing)
        for a, b, ia, ib in zip(ref_lps, lps, ref_ids, ids):
            n = next((k for k in range(min(len(ia), len(ib))) if ia[k] != ib[k]), min(len(ia), len(ib)))
            for x, y in zip(a[:n], b[:n]):
                m = max(m, abs(x - y))
        return m
    def report(name, calls):
        dev = [(i + 1, max_dlp(c[2], c[3])) for i, c in enumerate(calls) if c[1] != ref_texts]
        print(f"{name}: {len(calls)} calls, {len(dev)} deviate from control call 1"
              + (": " + ", ".join(f"call {i} max |dlogprob| {m:.3f} nats" for i, m in dev) if dev else ""))
        return dev
    d_ctl = report("control arm (before)", control)
    d_slp = report("sleep arm (two calls per wake)", slept)
    d_ct2 = report("control arm (after)", control2)
    worst = max((m for _, m in d_ctl + d_slp + d_ct2), default=0.0)
    if worst > MAX_DLP:
        print(f"PROBE FAIL: a call deviates by {worst:.2f} nats on its shared prefix (> {MAX_DLP}) — not a sporadic flip")
        sys.exit(1)
    if len(d_slp) == len(slept) and not d_ctl and not d_ct2:
        print("PROBE FAIL: EVERY sleep-arm call deviates and no control call does — a state that did not survive sleep/wake")
        sys.exit(1)
    if d_slp or d_ctl or d_ct2:
        arms = [n for n, d in (("control before", d_ctl), ("sleep", d_slp), ("control after", d_ct2)) if d]
        print(f"NOTE: deviations within {MAX_DLP} nats (worst {worst:.3f}) in {', '.join(arms)}: "
              "tolerated as the stack's batch-dependent nondeterminism.")
    print("PROBE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
