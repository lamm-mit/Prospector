#!/usr/bin/env python
"""Pick among k rollouts per item with a trained yes/no verifier.

    python scripts/select/verifier_select.py --verifier hf:models/verifier_sft_2b_flash20v/final \\
        --name q2b-verifier32-yn1 \\
        --runs data/runs/q2bpool32-s101--tsmc_core_agentic ... data/runs/q2bpool32-s132--tsmc_core_agentic

Scores locally, on CUDA when available: the verifier's adapter is merged into
its base in fp32, and each offered candidate takes two forward passes and no
generation (`graph_reasoner.select.verifier`). A candidate is a rollout's
answer with the chain it cited, and the pick is the largest margin
log P(yes) - log P(no). The offered candidates are an item's gate-passing,
non-abstaining rollouts (all of them when there is none), so the gate's
`cited subset seen` check comes before the verifier ranks anything, and the
selection reads nothing from the benchmark.

Writes `<out-root>/<name>--<benchmark>/`: `completions.jsonl` (the picked
answer per item), `scores.jsonl` (one row per scored candidate) and
`summary.json`, which records `runs_in_order`. The order of `--runs` matters:
a score row's `cand_idx` is a position in the pool built from those runs in
that order, and the paper's figure script rebuilds the pool from it.

Prints gated_acc and em_claim for the verifier's picks beside the first run,
the majority vote and the oracle, and the margin's rank AUC against each
candidate's gold-scored label. The AUC is a diagnostic and takes no part in
the selection.

The verifier's template comes from its run.json; `--chat-template` names one
for a verifier that records none, and must agree with the record otherwise.
"""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path


def _local_scorer(ref: str, chat_template: str | None):
    """(score_ids, tokenizer) for `select.verifier.score_pool`: `score_ids(ids)`
    -> log P(ids[j] | ids[:j]) per position, None at 0.

    `ref` is the verifier's adapter directory, which names its own base; the
    adapter is merged into the base in fp32.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from graph_reasoner.eval.local_model import merge_chain_into
    from graph_reasoner.train.adapters import (adapter_chain, chat_template_for, resolve_base,
                                                tokenizer_source)
    from graph_reasoner.train.tokenize import override_chat_template

    model_name, adapter = resolve_base(ref, None)
    tok_src = tokenizer_source(model_name, adapter)
    tok = AutoTokenizer.from_pretrained(tok_src)
    override_chat_template(tok, chat_template_for(model_name, adapter, chat_template), model=tok_src)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading local verifier: {model_name}" + (f" + {adapter}" if adapter else "")
          + f" on {device}, fp32")
    m = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32,
                                             attn_implementation="sdpa")
    m = merge_chain_into(m, adapter_chain(adapter))
    m.to(device).eval()

    @torch.no_grad()
    def score_ids(full: list[int]) -> list:
        ids = torch.tensor([full], device=device)
        logits = m(input_ids=ids).logits[0].float()
        lp = torch.log_softmax(logits[:-1], dim=-1)   # row j-1 predicts ids[j]
        return [None] + lp.gather(1, ids[0, 1:, None]).squeeze(1).tolist()

    return score_ids, tok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verifier", required=True,
                    help="hf:<adapter dir> of a TRAINED yes/no verifier")
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--name", default="verifier-select")
    ap.add_argument("--out-root", type=Path, default=Path("data/runs"),
                    help="where the scoring run directory goes, as run_agent_eval.py's")
    ap.add_argument("--batch", type=int, default=32,
                    help="items per progress line")
    ap.add_argument("--chat-template", default=None,
                    help="template override for the scoring tokenizer "
                         "(plain_chatml for Qwen3.5 verifiers, whose shipped "
                         "template rewrites history; MUST match the template "
                         "the verifier was trained under, or it scores "
                         "off-distribution with a weak AUC as the only symptom)")
    a = ap.parse_args()
    if not a.verifier.startswith("hf:"):
        ap.error("the verifier is scored locally: give hf:<adapter dir>")

    from graph_reasoner import select as bn
    from graph_reasoner.benchmarks import run_dir_name, split_run_name
    from graph_reasoner.select.verifier import score_pool, score_rows

    runs = [bn.load(r) for r in a.runs]
    # Only items present in every run are scored.
    ids = sorted(set(runs[0]).intersection(*(set(r) for r in runs)))
    if len(ids) < len(runs[0]):
        print(f"note: {len(runs[0]) - len(ids)} item(s) missing from some "
              f"run, excluded")
    score_ids, tok = _local_scorer(a.verifier[len("hf:"):], a.chat_template)

    # The output directory takes its benchmark suffix from the runs being scored.
    bench_suffix = split_run_name(a.runs[0].name)[1]
    out_dir = a.out_root / run_dir_name(a.name, bench_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    picked, n_asked, scores = [], 0, []
    with (out_dir / "completions.jsonl").open("w") as fh:
        for n_done, i in enumerate(ids, 1):
            pool = bn.offered([r[i] for r in runs])
            # A pool of one is not scored: its candidate is the pick.
            choice = pool[0]
            if len(pool) >= 2:
                margins = score_pool(score_ids, tok, runs[0][i]["question"], pool)
                n_asked += 2 * len(pool)
                choice = pool[bn.pick_by_margin(margins)]
                scores += score_rows(i, pool, margins)
            picked.append(choice)
            fh.write(json.dumps({"id": i, "completion": choice["answer"]},
                                ensure_ascii=False) + "\n")
            if n_done % a.batch == 0 or n_done == len(ids):
                print(f"  {n_done}/{len(ids)}", flush=True)

    if scores:
        with (out_dir / "scores.jsonl").open("w") as sfh:
            for srow in scores:
                sfh.write(json.dumps(srow) + "\n")
        for label in ("gated_acc", "em_claim"):
            pos = [s["margin"] for s in scores if s[label]]
            neg = sorted(s["margin"] for s in scores if not s[label])
            if pos and neg:
                # Rank AUC (Mann-Whitney), ties half credit. A diagnostic:
                # gold-scored labels are read here, never during selection.
                u = sum(bisect.bisect_left(neg, m)
                        + (bisect.bisect_right(neg, m) - bisect.bisect_left(neg, m)) / 2
                        for m in pos)
                print(f"\nmargin AUC vs {label} label: "
                      f"{u / (len(pos) * len(neg)):.3f} "
                      f"({len(pos)} pos / {len(neg)} neg candidates) — diagnostic, "
                      f"not part of selection")
    # The verifier and the --runs order, which cand_idx is a position in. Written
    # last and whole: the runner takes summary.json as the mark of a finished scoring.
    tmp = out_dir / "summary.json.tmp"
    tmp.write_text(json.dumps({
        "verifier": a.verifier, "chat_template": a.chat_template,
        "runs_in_order": [str(r) for r in a.runs],
    }, indent=2) + "\n")
    tmp.replace(out_dir / "summary.json")

    def rate(rows, key):
        return sum(bool(r[key]) for r in rows) / len(rows)
    first = [runs[0][i] for i in ids]
    vote = [bn.select([r[i] for r in runs]) for i in ids]
    # The selection ceiling is the oracle over the offered pool. The em_claim
    # oracle over every candidate is a separate row: it includes gate-fails
    # the selector never ranks.
    offered_pools = [bn.offered([r[i] for r in runs]) for i in ids]
    oracle = [{k: any(c[k] for c in pool) for k in ("gated_acc", "em_claim")}
              for pool in offered_pools]
    print(f"\n{len(ids)} items, {len(runs)} rollouts each, verifier asked on {n_asked}")
    print(f"{'selector':26s} {'gated_acc':>8s} {'em_claim':>8s}")
    for name, rows in (("run 0 (k=1)", first), ("majority vote", vote),
                       ("VERIFIER", picked), ("oracle (offered pool)", oracle)):
        print(f"{name:26s} {rate(rows, 'gated_acc'):8.3f} {rate(rows, 'em_claim'):8.3f}")
    every = sum(any(r[i]["em_claim"] for r in runs) for i in ids) / len(ids)
    print(f"{'oracle (all candidates)':26s} {'':8s} {every:8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
