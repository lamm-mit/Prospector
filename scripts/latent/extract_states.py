#!/usr/bin/env python
"""Read the residual stream after every decoder layer at each turn's read
position, for every episode of one or more stored runs. A GPU job (an Apple
GPU serves too, slowly).

    python scripts/latent/extract_states.py data/runs/<run> [data/runs/<run2> ...] \\
        --model Qwen/Qwen3.5-4B --adapter <stack> \\
        --layers 0-32 --out data/runs/_probe/pilot/policy_all

`--adapter` names the served stack; it is merged once into
`<last adapter>/hf_merged` (the merge the mlx: driver serves) and read from
there, or `--model` names a merged checkpoint directly. Each run gets
`<out>/<run>/` with index.jsonl, one L<n>.npy per read point, the per-episode
shards (an existing shard is kept, so a stopped job resumes), drops.jsonl and
meta.json. The reading is `graph_reasoner.latent.states`; the labels of the
reads' moves come next (scripts/latent/label_moves.py).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from graph_reasoner.latent.states import HFReader, extract_run, rendering_tokenizer


def parse_layers(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="stored run directories (data/runs/<run>)")
    ap.add_argument("--model", required=True, help="the base model, or a merged checkpoint when --adapter is absent")
    ap.add_argument("--adapter", default=None, help="the adapter chain to merge and read (comma-separated)")
    ap.add_argument("--tokenizer", default=None, help="where the rendering tokenizer loads from (default: the read weights)")
    ap.add_argument("--chat-template", default=None, help="a named template to install on the tokenizer (plain_chatml)")
    ap.add_argument("--out", type=Path, required=True, help="output root; each run gets a subdirectory")
    ap.add_argument("--layers", default="0-32", help="read points, as ranges and lists (0 = the embedding output)")
    ap.add_argument("--max-tokens", type=int, default=16384, help="transcripts longer than this are dropped and listed")
    ap.add_argument("--device", default=None, help="torch device (default: cuda, else mps, else cpu)")
    ap.add_argument("--limit", type=int, default=None, help="first N episodes of each run")
    ap.add_argument("--ids", default=None, help="comma-separated item ids to keep")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    weights = a.model
    if a.adapter:
        from graph_reasoner.eval.local_model import merged_hf_checkpoint
        weights = merged_hf_checkpoint(a.model, a.adapter, a.chat_template)
    layers = parse_layers(a.layers)
    tokenizer = rendering_tokenizer(a.tokenizer or weights, a.chat_template)
    print(f"Loading hf model: {weights}", flush=True)
    reader = HFReader(weights, device=a.device)
    if max(layers) > reader.n_layers:
        raise SystemExit(f"--layers names layer {max(layers)} but the model has {reader.n_layers}")
    ids = set(a.ids.split(",")) if a.ids else None
    for run in a.runs:
        print(f"{run.name}: reading", flush=True)
        extract_run(run, a.out / run.name, reader, tokenizer, layers, max_tokens=a.max_tokens,
                    limit=a.limit, ids=ids, log=lambda s: print(s, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
