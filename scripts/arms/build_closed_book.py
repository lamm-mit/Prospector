#!/usr/bin/env python
"""Build the closed-book arm: the same questions with no graph at all.

    python scripts/arms/build_closed_book.py data/benchmarks/tsmc/tsmc_core_agentic.jsonl
    # -> data/benchmarks/tsmc/tsmc_core__closed_book.jsonl, unless --out

The source is a graph's evaluation file. The output, which is built and not
tracked, is that file's questions with the evidence emptied, with a
`<out>.provenance.json` sidecar beside it. scripts/arms/run_static_api.py
answers it and scripts/arms/score_static_gated.py scores the answers against
the evaluation file's answer sets.

Output ids are `<base>#closed_book`. The scorer joins on the part before the
`#`, so the separator must stay.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from graph_reasoner import provenance
from graph_reasoner.benchmarks import load_benchmark
from graph_reasoner.schema import BenchmarkItem


def closed_book(item: BenchmarkItem) -> BenchmarkItem:
    """The same question with the evidence emptied.

    `meta.perturbation` marks the item so `static_prompts.build_eval_prompt`
    renders the question-only prompt. Its `expected` field records that a
    correct answer here means the model knew it without the graph.
    """
    meta = dict(item.meta)
    meta["perturbation"] = {
        "base_id": item.id,
        "variant": "closed_book",
        "prompt_mode": "question_only",
        "expected": "abstain_or_wrong; correct answer = parametric leakage",
    }
    return BenchmarkItem(
        id=f"{item.id}#closed_book",
        tier=item.tier,
        question=item.question,
        gold_answer=item.gold_answer,
        gold_path=item.gold_path,
        subgraph_edges=[],
        subgraph_text="",
        meta=meta,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmark", help="source benchmark JSONL")
    ap.add_argument("--out", default=None,
                    help="default: <dir>/<stem-without-_agentic>__closed_book.jsonl")
    args = ap.parse_args()

    src = Path(args.benchmark)
    items = load_benchmark(src)
    out = Path(args.out) if args.out else \
        src.parent / f"{src.stem.replace('_agentic', '')}__closed_book.jsonl"

    derived = [closed_book(it) for it in items]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for it in derived:
            f.write(json.dumps(it.model_dump(), ensure_ascii=False) + "\n")
    load_benchmark(out)          # parse back, so a malformed write fails here

    provenance.write_sidecar(out, provenance.block(
        "scripts/arms/build_closed_book.py", sys.argv[1:],
        extra={"source_benchmark": str(src), "n_items": len(derived)}))
    print(f"{len(derived)} items -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
