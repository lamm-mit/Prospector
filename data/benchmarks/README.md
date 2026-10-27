# data/benchmarks/: the questions

This folder holds the question files, one subfolder per graph. Two kinds of
file are committed:

| file | what it is |
|---|---|
| `<graph>/<graph>_core_agentic.jsonl` | the evaluation questions: 195 on the semiconductor graph (`tsmc`), 210 on `biograph`, 228 on `pfas`, 223 on `matprop` |
| `tsmc/tsmc_train_agentic_4x.jsonl` | the 3,971 training questions, all on the semiconductor graph. None shares an id, a question or a gold path's node set with an evaluation question (`tests/test_item_ids.py`) |

Each line is one question, with its `id`, its `tier`, the `question`, the
`gold_answer` and the `gold_path`. The tier is easy, medium or hard, for a
gold path of two, three or four hops. In `meta`, `anchor` is the entity the
question is about. `kb_answer_set` holds every answer that the gold relation
sequence reaches over the full graph, and an answer is scored against this
set. `subgraph_edges` holds the gold path among distractor edges, and
`subgraph_text` the same edges as text. The agent's prompt carries only the
question and never this subgraph.

The committed files are the ones the paper's runs read. Rebuilt from its
config under `configs/benchmark/`, a transfer graph's file has the same gold
paths, but the teacher model phrases the questions afresh because its cached
replies are not released. The semiconductor files do not rebuild. From the
seed in `tsmc_v0.yaml`, the current sampler draws gold paths that differ from
the committed ones, and no config holds the training questions' seeds.

The two baselines read files that are built in this folder and not
committed. Each carries the evaluation questions under their ids plus a `#`
suffix, which the scorers strip:

```bash
python scripts/arms/build_closed_book.py data/benchmarks/tsmc/tsmc_core_agentic.jsonl   # tsmc/tsmc_core__closed_book.jsonl
python scripts/arms/build_retrieval_benchmark.py --graph tsmc --rule khop --budget 400  # tsmc/tsmc_core__khop400anchor.jsonl
```

On another graph, name that graph's evaluation file in the first command and
pass `--graph <graph> --rule ppr` in the second. The main README's section
[Training on your own graph](../../README.md#training-on-your-own-graph)
shows how to build the evaluation file for a new graph.
