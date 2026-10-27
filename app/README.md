# Prospector app

The app is one page with two modes, live and replay, which the top-level
README introduces under "Serving Prospector". Live mode is the default, and
`#replay` in the URL opens replay mode.

    python -m uvicorn app.server:app --port 8321
    open http://localhost:8321          # or http://localhost:8321/#live

Live mode needs a model engine: mlx-lm in the `.venv-mac` environment on a
Mac, or vLLM in the `.venv` that `launch/setup.sh` builds on a machine with
an NVIDIA GPU. Setup in the top-level README builds both. In a `.venv` with
only `.[dev,app]`, the server starts and replay mode works, and live mode
reports in its status pill which engine is missing.

Stop the server with Ctrl+C. On a GPU machine it then stops live mode's
vLLM engine, which runs in a process of its own and would otherwise keep the
card's memory.

The server listens on localhost only. To use it on a remote GPU machine,
forward the port from your own computer and open the address there:

    ssh -L 8321:localhost:8321 <machine>
    open http://localhost:8321

## Replay mode

Replay mode lists the runs under `data/runs/` that carry a
`trajectories.jsonl` (the agentic runs), newest
first, and each run's items. The arrow keys step through an item's episode, and the space bar plays
it. The transcript matches each assistant turn to its observation by call
number, so a malformed or empty turn appears as a note at the step where it
happened. A run that served an adapter chain is labeled `Prospector-<size>`,
and any other run by the model string it stored.

Replay re-runs no tool. The server parses each call's new edges from the
recorded observation text and keeps only those in the record's own
`seen_edges`. A seen edge that the parser misses raises a warning badge. The
✓/✗ marks and the answer card show the verdicts stored in the run's
`results.jsonl`.

## The map

The map draws the whole graph under the episode's seen subgraph. The layout
packs the graph's Louvain communities as discs on a spiral, largest first,
and force-lays each community inside its own disc. Edges between communities
are drawn faintly. The server computes the layout from the backend's own
graph on the first request and caches it in `app/cache/`, keyed by the graph
fingerprint that the trajectory records. Node positions stay fixed across
episodes and runs.

Replay mode frames the whole map, live mode follows the seen subgraph, and ⛶
switches between the two framings. The page labels the current step's nodes,
the nodes the model called a tool on, the anchor and, at the final step, the
gold answer. Hovering shows any other node's name and a seen edge's relation.
Seen edges on the gold path are dashed gold, and the anchor has an ochre
ring. At the final step the map draws the gate's check: cited edges that were
seen turn teal, and cited edges never seen appear red and dotted. A gold
answer that the episode reached is then filled green.

The server finds the graph file by fingerprint, trying the run's own
`kb_path`, then live mode's graph, then every registered graph's default
file. A fingerprint that matches no file gets no map. A graph over 40,000
nodes or 60,000 edges is drawn as its 40,000 best-connected nodes. The layout
uses every edge among them, and the map draws a fixed sample of at most
60,000 of those edges. A node of the episode that is not on the map is still
drawn, off the layout. The page draws the graph with Cytoscape.js 3.33.1 (MIT
license), vendored as `static/vendor/cytoscape.min.js`.

## Live mode

Live mode serves the Prospector-4B chain that `configs/README.md` lists under
"The served stacks". The server loads the chain through one of the drivers
of `scripts/run_agent_eval.py`: `LocalMLXModel` (`mlx:`) on a Mac, or
`LocalVLLMModel` (`vllm:`) on a GPU machine, whichever engine is installed
unless `GR_LIVE_DRIVER` names one. It runs the chain under the paper's
protocol: greedy decoding, `plain_chatml`, 20 tool calls and 1,024 tokens a
turn. Each episode runs through `drive_episode`, so `advance()` applies
budgets, repeats and forced finals as in a scored run. The server streams
what changed to the page between generate calls.

- The graph loads before the model, so the map appears first. The model loads
  on the first entry to live mode and stays resident, and its first load
  merges the adapter chain. The status strip names the graph and the chain,
  and its pill shows the serving status.
- The transcript pane offers four benchmark questions as one-click starts.
  The ask box takes any question and offers the benchmark's questions as
  completions. A benchmark question is graded against its gold answer. A
  question outside the benchmark is ad hoc. The gate alone checks an ad-hoc
  question, and its answer card is marked "ungraded" in ochre.
- Each tool call appears as a card when the loop applies it. On a Mac, the
  model's text streams into a dashed card during decoding, and the Stop
  button aborts an episode between tokens. The vLLM driver returns a turn
  whole, so on a GPU machine the card shows the turn when it is done, and
  Stop takes effect between turns. A stopped episode is not saved.
- The server runs one episode at a time. A second question asked during an
  episode is refused with a message.
- A finished episode's URL is its replay link, `#replay/<run>/<id>`. A link
  with `?q=…&budget=…` fills the ask box and the budget box without starting
  an episode.
- The `?` in the header explains the tools, the seen contract and the gate.

The `budget` box sets the tool-call budget announced to the model, up to 200
calls. The chain was trained and scored at 20 calls, and a different budget
is a different policy, so every record and results row carries the budget it
ran under. The seen-edge cap is 400 edges up to 20 calls and 20 edges per
call above that, which is 2,000 edges at 100 calls.

The server appends every finished episode to `data/runs/live-mac--<benchmark>/`
on a Mac, or `data/runs/live-gpu--<benchmark>/` on a GPU machine, as a
`trajectory_record` and a results row, and replay mode lists it at once.
Live mode takes a call's new edges from the growth of the seen set, and each
turn event says whether the replay parser agrees. The run's summary
aggregates the benchmark questions at the default budget, adds one aggregate
per budget, and carries a `not_a_benchmark` note. An ad-hoc question's row
carries `gated_acc: null` and enters no aggregate.

Environment variables override live mode's defaults, and `app/live.py` reads
them when the server starts. The benchmark file decides the substrate, whose
default graph live mode loads unless `GR_LIVE_KB` names another file.

| variable | sets | default |
|---|---|---|
| `GR_LIVE_DRIVER` | the driver, `mlx` or `vllm` | `mlx` where mlx-lm is installed, else `vllm` where vLLM is |
| `GR_LIVE_MODEL` | the base model | `Qwen/Qwen3.5-4B` |
| `GR_LIVE_ADAPTER` | the adapter chain, in load order | the 4B chain of "The served stacks" |
| `GR_LIVE_CHAT_TEMPLATE` | the chat template | `plain_chatml` |
| `GR_LIVE_BENCHMARK` | the questions and the substrate | `data/benchmarks/tsmc/tsmc_core_agentic.jsonl` |
| `GR_LIVE_KB` | the graph file | the substrate's default graph |
| `GR_LIVE_MAX_TURNS` | the default budget | 20 calls |
| `GR_LIVE_MAX_MODEL_LEN` | the driver's context length | 131,072 tokens |
