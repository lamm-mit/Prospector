#!/usr/bin/env python
"""Run the agentic evaluation: the subgraph is hidden, the model explores it with tools.

The model string selects the driver:
    hf:<model-or-adapter-path>                   local HF model (HF generate)
    vllm:<model-or-adapter-path>                 same weights on a vLLM engine
    mlx:<model-or-adapter-path>                  same weights on Metal (Apple GPU;
                                                 needs .venv-mac)
    <anything else>                              OpenAI-compatible API

Every run ends with a latency probe (agent/eval.py::run_latency_probe): the
first 5 items of each tier, one question at a time, cache off, in
`<run>/latency/`. `--no-latency` opts out.

Examples:
    # a few items of an API model (--limit is per tier; spends API credit)
    python scripts/run_agent_eval.py --graph tsmc --model google/gemini-3.7-flash --limit 10 --name gemini37flash-probe

    # a base model with no adapter, the starting point the recipe trains from
    python scripts/run_agent_eval.py --graph tsmc --model hf:Qwen/Qwen3.5-0.8B --name base-agent

    # Prospector-4B: three adapters on the base, at 1,024 tokens a turn (the
    # default), one greedy seed of the paper's eight; the chat template comes
    # from the last adapter's run.json
    STACK=models/traj_sft_4b_flash20g2/final,models/traj_grpo_4b_flash20d1/checkpoint-120,models/traj_grpo_4b_flash20d2/checkpoint-120
    python scripts/run_agent_eval.py --graph tsmc --model vllm:Qwen/Qwen3.5-4B --adapter $STACK \\
        --turn-max-tokens 1024 --seed 0 --name 4b-s0

`--graph` names the substrate: its graph file and, without `--benchmark`, its
evaluation questions. `--benchmark` names any question file; the graph is then
inferred from it unless `--graph` says otherwise.
"""
import argparse

from graph_reasoner.agent.backend import available_backends, eval_benchmark
from graph_reasoner.agent.eval import run_agentic_eval
from graph_reasoner.agent.loop import EpisodeConfig
from graph_reasoner.agent.tools import ToolConfig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default=None,
                    help="a question file; without it, --graph's evaluation questions")
    ap.add_argument("--model", required=True,
                    help="hf:/vllm:/mlx:<model-or-adapter-path>, or an API model")
    ap.add_argument("--kb", default=None,
                    help="graph file; defaults to the one the inferred "
                         "substrate declares")
    ap.add_argument("--graph", default=None, choices=available_backends(),
                    help="the substrate: its graph file and, without --benchmark, its "
                         "evaluation questions; inferred from --benchmark when absent. "
                         "The choices come from the registry, so adding a KG never "
                         "edits this line")
    ap.add_argument("--adapter", default=None, help="comma-separated LoRA chain")
    ap.add_argument("--name", default=None, help="run name (default: derived from model)")
    ap.add_argument("--limit", type=int, default=None, help="items PER TIER")
    ap.add_argument("--shard", default=None,
                    help="'k/n': run only the k-th of n tier-balanced groups of "
                         "the benchmark (every n-th item per tier, offset k). "
                         "Groups partition the file; recorded in summary.json")
    ap.add_argument("--ids-file", default=None,
                    help="run only the benchmark items whose ids are listed in "
                         "this file (one per line); recorded in summary.json. "
                         "With --no-cache this is a fresh second draw of a "
                         "listed subset, e.g. a teacher pass's failures")
    ap.add_argument("--api-timeout", type=float, default=None,
                    help="API models only: per-call timeout in seconds (default 180)")

    # The budget and tool defaults are EpisodeConfig's and ToolConfig's. The
    # tool-call budget (--max-turns) is announced in the prompt and the policy
    # conditions its search on it, so another value is a different policy.
    ap.add_argument("--max-turns", type=int, default=EpisodeConfig.max_turns)
    ap.add_argument("--max-repeats", type=int, default=EpisodeConfig.max_repeats,
                    help="duplicate tool calls before the episode is force-ended "
                         "(see EpisodeConfig)")
    ap.add_argument("--max-edges", type=int, default=EpisodeConfig.max_edges_seen)
    ap.add_argument("--page-size", type=int, default=ToolConfig.page_size)
    ap.add_argument("--hub-threshold", type=int, default=ToolConfig.hub_threshold)
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="vllm: and mlx: drivers. The context window a conversation is cut at "
                         "(default: the window the graph's substrate declares, else 16384; the hf: "
                         "driver has no cut). PROTOCOL-CHANGING: recorded in summary.json")
    ap.add_argument("--hub-summary-limit", type=int, default=ToolConfig.hub_summary_limit,
                    help="relations listed in a hub's summary (ToolConfig.hub_summary_limit). "
                         "PROTOCOL-CHANGING below the widest hub a graph holds: disclosed in "
                         "the run's policy_dict")
    ap.add_argument("--connect-max-hops", type=int, default=ToolConfig.connect_max_hops)
    ap.add_argument("--turn-max-tokens", type=int, default=EpisodeConfig.turn_max_tokens,
                    help="tokens one assistant turn may use (default: the trained "
                         f"policies' budget, {EpisodeConfig.turn_max_tokens:,}). "
                         "PROTOCOL-CHANGING: disclosed in the run's policy_dict")
    ap.add_argument("--final-max-tokens", type=int, default=EpisodeConfig.final_max_tokens)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--seed", type=int, default=EpisodeConfig.seed,
                    help="seeds each item's tool RNG (expand's page order) and a local "
                         "driver's sampler; an API model samples as its provider does")
    ap.add_argument("--tools", default=None,
                    help="comma-separated allowlist of tools to offer, e.g. "
                         "find,expand (tool ablation). Default: find, expand and "
                         "connect (tools.DEFAULT_TOOLS). PROTOCOL-CHANGING: the "
                         "list is part of the system prompt; disclosed in the "
                         "run's policy_dict as `tools`")
    ap.add_argument("--user-note", default=None,
                    help="constant strategy sentence appended to the user turn "
                         "(perturbed best-of-n sub-pools; see "
                         "EpisodeConfig.user_note). PROTOCOL-CHANGING: "
                         "disclosed in the run's policy_dict")
    ap.add_argument("--chat-template", default=None,
                    help="named concatenative chat template (see "
                         "train/tokenize.CHAT_TEMPLATES). PROTOCOL-"
                         "CHANGING: a policy trained under an override must be "
                         "scored under the same one.")
    ap.add_argument("--prompt-version", default=EpisodeConfig.prompt_version,
                    choices=["agent_v1", "agent_v1_label",
                             "agent_v1_label_nothink", "agent_v1_label_native"],
                    help="system-prompt version. agent_v1 is the policy's and never "
                         "changes; agent_v1_label adds the copy-the-node-label rule "
                         "for the ANSWER line (prompts.AGENT_SYSTEM_V1_LABEL); "
                         "agent_v1_label_nothink is that prompt without the visible "
                         "reasoning block, for models whose provider restricts "
                         "reasoning disclosure (Anthropic); agent_v1_label_native "
                         "goes with --tool-protocol native-responses. "
                         "PROTOCOL-CHANGING: disclosed in the run's policy_dict")
    ap.add_argument("--tool-protocol", default=EpisodeConfig.tool_protocol,
                    choices=["text", "native-responses"],
                    help="how an API model calls the tools: the text block the loop "
                         "parses (every policy run), or the API's native function "
                         "channel on /v1/responses with server-side state, which "
                         "OpenAI's reasoning models require for function tools "
                         "(agent/native_tools.py; needs --prompt-version "
                         "agent_v1_label_native). It sends no temperature, since those "
                         "models sample at their own, so --temperature is refused. "
                         "PROTOCOL-CHANGING: disclosed in the "
                         "run's policy_dict")
    # Paired flags: --relation-vocab overrides an earlier
    # --no-relation-vocab on the same command line.
    ap.add_argument("--no-relation-vocab", dest="relation_vocab",
                    action="store_false",
                    help="omit the relation note from the tool docs (ablation)")
    ap.add_argument("--relation-vocab", dest="relation_vocab",
                    action="store_true",
                    help="keep it (default); use to override an earlier --no-relation-vocab")
    ap.set_defaults(relation_vocab=EpisodeConfig.relation_vocab)

    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="hf: driver only: episodes per generate call. vLLM and MLX "
                         "schedule their own batches and ignore it")
    ap.add_argument("--out-root", default="data/runs")
    ap.add_argument("--no-cache", dest="cache", action="store_false",
                    help="do not read or write the API response cache. REQUIRED "
                         "for a latency probe: the cache is keyed on (model, "
                         "messages, params), so re-running items already scored "
                         "replays at ~0 s/episode and reads as an infinitely "
                         "fast model. API models only; the local drivers "
                         "never touch it.")
    ap.set_defaults(cache=True)
    ap.add_argument("--sequential", action="store_true",
                    help="one question at a time, so wall_ms is a single-request "
                         "LATENCY. Without it the local drivers start "
                         "every item together and wall_ms is elapsed-since-batch- "
                         "start, and --concurrency does not reach them at all.")
    ap.add_argument("--no-latency", dest="latency", action="store_false",
                    help="skip the latency probe that every eval run ends "
                         "with: the first 5 items of each tier, one question "
                         "at a time with the response cache off, written to "
                         "<run>/latency/ and summarized in summary.json "
                         "(`latency`). A --sequential --no-cache run is its own "
                         "latency and skips it by itself.")
    ap.set_defaults(latency=True)
    args = ap.parse_args()

    cfg = EpisodeConfig(
        max_turns=args.max_turns,
        max_edges_seen=args.max_edges,
        max_repeats=args.max_repeats,
        turn_max_tokens=args.turn_max_tokens,
        final_max_tokens=args.final_max_tokens,
        temperature=args.temperature,
        tool_cfg=ToolConfig(page_size=args.page_size,
                            hub_threshold=args.hub_threshold,
                            hub_summary_limit=args.hub_summary_limit,
                            connect_max_hops=args.connect_max_hops,
                            tools=(tuple(t.strip() for t in args.tools.split(",") if t.strip())
                                   if args.tools else None)),
        seed=args.seed,
        relation_vocab=args.relation_vocab,
        user_note=args.user_note,
        prompt_version=args.prompt_version,
        tool_protocol=args.tool_protocol,
    )

    if args.benchmark is None and args.graph is None:
        ap.error("name the graph (--graph) or a question file (--benchmark)")
    run_agentic_eval(
        args.benchmark or eval_benchmark(args.graph), args.model,
        kb_path=args.kb, backend=args.graph or "auto",
        adapter=args.adapter, limit=args.limit, shard=args.shard,
        api_timeout=args.api_timeout,
        ids_file=args.ids_file, run_name=args.name,
        cfg=cfg, concurrency=args.concurrency, batch_size=args.batch_size,
        chat_template=args.chat_template, out_root=args.out_root,
        cache=args.cache, sequential=args.sequential,
        latency=args.latency,
        max_model_len=args.max_model_len,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
