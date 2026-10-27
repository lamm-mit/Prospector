"""Live mode: the served chain on this machine, driving one fresh episode
through the rollout loop and streaming it to the browser turn by turn.

`LiveEngine` holds a lazily loaded model, a loaded graph, and one episode at
a time. The model is an eval driver's: `LocalMLXModel` (`mlx:`) on a Mac, or
`LocalVLLMModel` (`vllm:`) on a machine with an NVIDIA GPU, chosen by what is
installed unless GR_LIVE_DRIVER names one (`default_driver`). The mlx driver
streams each turn's text as it decodes; the vLLM driver returns a turn whole,
so the page shows it when it is done, and a stop takes effect between turns.
The episode
runs through `drive_episode`, so budgets, repeats, forced finals and the seen
update happen in `advance()` as in a scored run: greedy, `plain_chatml`, 20
tool calls unless the viewer announces another budget, 1,024 tokens a turn.
Each turn's edges are the seen set's own growth; the replay parser runs on
the same record and every turn event says whether the two agree. A benchmark
question is scored with `agentic_metrics`; a typed question has no gold, so
only the gate verdict is reported and its row carries `gated_acc: null`. A
finished episode is appended as a `trajectory_record` to
`data/runs/live-mac--<benchmark>/` (mlx) or `live-gpu--<benchmark>/` (vLLM),
where replay mode lists it.

Environment overrides: GR_LIVE_DRIVER, GR_LIVE_MODEL, GR_LIVE_ADAPTER,
GR_LIVE_CHAT_TEMPLATE, GR_LIVE_BENCHMARK, GR_LIVE_KB (default: the graph the
benchmark's substrate declares), GR_LIVE_MAX_TURNS, GR_LIVE_MAX_MODEL_LEN.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Callable

from app.attribution import attribute_turn_edges, edge_key

REPO = Path(__file__).resolve().parent.parent

# The served 4B stack (configs/README.md, "The served stacks"): the recipe's
# names for (tsmc, 4b) at the paper's picks, which tests/test_recipe.py pins to
# graph_reasoner.recipe. Load order is training order: each adapter is a delta
# on the merge before it.
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_ADAPTER = ",".join([
    "models/traj_sft_4b_flash20g2/final",
    "models/traj_grpo_4b_flash20d1/checkpoint-120",
    "models/traj_grpo_4b_flash20d2/checkpoint-120",
])
DEFAULT_CHAT_TEMPLATE = "plain_chatml"
DEFAULT_BENCHMARK = "data/benchmarks/tsmc/tsmc_core_agentic.jsonl"
DEFAULT_MAX_TURNS = 20
#: The drivers live mode can serve through, with the run each one's episodes
#: are appended to.
RUN_NAMES = {"mlx": "live-mac", "vllm": "live-gpu"}
MAX_QUESTION_CHARS = 500
# A viewer may announce another budget, up to MAX_BUDGET. The chain was
# trained and scored at 20 calls and conditions its search on the announced
# budget, so every record carries the budget it ran under. Two harness limits
# leave room for a longer one: the seen-edge cap scales with it (`edges_for`)
# and the driver keeps a context of DEFAULT_MAX_MODEL_LEN tokens.
MAX_BUDGET = 200
DEFAULT_MAX_MODEL_LEN = 131072
# Seen edges allowed per announced call, the ratio of EpisodeConfig's
# defaults (400 edges, 20 calls). The cap is not announced to the policy.
EDGES_PER_CALL = 20


def default_driver() -> str:
    """`mlx` where mlx-lm is installed (Apple silicon), else `vllm` where vLLM
    is (a CUDA machine), else `mlx`, whose missing package the status then
    names."""
    import importlib.util
    if importlib.util.find_spec("mlx_lm") is not None:
        return "mlx"
    if importlib.util.find_spec("vllm") is not None:
        return "vllm"
    return "mlx"


def edges_for(max_turns: int, default: int) -> int:
    """The seen-edge cap for an announced budget: `default`, or
    EDGES_PER_CALL edges per call when that is larger."""
    return max(int(default), EDGES_PER_CALL * int(max_turns))
# Presentation only: what the status strip calls each graph. A graph without
# an entry is shown by its backend name.
GRAPH_LABELS = {
    "tsmc": "Semiconductor knowledge graph",
    "biograph": "Biomaterials knowledge graph",
    "pfas": "PFAS knowledge graph",
    "matprop": "Materials-properties knowledge graph",
}

Emit = Callable[[str, dict], None]


def _shown(path: Path) -> str:
    """A path for the UI: repo-relative when it is in the repo, else as is."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


class LiveCancelled(Exception):
    """The client went away mid-episode."""


class LiveEngine:
    """One model, one graph, one episode at a time."""

    def __init__(self) -> None:
        env = os.environ.get
        # Checked when the model loads, so a bad value is reported in the
        # status rather than stopping the server.
        self.driver = env("GR_LIVE_DRIVER") or default_driver()
        self.model = env("GR_LIVE_MODEL", DEFAULT_MODEL)
        self.adapter = env("GR_LIVE_ADAPTER", DEFAULT_ADAPTER) or None
        self.chat_template = env("GR_LIVE_CHAT_TEMPLATE", DEFAULT_CHAT_TEMPLATE) or None
        # None until the graph loads: then the file the benchmark's substrate
        # declares (agent.backend.resolve_kb), unless GR_LIVE_KB names one.
        self.kb_path = env("GR_LIVE_KB") or None
        self.backend: str | None = None
        self.benchmark = env("GR_LIVE_BENCHMARK", DEFAULT_BENCHMARK)
        self.max_turns = int(env("GR_LIVE_MAX_TURNS", str(DEFAULT_MAX_TURNS)))
        self.max_model_len = int(env("GR_LIVE_MAX_MODEL_LEN", str(DEFAULT_MAX_MODEL_LEN)))
        self.state = "cold"          # cold | loading | ready | error
        self.detail = ""
        self.load_s: float | None = None
        self._run_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._model = None
        self._graph = None
        self._items: list | None = None
        self._cfg = None

    # ------------------------------------------------------------ status

    @property
    def model_str(self) -> str:
        """The model as a run records it: the driver's prefix and the base."""
        return f"{self.driver}:{self.model}"

    def status(self) -> dict:
        graph = self._graph
        return {
            "state": self.state, "detail": self.detail, "load_s": self.load_s,
            "driver": self.driver, "model": self.model_str, "adapter": self.adapter,
            "chat_template": self.chat_template, "max_turns": self.max_turns,
            "max_budget": MAX_BUDGET, "max_model_len": self.max_model_len,
            "kb_path": self.kb_path, "benchmark": self.benchmark,
            # Set once the graph is loaded, before the model: the map is
            # keyed by this fingerprint.
            "kg_fingerprint": graph.fingerprint() if graph is not None else None,
            "backend": graph.name if graph is not None else None,
            "graph_label": (GRAPH_LABELS.get(graph.name, f"{graph.name} knowledge graph")
                            if graph is not None else None),
            "busy": self._run_lock.locked(),
            "run_dir": _shown(self.run_dir()),
        }

    def run_dir(self) -> Path:
        from graph_reasoner.benchmarks import run_dir_name
        return REPO / "data" / "runs" / run_dir_name(RUN_NAMES.get(self.driver, "live"), self.benchmark)

    def items(self) -> list:
        if self._items is None:
            from graph_reasoner.benchmarks import load_benchmark
            self._items = load_benchmark(REPO / self.benchmark)
        return self._items

    def questions(self) -> list[dict]:
        """The benchmark's questions, for the ask box's completion list. A
        picked question is scored against its gold; a typed one is gated
        only."""
        return [{"id": it.id, "tier": it.tier, "question": it.question}
                for it in self.items()]

    # ----------------------------------------------------------- loading

    def load_graph(self) -> None:
        """Load the graph and the default episode config. The map needs
        nothing else, so this runs before the model loads."""
        if self._graph is not None:
            return
        from graph_reasoner.agent.backend import load_backend, resolve_kb
        from graph_reasoner.agent.eval import infer_backend_kind

        kind = infer_backend_kind(self.benchmark, self.items())
        self.kb_path = resolve_kb(kind, self.kb_path)
        self._graph = load_backend(kind, str(REPO / self.kb_path))
        self.backend = kind
        self._cfg = self._config_for(self.max_turns)

    @staticmethod
    def _config_for(max_turns: int):
        """The episode config for an announced budget: the harness default
        with that budget and the seen-edge cap it scales to (`edges_for`).
        The server's default budget and a viewer's budget both go through
        here, so the same budget gets the same cap either way."""
        from dataclasses import replace
        from graph_reasoner.agent.loop import EpisodeConfig

        base = EpisodeConfig()
        return replace(base, max_turns=int(max_turns),
                       max_edges_seen=edges_for(int(max_turns), base.max_edges_seen))

    def load(self, emit: Emit | None = None) -> None:
        """Graph, then model. Idempotent; blocks until ready or failed."""
        with self._load_lock:
            if self.state == "ready":
                return
            self.state = "loading"
            started = time.monotonic()

            def step(detail: str) -> None:
                self.detail = detail
                if emit:
                    emit("status", self.status())

            try:
                if self.driver not in RUN_NAMES:
                    raise ValueError(f"GR_LIVE_DRIVER={self.driver!r}: live serves "
                                     f"through {' or '.join(RUN_NAMES)}")
                step("loading the knowledge graph")
                self.load_graph()
                self._model = self._load_model(step)
                self.load_s = round(time.monotonic() - started, 1)
                self.state = "ready"
                step(f"ready in {self.load_s}s")
            except Exception as exc:          # surfaced to the UI, not hidden
                self.state = "error"
                self.detail = f"{type(exc).__name__}: {exc}"
                if emit:
                    emit("status", self.status())
                raise

    def _load_model(self, step: Callable[[str], None]):
        """The driver's engine over the chain, under the paper's template and
        this engine's context window."""
        import importlib.util
        if self.driver == "mlx":
            if importlib.util.find_spec("mlx_lm") is None:
                raise ModuleNotFoundError(
                    "live needs an engine: on Apple silicon install the train and mlx extras "
                    "(uv pip install -e '.[train,mlx]' -c launch/constraints-mac.txt), on a "
                    "CUDA machine build the environment with launch/setup.sh and set "
                    "GR_LIVE_DRIVER=vllm")
            step(f"loading {self.model} on Metal (the first time fetches the base model from the "
                 "Hub, about 9 GB, unless it is cached, and merges the adapter chain, about 8 GB "
                 "on disk, which takes a few minutes)")
            from graph_reasoner.eval.local_model import LocalMLXModel
            return LocalMLXModel(self.model, adapter=self.adapter,
                                 chat_template=self.chat_template,
                                 max_model_len=self.max_model_len)
        if importlib.util.find_spec("vllm") is None:
            raise ModuleNotFoundError(
                "live on a GPU needs vLLM: build the environment with launch/setup.sh")
        step(f"loading {self.model} with vLLM (the first time fetches the base model from the Hub, "
             "about 9 GB, unless it is cached, and merges the adapter chain, about 8 GB on disk, "
             "which takes a few minutes)")
        from graph_reasoner.eval.local_model import LocalVLLMModel
        return LocalVLLMModel(self.model, adapter=self.adapter,
                              chat_template=self.chat_template,
                              max_model_len=self.max_model_len)

    def close(self) -> None:
        """Stop the model's engine, when its driver runs one in its own
        process (vLLM); the server calls this as it stops."""
        close = getattr(self._model, "close", None)
        if close is not None:
            close()
        self._model = None
        self.state = "cold"

    def load_in_background(self) -> None:
        if self.state in ("cold", "error"):
            threading.Thread(target=self._load_quietly, daemon=True).start()

    def _load_quietly(self) -> None:
        try:
            self.load()
        except Exception:
            pass                               # state/detail already carry it

    # ----------------------------------------------------------- episode

    def _item_for(self, question: str, item_id: str | None):
        """A benchmark item (by id, else by exact question text), or an
        ad-hoc item with no gold. The id gets a timestamp either way, so a
        question asked twice is two records in the run, not one overwritten."""
        from graph_reasoner.schema import BenchmarkItem

        stamp = time.strftime("%Y%m%d-%H%M%S")
        q = " ".join((question or "").split())
        for it in self.items():
            if (item_id and it.id == item_id) or (not item_id and it.question == q):
                return it.model_copy(update={"id": f"{it.id}--{stamp}"}), True
        if item_id:
            raise ValueError(f"unknown benchmark item {item_id!r}")
        if not q:
            raise ValueError("empty question")
        if len(q) > MAX_QUESTION_CHARS:
            raise ValueError(f"question over {MAX_QUESTION_CHARS} characters")
        return BenchmarkItem(
            id=f"adhoc--{stamp}", tier="adhoc", question=q, gold_answer="",
            gold_path={"nodes": [], "edges": []}, subgraph_edges=[],
            subgraph_text="", meta={"kb_answer_set": [], "live": True},
        ), False

    def run(self, *, question: str, item_id: str | None, emit: Emit,
            cancelled: Callable[[], bool], max_turns: int | None = None) -> dict | None:
        """One episode, start to record. Returns the results row, or None if
        the engine is busy or the client went away. Events go to
        `emit(name, payload)`.

        `max_turns` is the budget the viewer announced; None means the
        default. It is validated before anything loads and reaches the
        episode through its own EpisodeConfig, so the record's `policy`
        carries the budget it ran under."""
        if max_turns is not None and not (1 <= int(max_turns) <= MAX_BUDGET):
            raise ValueError(f"budget must be 1..{MAX_BUDGET} calls, got {max_turns}")
        if not self._run_lock.acquire(blocking=False):
            emit("failure", {"message": (
                "the model is still loading; ask again when the status says ready"
                if self.state == "loading" else
                "an episode is already running on the one model this machine serves; wait for it")})
            return None
        try:
            # The question is checked before the model loads, so a bad one is
            # refused at once on a cold engine.
            item, graded = self._item_for(question, item_id)
            self.load(emit)
            cfg = self._cfg
            if max_turns is not None and int(max_turns) != cfg.max_turns:
                cfg = self._config_for(int(max_turns))
            return self._run_locked(item, graded, emit, cancelled, cfg)
        except LiveCancelled:
            return None
        finally:
            self._run_lock.release()

    def _run_locked(self, item, graded: bool, emit: Emit,
                    cancelled: Callable[[], bool], cfg) -> dict:
        from graph_reasoner.agent.eval import local_generate_seam
        from graph_reasoner.agent.loop import (drive_episode, start_episode,
                                               trajectory_record)

        graph, engine = self._graph, self._model
        model_str = self.model_str
        # Only the mlx driver reports tokens as it decodes; the others return
        # a turn whole, counted afterwards with the same tokenizer.
        streams = self.driver == "mlx"
        ep = start_episode(item, graph, cfg)
        started = time.monotonic()

        def record() -> dict:
            return trajectory_record(ep, item, graph, cfg, model=model_str,
                                     adapter=self.adapter)

        emit("start", {
            "id": ep.item_id, "tier": ep.tier, "question": ep.question,
            "graded": graded, "anchor": item.meta.get("anchor"),
            "gold_answer": item.gold_answer if graded else None,
            "gold_path_edges": [e.model_dump() for e in item.gold_path.edges],
            "kb_answer_set": sorted(ep.answer_names),
            "backend": graph.name, "kg_fingerprint": graph.fingerprint(),
            "model": model_str, "adapter": self.adapter,
            "chat_template": self.chat_template, "max_turns": cfg.max_turns,
            "messages": list(ep.messages),
        })

        # The observer's cursor into the episode. `flush` reports every turn
        # advance() has added since the last generate call.
        seen = {"msgs": len(ep.messages), "calls": 0, "edges": 0, "turns": 0}
        gen_ms: dict[int, int] = {}
        gen_tok: dict[int, int] = {}
        ntok = {"n": 0}

        def flush() -> None:
            while seen["turns"] < ep.n_turns:
                msgs = ep.messages[seen["msgs"]:]
                # The turn is the first assistant message not yet reported;
                # the user message after it is its observation or nudge.
                a = next((i for i, m in enumerate(msgs) if m["role"] == "assistant"), None)
                if a is None:
                    return
                text = msgs[a]["content"]
                obs = next((m["content"] for m in msgs[a + 1:] if m["role"] == "user"), None)
                seen["msgs"] = len(ep.messages)
                seen["turns"] += 1
                turn = seen["turns"]
                if ep.n_calls > seen["calls"]:
                    seen["calls"] = ep.n_calls
                    k = ep.n_calls
                    c = ep.calls[-1]
                    # ep.seen is an insertion-ordered dict that only grows,
                    # so the entries past the cursor are this call's new edges.
                    new_edges = [{"source": e.source, "relation": e.relation,
                                  "target": e.target}
                                 for e in list(ep.seen.values())[seen["edges"]:]]
                    seen["edges"] = len(ep.seen)
                    parsed, _, focus = attribute_turn_edges(record())
                    agrees = ({edge_key(e) for e in parsed[k - 1]}
                              == {edge_key(e) for e in new_edges})
                    emit("turn", {
                        "kind": "call", "turn": turn, "k": k,
                        "assistant": text, "observation": obs,
                        "call": {"turn": c.turn, "tool": c.tool, "args": c.args,
                                 "ok": c.ok, "error": c.error, "n_edges": c.n_edges,
                                 "n_new_edges": c.n_new_edges,
                                 "cum_edges_seen": c.cum_edges_seen,
                                 "repeat_of": c.repeat_of},
                        "new_edges": new_edges, "focus": focus[k - 1],
                        "attribution_agrees": agrees,
                        "n_edges_seen": len(ep.seen), "status": ep.status,
                        "forced_next": ep.forced_pending,
                        "gen_ms": gen_ms.get(turn), "n_tokens": gen_tok.get(turn),
                        "elapsed_ms": int((time.monotonic() - started) * 1000),
                    })
                elif ep.final_text and text == ep.final_text:
                    pass                       # the final event carries it
                else:
                    emit("turn", {
                        "kind": "empty" if not text else "malformed",
                        "turn": turn, "k": ep.n_calls, "assistant": text,
                        "observation": obs, "gen_ms": gen_ms.get(turn),
                        "n_tokens": gen_tok.get(turn),
                        "elapsed_ms": int((time.monotonic() - started) * 1000),
                    })

        seam = local_generate_seam(engine, cfg)
        tick = {"at": 0.0, "n": 0}

        def on_token(_i: int, tokens: list[int]) -> None:
            if cancelled():
                raise LiveCancelled()
            ntok["n"] = len(tokens)          # every token, so the count is exact
            now = time.monotonic()
            if len(tokens) - tick["n"] >= 6 or now - tick["at"] > 0.15:
                tick["at"], tick["n"] = now, len(tokens)
                emit("delta", {"turn": ep.n_turns + 1,
                               "text": engine.tokenizer.decode(tokens),
                               "n_tokens": len(tokens)})

        def generate(conversations, max_new_tokens):
            flush()
            if cancelled():
                raise LiveCancelled()
            turn = ep.n_turns + 1
            emit("generating", {"turn": turn, "k": ep.n_calls + 1,
                                "forced": ep.forced, "max_new_tokens": max_new_tokens})
            tick["at"], tick["n"] = time.monotonic(), 0
            ntok["n"] = 0
            t0 = time.monotonic()
            out = seam(conversations, max_new_tokens)
            gen_ms[turn] = int((time.monotonic() - t0) * 1000)
            gen_tok[turn] = (ntok["n"] if streams else
                             len(engine.tokenizer.encode(out[0], add_special_tokens=False)))
            return out

        if streams:
            engine.on_token = on_token
        try:
            drive_episode(ep, graph, generate, cfg)
        finally:
            if streams:
                engine.on_token = None
        flush()

        rec = record()
        _, checks, _ = attribute_turn_edges(rec)
        seen_keys = {edge_key(e) for e in rec["seen_edges"]}
        cited = [{**e, "in_seen": edge_key(e) in seen_keys} for e in rec["cited_edges"]]
        row = self._score(ep, item, graded)
        # The budget this episode announced, on the row as well as in the
        # record's policy block: two rows of one run can differ here.
        row["max_turns"] = cfg.max_turns
        out_dir = self._save(rec, row, graded)
        emit("final", {
            "id": ep.item_id, "status": ep.status, "answer": rec["answer"],
            "final_text": rec["final_text"], "cited_edges": cited,
            "result": row, "checks": checks, "graded": graded,
            "n_calls": ep.n_calls, "n_turns": ep.n_turns,
            "n_edges_seen": len(ep.seen), "wall_ms": ep.wall_ms,
            "gen_ms_total": sum(gen_ms.values()),
            "gen_tokens_total": sum(gen_tok.values()),
            "saved": _shown(out_dir),
        })
        return row

    # ----------------------------------------------------------- scoring

    @staticmethod
    def _score(ep, item, graded: bool) -> dict:
        """The eval's metrics row for a benchmark question. An ad-hoc one
        gets the gate verdict alone: with no gold, its correctness fields are
        null, not False."""
        from graph_reasoner.agent.gate import edge_dicts, gate_assertion, gated_fields
        from graph_reasoner.eval.parse import extract_final_answer, extract_graph_json

        if graded:
            from graph_reasoner.agent.metrics import agentic_metrics
            row = {"id": ep.item_id, "tier": ep.tier}
            row.update(agentic_metrics(ep, item))
            return row
        cited = extract_graph_json(ep.final_text)
        answer = extract_final_answer(ep.final_text)
        verdict = gate_assertion(answer, edge_dicts(cited.edges if cited else None),
                                 edge_dicts(ep.seen.values()))
        fields = gated_fields(verdict)
        return {
            "id": ep.item_id, "tier": ep.tier, "status": ep.status,
            "answer": answer, "format_ok": cited is not None,
            "em_claim": None, "gated_acc": None,
            "n_turns": ep.n_turns, "n_calls": ep.n_calls,
            "n_tool_errors": ep.n_tool_errors, "n_malformed": ep.n_malformed,
            "n_repeats": ep.n_repeats, "n_edges_seen": len(ep.seen),
            **fields,
        }

    def _save(self, rec: dict, row: dict, graded: bool) -> Path:
        """Append to the live run dir and rewrite its summary, so the episode
        is listed in replay mode like any stored run. The aggregate covers
        benchmark questions at the default budget."""
        from graph_reasoner.agent.eval import timing_block, write_jsonl
        from graph_reasoner.agent.metrics import aggregate_agentic

        out = self.run_dir()
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "trajectories.jsonl", [rec], mode="a")
        write_jsonl(out / "results.jsonl", [row], mode="a")
        rows = []
        with (out / "results.jsonl").open(encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        gold_rows = [r for r in rows if r.get("gated_acc") is not None]
        # Only graded rows at the default budget are comparable to a scored
        # run; every budget also gets its own bucket.
        default_budget = self._cfg.max_turns
        default_rows = [r for r in gold_rows if r.get("max_turns") == default_budget]
        budgets = sorted({r.get("max_turns") for r in gold_rows if r.get("max_turns") is not None})
        graph = self._graph
        summary = {
            "model": self.model_str, "benchmark": self.benchmark, "agentic": True,
            "live": True, "adapter": self.adapter,
            "backend": graph.name, "kg_fingerprint": graph.fingerprint(),
            "kb_path": self.kb_path,
            "chat_template": self.chat_template,
            "timing": {**timing_block("batched", sequential=True,
                                      concurrency=1, batch_size=None,
                                      n_episodes=len(rows)),
                       "note": "one live episode at a time from app/live.py; "
                               "wall_ms is that episode's own clock"},
            # The default policy. An episode's own budget is in its record's
            # `policy.max_turns` and its row's `max_turns`.
            "policy": self._cfg.policy_dict(),
            "max_turns_seen": sorted({r.get("max_turns") for r in rows
                                      if r.get("max_turns") is not None}),
            "n_episodes": len(rows), "n_graded": len(gold_rows),
            "not_a_benchmark": ("hand-picked questions, one greedy read each, at whatever "
                                "budget the viewer announced; the README and the "
                                "paper own the numbers"),
            "aggregate_over": f"benchmark questions at the default budget ({default_budget} calls) only",
            "aggregate": aggregate_agentic(default_rows) if default_rows else {},
            "aggregate_by_budget": {
                str(b): aggregate_agentic([r for r in gold_rows if r.get("max_turns") == b])
                for b in budgets},
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
        return out


ENGINE = LiveEngine()
