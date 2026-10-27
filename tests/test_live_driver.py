"""The live app's episode driver (app/live.py): what it streams, and that it
is the same episode an eval run would score.

No model, no Metal and no GPU: a scripted engine stands in for
`LocalMLXModel`, with the one attribute the driver reads (`tokenizer.decode`)
and the one hook it sets (`on_token`), and a second one for `LocalVLLMModel`,
which returns a turn whole. The graph is the loop tests' film graph (conftest).
"""
from __future__ import annotations

import json
import os

import pytest

from app.attribution import attribute_turn_edges, edge_key
from app.live import LiveEngine, LiveCancelled
from conftest import final_answer, tool_call
from graph_reasoner.agent.loop import EpisodeConfig


@pytest.fixture(autouse=True)
def _engine_defaults(monkeypatch):
    """`LiveEngine()` reads GR_LIVE_* overrides from the environment; a shell
    that exported one (to run the app) must not change what these tests pin."""
    for name in [k for k in os.environ if k.startswith("GR_LIVE_")]:
        monkeypatch.delenv(name)


class _Tok:
    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


class _Engine:
    """Replays scripted turns; feeds `on_token` one character at a time so the
    delta path runs, exactly where the real driver calls it."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.tokenizer = _Tok()
        self.on_token = None
        self.calls = []

    def generate_chat(self, conversations, max_new_tokens, temperature=None,
                      stop=None, desc=""):
        assert len(conversations) == 1, "the live driver runs one episode"
        self.calls.append((len(conversations[0]), max_new_tokens, tuple(stop or ())))
        text = self.turns.pop(0)
        toks = []
        for ch in text:
            toks.append(ord(ch))
            if self.on_token:
                self.on_token(0, toks)
        return [text]


def _engine(tmp_path, kb, item, turns, cfg=None, driver="mlx"):
    eng = LiveEngine()
    eng.driver = driver          # pinned: the machine's own default must not decide
    eng._graph = kb
    eng._cfg = cfg or EpisodeConfig(max_turns=4)
    eng._model = _Engine(turns) if driver == "mlx" else _WholeTurnEngine(turns)
    eng._items = [item]
    eng.state = "ready"
    eng.adapter = "models/a,models/b"
    eng.run_dir = lambda: tmp_path / "live-run"
    return eng


class _EncodingTok(_Tok):
    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]


class _WholeTurnEngine:
    """Replays scripted turns the way the vLLM driver returns them: whole, with
    no per-token hook, through the same `generate_chat` contract."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.tokenizer = _EncodingTok()
        self.calls = []

    def generate_chat(self, conversations, max_new_tokens, temperature=None,
                      stop=None, desc=""):
        assert len(conversations) == 1, "the live driver runs one episode"
        self.calls.append((len(conversations[0]), max_new_tokens, tuple(stop or ())))
        return [self.turns.pop(0)]


def _drive(eng, question, item_id=None, cancel_after_tokens=None):
    events = []
    n = {"tok": 0}

    def emit(name, data):
        events.append((name, data))
        if name == "delta":
            n["tok"] = data["n_tokens"]

    def cancelled():
        return cancel_after_tokens is not None and n["tok"] >= cancel_after_tokens

    row = eng.run(question=question, item_id=item_id, emit=emit, cancelled=cancelled)
    return events, row


GOOD = [
    "<think>look</think>\n" + tool_call("expand", {"node": "Film A"}),
    tool_call("expand", {"node": "Jane Doe"}),
    final_answer("Film B", [{"source": "Film B", "relation": "written_by", "target": "Jane Doe"},
                       {"source": "Film A", "relation": "written_by", "target": "Jane Doe"}]),
]


def test_stream_is_start_turns_final_and_the_edges_are_the_seen_sets_growth(tmp_path, film_kb, film_item):
    eng = _engine(tmp_path, film_kb, film_item, GOOD)
    events, row = _drive(eng, film_item.question, item_id=film_item.id)
    names = [n for n, _ in events]
    assert names[0] == "start"
    assert names[-1] == "final"
    turns = [d for n, d in events if n == "turn"]
    assert [u["k"] for u in turns] == [1, 2]
    assert all(u["kind"] == "call" for u in turns)
    # Every turn's `new_edges` is the seen dict's growth; the parser, run on
    # the same record, must agree with it call by call.
    assert all(u["attribution_agrees"] for u in turns)
    assert sum(len(u["new_edges"]) for u in turns) == events[-1][1]["n_edges_seen"]
    # `generating` precedes every turn, and deltas arrive while it decodes.
    assert names.count("generating") == 3
    assert names.count("delta") >= 3
    assert names.index("generating") < names.index("delta") < names.index("turn")
    # The same generate seam as an eval run: the tool-call stop, the
    # per-turn budget, the whole conversation each turn.
    assert all(stop == ("</tool_call>",) for _, _, stop in eng._model.calls)
    assert eng._model.on_token is None, "the observer is unset after the episode"


def test_a_benchmark_question_is_scored_like_an_eval_run(tmp_path, film_kb, film_item):
    eng = _engine(tmp_path, film_kb, film_item, GOOD)
    events, row = _drive(eng, film_item.question, item_id=film_item.id)
    final = events[-1][1]
    assert final["graded"] is True
    assert row["gated_acc"] is True and row["em_claim"] is True
    assert final["result"] == row
    assert all(e["in_seen"] for e in final["cited_edges"])
    assert final["checks"]["attribution_exact"]
    # The id carries a timestamp so a re-ask is a second record, not an
    # overwrite; the item is still recognisable by its prefix.
    assert final["id"].startswith("medium-0000--")


def test_the_question_text_alone_finds_the_benchmark_item(tmp_path, film_kb, film_item):
    """The datalist hands the UI a question, not an id."""
    eng = _engine(tmp_path, film_kb, film_item, GOOD)
    events, row = _drive(eng, "  which movies share a writer   with Film A ")
    assert events[0][1]["graded"] is True
    assert row["gated_acc"] is True


def test_an_adhoc_question_gets_the_gate_and_no_verdict(tmp_path, film_kb, film_item):
    eng = _engine(tmp_path, film_kb, film_item, GOOD)
    events, row = _drive(eng, "who wrote Film A?")
    start = events[0][1]
    assert start["graded"] is False and start["gold_answer"] is None
    assert start["kb_answer_set"] == []
    assert row["gated_acc"] is None and row["em_claim"] is None
    assert row["gate_pass"] is True and row["gate_reason"] == "ok"
    assert row["tier"] == "adhoc" and row["id"].startswith("adhoc--")


def test_a_hallucinated_citation_fails_the_gate_in_the_row_and_the_edges(tmp_path, film_kb, film_item):
    turns = [GOOD[0], final_answer("Film B", [{"source": "Film B", "relation": "written_by",
                                           "target": "Nobody"}])]
    eng = _engine(tmp_path, film_kb, film_item, turns)
    events, row = _drive(eng, "who wrote Film A?")
    final = events[-1][1]
    assert row["gate_pass"] is False and row["gate_reason"] == "hallucinated_citation"
    assert [e["in_seen"] for e in final["cited_edges"]] == [False]


def test_the_record_replays_through_the_same_attribution(tmp_path, film_kb, film_item):
    eng = _engine(tmp_path, film_kb, film_item, GOOD)
    events, row = _drive(eng, film_item.question, item_id=film_item.id)
    out = tmp_path / "live-run"
    recs = [json.loads(l) for l in (out / "trajectories.jsonl").read_text().splitlines()]
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    assert len(recs) == len(rows) == 1
    assert recs[0]["id"] == rows[0]["id"] == row["id"]
    assert recs[0]["schema"] == "agent_trajectory_v1"
    assert recs[0]["model"] == "mlx:Qwen/Qwen3.5-4B" and recs[0]["adapter"] == "models/a,models/b"
    per_call, checks, _ = attribute_turn_edges(recs[0])
    assert checks["attribution_exact"]
    live = [{edge_key(e) for e in d["new_edges"]} for n, d in events if n == "turn"]
    assert [{edge_key(e) for e in batch} for batch in per_call] == live
    summ = json.loads((out / "summary.json").read_text())
    assert summ["live"] is True and summ["n_episodes"] == 1 and summ["n_graded"] == 1
    assert summ["aggregate"]["ALL"]["gated_acc"] == 1.0
    assert summ["timing"]["wall_ms_semantics"] == "per_episode"


def test_a_second_ask_appends_and_adhoc_rows_stay_out_of_the_aggregate(tmp_path, film_kb, film_item):
    eng = _engine(tmp_path, film_kb, film_item, GOOD + GOOD)
    _drive(eng, film_item.question, item_id=film_item.id)
    _drive(eng, "who wrote Film A?")
    out = tmp_path / "live-run"
    assert len((out / "trajectories.jsonl").read_text().splitlines()) == 2
    summ = json.loads((out / "summary.json").read_text())
    assert summ["n_episodes"] == 2 and summ["n_graded"] == 1
    assert summ["aggregate"]["ALL"]["n"] == 1


def test_a_malformed_turn_is_a_note_not_a_call(tmp_path, film_kb, film_item):
    turns = ["I will now look at the graph.", GOOD[0], GOOD[1], GOOD[2]]
    eng = _engine(tmp_path, film_kb, film_item, turns)
    events, row = _drive(eng, film_item.question, item_id=film_item.id)
    turns_out = [d for n, d in events if n == "turn"]
    assert [u["kind"] for u in turns_out] == ["malformed", "call", "call"]
    assert turns_out[0]["k"] == 0 and "observation" in turns_out[0]
    assert [u["k"] for u in turns_out[1:]] == [1, 2]
    assert row["n_malformed"] == 1


def test_a_forced_final_is_generated_with_the_final_budget(tmp_path, film_kb, film_item):
    cfg = EpisodeConfig(max_turns=1, turn_max_tokens=64, final_max_tokens=512)
    turns = [GOOD[0], GOOD[2]]
    eng = _engine(tmp_path, film_kb, film_item, turns, cfg=cfg)
    events, row = _drive(eng, film_item.question, item_id=film_item.id)
    gens = [d for n, d in events if n == "generating"]
    assert [g["forced"] for g in gens] == [False, True]
    assert [g["max_new_tokens"] for g in gens] == [64, 512]
    assert [d["forced_next"] for n, d in events if n == "turn"] == [True]
    assert events[-1][1]["status"] == "budget"


def test_cancel_aborts_between_tokens_and_writes_nothing(tmp_path, film_kb, film_item):
    eng = _engine(tmp_path, film_kb, film_item, GOOD)
    events, row = _drive(eng, film_item.question, item_id=film_item.id, cancel_after_tokens=5)
    assert row is None
    assert [n for n, _ in events][-1] != "final"
    assert not (tmp_path / "live-run").exists()
    assert eng._model.on_token is None
    assert not eng._run_lock.locked(), "a cancelled episode releases the model"


def test_the_engine_refuses_a_second_concurrent_episode(tmp_path, film_kb, film_item):
    eng = _engine(tmp_path, film_kb, film_item, GOOD)
    eng._run_lock.acquire()
    try:
        events, row = _drive(eng, film_item.question, item_id=film_item.id)
    finally:
        eng._run_lock.release()
    assert row is None and events == [("failure", events[0][1])]
    assert "already running" in events[0][1]["message"]


def test_bad_questions_are_refused_before_the_model_runs(tmp_path, film_kb, film_item):
    eng = _engine(tmp_path, film_kb, film_item, GOOD)
    with pytest.raises(ValueError):
        eng.run(question="   ", item_id=None, emit=lambda *_: None, cancelled=lambda: False)
    with pytest.raises(ValueError):
        eng.run(question="x" * 501, item_id=None, emit=lambda *_: None, cancelled=lambda: False)
    with pytest.raises(ValueError):
        eng.run(question="q", item_id="nope-0000", emit=lambda *_: None, cancelled=lambda: False)
    assert eng._model.calls == []


def test_the_token_observer_can_abort_the_real_driver_loop():
    """`LocalMLXModel._many` closes its generator when the token observer
    raises, which is how the stop button aborts a turn."""
    from types import SimpleNamespace

    from graph_reasoner.eval.local_model import LocalMLXModel
    from graph_reasoner.train.generate import cut_at_stop, split_stops, turn_text

    class Tok:
        eos_token_id = 0
        def __call__(self, s, add_special_tokens=False):
            return {"input_ids": [ord(c) for c in s]}
        def decode(self, toks):
            return "".join(chr(t) for t in toks)
        chat_template = None

    class Gen:
        closed = False
        def __init__(self, model, **kw): self.n = 0
        def insert(self, prompts, max_tokens=None, caches=None): return [0]
        def next_generated(self):
            self.n += 1
            return [SimpleNamespace(uid=0, token=ord("a"), finish_reason=None,
                                    match_sequence=None, prompt_cache=None, all_tokens=None)]
        def remove(self, uids, return_prompt_caches=False): return {}
        def close(self): Gen.closed = True

    obj = LocalMLXModel.__new__(LocalMLXModel)
    obj.model = object(); obj._caches = []; obj.tokenizer = Tok()
    obj._mlx_tok = SimpleNamespace(eos_token_ids={0})
    obj.prefill_batch_size = 2; obj.prefix_cache = True
    seen = []

    def on_token(i, toks):
        seen.append(len(toks))
        if len(toks) == 3:
            raise LiveCancelled()

    obj.on_token = on_token
    with pytest.raises(LiveCancelled):
        LocalMLXModel._many(obj, [[1, 2]], 64, None, [], Gen, lambda m: object(),
                            turn_text, split_stops, cut_at_stop)
    assert seen == [1, 2, 3]
    assert Gen.closed, "an aborted turn must close the engine's sequences"


def test_the_budget_is_per_episode_and_disclosed(tmp_path, film_kb, film_item):
    """The ask box can announce another budget. It reaches the episode as its
    own config (the record's policy says so), never by mutating the default
    the summary reports."""
    turns = [GOOD[0], GOOD[2]]
    eng = _engine(tmp_path, film_kb, film_item, turns)
    events, row = _drive_with(eng, film_item.question, film_item.id, max_turns=1)
    gens = [d for n, d in events if n == "generating"]
    assert [g["forced"] for g in gens] == [False, True]
    assert events[0][1]["max_turns"] == 1
    assert row["max_turns"] == 1 and events[-1][1]["status"] == "budget"
    out = tmp_path / "live-run"
    rec = json.loads((out / "trajectories.jsonl").read_text().splitlines()[0])
    assert rec["policy"]["max_turns"] == 1
    assert "Budget: 1 tool calls" in rec["messages"][1]["content"]
    summ = json.loads((out / "summary.json").read_text())
    assert summ["policy"]["max_turns"] == 4 and summ["max_turns_seen"] == [1]
    assert eng._cfg.max_turns == 4, "the default config is untouched"


def _drive_with(eng, question, item_id, **kw):
    events = []
    row = eng.run(question=question, item_id=item_id, emit=lambda n, d: events.append((n, d)),
                  cancelled=lambda: False, **kw)
    return events, row


def test_an_out_of_range_budget_is_refused_before_anything_runs(tmp_path, film_kb, film_item):
    eng = _engine(tmp_path, film_kb, film_item, GOOD)
    for bad in (0, 201, -3):
        with pytest.raises(ValueError):
            eng.run(question=film_item.question, item_id=film_item.id, emit=lambda *_: None,
                    cancelled=lambda: False, max_turns=bad)
    assert eng._model.calls == [] and not eng._run_lock.locked()


def test_status_reports_the_map_key_once_the_graph_is_loaded(tmp_path, film_kb, film_item):
    eng = LiveEngine()
    assert eng.status()["kg_fingerprint"] is None
    eng._graph = film_kb
    st = eng.status()
    assert st["kg_fingerprint"] == film_kb.fingerprint() and st["backend"] == film_kb.name
    assert st["max_budget"] == 200 and st["max_model_len"] == 131072


def test_a_long_budget_scales_the_edge_cap_and_a_short_one_keeps_it(tmp_path, film_kb, film_item):
    """The seen-edge cap is a harness limit the policy is never told about, so
    a longer budget lifts it in proportion; at or below the shipped budget it
    keeps the shipped value."""
    from app.live import edges_for
    assert edges_for(20, 400) == 400 and edges_for(4, 400) == 400
    assert edges_for(100, 400) == 2000 and edges_for(21, 400) == 420
    eng = _engine(tmp_path, film_kb, film_item, GOOD + GOOD)
    _drive_with(eng, film_item.question, film_item.id, max_turns=100)
    _drive_with(eng, film_item.question, film_item.id, max_turns=2)
    recs = [json.loads(l) for l in (tmp_path / "live-run" / "trajectories.jsonl").read_text().splitlines()]
    assert [(r["policy"]["max_turns"], r["policy"]["max_edges_seen"]) for r in recs] == [(100, 2000), (2, 400)]
    # Neither ran at the default budget (4 here), so the headline aggregate is
    # empty and each budget has its own bucket.
    summ = json.loads((tmp_path / "live-run" / "summary.json").read_text())
    assert summ["aggregate"] == {} and "not_a_benchmark" in summ
    assert set(summ["aggregate_by_budget"]) == {"100", "2"}
    assert summ["aggregate_by_budget"]["100"]["ALL"]["n"] == 1


def test_turn_events_carry_exact_token_counts(tmp_path, film_kb, film_item):
    """The scripted engine feeds one token per character, so the counts the
    UI turns into tokens-per-second are checkable to the character."""
    eng = _engine(tmp_path, film_kb, film_item, GOOD)
    events, row = _drive(eng, film_item.question, item_id=film_item.id)
    turns = [d for n, d in events if n == "turn"]
    assert [u["n_tokens"] for u in turns] == [len(GOOD[0]), len(GOOD[1])]
    assert events[-1][1]["gen_tokens_total"] == sum(len(t) for t in GOOD)


def test_an_ask_during_the_first_load_is_told_to_wait(tmp_path, film_kb, film_item):
    eng = _engine(tmp_path, film_kb, film_item, GOOD)
    eng.state = "loading"
    eng._run_lock.acquire()
    try:
        events, row = _drive(eng, film_item.question, item_id=film_item.id)
    finally:
        eng._run_lock.release()
    assert row is None and "loading" in events[0][1]["message"]


def test_the_seen_edge_cap_scales_with_the_servers_default_budget(monkeypatch):
    """A server started with GR_LIVE_MAX_TURNS=100 runs its default episodes
    under the same cap a viewer gets by typing budget 100: 20 edges a call,
    2,000, not the harness's 400. The loaders are stubbed; only the config
    matters here."""
    monkeypatch.setenv("GR_LIVE_MAX_TURNS", "100")
    monkeypatch.setattr("graph_reasoner.agent.eval.infer_backend_kind", lambda benchmark, items: "tsmc")
    monkeypatch.setattr("graph_reasoner.agent.backend.resolve_kb", lambda kind, kb: "unused.graphml")
    monkeypatch.setattr("graph_reasoner.agent.backend.load_backend", lambda kind, path: object())
    monkeypatch.setattr(LiveEngine, "items", lambda self: [])
    eng = LiveEngine()
    eng.load_graph()
    assert (eng._cfg.max_turns, eng._cfg.max_edges_seen) == (100, 2000)
    # the per-request path agrees, and the default budget keeps the harness cap
    assert eng._config_for(100).max_edges_seen == 2000
    assert eng._config_for(20).max_edges_seen == EpisodeConfig().max_edges_seen == 400
    assert eng._config_for(5).max_edges_seen == 400


def test_on_a_gpu_the_turns_arrive_whole_and_the_record_names_vllm(tmp_path, film_kb, film_item):
    """The vLLM driver returns each turn whole: no deltas, the same turns, the
    same score, token counts from the tokenizer, and the episode's record
    and run say which driver served it."""
    eng = _engine(tmp_path, film_kb, film_item, GOOD, driver="vllm")
    events, row = _drive(eng, film_item.question, item_id=film_item.id)
    names = [n for n, _ in events]
    assert "delta" not in names and names.count("generating") == 3
    turns = [d for n, d in events if n == "turn"]
    assert [u["k"] for u in turns] == [1, 2] and all(u["attribution_agrees"] for u in turns)
    assert [u["n_tokens"] for u in turns] == [len(GOOD[0]), len(GOOD[1])]
    assert row["gated_acc"] is True
    rec = json.loads((tmp_path / "live-run" / "trajectories.jsonl").read_text().splitlines()[0])
    assert rec["model"] == "vllm:Qwen/Qwen3.5-4B" and eng.status()["driver"] == "vllm"
    summ = json.loads((tmp_path / "live-run" / "summary.json").read_text())
    assert summ["model"] == "vllm:Qwen/Qwen3.5-4B"
    assert all(stop == ("</tool_call>",) for _, _, stop in eng._model.calls)


def test_on_a_gpu_a_stop_takes_effect_between_turns(tmp_path, film_kb, film_item):
    eng = _engine(tmp_path, film_kb, film_item, GOOD, driver="vllm")
    asked = {"n": 0}

    def cancelled():
        asked["n"] += 1
        return len(eng._model.calls) >= 1          # after the first turn returns

    row = eng.run(question=film_item.question, item_id=film_item.id,
                  emit=lambda *_: None, cancelled=cancelled)
    assert row is None and len(eng._model.calls) == 1
    assert not (tmp_path / "live-run").exists() and not eng._run_lock.locked()


def test_each_driver_appends_to_its_own_run():
    from app.live import RUN_NAMES
    eng = LiveEngine()
    eng.benchmark = "data/benchmarks/tsmc/tsmc_core_agentic.jsonl"
    for driver, name in RUN_NAMES.items():
        eng.driver = driver
        assert eng.run_dir().name == f"{name}--tsmc_core_agentic"
    assert RUN_NAMES == {"mlx": "live-mac", "vllm": "live-gpu"}


def test_the_driver_follows_what_is_installed_unless_the_environment_names_one(monkeypatch):
    import importlib.util

    from app.live import default_driver
    installed = set()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name in installed else None)
    assert default_driver() == "mlx"            # neither: the status names mlx-lm as missing
    installed.add("vllm")
    assert default_driver() == "vllm"           # a CUDA machine
    installed.add("mlx_lm")
    assert default_driver() == "mlx"            # Apple silicon
    monkeypatch.setenv("GR_LIVE_DRIVER", "vllm")
    assert LiveEngine().driver == "vllm"


def test_an_unknown_driver_is_reported_in_the_status_not_raised_at_start(monkeypatch):
    monkeypatch.setenv("GR_LIVE_DRIVER", "metal")
    eng = LiveEngine()                           # the server still starts
    with pytest.raises(ValueError, match="mlx or vllm"):
        eng.load()
    assert eng.state == "error" and "GR_LIVE_DRIVER='metal'" in eng.status()["detail"]


def test_closing_the_engine_stops_the_drivers_process_and_frees_the_slot():
    class _Closing:
        closed = False
        def close(self):
            _Closing.closed = True

    eng = LiveEngine()
    eng._model, eng.state = _Closing(), "ready"
    eng.close()
    assert _Closing.closed and eng._model is None and eng.state == "cold"
    eng.close()                                   # no model: nothing to stop


def test_the_server_stops_the_live_engine_as_it_stops(monkeypatch):
    """The vLLM engine runs in a process of its own; the server's lifespan
    closes it, so stopping the app frees the card."""
    pytest.importorskip("fastapi", reason="the app extras (.[app]) are not installed")
    pytest.importorskip("igraph", reason="the app extras (.[app]) are not installed")
    import asyncio

    from app import server
    calls = []
    monkeypatch.setattr(server.LIVE, "close", lambda: calls.append("close"))

    async def serve_and_stop():
        async with server.app.router.lifespan_context(server.app):
            assert calls == []
    asyncio.run(serve_and_stop())
    assert calls == ["close"]


def test_the_vllm_driver_closes_its_engine_client():
    from types import SimpleNamespace

    from graph_reasoner.eval.local_model import LocalVLLMModel
    stopped = []
    obj = LocalVLLMModel.__new__(LocalVLLMModel)
    obj.llm = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(
        shutdown=lambda: stopped.append(True))))
    obj.close()
    assert stopped == [True]
    bare = LocalVLLMModel.__new__(LocalVLLMModel)
    bare.close()                                  # never loaded: nothing to stop
