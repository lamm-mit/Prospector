"""A run records which driver produced its `wall_ms`. Guards `agent/eval.py`.

`Episode.wall_ms` has two meanings: on the async (API) driver it is that
episode's own latency (`loop.run_episode_async`), and on the batched driver it
is the time since the whole batch started (`loop._drive_batched`).
`driver_kind` is the predicate the dispatch branches on, and its value is what
`summary.json` records under `timing`, so the label cannot disagree with the
driver that ran.
"""
import json

import networkx as nx
import pytest

from graph_reasoner.agent.eval import driver_kind, run_agentic_eval


@pytest.mark.parametrize("model,expected", [
    ("hf:Qwen/Qwen3.5-4B", "batched"),
    ("vllm:Qwen/Qwen3.5-4B", "batched"),
    ("mlx:Qwen/Qwen3.5-4B", "batched"),
    ("google/gemini-3.7-flash", "async"),
    ("gpt-5-mini", "async"),
    ("qwen/qwen3.5-35b:free", "async"),
])
def test_driver_kind_matches_the_branch_that_will_run(model, expected):
    """A local engine prefix (hf:, vllm:, mlx:) selects the batched driver and
    anything else the async one; an API id with a `:variant` suffix stays an
    API id."""
    assert driver_kind(model) == expected


def _toy_run(tmp_path):
    """One batched episode over a four-edge graph. Callers request the
    `stub_local_hf` fixture, so it needs no GPU and no API key."""
    g = nx.Graph()
    for a, b, rel in [("anchor", "mid", "feeds"), ("mid", "answer", "controls"),
                      ("anchor", "swamp", "borders"), ("swamp", "bog", "touches")]:
        g.add_node(a, group=1)
        g.add_node(b, group=1)
        g.add_edge(a, b, title=rel, chunk_id="ab" * 16)
    kb = tmp_path / "toy.graphml"
    nx.write_graphml(g, kb)

    item = {
        "id": "easy-0000", "tier": "easy", "question": "what does anchor reach?",
        "gold_answer": "answer",
        "gold_path": {"nodes": ["anchor", "mid", "answer"],
                      "edges": [{"source": "anchor", "relation": "feeds",
                                 "target": "mid"},
                                {"source": "mid", "relation": "controls",
                                 "target": "answer"}]},
        "subgraph_edges": [], "subgraph_text": "",
        "meta": {"anchor": "anchor", "kb_answer_set": ["answer"]},
    }
    bench = tmp_path / "toy_bench.jsonl"
    bench.write_text(json.dumps(item) + "\n", encoding="utf-8")

    return run_agentic_eval(str(bench), "hf:fake", kb_path=str(kb),
                            backend="biograph", limit=1, run_name="toy",
                            out_root=str(tmp_path / "runs"))


def test_a_run_records_the_driver_that_produced_its_wall_ms(tmp_path, stub_local_hf):
    """A batched run's timing block names its driver, what `wall_ms` means and
    the knob that reached the driver. The full key set is not pinned, so a new
    key does not fail this test."""
    summary = _toy_run(tmp_path)
    t = summary["timing"]
    assert t["driver"] == "batched"
    assert t["wall_ms_semantics"] == "since_batch_start"
    assert t["n_episodes"] == 1
    assert t["total_wall_ms"] >= 0
    # Only the knob that reached the driver is recorded: `concurrency` never
    # leaves the async path, `batch_size` never leaves HF.
    assert t["concurrency"] is None
    assert isinstance(t["batch_size"], int)


def test_the_timing_block_survives_the_round_trip_to_disk(tmp_path, stub_local_hf):
    """The timing block in `summary.json` on disk equals the one in the
    returned summary."""
    summary = _toy_run(tmp_path)
    written = json.loads(
        (tmp_path / "runs" / "toy--toy_bench" / "summary.json")
        .read_text(encoding="utf-8"))
    assert written["timing"] == summary["timing"]


def test_provenance_surfaces_the_keys_that_decide_comparability(tmp_path, stub_local_hf):
    """`describe_run` surfaces the timing keys that decide whether two runs'
    timings are comparable: the driver, the concurrency and the cache hits.
    The API run's summary is built from `timing_block`, since a real API run
    needs a key."""
    from graph_reasoner.agent.eval import timing_block
    from graph_reasoner.provenance import describe_run

    _toy_run(tmp_path)
    assert describe_run(tmp_path / "runs" / "toy--toy_bench")["driver"] == "batched"

    api = tmp_path / "runs" / "api--toy_bench"
    api.mkdir()
    timing = timing_block("async", sequential=True, concurrency=1,
                          batch_size=None, n_episodes=1, usage={"calls": 3, "cached": 0})
    (api / "summary.json").write_text(json.dumps({"model": "gpt-5-mini", "timing": timing}))
    info = describe_run(api)
    # cache_hits 0 is the reading that matters: a falsy value must survive.
    assert (info["driver"], info["concurrency"], info["cache_hits"]) == ("async", 1, 0)


def test_sequential_runs_one_episode_per_call_on_the_api_path(monkeypatch):
    """With `sequential`, the async driver is handed one episode per call.

    `concurrency` limits in-flight calls, not episodes: `run_episodes_async`
    starts every episode it is handed as a task, so at concurrency 1 each
    would wait on the others and its `wall_ms` would cover the whole run.
    """
    from graph_reasoner.agent import eval as E

    calls = []

    class _Client:
        usage = {"calls": 0, "cached": 0}

        def __init__(self, *a, **kw):
            calls.append(("client", kw.get("concurrency")))

    async def _fake_async(items, backend, client, cfg, **kw):
        # a coroutine, because the caller hands it to asyncio.run
        calls.append(("run", len(items)))
        return []

    monkeypatch.setattr(E, "LLMClient", _Client)
    monkeypatch.setattr(E, "run_episodes_async", _fake_async)

    items = [object(), object(), object()]
    E.run_episodes(items, None, "gpt-5-mini", E.EpisodeConfig(), sequential=True)
    assert [c for c in calls if c[0] == "run"] == [("run", 1)] * 3, \
        "sequential must hand the async driver one episode at a time"
    assert ("client", 1) in calls

    calls.clear()
    E.run_episodes(items, None, "gpt-5-mini", E.EpisodeConfig(), concurrency=8)
    assert [c for c in calls if c[0] == "run"] == [("run", 3)], \
        "without --sequential the whole item set goes in one call"
