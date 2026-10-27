"""Rollout-loop tests: `agent/loop.py`, with `agent/prompts.py` for what the
episode is started with and `agent/metrics.py` for what it is scored on.

A scripted generator stands in for the model — no GPU, no API.
"""
import hashlib
import json

import pytest

from conftest import final_answer, run_turns, tool_call
from graph_reasoner.agent.loop import (EpisodeConfig, parse_tool_call, run_episodes_batched,
                                       start_episode, trajectory_record, truncate_turn)
from graph_reasoner.agent.metrics import agentic_metrics
from graph_reasoner.agent.prompts import build_agent_system, build_agent_user
from graph_reasoner.agent.tools import ToolConfig


# ---------------------------------------------------------------- protocol

def test_parse_tool_call_tolerant():
    assert parse_tool_call(tool_call("expand", {"node": "X"})) == ("expand", {"node": "X"})
    # OpenAI-style key names are accepted too
    assert parse_tool_call('<tool_call>{"name": "find", "arguments": {"name": "X"}}</tool_call>') \
        == ("find", {"name": "X"})
    assert parse_tool_call("no call here") is None
    assert parse_tool_call("<tool_call>not json</tool_call>") is None


def test_truncate_turn_strips_forged_observation():
    """An OBSERVATION block the model writes itself is cut from its turn, so
    the transcript holds only observations the tools produced."""
    forged = (tool_call("expand", {"node": "Film A"})
              + "\nOBSERVATION (call 2/8): Film A --[directed_by]--> Someone Fake")
    assert "Someone Fake" not in truncate_turn(forged)

    bare = "<think>t</think>\nOBSERVATION: invented stuff"
    assert truncate_turn(bare) == "<think>t</think>"
    assert truncate_turn("plain answer <|im_end|>junk") == "plain answer"


# ---------------------------------------------------------------- seen set

def test_cited_not_seen_is_hallucination(film_kb, film_item):
    """Citing an edge no tool returned is a hallucination even when the edge
    is real: `Secret Film --[directed_by]--> Hidden Person` is in the graph,
    and the episode never fetches it."""
    ep = run_turns(film_kb, film_item, [
        tool_call("expand", {"node": "Film A", "relation": "written_by"}),
        final_answer("Film B", [{"source": "Secret Film", "relation": "directed_by",
                            "target": "Hidden Person"}]),
    ])
    metrics = agentic_metrics(ep, film_item)
    assert metrics["halluc_strict"] == 1.0
    assert film_kb.digraph().has_edge("Secret Film", "Hidden Person")   # genuinely real


def test_seen_grows_only_from_tool_results(film_kb, film_item):
    ep = run_turns(film_kb, film_item, [
        tool_call("find", {"name": "Film A"}),                         # nodes only, no edges
        tool_call("expand", {"node": "Film A", "relation": "written_by"}),
        final_answer("Film B", []),
    ])
    assert {(e.source, e.relation, e.target) for e in ep.seen.values()} == {
        ("Film A", "written_by", "Jane Doe")}


# ---------------------------------------------------------------- budgets

def test_budget_forces_a_scoreable_answer(film_kb, film_item):
    """When the call budget runs out the episode gets one more turn to answer,
    so a budget-terminated episode still has an answer to score."""
    cfg = EpisodeConfig(max_turns=2)
    turns = [tool_call("expand", {"node": "Film A"}),
             tool_call("expand", {"node": "Jane Doe"}),
             final_answer("Film B", [])]
    ep = run_turns(film_kb, film_item, turns, cfg)
    assert ep.status == "budget" and ep.n_calls == 2
    assert ep.final_text and "Film B" in ep.final_text


def test_repeat_call_adds_no_edges(film_kb, film_item):
    ep = run_turns(film_kb, film_item, [
        tool_call("expand", {"node": "Film A", "relation": "written_by"}),
        tool_call("expand", {"node": "Film A", "relation": "written_by"}),
        final_answer("Film B", []),
    ])
    assert ep.n_repeats == 1
    assert ep.calls[1].repeat_of == 1 and ep.calls[1].n_new_edges == 0
    assert len(ep.seen) == 1
    # The reply to a repeat re-renders the prior call's page.
    note = next(m["content"] for m in ep.messages
                if m["role"] == "user" and "repeat of call #1" in m["content"])
    assert ep.calls[0].result_text and ep.calls[0].result_text in note


def test_repeat_cap_is_the_binding_budget_and_is_recorded(film_kb, film_item):
    """The repeat cap ends an episode on its own, before the turn budget is
    reached, and `policy_dict` records it with the two token budgets."""
    cfg = EpisodeConfig(max_turns=14, max_repeats=3)
    dup = tool_call("expand", {"node": "Film A", "relation": "written_by"})
    ep = run_turns(film_kb, film_item, [dup] * 6 + [final_answer("Film B", [])], cfg)
    assert ep.status == "budget"
    assert ep.n_repeats == 3 and ep.n_turns < 14, "cap must bind before turns"
    assert cfg.policy_dict()["max_repeats"] == 3
    # The token budgets change the policy's behaviour, so a run records them.
    assert cfg.policy_dict()["turn_max_tokens"] == cfg.turn_max_tokens
    assert cfg.policy_dict()["final_max_tokens"] == cfg.final_max_tokens


def test_malformed_then_recover(film_kb, film_item):
    cfg = EpisodeConfig(max_malformed=3)
    ep = run_turns(film_kb, film_item, [
        "I will think about it.",
        "<tool_call>still not json</tool_call>",
        tool_call("expand", {"node": "Film A", "relation": "written_by"}),
        final_answer("Film B", []),
    ], cfg)
    assert ep.n_malformed == 2 and ep.status == "answered" and len(ep.seen) == 1


def test_bare_json_tool_call_is_accepted(film_kb, film_item):
    """A message that is only the call's JSON object, with no `<tool_call>`
    tags, is accepted as a call."""
    bare = '{"tool": "expand", "args": {"node": "Film A", "relation": "written_by"}}'
    assert parse_tool_call(bare) == ("expand", {"node": "Film A", "relation": "written_by"})

    ep = run_turns(film_kb, film_item, [bare, final_answer("Film B", [])])
    assert ep.n_calls == 1 and ep.n_malformed == 0
    assert len(ep.seen) == 1

    # Prose that merely contains braces is still not a call.
    assert parse_tool_call('I think {"tool": "expand"} might work, but first...') is None


def test_system_prompt_shows_single_braces(film_kb):
    """The template is filled with `replace()`, not `format()`, so a `{{`
    escape in it would reach the model verbatim."""
    system = build_agent_system(film_kb, ToolConfig())
    # A doubled opening brace is never valid here; `}}` legitimately appears
    # where the inner args object and the outer call object both close.
    assert "{{" not in system
    assert '<tool_call>{"tool": "<name>", "args": {...}}</tool_call>' in system
    assert json.loads('{"tool": "expand", "args": {"node": "X"}}')


def test_prompt_versions_are_additive_and_disclosed(film_kb):
    """The API arms' prompt variants extend agent_v1 and leave it unchanged.

    agent_v1_label is agent_v1 plus the copy-the-node-label rule. The version
    is recorded in `policy_dict`, and an unknown version raises.
    """
    v1 = build_agent_system(film_kb, ToolConfig())
    v1x = build_agent_system(film_kb, ToolConfig(), version="agent_v1")
    assert v1 == v1x
    with pytest.raises(ValueError):
        build_agent_system(film_kb, ToolConfig(), version="agent_v3")
    assert EpisodeConfig().policy_dict()["prompt_version"] == "agent_v1"
    v1l = build_agent_system(film_kb, ToolConfig(), version="agent_v1_label")
    assert v1l.startswith(v1)
    assert "6. In ANSWER, name the answer entity exactly" in v1l
    assert "6. In ANSWER" not in v1
    assert (EpisodeConfig(prompt_version="agent_v1_label").policy_dict()["prompt_version"]
            == "agent_v1_label")
    # agent_v1_label_nothink (the Anthropic arms) is agent_v1_label without the
    # reasoning block: no "think" in the system prompt or the nudges, the label
    # rule kept. The two versions above use the module constants as nudges.
    from graph_reasoner.agent import prompts as P
    v1n = build_agent_system(film_kb, ToolConfig(), version="agent_v1_label_nothink")
    assert "think" not in v1n and "6. In ANSWER" in v1n
    assert "<tool_call>" in v1n and "<graph_json>" in v1n and "ANSWER:" in v1n
    for name, const in (("FORCED_ANSWER", P.FORCED_ANSWER), ("EMPTY_TURN_NUDGE", P.EMPTY_TURN_NUDGE),
                        ("SCHEMA_REMINDER", P.SCHEMA_REMINDER)):
        assert P.nudge(name, "agent_v1") == const
        assert P.nudge(name, "agent_v1_label") == const
        assert "think" not in P.nudge(name, "agent_v1_label_nothink")
    assert "<graph_json>" in P.nudge("FORCED_ANSWER", "agent_v1_label_nothink")


# The policy's prompt, pinned by digest. Every served policy was trained and
# evaluated under these bytes, so a change belongs under a new version name in
# prompts._SYSTEMS, not in a new digest here.
_SHA = {
    "AGENT_SYSTEM": "b80c6650494bac53114a7d55c368c392bba736ac84ddb034c203b105c9332832",
    "system": "b090af9d04fd0dca224ba3718263cf1e5b91aa2094399d9a26374bf76a191364",
    "FORCED_ANSWER": "3a572acc26ad3e00ba92b02adfb3a07a2651afe9358635f40ea60525afc09bc2",
    "EMPTY_TURN_NUDGE": "bddedcb19a804786ecc2a31580eec7685df8a9e511961d3730ae31198361eb13",
    "SCHEMA_REMINDER": "e370349a5f89d12259439e68584eee3eddfc7d40e9faf68aabe8845ad57b9c32",
    "user": "f618e2face357bfc332dd5c1ea49bda87aea3519d715f8bc310b09cef88be861",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_the_policy_prompt_is_byte_pinned(film_kb, film_item):
    """agent_v1's system template, its rendering over the film graph (tool
    table and relation list included), the three nudges and the user turn at
    the served budget of 20 calls."""
    from graph_reasoner.agent import prompts as P
    assert _sha(P.AGENT_SYSTEM) == _SHA["AGENT_SYSTEM"]
    assert _sha(build_agent_system(film_kb, ToolConfig())) == _SHA["system"]
    for name in ("FORCED_ANSWER", "EMPTY_TURN_NUDGE", "SCHEMA_REMINDER"):
        assert _sha(P.nudge(name, "agent_v1")) == _SHA[name], name
    assert _sha(build_agent_user(film_item, max_turns=20)) == _SHA["user"]


def test_registering_a_tool_leaves_the_default_prompt_alone(film_kb, monkeypatch):
    """A tool added to the registry is opt-in: a run that names no tools is
    offered DEFAULT_TOOLS, so its system prompt does not move."""
    from graph_reasoner.agent import tools as T
    before = build_agent_system(film_kb, ToolConfig())
    extra = dict(T.TOOLS)
    extra["teleport"] = T.ToolSpec(lambda backend, cfg, **kw: None, ("node",), ("node",),
                                   "teleport(node)                   -> goes anywhere.")
    monkeypatch.setattr(T, "TOOLS", extra)
    assert build_agent_system(film_kb, ToolConfig()) == before
    assert "teleport(node)" in build_agent_system(
        film_kb, ToolConfig(tools=("find", "expand", "connect", "teleport")))


def test_batched_forced_final_is_generated_in_the_same_run(film_kb, film_item):
    """The batched driver generates a budget-exhausted episode's forced final,
    at the final token budget. Forcing sets status to 'budget' at once, so the
    episode is not `.active` when the driver must still give it a turn."""
    seen_edge = {"source": "Film A", "relation": "written_by", "target": "Jane Doe"}
    cfg = EpisodeConfig(max_turns=1, turn_max_tokens=111, final_max_tokens=999)
    budgets = []
    script = iter([
        [tool_call("expand", {"node": "Film A"})],     # spends the only turn -> forced
        [final_answer("Jane Doe", [seen_edge])],       # the forced final
    ])

    def generate(messages_list, max_tokens):
        budgets.append(max_tokens)
        return next(script)

    (ep,) = run_episodes_batched([film_item], film_kb, generate, cfg)
    assert ep.status == "budget" and ep.forced
    assert "Jane Doe" in (ep.final_text or "")   # the final was generated
    assert budgets == [111, 999]                 # and at the final budget


def test_empty_turn_is_not_malformed(film_kb, film_item):
    """An empty turn counts against `max_empty`, not `max_malformed`: a
    reasoning model can spend its whole turn budget on hidden reasoning and
    emit nothing, which is not a protocol violation."""
    ep = run_turns(film_kb, film_item, [
        "", "",
        tool_call("expand", {"node": "Film A", "relation": "written_by"}),
        final_answer("Film B", []),
    ], EpisodeConfig(max_malformed=2, max_empty=3))
    assert ep.n_empty == 2 and ep.n_malformed == 0
    assert ep.status == "answered" and len(ep.seen) == 1


def test_repeated_empty_turns_force_answer(film_kb, film_item):
    ep = run_turns(film_kb, film_item, ["", "", "", final_answer("Film B", [])],
                    EpisodeConfig(max_empty=3))
    assert ep.n_empty == 3 and ep.status == "empty"
    assert "Film B" in ep.final_text


def test_consecutive_malformed_forces_answer(film_kb, film_item):
    ep = run_turns(film_kb, film_item, ["junk", "more junk", final_answer("Film B", [])],
                    EpisodeConfig(max_malformed=2))
    assert ep.status == "malformed" and "Film B" in ep.final_text


# ---------------------------------------------------------------- prompts

def test_prompt_hides_the_subgraph(film_kb, film_item):
    """The prompt carries the question, and not the subgraph, a gold
    intermediate or the gold answer."""
    cfg = EpisodeConfig()
    system = build_agent_system(film_kb, ToolConfig())
    user = build_agent_user(film_item, max_turns=cfg.max_turns)
    blob = system + user
    assert film_item.subgraph_text not in blob
    assert "Jane Doe" not in blob          # a gold intermediate
    assert film_item.gold_answer not in blob
    assert film_item.question in blob


def test_user_note_is_appended_verbatim_and_default_off(film_kb, film_item):
    """`user_note` is appended verbatim to the user turn and recorded in
    `policy_dict`; by default there is none and the user turn is unchanged."""
    note = "Open with the most specific noun phrase from the question."
    ep_off = start_episode(film_item, film_kb, EpisodeConfig())
    ep_on = start_episode(film_item, film_kb, EpisodeConfig(user_note=note))
    assert ep_on.messages[1]["content"] == \
        ep_off.messages[1]["content"] + "\n\n" + note
    assert ep_off.messages[1]["content"] == build_agent_user(
        film_item, max_turns=EpisodeConfig().max_turns)          # default unchanged
    assert EpisodeConfig(user_note=note).policy_dict()["user_note"] == note


# ---------------------------------------------------------------- records

def test_trajectory_record_round_trips(film_kb, film_item):
    ep = run_turns(film_kb, film_item, [
        tool_call("expand", {"node": "Film A", "relation": "written_by"}),
        final_answer("Film B", [{"source": "Film A", "relation": "written_by",
                            "target": "Jane Doe"}]),
    ])
    rec = trajectory_record(ep, film_item, film_kb, EpisodeConfig(), model="scripted")
    assert json.loads(json.dumps(rec))["id"] == film_item.id
    assert rec["seen_edges"] == [{"source": "Film A", "relation": "written_by",
                                  "target": "Jane Doe"}]
    assert rec["answer"] == "Film B" and rec["status"] == "answered"
    assert rec["messages"][0]["role"] == "system"



# ---------------------------------------------------------------- one loop

def test_one_rollout_loop_and_native_calling_is_an_adapter():
    """One rollout loop, checked on the call graph: only loop.py applies a
    turn (`advance`) or closes an episode (`finalize`); a tool runs through
    `dispatch` from loop.py, or from traj/error_pairs.py, which builds detour
    observations and runs no episode; an episode starts in loop.py or in the
    live app, which drives it with `drive_episode`; and only loop.py imports
    the native function-calling adapter."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    allowed = {
        "advance": {"src/graph_reasoner/agent/loop.py"},
        "finalize": {"src/graph_reasoner/agent/loop.py"},
        "dispatch": {"src/graph_reasoner/agent/loop.py", "src/graph_reasoner/traj/error_pairs.py"},
        "start_episode": {"src/graph_reasoner/agent/loop.py", "app/live.py"},
    }
    calls, natives = [], []
    for path in [*root.glob("src/**/*.py"), *root.glob("scripts/**/*.py"), *root.glob("app/**/*.py")]:
        rel = path.relative_to(root).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
                if name in allowed and rel not in allowed[name]:
                    calls.append(f"{rel}:{node.lineno} {name}()")
            # Every spelling of the import: `from .native_tools import x`,
            # `from graph_reasoner.agent import native_tools`, `import ....native_tools`.
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                names = []
            if any(n.endswith("native_tools") for n in names) \
                    and rel != "src/graph_reasoner/agent/loop.py":
                natives.append(f"{rel}:{node.lineno}")
    assert not calls, calls
    assert not natives, natives
