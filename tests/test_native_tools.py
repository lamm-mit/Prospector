"""The native function-calling adapter (agent/native_tools.py): the API's
structured call round-trips into the text block the loop parses, the
transcript reaches /v1/responses as function-call outputs, the schemas mirror
`tools.TOOLS`, and the text protocol is unchanged."""
import json

from graph_reasoner.agent import prompts as P
from graph_reasoner.agent.loop import EpisodeConfig, parse_tool_call
from graph_reasoner.agent.native_tools import compose_text, tool_schemas
from graph_reasoner.agent.tools import TOOLS, ToolConfig, format_doc


def test_schemas_mirror_the_tool_specs():
    s = tool_schemas(ToolConfig(tools=("find", "expand", "connect")))
    assert [t["function"]["name"] for t in s] == ["find", "expand", "connect"]
    for t in s:
        spec = TOOLS[t["function"]["name"]]
        params = t["function"]["parameters"]
        assert tuple(params["properties"]) == spec.args
        assert tuple(params["required"]) == spec.required
        assert "->" not in t["function"]["description"] and t["function"]["description"]
    assert tool_schemas(ToolConfig(tools=("find",)))[0]["function"]["parameters"]["properties"]["name"] == {"type": "string"}
    assert [p for p in tool_schemas(ToolConfig())[1]["function"]["parameters"]["properties"].items()][2] == ("offset", {"type": "integer"})


def test_schema_descriptions_carry_the_runs_limits():
    """The schema and the prompt's tool table are filled by one function, so a
    run that changes find_k or connect_limit states the same limit in both."""
    cfg = ToolConfig(find_k=5, connect_limit=2)
    find, _, connect = tool_schemas(cfg)
    assert "up to 5 nodes" in find["function"]["description"] and "up to 5 nodes" in format_doc("find", cfg)
    assert "up to 2 shortest paths" in connect["function"]["description"]


def test_the_native_prompt_names_the_offered_tools(tmp_path):
    """The native protocol sentence and reminder list the run's tools, so a
    withheld tool is not advertised; the default list is the three."""
    from graph_reasoner.substrates.triples import TextKBBackend
    p = tmp_path / "kb.txt"
    p.write_text("A|r|B\n", encoding="utf-8")
    kb = TextKBBackend(str(p))
    v = "agent_v1_label_native"
    full = P.build_agent_system(kb, ToolConfig(), version=v)
    assert "provided functions (find, expand, connect);" in full
    two = P.build_agent_system(kb, ToolConfig(tools=("find", "expand")), version=v)
    assert "provided functions (find, expand);" in two
    assert "function (find, expand, connect)," in P.nudge("SCHEMA_REMINDER", v)
    assert "function (find, expand)," in P.nudge("SCHEMA_REMINDER", v, ("find", "expand"))


def test_native_call_becomes_the_block_the_loop_parses():
    call = {"id": "call_1", "name": "expand", "arguments": json.dumps({"node": "wafer", "offset": 40})}
    text = compose_text("Let me look at wafer.", call)
    assert parse_tool_call(text) == ("expand", {"node": "wafer", "offset": 40})
    assert compose_text("final text", None) == "final text"
    bad = compose_text("", {"id": "c", "name": "find", "arguments": "{not json"})
    assert parse_tool_call(bad) is None          # malformed stays malformed


def test_text_protocol_untouched_and_native_disclosed():
    assert EpisodeConfig().policy_dict()["tool_protocol"] == "text"
    assert (EpisodeConfig(tool_protocol="native-responses").policy_dict()["tool_protocol"]
            == "native-responses")
    native = P.AGENT_SYSTEM_V1_LABEL_NATIVE
    assert "<tool_call>" not in native and "<graph_json>" not in native and "6. In `answer`" in native
    assert P.AGENT_SYSTEM_V1_LABEL.startswith(P.AGENT_SYSTEM)
    for name in ("FORCED_ANSWER", "EMPTY_TURN_NUDGE", "SCHEMA_REMINDER"):
        assert "<tool_call>" not in P.nudge(name, "agent_v1_label_native")
        assert P.nudge(name, "agent_v1") == getattr(P, name)


def test_responses_delta_sends_only_what_the_server_lacks():
    from graph_reasoner.agent.native_tools import responses_delta, responses_tools
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}]
    ins, items, prev = responses_delta(msgs)
    assert ins == "S" and items == [{"role": "user", "content": "Q"}] and prev is None
    # a native call answered by an observation, then a nudge
    msgs += [{"role": "assistant", "content": '<tool_call>{"tool": "find", "args": {"name": "w"}}</tool_call>',
              "_native_call": {"id": "c1", "name": "find", "arguments": '{"name": "w"}'}, "_response_id": "resp_1"},
             {"role": "user", "content": "OBSERVATION (call 1/20, find):\nw"},
             {"role": "user", "content": "nudge"}]
    ins, items, prev = responses_delta(msgs)
    assert ins is None and prev == "resp_1"
    assert items == [{"type": "function_call_output", "call_id": "c1", "output": "OBSERVATION (call 1/20, find):\nw"},
                     {"role": "user", "content": "nudge"}]
    # a plain assistant turn (no call) followed by a user message
    msgs += [{"role": "assistant", "content": "prose", "_response_id": "resp_2"},
             {"role": "user", "content": "reminder"}]
    assert responses_delta(msgs) == (None, [{"role": "user", "content": "reminder"}], "resp_2")
    flat = responses_tools(tool_schemas(ToolConfig(tools=("find",))))
    assert flat[0]["type"] == "function" and flat[0]["name"] == "find" and "function" not in flat[0]


def test_answer_function_renders_to_the_final_answer_the_gate_reads():
    from graph_reasoner.agent.native_tools import ANSWER_TOOL, render_answer_call
    from graph_reasoner.eval.parse import extract_final_answer, extract_graph_json
    schemas = tool_schemas(ToolConfig(tools=("find",)), with_answer=True)
    assert [t["function"]["name"] for t in schemas] == ["find", "answer"]
    assert ANSWER_TOOL["function"]["parameters"]["required"] == ["edges", "answer"]
    call = {"id": "c", "name": "answer", "arguments": json.dumps(
        {"edges": [{"source": "A", "relation": "uses", "target": "B"}], "answer": "B"})}
    text = compose_text("done", call)
    gj = extract_graph_json(text)
    assert [ (e.source, e.relation, e.target) for e in gj.edges ] == [("A", "uses", "B")]
    assert extract_final_answer(text).strip() == "B"
    assert parse_tool_call(text) is None           # an answer, not a tool call
    assert render_answer_call({"edges": [], "answer": "x"}).endswith("ANSWER: x")
