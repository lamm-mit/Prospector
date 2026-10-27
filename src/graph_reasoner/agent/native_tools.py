"""Native function calling for API arms.

By default a model is driven through a text protocol: tools described in the
system prompt, a `<tool_call>{...}</tool_call>` block the loop parses, results
fed back as OBSERVATION user messages. This adapter lets an API arm use its
function-calling channel without a second rollout loop. The request carries
the same tools as JSON schemas, the structured call the model returns is
rendered into the block the loop parses, and the loop's OBSERVATION goes back
as that call's `function_call_output` on OpenAI's `/v1/responses` endpoint
(`responses_delta`). `advance`, the gate and the metrics see only the text
form.

The stored transcript is the text form plus a `_native_call` field on each
assistant turn that came through the channel. One call per turn is enforced
at the API (`parallel_tool_calls: false`).
"""
from __future__ import annotations

import json
import re

from .tools import TOOLS, ToolConfig, available_tools, format_doc

_SIG_RE = re.compile(r"^\S+\([^)]*\)\s*->\s*", re.S)


def _description(name: str, cfg: ToolConfig) -> str:
    """The tool's doc line without the `name(args) ->` prefix, on one line,
    filled with this run's limits exactly as the prompt's tool table is."""
    text = _SIG_RE.sub("", format_doc(name, cfg))
    return " ".join(text.split())


def tool_schemas(cfg: ToolConfig, *, with_answer: bool = False) -> list[dict]:
    """OpenAI-style `tools` for the tools this run offers, mirroring
    `tools.TOOLS` (same names, arguments and required set; `int_args` typed
    integer, the rest string), plus the `answer` function when asked."""
    out = []
    for name in available_tools(cfg):
        spec = TOOLS[name]
        props = {a: ({"type": "integer"} if a in spec.int_args else {"type": "string"})
                 for a in spec.args}
        out.append({"type": "function", "function": {
            "name": name,
            "description": _description(name, cfg),
            "parameters": {"type": "object", "properties": props,
                           "required": list(spec.required), "additionalProperties": False},
        }})
    if with_answer:
        out.append(ANSWER_TOOL)
    return out


# The final answer as a function: its arguments are the cited edges and the
# answer, which `render_answer_call` turns into the <graph_json> block and
# ANSWER line the gate and metrics read.
ANSWER_TOOL = {"type": "function", "function": {
    "name": "answer",
    "description": "Give your final answer. Cite only edges you actually saw in a tool result, "
                   "copied verbatim (same spelling, same relation, same direction): the chain from "
                   "the entity in the question to the answer. Name the answer exactly as its node "
                   "label appears in a tool result. If the evidence is insufficient, say so in `answer`.",
    "parameters": {"type": "object", "properties": {
        "edges": {"type": "array", "items": {"type": "object", "properties": {
            "source": {"type": "string"}, "relation": {"type": "string"}, "target": {"type": "string"}},
            "required": ["source", "relation", "target"], "additionalProperties": False}},
        "answer": {"type": "string"}},
        "required": ["edges", "answer"], "additionalProperties": False}}}


def render_answer_call(args: dict) -> str:
    """The `answer` call as the protocol's final-answer text."""
    edges = [{"source": str(e.get("source", "")), "relation": str(e.get("relation", "")),
              "target": str(e.get("target", ""))} for e in (args.get("edges") or []) if isinstance(e, dict)]
    nodes: list[str] = []
    for e in edges:
        for n in (e["source"], e["target"]):
            if n and n not in nodes:
                nodes.append(n)
    gj = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)
    return f"<graph_json>{gj}</graph_json>\nANSWER: {str(args.get('answer', '')).strip()}"


def compose_text(content: str, call: dict | None) -> str:
    """The assistant text the loop parses: the model's content plus, when it
    made a native call, that call rendered as the protocol's block (or, for
    the `answer` function, as the final-answer text)."""
    text = (content or "").rstrip()
    if call is None:
        return text
    if call.get("name") == "answer":
        try:
            args = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError:
            return text or "answer()"      # malformed: the loop counts it as such
        final = render_answer_call(args if isinstance(args, dict) else {})
        return f"{text}\n{final}" if text else final
    try:
        args = json.loads(call.get("arguments") or "{}")
    except json.JSONDecodeError:
        # Leave the block unparseable: the loop counts it as malformed.
        args = None
    block = json.dumps({"tool": call["name"], "args": args}, ensure_ascii=False) if args is not None \
        else f'{{"tool": {json.dumps(call["name"])}, "args": {call.get("arguments")!s}'
    return f"{text}\n<tool_call>{block}</tool_call>" if text else f"<tool_call>{block}</tool_call>"


def mark_native_call(messages: list[dict], call: dict | None) -> None:
    """Record on the last assistant message that it came through the native
    channel, so `responses_delta` can send the reply it gets as that call's
    output. `call` is {id, name, arguments}."""
    if call is None:
        return
    for m in reversed(messages):
        if m.get("role") == "assistant":
            m["_native_call"] = {"id": call["id"], "name": call["name"],
                                 "arguments": call.get("arguments") or "{}"}
            return


def responses_tools(schemas: list[dict]) -> list[dict]:
    """The same schemas in the Responses API's flat tool shape."""
    return [{"type": "function", "name": s["function"]["name"],
             "description": s["function"]["description"],
             "parameters": s["function"]["parameters"], "strict": False} for s in schemas]


def responses_delta(messages: list[dict]) -> tuple[str | None, list[dict], str | None]:
    """What to send the Responses API for the next turn, given the transcript.

    The conversation lives server-side under the previous response id, which
    the driver stores on the assistant turn as `_response_id`. The first turn
    sends the system prompt as `instructions` and the question. Each later
    turn sends only the items since: the OBSERVATION that answered its native
    call as a `function_call_output`, and any nudge as a user message.

    Returns (instructions, input_items, previous_response_id)."""
    last = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant" and messages[i].get("_response_id"):
            last = i
            break
    if last is None:
        system = next((m["content"] for m in messages if m.get("role") == "system"), None)
        items = [{"role": m["role"], "content": m["content"]} for m in messages
                 if m.get("role") in ("user", "assistant")]
        return system, items, None
    prev = messages[last]
    items: list[dict] = []
    pending_call = (prev.get("_native_call") or {}).get("id")
    for m in messages[last + 1:]:
        if m.get("role") == "user":
            if pending_call is not None:
                items.append({"type": "function_call_output", "call_id": pending_call,
                              "output": m.get("content") or ""})
                pending_call = None
            else:
                items.append({"role": "user", "content": m.get("content") or ""})
        elif m.get("role") == "assistant":
            # an assistant turn without a response id is resent as text
            items.append({"role": "assistant", "content": m.get("content") or ""})
    if pending_call is not None:
        # the loop ended the episode before answering the call (forced final):
        # the API still requires an output for every call it made
        items.insert(0, {"type": "function_call_output", "call_id": pending_call, "output": ""})
    return None, items, prev["_response_id"]
