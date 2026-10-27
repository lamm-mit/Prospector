"""Prompts for the exploring agent.

The final-answer format (`<think>` / `<graph_json>` / `ANSWER:`) is the static
arms', so `eval.parse` reads both, and an answer read off a given subgraph and
one found by exploring are scored by the same code. The agentic task adds the
turn protocol: one tool call per message, results arriving as OBSERVATION
messages.

The policy's prompt is `agent_v1`, pinned by digest in
tests/test_agent_loop.py. The other versions are variants for the API arms,
recorded per run as `prompt_version`. A change to `agent_v1` is a new version,
not an edit.
"""
from __future__ import annotations

import re

from ..schema import BenchmarkItem
from .backend import GraphBackend
from .tools import DEFAULT_TOOLS, ToolConfig, ToolResult, available_tools, tool_docs

PROMPT_VERSION = "agent_v1"

# Filled with str.replace, not str.format: braces are literal and must not be
# doubled.
AGENT_SYSTEM = """You are a graph explorer. You answer questions by exploring a \
knowledge graph with tools. You cannot see the graph — you only see what the tools \
return.

TOOLS
{TOOL_DOCS}

PROTOCOL
On every turn, either call ONE tool or give your final answer.

To call a tool, end your message with exactly one block:
<tool_call>{"tool": "<name>", "args": {...}}</tool_call>
You may reason first inside <think>...</think>. Keep it brief, and write nothing \
after the tool call — the result comes back to you as an OBSERVATION message.

When you have enough evidence, end with your final answer instead:
<think>...</think>
<graph_json>{"nodes": ["..."], "edges": [{"source": "...", "relation": "...", \
"target": "..."}]}</graph_json>
ANSWER: <concise answer>

RULES
1. Every edge in <graph_json> must be an edge you actually saw in an OBSERVATION, \
copied verbatim — same spelling, same relation, same direction.
2. Never invent an edge, a node name, or a relation. If you did not see it, it does \
not exist for you.
3. Cite only the edges that support your answer: the chain from the entity in the \
question to the answer.
4. Tool calls are limited. Prefer the fewest expansions that reach the answer.
5. If you run out of budget, answer with what you have and say what is missing."""

# agent_v1_label: agent_v1 plus rule 6, which states what the scorer checks, a
# substring match of the gold node label in the ANSWER line. For the API arms.
AGENT_SYSTEM_V1_LABEL = AGENT_SYSTEM + """
6. In ANSWER, name the answer entity exactly as its node label appears in an \
OBSERVATION: copy the spelling, and do not abbreviate, expand, pluralize or \
rephrase it."""

# agent_v1_label_nothink: the label prompt without the invitation to write a
# reasoning block, and without the block in the answer template and the
# nudges. For models that refuse a prompt inviting visible reasoning
# (Anthropic's).
_THINK_INVITE = ("You may reason first inside <think>...</think>. Keep it brief, and write "
                 "nothing after the tool call")
assert _THINK_INVITE in AGENT_SYSTEM_V1_LABEL


def _strip_think(text: str) -> str:
    """Drop every line that is exactly the reasoning-block placeholder."""
    return "\n".join(l for l in text.split("\n") if l.strip() != "<think>...</think>")


AGENT_SYSTEM_V1_LABEL_NOTHINK = _strip_think(
    AGENT_SYSTEM_V1_LABEL.replace(_THINK_INVITE, "Write nothing after the tool call"))
assert "think" not in AGENT_SYSTEM_V1_LABEL_NOTHINK

# agent_v1_label_native: the label prompt for native function calling
# (agent/native_tools.py). The API's function channel replaces the text
# tool-call block, and an `answer` function replaces the final-answer
# template; the tool docs stay.
_TEXT_CALL_PROTOCOL = (
    "To call a tool, end your message with exactly one block:\n"
    '<tool_call>{"tool": "<name>", "args": {...}}</tool_call>\n'
    "You may reason first inside <think>...</think>. Keep it brief, and write nothing "
    "after the tool call — the result comes back to you as an OBSERVATION message.")
assert _TEXT_CALL_PROTOCOL in AGENT_SYSTEM_V1_LABEL
# {TOOL_NAMES} is the run's tool list (tools.available_tools), filled by
# build_agent_system.
_NATIVE_CALL_PROTOCOL = (
    "To call a tool, call ONE of the provided functions ({TOOL_NAMES}); its result "
    "comes back to you as the function's result. You may reason first inside "
    "<think>...</think>. Keep it brief.")
_TEXT_FINAL_TEMPLATE = (
    "When you have enough evidence, end with your final answer instead:\n"
    "<think>...</think>\n"
    '<graph_json>{"nodes": ["..."], "edges": [{"source": "...", "relation": "...", '
    '"target": "..."}]}</graph_json>\n'
    "ANSWER: <concise answer>")
assert _TEXT_FINAL_TEMPLATE in AGENT_SYSTEM_V1_LABEL
# On the native channel the final answer is a function call
# (native_tools.ANSWER_TOOL), which the adapter renders into the block and
# ANSWER line the gate reads.
_NATIVE_FINAL = (
    "When you have enough evidence, call the `answer` function: `edges` is the chain of "
    "edges you cite and `answer` is your concise answer. Every episode ends with an `answer` "
    "call; a plain message is not an answer.")
AGENT_SYSTEM_V1_LABEL_NATIVE = (AGENT_SYSTEM_V1_LABEL
                                .replace(_TEXT_CALL_PROTOCOL, _NATIVE_CALL_PROTOCOL)
                                .replace(_TEXT_FINAL_TEMPLATE, _NATIVE_FINAL)
                                .replace("1. Every edge in <graph_json> must be", "1. Every edge you pass to `answer` must be")
                                .replace("6. In ANSWER, name the answer entity", "6. In `answer`, name the answer entity"))
assert "<tool_call>" not in AGENT_SYSTEM_V1_LABEL_NATIVE and "<graph_json>" not in AGENT_SYSTEM_V1_LABEL_NATIVE
assert "`answer` function" in AGENT_SYSTEM_V1_LABEL_NATIVE and "6. In `answer`" in AGENT_SYSTEM_V1_LABEL_NATIVE

_SYSTEMS = {"agent_v1": AGENT_SYSTEM,
            "agent_v1_label": AGENT_SYSTEM_V1_LABEL,
            "agent_v1_label_nothink": AGENT_SYSTEM_V1_LABEL_NOTHINK,
            "agent_v1_label_native": AGENT_SYSTEM_V1_LABEL_NATIVE}
NOTHINK_VERSIONS = {"agent_v1_label_nothink"}
NATIVE_VERSIONS = {"agent_v1_label_native"}

AGENT_USER = """Question: {QUESTION}

Find the entity named in the question, then explore outward.
Budget: {MAX_TURNS} tool calls."""

FORCED_ANSWER = """You have no tool calls left. Answer now with the evidence you \
already have, in the required format:
<think>...</think>
<graph_json>{"nodes": ["..."], "edges": [...]}</graph_json>
ANSWER: <concise answer>
If the evidence is insufficient, say so in the answer."""

EMPTY_TURN_NUDGE = """Your last message was empty — you may have spent the whole \
turn thinking. Keep reasoning to one or two short sentences, then emit the block:
<tool_call>{"tool": "<name>", "args": {...}}</tool_call>
or your final ANSWER."""

SCHEMA_REMINDER = """That message contained neither a valid tool call nor a final \
answer. Reply with exactly one of:
<tool_call>{"tool": "<name>", "args": {...}}</tool_call>
or
<graph_json>{"nodes": [...], "edges": [...]}</graph_json>
ANSWER: <concise answer>"""


_EMPTY_NUDGE_THINK = ("Your last message was empty — you may have spent the whole turn "
                      "thinking. Keep reasoning to one or two short sentences, then emit the block:")
assert _EMPTY_NUDGE_THINK in EMPTY_TURN_NUDGE


def nudge(name: str, version: str = PROMPT_VERSION,
          tools: tuple[str, ...] = DEFAULT_TOOLS) -> str:
    """One of the loop's mid-episode nudges, for a prompt version: the module
    constant, except that the no-think versions drop the reasoning-block
    placeholder and the "spent the turn thinking" sentence, and the native
    versions ask for a function call. `tools` is the run's tool list, which
    the native reminder names."""
    text = {"FORCED_ANSWER": FORCED_ANSWER, "EMPTY_TURN_NUDGE": EMPTY_TURN_NUDGE,
            "SCHEMA_REMINDER": SCHEMA_REMINDER}[name]
    if version in NATIVE_VERSIONS:
        # The block the text nudges show does not exist on the native channel.
        return {"FORCED_ANSWER": "You have no tool calls left. Call the `answer` function now with "
                                 "the evidence you already have. If the evidence is insufficient, say "
                                 "so in `answer`.",
                "EMPTY_TURN_NUDGE": "Your last message was empty — you may have spent the whole "
                                    "turn thinking. Keep reasoning to one or two short sentences, "
                                    "then call a function: a tool, or `answer`.",
                "SCHEMA_REMINDER": "That message was not a function call. Either call ONE tool "
                                   f"function ({', '.join(tools)}), or call `answer` with your "
                                   "cited edges and your answer."}[name]
    if version not in NOTHINK_VERSIONS:
        return text
    return _strip_think(text.replace(_EMPTY_NUDGE_THINK,
                                     "Your last message was empty. Emit the block:"))


def build_agent_system(
    backend: GraphBackend, cfg: ToolConfig, *, relation_vocab: bool = True,
    version: str = PROMPT_VERSION,
) -> str:
    if version not in _SYSTEMS:
        raise ValueError(f"unknown prompt version {version!r}; "
                         f"have {sorted(_SYSTEMS)}")
    return (_SYSTEMS[version]
            .replace("{TOOL_NAMES}", ", ".join(available_tools(cfg)))
            .replace("{TOOL_DOCS}", tool_docs(backend, cfg, relation_vocab=relation_vocab)))


def build_agent_user(item: BenchmarkItem, *, max_turns: int) -> str:
    """The user turn: the question and the announced budget. Reads
    `item.question` only, never the subgraph, the gold path or the answer."""
    return (AGENT_USER
            .replace("{QUESTION}", item.question)
            .replace("{MAX_TURNS}", str(max_turns)))


def render_observation(
    result: ToolResult, *, call_no: int, max_turns: int, tool: str,
    n_new: int, n_seen: int,
) -> str:
    calls_left = max(0, max_turns - call_no)
    return (f"OBSERVATION (call {call_no}/{max_turns}, {tool}):\n{result.text}\n"
            f"[{n_new} new edges; {n_seen} edges seen; {calls_left} tool calls left]")


def render_error(message: str, *, call_no: int, max_turns: int) -> str:
    calls_left = max(0, max_turns - call_no)
    return (f"OBSERVATION (call {call_no}/{max_turns}, error):\n{message}\n"
            f"[0 new edges; {calls_left} tool calls left]")


#: The header both renderers above write: call number, announced budget, and
#: the tool name (or `error`).
OBSERVATION_HEADER_RE = re.compile(r"OBSERVATION \(call (\d+)/(\d+), (\w+)\):")


def parse_observation_header(content: str) -> tuple[int, int, str] | None:
    """`(call_no, max_turns, tool)` from a rendered observation, or None for a
    message that carries no header (a nudge)."""
    m = OBSERVATION_HEADER_RE.search(content)
    return (int(m.group(1)), int(m.group(2)), m.group(3)) if m else None


__all__ = [
    "PROMPT_VERSION", "AGENT_SYSTEM", "FORCED_ANSWER", "SCHEMA_REMINDER",
    "EMPTY_TURN_NUDGE",
    "build_agent_system", "build_agent_user", "render_observation", "render_error",
    "OBSERVATION_HEADER_RE", "parse_observation_header",
]
