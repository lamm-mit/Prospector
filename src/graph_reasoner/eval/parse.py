"""Parsing model outputs: the <graph_json> citation block, the final answer,
its asserted claim, and the outermost JSON object of a reply.

The gate, the metrics and the vote read model text through these functions
(agent/gate.py, agent/metrics.py, select/vote.py), and stored runs are
rescored from their text, so a change here changes their scores too.
"""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from ..schema import CitedGraph

_GRAPH_JSON_RE = re.compile(r"<graph_json>(.*?)</graph_json>", re.DOTALL | re.IGNORECASE)
_THINK_END_RE = re.compile(r"</think>", re.IGNORECASE)
_ANSWER_RE = re.compile(r"^\s*ANSWER\s*:\s*(.*)", re.IGNORECASE | re.DOTALL | re.MULTILINE)


def extract_json_object(text: str) -> dict | None:
    """The outermost JSON object in a string, from its first '{' to its last
    '}', or None when there is none, it does not parse, or it is not an
    object. Tool calls (`agent/loop.parse_tool_call`), citation blocks and
    the question generator's replies (`bench/qa_gen.py`) parse through it."""
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j <= i:
        return None
    try:
        obj = json.loads(text[i: j + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def extract_graph_json(text: str) -> CitedGraph | None:
    """The last <graph_json>...</graph_json> block as a CitedGraph, or None.

    The tags match case-insensitively and the JSON is taken from the block's
    first '{' to its last '}'. Any failure (no block, invalid JSON, schema
    mismatch) returns None, which the gate fails as `no_citation`.
    """
    blocks = _GRAPH_JSON_RE.findall(text)
    if not blocks:
        return None
    obj = extract_json_object(blocks[-1])
    if obj is None:
        return None
    try:
        return CitedGraph(**obj)
    except ValidationError:
        return None


_ABSTAIN_RE = re.compile(
    r"insufficient|not enough information|no (?:relevant |such )?(?:information|edge|path)"
    r"|does not (?:contain|provide|include|specify|support)"
    r"|doesn'?t (?:contain|provide|include|specify|support)"
    r"|cannot (?:be )?(?:determin|answer|establish|conclud|verif)"
    r"|can'?t (?:be )?(?:determin|answer|establish|conclud|verif)"
    r"|unable to (?:determine|answer|establish)|no answer",
    re.IGNORECASE,
)


def detect_abstention(answer: str) -> bool:
    """Whether an answer declines to answer: a regex, no model.

    It decides the gate's `voluntary_abstention` verdict (agent/gate.py), so
    a phrase added here reclassifies stored runs.
    """
    return bool(_ABSTAIN_RE.search(answer))


def extract_final_answer(text: str) -> str:
    """The text after the first 'ANSWER:', else after the last </think>, else
    the whole text; any <graph_json> block and stray lowercase tag is then
    stripped.

    The first marker, not the last: `_ANSWER_RE` is DOTALL, so its match runs
    to the end of the text and takes any later 'ANSWER:' line with it.
    """
    matches = list(_ANSWER_RE.finditer(text))
    if matches:
        ans = matches[-1].group(1)
    else:
        parts = _THINK_END_RE.split(text)
        ans = parts[-1] if len(parts) > 1 else text
    ans = _GRAPH_JSON_RE.sub("", ans)
    ans = re.sub(r"</?[a-z_]+>", "", ans)  # strip stray schema tags
    return ans.strip()


_CLAIM_RE = re.compile(r"the answer is", re.IGNORECASE)


def extract_claim(answer: str) -> str:
    """The asserted tail of a final answer: the text after the last "the
    answer is", or the whole answer when the marker is absent.

    An answer may restate its cited chain before the claim ("Because A rel B;
    B rel C, the answer is X"), and matching gold against the full text would
    credit a chain that merely passes through the gold entity.
    """
    parts = _CLAIM_RE.split(answer or "")
    return parts[-1].strip() if len(parts) > 1 else (answer or "").strip()
