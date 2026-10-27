"""Error-correction rows: a dead end followed by the recovery from it.

The idea is KG-Reasoner's error-correction pairs (section 4.2.3 of its paper):
a correct reasoning chain is corrupted, then revised back. For one mid-episode
turn of a kept episode, `build_error_pair_rows` adds one row whose prompt ends
in a detour (a templated `expand` of an off-path node, and the observation the
backend returns for it, rendered as the loop renders it) and whose completion
is a templated one-sentence recognition spliced onto the turn the episode took
next. The rows teach backing up after an unhelpful observation instead of
repeating the call.

The detour call is trained, not masked: a detour row's prefix diverges from
its episode's, so the SFT trainer (`train/transcripts.py`) keeps it as a
transcript of its own. The `[k new edges; n seen]` footer of the injected
observation is approximate: it is computed from the edge lines printed in the
prompt, not by replaying the episode's seen set.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict

from ..agent.backend import GraphBackend
from ..agent.loop import parse_tool_call
from ..agent.prompts import parse_observation_header, render_error, render_observation
from ..agent.tools import ToolConfig, dispatch
from ..kg.serialize import parse_edge_line
from .sft_rows import TRACE_SCHEMA

# The recognition sentences; one is drawn per episode with its seeded
# generator. Each says to notice the dead end, not repeat the call, and return
# to the productive branch. None names an answer or a gold node.
RECOGNITIONS = (
    "None of those edges move toward the question — that branch is a dead "
    "end. Rather than expanding it again, I return to the thread that was "
    "working. ",
    "That expansion surfaced nothing relevant. Repeating the call would give "
    "the same edges, so I back up to the earlier line of search. ",
    "This node's neighbourhood is unrelated to what the question needs. I "
    "abandon it and pick up where the productive path left off. ",
    "A dead end — nothing there connects onward. Backing up rather than "
    "asking again. ",
)


def _episode_key(row_id: str) -> str:
    """`<item id>#<run tag>#t2` -> everything before the turn tag."""
    head, sep, tail = row_id.rpartition("#t")
    return head if sep and tail.isdigit() else row_id


def _printed_edges(messages: list[dict]) -> list[tuple[str, str, str]]:
    """Every edge line already shown, in order, deduplicated."""
    seen: dict[tuple[str, str, str], None] = {}
    for m in messages:
        if m.get("role") != "user":
            continue
        for line in m.get("content", "").split("\n"):
            edge = parse_edge_line(line)
            if edge:
                seen[edge] = None
    return list(seen)


def _last_call_header(messages: list[dict]) -> tuple[int, int] | None:
    """(call_no, max_turns) of the most recent observation, or None."""
    for m in reversed(messages):
        if m.get("role") == "user":
            head = parse_observation_header(m.get("content", ""))
            if head:
                return head[0], head[1]
    return None


def _expanded_nodes(messages: list[dict]) -> set[str]:
    out = set()
    for m in messages:
        if m.get("role") != "assistant":
            continue
        call = parse_tool_call(m.get("content", ""))
        if call and call[0] == "expand":
            out.add(str(call[1].get("node", "")))
    return out


def build_error_pair_rows(
    rows: list[dict],
    backend: GraphBackend,
    tool_cfg: ToolConfig,
    gold_nodes_by_item: dict[str, set[str]],
    *,
    seed: int = 0,
    max_wrong_degree: int = 60,
) -> tuple[list[dict], dict]:
    """Keeps rows -> (at most one detour-recovery row per episode, a tally of
    what was built and skipped).

    `gold_nodes_by_item` maps item_id to every node that must not be picked as
    the wrong turn: the gold path plus the kb answer set. An off-path node that
    is an answer would make the "dead end" text false.
    """
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("schema") == TRACE_SCHEMA:
            by_episode[_episode_key(r["id"])].append(r)

    out, tally = [], defaultdict(int)
    for ep_key in sorted(by_episode):
        ep_rows = sorted(by_episode[ep_key], key=lambda r: int(r.get("turn", 0)))
        rng = random.Random(f"{seed}:{ep_key}")
        item_id = ep_rows[0].get("item_id", "")
        forbidden = set(gold_nodes_by_item.get(item_id, set()))
        if not forbidden:
            tally["no_gold_lookup"] += 1
            continue

        # Candidate injection points: tool-call turns with at least one
        # observation already behind them (turn >= 2), tried latest first.
        def is_call_row(r: dict) -> bool:
            c = (r.get("completion") or [{}])[0].get("content", "")
            return parse_tool_call(c) is not None and c.startswith("<think>")

        candidates = [r for r in ep_rows if int(r.get("turn", 0)) >= 2
                      and is_call_row(r)]
        made = False
        for row in reversed(candidates):
            prompt = row["prompt"]
            header = _last_call_header(prompt)
            if header is None:
                continue
            call_no, max_turns = header
            if call_no + 1 >= max_turns:      # no budget left for a detour
                continue

            shown = _printed_edges(prompt)
            already = _expanded_nodes(prompt)
            orig_completion = row["completion"][0]["content"]
            orig_call = parse_tool_call(orig_completion)
            if orig_call and orig_call[0] == "expand":
                already.add(str(orig_call[1].get("node", "")))

            pool = []
            for s, _r, t in shown:
                for n in (s, t):
                    if n in forbidden or n in already or n in pool:
                        continue
                    if backend.canonical(n) != n:
                        continue          # only nodes expand() will accept
                    if not 1 <= backend.degree(n) <= max_wrong_degree:
                        continue
                    pool.append(n)
            if not pool:
                continue
            wrong = rng.choice(pool)

            result = dispatch(backend, tool_cfg, "expand", {"node": wrong},
                              rng=random.Random(0))
            wrong_turn = (
                f"<think>{wrong!r} appeared in an earlier observation; "
                f"checking whether it links toward the question.</think>\n"
                f'<tool_call>{{"tool": "expand", "args": {{"node": '
                f'{json.dumps(wrong, ensure_ascii=False)}}}}}</tool_call>')
            if result.ok:
                new = [e for e in result.edges
                       if (e.source, e.relation, e.target) not in set(shown)]
                obs = render_observation(
                    result, call_no=call_no + 1, max_turns=max_turns,
                    tool="expand", n_new=len(new),
                    n_seen=len(shown) + len(new))
            else:
                obs = render_error(result.error or "tool error",
                                   call_no=call_no + 1, max_turns=max_turns)

            recognition = RECOGNITIONS[rng.randrange(len(RECOGNITIONS))]
            recovered = orig_completion.replace(
                "<think>", "<think>" + recognition, 1)

            out.append({
                "schema": TRACE_SCHEMA,
                "id": f"{ep_key}#ec",
                "item_id": item_id,
                "tier": row.get("tier", ""),
                "turn": int(row.get("turn", 0)) + 1,
                "prompt": list(prompt) + [
                    {"role": "assistant", "content": wrong_turn},
                    {"role": "user", "content": obs},
                ],
                "completion": [{"role": "assistant", "content": recovered}],
                "source": "error_pair",
            })
            tally["built"] += 1
            made = True
            break
        if not made:
            tally["no_viable_detour"] += 1
    return out, dict(tally)
