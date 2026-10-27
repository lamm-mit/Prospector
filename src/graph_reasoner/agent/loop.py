"""The rollout loop: one state machine and two drivers.

`advance()` holds every rule about what a turn means (budgets, repeats,
malformed calls, the seen set) and does not know where the assistant text came
from. The two drivers, API and batched generate, each produce a string and
hand it to `advance()`; the batched one serves local HF, vLLM, MLX and the
GRPO trainer's rollouts. An API model and a trained model are therefore
measured by the same code.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable

from ..eval.parse import extract_final_answer, extract_graph_json, extract_json_object
from ..llm import LLMClient, gather_with_progress
from ..schema import BenchmarkItem
from .backend import GraphBackend, SeenEdge
from .gate import edge_dicts
from .prompts import (
    EMPTY_TURN_NUDGE,   # seed_episode recounts a resumed prefix's nudges by equality
    FORCED_ANSWER,
    PROMPT_VERSION,
    SCHEMA_REMINDER,
    build_agent_system,
    build_agent_user,
    nudge,
    parse_observation_header,
    render_error,
    render_observation,
)
from .tools import ToolConfig, ToolResult, available_tools, dispatch

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_TOOL_OPEN_RE = re.compile(r"<tool_call>", re.IGNORECASE)
_ANSWER_RE = re.compile(r"^\s*ANSWER\s*:", re.IGNORECASE | re.MULTILINE)
_FORGED_OBS_RE = re.compile(r"^\s*OBSERVATION\b", re.IGNORECASE | re.MULTILINE)
_CHAT_TOKEN_RE = re.compile(r"<\|im_(?:start|end)\|>")

# Tokens one assistant turn may use, for the trained policies: the budget of
# the paper's runs, of the live app, and the default of `prospector`'s TURN_CAP
# (recipe/plan.py).
# Run a trained policy at this value: a lower cap cuts its longer turns short.
SERVED_TURN_MAX_TOKENS = 1024


@dataclass(frozen=True)
class EpisodeConfig:
    # The announced budget is part of the policy: it is rendered into the user
    # prompt ("Budget: N tool calls") and into every OBSERVATION header, so run
    # a policy at the budget it was trained with.
    max_turns: int = 20
    max_edges_seen: int = 400
    max_malformed: int = 2
    max_empty: int = 3
    # At this many repeated calls (same tool and arguments as an earlier call)
    # the episode is forced to answer, with status "budget". Until then a
    # repeat is not dispatched: the earlier call's result is shown again.
    max_repeats: int = 3
    # Tokens per ordinary turn. A reasoning model spends them on hidden
    # reasoning first, and a turn with no visible text is an empty turn: give
    # reasoning models 3000 or more. The default is the trained policies'.
    turn_max_tokens: int = SERVED_TURN_MAX_TOKENS
    final_max_tokens: int = 900
    temperature: float | None = None
    tool_cfg: ToolConfig = ToolConfig()
    seed: int = 0
    relation_vocab: bool = True
    # A constant sentence appended to the user turn, for perturbed sub-pools in
    # best-of-n sampling (task-level instruction perturbation, arXiv
    # 2502.11027). It carries no item fields. Off by default.
    user_note: str | None = None
    # The policy's prompt is agent_v1; the other versions are the API arms'
    # additive variants (prompts._SYSTEMS).
    prompt_version: str = PROMPT_VERSION
    # "text" (the default, and every policy run): the tools are described in
    # the system prompt and called through the <tool_call> block.
    # "native-responses": the same tools sent as function schemas on OpenAI's
    # /v1/responses endpoint, the model's structured call read back into the
    # text block the loop parses (agent/native_tools.py). API driver only, and
    # it needs a prompt version in prompts.NATIVE_VERSIONS.
    tool_protocol: str = "text"

    def policy_dict(self) -> dict:
        tool = asdict(self.tool_cfg)        # every ToolConfig field is disclosed
        tool["tools"] = list(self.tool_cfg.tools) if self.tool_cfg.tools is not None else None
        return {
            "prompt_version": self.prompt_version,
            "tool_protocol": self.tool_protocol,
            "max_turns": self.max_turns,
            # The token budgets are part of the protocol, so they are recorded.
            "turn_max_tokens": self.turn_max_tokens,
            "final_max_tokens": self.final_max_tokens,
            "max_edges_seen": self.max_edges_seen,
            "max_repeats": self.max_repeats,
            # The two caps that force a final answer besides the budget.
            "max_malformed": self.max_malformed,
            "max_empty": self.max_empty,
            **tool,
            "temperature": self.temperature,
            "seed": self.seed,
            "relation_vocab": self.relation_vocab,
            "user_note": self.user_note,
        }


@dataclass
class CallRecord:
    turn: int
    tool: str
    args: dict
    ok: bool
    error: str | None
    n_edges: int
    n_new_edges: int
    n_new_gold: int
    cum_edges_seen: int
    repeat_of: int | None
    result_text: str


@dataclass
class Episode:
    item_id: str
    tier: str
    question: str
    messages: list[dict]
    gold_keys: set[tuple[str, str, str]] = field(default_factory=set)
    answer_names: set[str] = field(default_factory=set)
    seen: dict[tuple[str, str, str], SeenEdge] = field(default_factory=dict)
    calls: list[CallRecord] = field(default_factory=list)
    # running|answered|budget|edge_budget|malformed|empty|error. "budget"
    # covers two causes, the turn cap and the repeat cap; `n_repeats` tells
    # them apart. "error" is an API call that failed past its retries.
    status: str = "running"
    n_turns: int = 0
    n_malformed: int = 0
    n_consecutive_malformed: int = 0
    n_empty: int = 0
    n_repeats: int = 0
    n_tool_errors: int = 0
    final_text: str = ""
    forced: bool = False
    answer_first_seen_call: int | None = None
    gold_complete_at_call: int | None = None
    # A forced final that has not been generated yet. `_force_answer` sets the
    # terminal status before the final turn, so the episode is inactive while
    # its answer is still owed; an episode resumed at its forced final
    # (seed_episode) starts in that state. advance() clears the flag.
    forced_pending: bool = False
    # Assistant turns the seeded prefix already carried (seed_episode). A
    # trainer masks them out of the loss: the policy under training did not
    # produce them.
    resumed_turns: int = 0
    wall_ms: int = 0
    # The failure behind status "error": the episode is scored as a failure
    # and the run goes on.
    error: str | None = None
    _rng: random.Random = field(default_factory=lambda: random.Random(0))
    _call_index: dict[str, int] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.status == "running"

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    def seen_edges(self) -> list[SeenEdge]:
        return list(self.seen.values())


def is_answer(text: str) -> bool:
    """Whether an assistant turn is a final answer: it carries an `ANSWER:` line
    and no tool call. The loop's own test, shared with the readers of stored
    episodes (`graph_reasoner.latent.moves`)."""
    return bool(_ANSWER_RE.search(text)) and not _TOOL_OPEN_RE.search(text)


def truncate_turn(text: str) -> str:
    """Cut an assistant turn where it stops being the model's own: at a chat
    control token, then after the first complete tool call, or else before an
    OBSERVATION block the model wrote itself."""
    text = _CHAT_TOKEN_RE.split(text)[0]
    match = _TOOL_CALL_RE.search(text)
    if match:
        return text[: match.end()].strip()
    forged = _FORGED_OBS_RE.search(text)
    if forged:
        return text[: forged.start()].strip()
    return text.strip()


def _call_from_json(blob: str) -> tuple[str, dict] | None:
    obj = extract_json_object(blob)
    if obj is None:
        return None
    tool = obj.get("tool") or obj.get("name")
    args = obj.get("args") if obj.get("args") is not None else obj.get("arguments", {})
    if not isinstance(tool, str):
        return None
    return tool, (args if isinstance(args, dict) else {})


def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """`(tool, args)` from the last `<tool_call>...</tool_call>` block, or from
    a bare JSON object when the whole message is one; None otherwise. The
    object may name the tool `tool` or `name`, its arguments `args` or
    `arguments`."""
    blocks = _TOOL_CALL_RE.findall(text)
    if blocks:
        return _call_from_json(blocks[-1])
    # Bare-JSON fallback: only when the message is essentially just the object,
    # so a stray brace inside prose can never be read as a call.
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return _call_from_json(stripped)
    return None


def start_episode(
    item: BenchmarkItem, backend: GraphBackend, cfg: EpisodeConfig
) -> Episode:
    system = build_agent_system(backend, cfg.tool_cfg, relation_vocab=cfg.relation_vocab,
                                version=cfg.prompt_version)
    user = build_agent_user(item, max_turns=cfg.max_turns)
    if cfg.user_note:
        user = f"{user}\n\n{cfg.user_note}"
    gold_keys = {SeenEdge(e.source, e.relation, e.target).key() for e in item.gold_path.edges}
    return Episode(
        item_id=item.id,
        tier=item.tier,
        question=item.question,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        gold_keys=gold_keys,
        answer_names=set(item.answer_set()),
        _rng=random.Random(f"{cfg.seed}:{item.id}"),
    )


def _observation_mentions_answer(ep: Episode, text: str) -> bool:
    low = text.casefold()
    return any(a.casefold() in low for a in ep.answer_names)


def advance(
    ep: Episode, assistant_text: str, backend: GraphBackend, cfg: EpisodeConfig
) -> bool:
    """Apply one assistant turn. Returns True while another generation is owed:
    an ordinary turn or the forced final."""
    text = truncate_turn(assistant_text)
    ep.messages.append({"role": "assistant", "content": text})
    ep.n_turns += 1

    if ep.forced:                       # this generation was the forced answer
        ep.final_text = text
        ep.forced_pending = False
        return False

    # An empty turn is the model running out of tokens, not a protocol error:
    # it is counted apart from malformed turns and has its own cap.
    if not text:
        ep.n_empty += 1
        if ep.n_empty >= cfg.max_empty:
            return _force_answer(ep, "empty", cfg)
        ep.messages.append({"role": "user", "content": nudge(
            "EMPTY_TURN_NUDGE", cfg.prompt_version, available_tools(cfg.tool_cfg))})
        return True

    call = parse_tool_call(text)
    answered = is_answer(text)

    if answered:
        ep.final_text = text
        ep.status = "answered"
        return False

    if call is None:
        ep.n_malformed += 1
        ep.n_consecutive_malformed += 1
        if ep.n_consecutive_malformed >= cfg.max_malformed:
            return _force_answer(ep, "malformed", cfg)
        ep.messages.append({"role": "user", "content": nudge(
            "SCHEMA_REMINDER", cfg.prompt_version, available_tools(cfg.tool_cfg))})
        return True

    ep.n_consecutive_malformed = 0
    tool, args = call
    call_no = ep.n_calls + 1

    signature = f"{tool}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
    repeat_of = ep._call_index.get(signature)
    if repeat_of is not None:
        prior = ep.calls[repeat_of - 1]
        ep.n_repeats += 1
        ep.calls.append(CallRecord(
            turn=ep.n_turns, tool=tool, args=args, ok=True, error=None,
            n_edges=0, n_new_edges=0, n_new_gold=0, cum_edges_seen=len(ep.seen),
            repeat_of=repeat_of, result_text=prior.result_text,
        ))
        if ep.n_repeats >= cfg.max_repeats:
            return _force_answer(ep, "budget", cfg)
        note = (f"(repeat of call #{repeat_of} — you already have those edges)"
                f"\n{prior.result_text}")
        ep.messages.append({"role": "user", "content": render_error(
            note, call_no=call_no, max_turns=cfg.max_turns)})
        return _check_budgets(ep, cfg)

    result: ToolResult = dispatch(backend, cfg.tool_cfg, tool, args, rng=ep._rng)
    ep._call_index[signature] = call_no

    if not result.ok:
        ep.n_tool_errors += 1
        ep.calls.append(CallRecord(
            turn=ep.n_turns, tool=tool, args=args, ok=False, error=result.error,
            n_edges=0, n_new_edges=0, n_new_gold=0, cum_edges_seen=len(ep.seen),
            repeat_of=None, result_text=result.error or "",
        ))
        ep.messages.append({"role": "user", "content": render_error(
            result.error or "tool error", call_no=call_no, max_turns=cfg.max_turns)})
        return _check_budgets(ep, cfg)

    # The seen update: during a rollout an edge enters ep.seen only here, from
    # ToolResult.edges.
    new_keys = []
    for edge in result.edges:
        key = edge.key()
        if key not in ep.seen:
            ep.seen[key] = edge
            new_keys.append(key)
    n_new_gold = sum(1 for k in new_keys if k in ep.gold_keys)

    if ep.answer_first_seen_call is None and _observation_mentions_answer(ep, result.text):
        ep.answer_first_seen_call = call_no
    if ep.gold_complete_at_call is None and ep.gold_keys <= set(ep.seen):
        ep.gold_complete_at_call = call_no

    ep.calls.append(CallRecord(
        turn=ep.n_turns, tool=tool, args=args, ok=True, error=None,
        n_edges=len(result.edges), n_new_edges=len(new_keys), n_new_gold=n_new_gold,
        cum_edges_seen=len(ep.seen), repeat_of=None, result_text=result.text,
    ))
    ep.messages.append({"role": "user", "content": render_observation(
        result, call_no=call_no, max_turns=cfg.max_turns, tool=tool,
        n_new=len(new_keys), n_seen=len(ep.seen))})
    return _check_budgets(ep, cfg)


def _check_budgets(ep: Episode, cfg: EpisodeConfig) -> bool:
    if len(ep.seen) >= cfg.max_edges_seen:
        return _force_answer(ep, "edge_budget", cfg)
    if ep.n_calls >= cfg.max_turns:
        return _force_answer(ep, "budget", cfg)
    return True


def _force_answer(ep: Episode, status: str, cfg: EpisodeConfig) -> bool:
    """End the search with `status` and ask for one last generation, so an
    episode that ran out of budget still yields an answer to score."""
    ep.status = status
    ep.forced = True
    ep.forced_pending = True
    ep.messages.append({"role": "user", "content": nudge(
        "FORCED_ANSWER", cfg.prompt_version, available_tools(cfg.tool_cfg))})
    return True


def finalize(ep: Episode) -> None:
    if ep.status == "running":
        ep.status = "answered" if ep.final_text else "budget"


# ---------------------------------------------------------------- drivers

def max_tokens_for(ep: Episode, cfg: EpisodeConfig) -> int:
    """The forced final gets the final budget; every other turn the turn budget."""
    return cfg.final_max_tokens if ep.forced else cfg.turn_max_tokens


async def run_episode_async(
    item: BenchmarkItem, backend: GraphBackend, client: LLMClient, cfg: EpisodeConfig,
) -> Episode:
    """One episode against an API model. Turns are sequential; episodes are not."""
    ep = start_episode(item, backend, cfg)
    started = time.monotonic()
    if cfg.tool_protocol not in ("text", "native-responses"):
        raise ValueError(f"tool_protocol must be 'text' or 'native-responses', "
                         f"got {cfg.tool_protocol!r}")
    native = cfg.tool_protocol == "native-responses"
    if native:
        from .native_tools import (compose_text, mark_native_call, responses_delta,
                                   responses_tools, tool_schemas)
        from .prompts import NATIVE_VERSIONS
        if cfg.prompt_version not in NATIVE_VERSIONS:
            raise ValueError(f"tool_protocol={cfg.tool_protocol} needs a prompt version in "
                             f"{sorted(NATIVE_VERSIONS)}, got {cfg.prompt_version!r}")
        r_tools = responses_tools(tool_schemas(cfg.tool_cfg, with_answer=True))
        if cfg.temperature is not None:
            raise ValueError("tool_protocol=native-responses sends no temperature: the "
                             "models behind /v1/responses sample at their own, so run "
                             "it without --temperature")
    try:
        while True:
            if native:
                instructions, items, prev = responses_delta(ep.messages)
                reply = await client.chat_responses(
                    instructions=instructions, input_items=items, previous_response_id=prev,
                    tools=r_tools, max_tokens=max_tokens_for(ep, cfg),
                    cache_key=[{k: v for k, v in m.items() if not k.startswith("_")} for m in ep.messages],
                )
                call = (reply.get("tool_calls") or [None])[0]
                text = compose_text(reply.get("content") or "", call)
                running = advance(ep, text, backend, cfg)
                mark_native_call(ep.messages, call)
                if reply.get("response_id"):
                    for m in reversed(ep.messages):
                        if m.get("role") == "assistant":
                            m["_response_id"] = reply["response_id"]
                            break
                if not running:
                    break
                continue
            text = await client.chat(
                ep.messages,
                temperature=cfg.temperature,
                max_tokens=max_tokens_for(ep, cfg),
            )
            if not advance(ep, text or "", backend, cfg):
                break
    except Exception as e:  # noqa: BLE001 -- one episode's failure must not discard the run
        ep.status = "error"
        ep.error = f"{type(e).__name__}: {e}"
        ep.final_text = ""
        print(f"\nEPISODE {item.id} FAILED: {ep.error}", flush=True)
    finalize(ep)
    ep.wall_ms = int((time.monotonic() - started) * 1000)
    return ep


async def run_episodes_async(
    items: Iterable[BenchmarkItem], backend: GraphBackend, client: LLMClient,
    cfg: EpisodeConfig, *, desc: str = "episodes",
) -> list[Episode]:
    """Every item's episode, gathered: the episodes start together."""
    items = list(items)
    return await gather_with_progress(
        [run_episode_async(it, backend, client, cfg) for it in items],
        desc=desc,
    )


ChatGenerator = Callable[[list[list[dict]], int], list[str]]


def _drive_batched(episodes: list[Episode], backend: GraphBackend,
                   generate: ChatGenerator, cfg: EpisodeConfig,
                   progress: Callable[[int, int], None] | None = None
                   ) -> list[Episode]:
    """The turn-synchronous loop over episodes already built, by
    `start_episode` (a fresh item) or `seed_episode` (a stored prefix), so a
    resumed episode meets the same budgets, repeat cap and forced-final
    handling as a fresh one."""
    started = time.monotonic()
    while True:
        active = [ep for ep in episodes if ep.active or ep.forced_pending]
        # Stamp each episode when it finishes. Every episode starts with the
        # batch, so `wall_ms` is the time from batch start to that episode's end.
        for ep in episodes:
            if not ep.active and not ep.forced_pending and not ep.wall_ms:
                ep.wall_ms = int((time.monotonic() - started) * 1000)
        if not active:
            break
        # Ordinary turns first, then forced finals, over the same `active`
        # list: an episode forced in the first pass gets its final in the
        # second. Token budgets come from max_tokens_for, the rule the async
        # driver uses too.
        for forced in (False, True):
            group = [ep for ep in active if ep.forced is forced]
            if not group:
                continue
            budgets: dict[int, list[Episode]] = {}
            for ep in group:
                budgets.setdefault(max_tokens_for(ep, cfg), []).append(ep)
            for budget, sub in budgets.items():
                texts = generate([ep.messages for ep in sub], budget)
                for ep, text in zip(sub, texts):
                    advance(ep, text or "", backend, cfg)
        if progress:
            progress(sum(1 for ep in episodes if not ep.active), len(episodes))
    for ep in episodes:
        finalize(ep)
        if not ep.wall_ms:          # never entered the loop (0 turns)
            ep.wall_ms = int((time.monotonic() - started) * 1000)
    return episodes


def run_episodes_batched(
    items: Iterable[BenchmarkItem], backend: GraphBackend, generate: ChatGenerator,
    cfg: EpisodeConfig, *, progress: Callable[[int, int], None] | None = None,
) -> list[Episode]:
    """Turn-synchronous driver for any batched `generate`: local HF, vLLM, MLX
    or the GRPO trainer's rollout.

    Each turn batches the episodes still running, so finished ones drop out.
    Forced-answer turns are generated separately, under `final_max_tokens`.
    """
    episodes = [start_episode(it, backend, cfg) for it in items]
    return _drive_batched(episodes, backend, generate, cfg, progress)


def drive_episode(ep: Episode, backend: GraphBackend, generate: ChatGenerator,
                  cfg: EpisodeConfig) -> Episode:
    """One already-started episode through the same loop as a batch of them,
    for a caller that observes the episode between generate calls (the live
    app). The caller builds the episode and so holds the reference."""
    return _drive_batched([ep], backend, generate, cfg)[0]


def _observation_body(content: str) -> str:
    """The tool text inside a rendered OBSERVATION message: the header line off
    the top, the `[k new edges; ...]` footer off the bottom (see
    prompts.render_observation / render_error)."""
    lines = content.split("\n")
    if lines and parse_observation_header(lines[0]):
        lines = lines[1:]
    if lines and lines[-1].startswith("[") and lines[-1].endswith("]"):
        lines = lines[:-1]
    return "\n".join(lines)


def seed_episode(record: dict, item: BenchmarkItem, cfg: EpisodeConfig, *,
                 cut: int | None = None, episode_id: str | None = None) -> Episode:
    """A live Episode resumed from a stored record.

    A full trajectory record is cut before its final assistant turn; a prefix
    record (ending in a user turn, `n_prefix_turns` set, as
    scripts/traj/build_prefixes.py writes) is resumed where it ends; `cut` (a
    1-based assistant turn) cuts either kind earlier. The policy continues
    from the cut under the same `advance` rules and budgets. `episode_id`
    replaces the record's id.

    Rebuilt from the record, up to the cut:
      * `messages`. The message before the cut must be a user turn; if it is
        the FORCED_ANSWER nudge, the episode resumes owed its forced final,
        with the record's terminal status.
      * `seen`, `calls`, the repeat index and the counters, from `seen_edges`
        and `calls`. `CallRecord.result_text` is not stored and is read back
        from the rendered OBSERVATION message, without its header and footer.
      * `resumed_turns`: assistant turns before the cut, for the loss mask.

    Raises ValueError when the record's announced `max_turns` differs from
    `cfg.max_turns`: every observation in the prefix carries that budget.
    """
    policy = record.get("policy") or {}
    if policy.get("max_turns") is not None and int(policy["max_turns"]) != cfg.max_turns:
        raise ValueError(f"record announced max_turns={policy['max_turns']} but "
                         f"cfg.max_turns={cfg.max_turns}: a prefix carries its "
                         f"budget in every observation")
    msgs = record["messages"]
    positions = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    if not msgs:
        raise ValueError("record has no messages")
    already_prefix = msgs[-1].get("role") == "user"
    if cut is None:
        if already_prefix:
            # A prefix record: nothing to cut, every assistant turn belongs to
            # the prefix.
            cut_idx, n_prefix_turns = len(msgs), len(positions)
        else:
            if not positions:
                raise ValueError("record has no assistant turn to cut before")
            cut_idx, n_prefix_turns = positions[-1], len(positions) - 1
    else:
        cut_turn = int(cut)
        if not 1 <= cut_turn <= len(positions):
            raise ValueError(f"cut={cut_turn} outside 1..{len(positions)}")
        cut_idx, n_prefix_turns = positions[cut_turn - 1], cut_turn - 1
    declared = record.get("n_prefix_turns")
    if cut is None and declared is not None and int(declared) != n_prefix_turns:
        raise ValueError(f"record declares n_prefix_turns={declared} but its "
                         f"messages carry {n_prefix_turns} assistant turns before the cut")
    prefix = [dict(m) for m in msgs[:cut_idx]]
    if not prefix or prefix[-1].get("role") != "user":
        raise ValueError("the message before the cut must be a user turn")
    last = prefix[-1]["content"]
    forced = last == FORCED_ANSWER

    calls: list[CallRecord] = []
    call_index: dict[str, int] = {}
    for c in record.get("calls", []):
        if int(c["turn"]) > n_prefix_turns:
            break
        pos = positions[int(c["turn"]) - 1]
        obs = msgs[pos + 1]["content"] if pos + 1 < len(msgs) and msgs[pos + 1].get("role") == "user" else ""
        repeat_of = c.get("repeat_of")
        text = (calls[repeat_of - 1].result_text if repeat_of and repeat_of - 1 < len(calls)
                else _observation_body(obs))
        calls.append(CallRecord(
            turn=int(c["turn"]), tool=c["tool"], args=dict(c.get("args") or {}),
            ok=bool(c.get("ok", True)), error=c.get("error"),
            n_edges=int(c.get("n_edges", 0)), n_new_edges=int(c.get("n_new_edges", 0)),
            n_new_gold=int(c.get("n_new_gold", 0)), cum_edges_seen=int(c.get("cum_edges_seen", 0)),
            repeat_of=repeat_of, result_text=text))
        if repeat_of is None:
            sig = f"{c['tool']}:{json.dumps(dict(c.get('args') or {}), sort_keys=True, ensure_ascii=False)}"
            call_index[sig] = len(calls)
    n_seen = calls[-1].cum_edges_seen if calls else 0
    if cut is None:
        n_seen = len(record["seen_edges"])
    seen: dict = {}
    for e in record["seen_edges"][:n_seen]:
        edge = SeenEdge(e["source"], e["relation"], e["target"])
        seen[edge.key()] = edge
    n_calls = len(calls)

    def _upto(v):
        return v if (v is not None and int(v) <= n_calls) else None

    gold_keys = {SeenEdge(e.source, e.relation, e.target).key() for e in item.gold_path.edges}
    eid = episode_id or str(record["id"])
    ep = Episode(
        item_id=eid,
        tier=record.get("tier") or item.tier,
        question=record.get("question") or item.question,
        messages=prefix,
        gold_keys=gold_keys,
        answer_names=set(item.answer_set()),
        seen=seen,
        calls=calls,
        status=str(record.get("status") or "budget") if forced else "running",
        n_turns=n_prefix_turns,
        n_malformed=sum(1 for m in prefix if m.get("role") == "user" and m["content"] == SCHEMA_REMINDER),
        n_consecutive_malformed=1 if last == SCHEMA_REMINDER else 0,
        n_empty=sum(1 for m in prefix if m.get("role") == "user" and m["content"] == EMPTY_TURN_NUDGE),
        n_repeats=sum(1 for c in calls if c.repeat_of is not None),
        n_tool_errors=sum(1 for c in calls if not c.ok),
        forced=forced,
        forced_pending=forced,
        answer_first_seen_call=_upto(record.get("answer_first_seen_call")),
        gold_complete_at_call=_upto(record.get("gold_complete_at_call")),
        resumed_turns=n_prefix_turns,
        _rng=random.Random(f"{cfg.seed}:{eid}"),
        _call_index=call_index,
    )
    return ep


def resume_episodes_batched(
    episodes: list[Episode], backend: GraphBackend, generate: ChatGenerator,
    cfg: EpisodeConfig, *, progress: Callable[[int, int], None] | None = None,
) -> list[Episode]:
    """Drive episodes built by `seed_episode` to completion, through the same
    loop as `run_episodes_batched`. A prefix cut at its forced final gets
    exactly one generation and keeps the source episode's terminal status."""
    for ep in episodes:
        if not (ep.active or ep.forced_pending):
            raise ValueError(f"{ep.item_id}: a resumed episode must be running "
                             f"or owed its forced final")
    return _drive_batched(episodes, backend, generate, cfg, progress)



# ---------------------------------------------------------------- persistence

TRAJECTORY_SCHEMA = "agent_trajectory_v1"


def trajectory_record(
    ep: Episode, item: BenchmarkItem, backend: GraphBackend, cfg: EpisodeConfig,
    *, model: str, adapter: str | None = None,
) -> dict:
    cited = extract_graph_json(ep.final_text)
    return {
        "schema": TRAJECTORY_SCHEMA,
        "id": ep.item_id,
        "tier": ep.tier,
        "question": ep.question,
        "anchor": item.meta.get("anchor"),
        "gold_answer": item.gold_answer,
        "gold_path_edges": [e.model_dump() for e in item.gold_path.edges],
        "kb_answer_set": sorted(ep.answer_names),
        "backend": backend.name,
        "kg_fingerprint": backend.fingerprint(),
        "model": model,
        "adapter": adapter,
        "policy": cfg.policy_dict(),
        "status": ep.status,
        "error": ep.error,
        "n_turns": ep.n_turns,
        "n_calls": ep.n_calls,
        "n_tool_errors": ep.n_tool_errors,
        "n_malformed": ep.n_malformed,
        "n_empty": ep.n_empty,
        "n_repeats": ep.n_repeats,
        "n_edges_seen": len(ep.seen),
        "seen_edges": edge_dicts(ep.seen.values()),
        "calls": [
            {"turn": c.turn, "tool": c.tool, "args": c.args, "ok": c.ok, "error": c.error,
             "n_edges": c.n_edges, "n_new_edges": c.n_new_edges, "n_new_gold": c.n_new_gold,
             "cum_edges_seen": c.cum_edges_seen, "repeat_of": c.repeat_of}
            for c in ep.calls
        ],
        "answer_first_seen_call": ep.answer_first_seen_call,
        "gold_complete_at_call": ep.gold_complete_at_call,
        "messages": ep.messages,
        "final_text": ep.final_text,
        "answer": extract_final_answer(ep.final_text),
        "cited_edges": [e.model_dump() for e in cited.edges] if cited else [],
        "wall_ms": ep.wall_ms,
    }


__all__ = [
    "EpisodeConfig", "Episode", "CallRecord", "TRAJECTORY_SCHEMA",
    "advance", "finalize", "start_episode", "truncate_turn", "parse_tool_call", "is_answer",
    "run_episode_async", "run_episodes_async", "run_episodes_batched",
    "drive_episode",
    "seed_episode", "resume_episodes_batched",
    "trajectory_record",
]
