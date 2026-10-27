"""The three tools the agent explores with, and the seen contract they enforce.

An edge is seen only if a tool returned it in `ToolResult.edges`, and every
edge in `ToolResult.edges` is rendered by `render_edge` into the same result's
text. During a rollout only `loop.advance()` adds to the episode's seen set,
and it adds `result.edges`.

The converse does not hold. A result's text can name nodes and relations it
does not return as edges (a hub's relation summary, a paging note), so a
mention is not a seen edge. Edges withheld by paging or by the hub cap are
counted in the text and are not seen: `cited ⊆ seen` credits only edges the
model was shown.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..kg.paths import find_paths
from ..kg.serialize import render_edge_line
from .backend import GraphBackend, SeenEdge


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    text: str
    edges: tuple[SeenEdge, ...] = ()
    nodes: tuple[str, ...] = ()
    total: int | None = None
    truncated: bool = False
    error: str | None = None


@dataclass(frozen=True)
class ToolConfig:
    page_size: int = 40
    hub_threshold: int = 150
    # A hub's per-relation summary lists at most this many relations, most
    # frequent first, then one line saying how many more exist.
    hub_summary_limit: int = 400
    connect_max_hops: int = 3
    connect_limit: int = 3
    find_k: int = 10
    # None offers DEFAULT_TOOLS; a tuple names the tools the model is shown
    # and may call. The tool list is part of the system prompt, so another set
    # is a different protocol, recorded per run by `policy_dict` as `tools`.
    tools: tuple[str, ...] | None = None


def render_edge(e: SeenEdge) -> str:
    """The edge line shown to the model, byte-identical to the static setting's
    (`kg.serialize.serialize_subgraph`). The model copies these lines verbatim
    into `<graph_json>`."""
    return render_edge_line(e.source, e.relation, e.target)


def _edge_block(edges) -> str:
    return "\n".join(render_edge(e) for e in edges)


# ---------------------------------------------------------------- find

def tool_find(backend: GraphBackend, cfg: ToolConfig, *, name: str) -> ToolResult:
    """Entity linking. Returns nodes only — never edges."""
    if not str(name).strip():
        return ToolResult(False, "", error='find needs a non-empty "name".')

    hits = backend.search(str(name), cfg.find_k)
    if not hits:
        return ToolResult(
            True,
            f'NO MATCH for "{name}". Try a shorter name, or different spelling.',
            nodes=(),
        )
    lines = [f'FOUND {len(hits)} node(s) matching "{name}":']
    lines += [f"  {n}  [{t}, {d} connections]" for n, t, d in hits]
    if cfg.tools is None or "expand" in cfg.tools:
        lines.append("No edges returned — use expand() to see connections.")
    else:
        lines.append("No edges returned.")
    return ToolResult(True, "\n".join(lines), nodes=tuple(n for n, _, _ in hits))


# ---------------------------------------------------------------- expand

def tool_expand(
    backend: GraphBackend,
    cfg: ToolConfig,
    *,
    node: str,
    relation: str | None = None,
    offset: int = 0,
    rng: random.Random | None = None,
) -> ToolResult:
    """One hop out of `node`, in either direction, paged.

    A node with more than `hub_threshold` connections and no relation filter
    returns a per-relation summary plus its `page_size` most important edges,
    and reports how many were withheld. The withheld edges are not seen.
    """
    canonical = backend.canonical(str(node))
    if canonical is None:
        return ToolResult(
            False, "",
            error=f'unknown node "{node}". Use find() to get the exact name first.',
        )

    neighbours = backend.neighbors(canonical)
    all_edges: list[SeenEdge] = []
    for other in neighbours:
        all_edges.extend(backend.edges_between(canonical, other))

    if relation:
        wanted = str(relation).strip().casefold()
        all_edges = [e for e in all_edges if e.relation.casefold() == wanted]
        if not all_edges:
            available = sorted({e.relation for other in neighbours
                                for e in backend.edges_between(canonical, other)})
            # The message names the argument the list belongs to: free-text
            # relation names read like entity names.
            listed = ", ".join(f'"{r}"' for r in available) or "(none)"
            return ToolResult(
                True,
                f'"{canonical}" has no "{relation}" edges.\n'
                f'Other relations available ON THIS NODE — pass one as the '
                f'`relation` argument, with node="{canonical}": {listed}\n'
                f'These are relation names, not entities: do not pass them as `node`.',
            )
    elif backend.degree(canonical) > cfg.hub_threshold:
        # The page rendered here is seen; the withheld remainder is not.
        # Ranking is by node importance alone, independent of the question, so
        # the tool does no retrieval on the model's behalf.
        counts: dict[str, int] = {}
        for e in all_edges:
            counts[e.relation] = counts.get(e.relation, 0) + 1
        # Tie-break on the relation name, so the summary is the same in every
        # process.
        ranked_rels = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        # Capped: see ToolConfig.hub_summary_limit.
        shown = ranked_rels[: cfg.hub_summary_limit]
        summary = "\n".join(f"  {r}: {n}" for r, n in shown)
        if len(ranked_rels) > len(shown):
            summary += f"\n  ({len(ranked_rels) - len(shown)} more relations not listed)"
        ranked = sorted(all_edges, key=lambda e: (
            -backend.importance(e.target if e.source == canonical else e.source),
            e.relation, e.target, e.source))
        page = ranked[: cfg.page_size]
        (rng or random.Random(0)).shuffle(page)
        withheld = len(all_edges) - len(page)
        lines = [render_edge(e) for e in page]
        return ToolResult(
            True,
            f'"{canonical}" has {len(all_edges)} connections — too many to list. '
            f"Relations available:\n{summary}\n"
            f"Showing the {len(page)} most connected, {withheld} not shown:\n"
            + "\n".join(lines)
            + '\nCall expand again with a "relation" argument to see others.',
            edges=tuple(page),
            total=len(all_edges),
            truncated=True,
        )

    # Stable order so paging is coherent, then shuffle within the page so
    # position carries no signal (same rationale as serialize_subgraph).
    all_edges.sort(key=lambda e: (e.relation, e.target, e.source))
    total = len(all_edges)
    offset = max(0, int(offset))
    page = all_edges[offset: offset + cfg.page_size]
    if not page:
        return ToolResult(True, f'No more edges for "{canonical}" at offset {offset} '
                                f"(it has {total}).", total=total)
    (rng or random.Random(0)).shuffle(page)

    text = _edge_block(page)
    truncated = total > offset + len(page)
    if truncated:
        text += (f"\n(showing {offset + 1}-{offset + len(page)} of {total} — "
                 f'call again with "offset": {offset + len(page)} for more)')
    return ToolResult(True, text, edges=tuple(page), total=total, truncated=truncated)


# ---------------------------------------------------------------- connect

def tool_connect(
    backend: GraphBackend,
    cfg: ToolConfig,
    *,
    a: str,
    b: str,
    max_hops: int | None = None,
) -> ToolResult:
    """Up to `connect_limit` shortest paths between two known nodes, of at most
    `connect_max_hops` hops (`max_hops` can only lower that)."""
    src, dst = backend.canonical(str(a)), backend.canonical(str(b))
    missing = [raw for raw, res in ((a, src), (b, dst)) if res is None]
    if missing:
        return ToolResult(
            False, "",
            error=f"unknown node(s): {', '.join(repr(m) for m in missing)}. "
                  "Use find() to get exact names.",
        )
    if src == dst:
        return ToolResult(True, f'"{src}" is the same node as "{dst}".')

    hops_cap = min(int(max_hops or cfg.connect_max_hops), cfg.connect_max_hops)
    G = backend.digraph()
    for hops in range(1, hops_cap + 1):
        paths = find_paths(G, src, dst, hops, limit=cfg.connect_limit,
                           max_expand=cfg.hub_threshold)
        if not paths:
            continue
        seen: list[SeenEdge] = []
        blocks = []
        for i, nodes in enumerate(paths[: cfg.connect_limit], start=1):
            edges: list[SeenEdge] = []
            for u, v in zip(nodes, nodes[1:]):
                edges.extend(backend.edges_between(u, v))
            seen.extend(edges)
            blocks.append(f"PATH {i} ({hops} hop{'s' if hops > 1 else ''}):\n"
                          + _edge_block(edges))
        return ToolResult(True, "\n".join(blocks), edges=tuple(seen))

    return ToolResult(
        True,
        f'No path of {hops_cap} hops or fewer between "{src}" and "{dst}".',
    )



# ---------------------------------------------------------------- dispatch

@dataclass(frozen=True)
class ToolSpec:
    """One tool: its function, its argument names, the ones it requires, the
    doc line the prompt renders, the arguments `dispatch` coerces to int (the
    native adapter types them `integer`), and whether the function takes the
    episode's rng."""
    fn: object
    args: tuple[str, ...]
    required: tuple[str, ...]
    doc: str
    int_args: tuple[str, ...] = ()
    takes_rng: bool = False


TOOLS: dict[str, ToolSpec] = {
    "find": ToolSpec(
        tool_find, ("name",), ("name",),
        'find(name)                       -> up to {k} nodes whose name matches `name`, with\n'
        "                                    type and connection count. Returns no edges. Use it\n"
        "                                    first to get the exact spelling of an entity.",
    ),
    # A hub returns its relation summary plus a page of its most-connected
    # edges (tool_expand), not "a relation summary instead of edges". The
    # wording is part of the `agent_v1` system prompt the policies were trained
    # under, pinned in tests/test_agent_loop.py; changing it is a new version.
    "expand": ToolSpec(
        tool_expand, ("node", "relation", "offset"), ("node",),
        "expand(node, relation?, offset?) -> the edges connected to `node` in either direction,\n"
        "                                    at most {page} per call. `relation` filters, `offset`\n"
        "                                    pages. A node with more than {hub} connections returns\n"
        "                                    a relation summary instead of edges — narrow first.",
        int_args=("offset",), takes_rng=True,
    ),
    "connect": ToolSpec(
        tool_connect, ("a", "b", "max_hops"), ("a", "b"),
        "connect(a, b, max_hops?)         -> up to {limit} shortest paths between two nodes, as\n"
        "                                    edges. Use it when you already know both ends.",
        int_args=("max_hops",),
    ),
}

#: The tools a run offers when its config names none. A tool registered in
#: TOOLS beyond these is opt-in (`--tools`, or a `tools:` config key).
DEFAULT_TOOLS: tuple[str, ...] = ("find", "expand", "connect")


def available_tools(cfg: "ToolConfig | None" = None) -> tuple[str, ...]:
    """The tools this run offers, in registry order: DEFAULT_TOOLS, or
    `cfg.tools` when a run names its own. An unknown name raises ValueError."""
    if cfg is None or cfg.tools is None:
        return DEFAULT_TOOLS
    unknown = [t for t in cfg.tools if t not in TOOLS]
    if unknown:
        raise ValueError(f"unknown tool(s) in the allowlist: {', '.join(unknown)}; "
                         f"known: {', '.join(TOOLS)}")
    return tuple(n for n in TOOLS if n in cfg.tools)


def _unknown_tool_error(tool: str, cfg: "ToolConfig | None" = None) -> str:
    return (f'unknown tool "{tool}". Available: {", ".join(available_tools(cfg))}.')


def format_doc(name: str, cfg: ToolConfig) -> str:
    """A tool's doc line with this run's limits filled in, shared by the
    prompt's tool table and the native adapter's schema descriptions."""
    return TOOLS[name].doc.format(k=cfg.find_k, page=cfg.page_size,
                                  hub=cfg.hub_threshold, limit=cfg.connect_limit)


#: Above this many distinct relations, the prompt describes the vocabulary
#: instead of listing it.
MAX_VOCAB_IN_PROMPT = 64


def tool_docs(backend: GraphBackend, cfg: ToolConfig, *, relation_vocab: bool = True) -> str:
    """The tool table rendered with this run's limits."""
    lines = [format_doc(name, cfg) for name in available_tools(cfg)]
    if relation_vocab:
        vocab = backend.relation_vocab()
        if vocab and len(vocab) <= MAX_VOCAB_IN_PROMPT:
            # Short enough to list, which saves the model from guessing
            # relation spellings.
            lines.append("relations in this graph: " + ", ".join(vocab))
        elif vocab:
            # Too long to list: the prompt says the vocabulary is open.
            lines.append(
                f"This graph has {len(vocab):,} distinct relation names — too "
                "many to list, and they are free text rather than a fixed "
                "schema. Read the exact relation spelling off the OBSERVATION "
                "lines before using it in a filter or a citation.")
    return "\n".join(lines)


def dispatch(
    backend: GraphBackend,
    cfg: ToolConfig,
    tool: str,
    args: dict,
    *,
    rng: random.Random | None = None,
) -> ToolResult:
    """Validate and run one call. Never raises — a bad call is a ToolResult."""
    spec = TOOLS.get(tool)
    if spec is None or tool not in available_tools(cfg):
        return ToolResult(False, "", error=_unknown_tool_error(tool, cfg))
    if not isinstance(args, dict):
        return ToolResult(False, "", error=f'"args" must be a JSON object for {tool}.')

    unknown = [k for k in args if k not in spec.args]
    if unknown:
        return ToolResult(False, "", error=f"{tool} got unexpected argument(s): "
                                           f"{', '.join(unknown)}. Expected: {', '.join(spec.args)}.")
    missing = [k for k in spec.required if args.get(k) in (None, "")]
    if missing:
        return ToolResult(False, "", error=f"{tool} needs: {', '.join(missing)}.")

    kwargs = {k: v for k, v in args.items() if v is not None}
    for name in spec.int_args:
        if name in kwargs:
            try:
                kwargs[name] = int(kwargs[name])
            except (TypeError, ValueError):
                return ToolResult(False, "", error=f'{tool} "{name}" must be an integer.')
    if spec.takes_rng:
        kwargs["rng"] = rng

    try:
        return spec.fn(backend, cfg, **kwargs)
    except Exception as exc:                      # a tool bug must not kill a rollout
        # It is counted in `n_tool_errors` like a bad call; the error string
        # is what tells a tool bug from a model error.
        return ToolResult(False, "", error=f"{tool} failed: {type(exc).__name__}: {exc}")


__all__ = [
    "ToolResult", "ToolConfig", "TOOLS", "DEFAULT_TOOLS", "dispatch", "tool_docs",
    "available_tools", "format_doc", "render_edge",
    "tool_find", "tool_expand", "tool_connect",
]
