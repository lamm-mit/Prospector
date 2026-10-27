"""The graph-access contract, and the registry that resolves a substrate by name.

No concrete backend lives here. A substrate is a plugin: `substrates/<name>/`
implements `GraphBackend` and registers itself, and nothing in `agent/`,
`train/`, `select/` or `eval/` names a graph.

A backend returns `SeenEdge`s, and the set of edges the tools returned is the
only thing a citation is checked against (`cited ⊆ seen`, by set membership),
so a backend must not invent, alter or silently truncate the edges it returns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from difflib import SequenceMatcher
from typing import Callable, Protocol

import networkx as nx

from ..kg.serialize import FUZZY_THRESHOLD, edge_key, normalize


@dataclass(frozen=True)
class SeenEdge:
    """An edge the model has actually been shown, in the KG's true direction."""

    source: str
    relation: str
    target: str

    def key(self) -> tuple[str, str, str]:
        return edge_key(self.source, self.relation, self.target)


class GraphBackend(Protocol):
    """What the tools need from a graph."""

    name: str

    def canonical(self, name: str) -> str | None: ...
    def search(self, query: str, k: int = 10) -> list[tuple[str, str, int]]: ...
    def degree(self, node: str) -> int: ...
    def importance(self, node: str) -> float: ...
    def node_type(self, node: str) -> str: ...
    def neighbors(self, node: str) -> set[str]: ...
    def edges_between(self, u: str, v: str) -> tuple[SeenEdge, ...]: ...
    def relation_vocab(self) -> tuple[str, ...]: ...
    def digraph(self) -> nx.DiGraph: ...
    def fingerprint(self) -> str: ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EntityIndex:
    """Name -> node lookup: verbatim, then case-folded, then fuzzy.

    A verbatim match wins, because two distinct nodes can share a case-folded
    name and the model copies names exactly as observations show them. An
    ambiguous case-folded lookup resolves to the candidate ranked highest by
    `rank` (the backends pass node degree), and `search` returns every such
    candidate. `search` shortlists names by shared tokens before fuzzy
    matching, which keeps a lookup fast on a large graph.
    """

    def __init__(self, names, rank=None) -> None:
        self._names: set[str] = set()
        self._by_norm: dict[str, list[str]] = {}
        self._tokens: dict[str, set[str]] = {}
        for name in names:
            self._names.add(name)
            norm = normalize(name)
            self._by_norm.setdefault(norm, []).append(name)
            for token in _TOKEN_RE.findall(norm):
                self._tokens.setdefault(token, set()).add(name)
        # Deterministic collision order: highest rank first, then by name.
        rank = rank or (lambda _n: 0)
        for norm, group in self._by_norm.items():
            if len(group) > 1:
                group.sort(key=lambda n: (-rank(n), n))

    def canonical(self, name: str) -> str | None:
        if name in self._names:
            return name
        group = self._by_norm.get(normalize(name))
        return group[0] if group else None

    def search(self, query: str, k: int = 10) -> list[str]:
        norm = normalize(query)
        if not norm:
            return []

        candidates: set[str] = set()
        for token in _TOKEN_RE.findall(norm):
            candidates |= self._tokens.get(token, set())
        if not candidates:  # no shared token: substring scan over normalized keys
            candidates = {n for kk, group in self._by_norm.items() if norm in kk
                          for n in group}

        scored: list[tuple[float, str]] = []
        for cand in candidates:
            cnorm = normalize(cand)
            if cnorm == norm:
                ratio = 1.0
            elif norm in cnorm or cnorm in norm:
                ratio = 0.9 + 0.1 * (min(len(norm), len(cnorm)) / max(len(norm), len(cnorm)))
            else:
                ratio = SequenceMatcher(None, norm, cnorm).ratio()
            if ratio >= FUZZY_THRESHOLD:
                scored.append((ratio, cand))

        # The raw name is the final tie-break. Names that differ only in case
        # normalize equal, and without it their order would follow set
        # iteration, which varies between processes.
        scored.sort(key=lambda x: (-x[0], normalize(x[1]), x[1]))
        out = [name for _, name in scored[:k]]
        # An exact hit is never ranked away, and every case-collision sibling
        # stays visible.
        for exact in reversed(self._by_norm.get(norm, [])):
            if exact not in out:
                out = [exact] + out[: k - 1]
        return out


# ---------------------------------------------------------------- registry

_BACKENDS: dict[str, Callable[..., "GraphBackend"]] = {}
_DEFAULT_KB: dict[str, str] = {}
_BENCHMARKS: dict[str, dict[str, str]] = {}
_WINDOW: dict[str, int] = {}


def register_backend(name: str, factory: Callable[..., "GraphBackend"],
                     *, default_kb: str | None = None,
                     eval_benchmark: str | None = None,
                     train_benchmark: str | None = None,
                     max_model_len: int | None = None) -> None:
    """Called by each substrate package at import time.

    `default_kb` is the graph file the substrate reads when a caller names no
    other, so a substrate and its file are resolved together.
    `eval_benchmark` and `train_benchmark` are its question files when they
    are not at the default paths `data/benchmarks/<name>/<name>_core_agentic.jsonl`
    and `<name>_train_agentic.jsonl`; the recipe (`graph_reasoner.recipe`)
    reads them through `eval_benchmark()` and `train_benchmark()`.
    `max_model_len` is the context window a local engine needs on this graph,
    when its conversations outgrow the engine's default; every vllm: and mlx:
    engine built for the graph uses it unless the caller names another
    (`context_window()`).
    """
    _BACKENDS[name] = factory
    if max_model_len:
        _WINDOW[name] = int(max_model_len)
    if default_kb:
        _DEFAULT_KB[name] = default_kb
    declared = {k: v for k, v in (("eval", eval_benchmark), ("train", train_benchmark)) if v}
    if declared:
        _BENCHMARKS[name] = declared


def default_kb(kind: str) -> str | None:
    """The graph file a substrate declares for itself, or None."""
    _load_substrates()
    return _DEFAULT_KB.get(kind)


def context_window(kind: str | None) -> int | None:
    """The context window a substrate declares for local engines, or None
    for the engine's default."""
    if not kind:
        return None
    _load_substrates()
    return _WINDOW.get(kind)


def _benchmark(kind: str, role: str) -> str:
    _load_substrates()
    if kind not in _BACKENDS:
        raise ValueError(f"unknown backend: {kind!r} (registered: "
                         f"{', '.join(sorted(_BACKENDS)) or 'none'})")
    declared = _BENCHMARKS.get(kind, {}).get(role)
    if declared:
        return declared
    suffix = "core_agentic" if role == "eval" else "train_agentic"
    return f"data/benchmarks/{kind}/{kind}_{suffix}.jsonl"


def eval_benchmark(kind: str) -> str:
    """The substrate's evaluation questions: the file it declares, else
    `data/benchmarks/<kind>/<kind>_core_agentic.jsonl`."""
    return _benchmark(kind, "eval")


def train_benchmark(kind: str) -> str:
    """The substrate's training questions: the file it declares, else
    `data/benchmarks/<kind>/<kind>_train_agentic.jsonl`."""
    return _benchmark(kind, "train")


def resolve_kb(kind: str, explicit: str | None) -> str:
    """The graph file to load: an explicit path wins, else the substrate's own.

    Callers pass whatever the user gave them, including None, and never carry a
    default of their own.
    """
    if explicit:
        return explicit
    path = default_kb(kind)
    if path:
        return path
    raise ValueError(
        f"substrate {kind!r} declares no default graph file: pass --kb. "
        f"Substrates that declare one: "
        f"{', '.join(sorted(_DEFAULT_KB)) or 'none'}")


def _load_substrates() -> None:
    """Import the substrate packages so they register themselves. Deferred to
    first use, so importing `agent/` imports no substrate."""
    from .. import substrates  # noqa: F401


def available_backends() -> list[str]:
    _load_substrates()
    return sorted(_BACKENDS)


def load_backend(kind: str, path: str) -> "GraphBackend":
    """The backend registered under `kind`, built from the graph file `path`.
    A missing file is refused here, by name, before a parser's traceback."""
    _load_substrates()
    try:
        factory = _BACKENDS[kind]
    except KeyError:
        raise ValueError(
            f"unknown backend: {kind!r} (registered: "
            f"{', '.join(sorted(_BACKENDS)) or 'none'})") from None
    if not Path(path).exists():
        raise FileNotFoundError(f"no graph file at {path}: data/raw/README.md says how to fetch each graph")
    return factory(path)
