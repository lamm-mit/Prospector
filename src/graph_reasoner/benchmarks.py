"""Resolve a benchmark name to its file, and to its gold answer sets.

A stored run is a directory named `<tag>--<benchmark stem>`: `run_dir_name`
writes the name and `split_run_name` reads it back, so a run names its
benchmark by stem and never by path. `resolve` turns the stem back into a file
by searching `data/benchmarks/<substrate>/`, so a benchmark's stem must be
unique across substrates.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = ROOT / "data" / "benchmarks"

_GOLD_CACHE: dict[str, dict[str, list[str]]] = {}

#: Between a run's tag and its benchmark stem in a run directory's name.
RUN_SEP = "--"


def run_dir_name(tag: str, benchmark: str | Path) -> str:
    """`<tag>--<benchmark stem>`: the name of a stored run's directory. A
    writer of a run directory names it with this, so `split_run_name` can read
    it back."""
    return f"{tag}{RUN_SEP}{Path(benchmark).stem}"


def split_run_name(name: str) -> tuple[str, str | None]:
    """A run directory's name read back: `(tag, benchmark stem)`, the stem None
    for a name with no separator. The tag ends at the first separator."""
    tag, sep, bench = name.partition(RUN_SEP)
    return tag, (bench if sep else None)


def repo_path(path: str | Path) -> str:
    """A path as a stored record names it: relative to the repository when it
    lies inside it, so the record names no machine's directories."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve(name_or_path: str | Path) -> Path:
    """Benchmark stem (or path) -> the file. Raises `FileNotFoundError` if it
    cannot be found.

    In order: the argument itself if it is an existing `.jsonl` path; then
    `data/benchmarks/<substrate>/<stem>.jsonl` for each substrate directory in
    sorted order; then `data/benchmarks/<stem>.jsonl`. The first hit wins, so a
    stem reused under two substrates resolves to whichever sorts first.
    """
    p = Path(name_or_path)
    if p.suffix == ".jsonl" and p.exists():
        return p
    stem = p.name[:-6] if p.name.endswith(".jsonl") else p.name
    searched = []
    for sub in sorted(d for d in BENCHMARK_ROOT.glob("*") if d.is_dir()):
        cand = sub / f"{stem}.jsonl"
        searched.append(cand)
        if cand.exists():
            return cand
    flat = BENCHMARK_ROOT / f"{stem}.jsonl"          # in no substrate directory
    searched.append(flat)
    if flat.exists():
        return flat
    raise FileNotFoundError(
        f"no benchmark named {stem!r}; looked in:\n  "
        + "\n  ".join(str(s) for s in searched))


def load_benchmark(path: str | Path) -> list:
    """A benchmark file's items, as `BenchmarkItem`s."""
    from .schema import BenchmarkItem

    items = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(BenchmarkItem(**json.loads(line)))
    return items


def answers_for(static_path: str | Path) -> Path:
    """The benchmark holding the answer sets for a static benchmark's items.

    A static benchmark is the agentic one with a variant suffix (`ppr400anchor`,
    `khop400anchor`, `closed_book`) appended after a double underscore, so
    `<stem>__<variant>.jsonl` answers to `<stem>_agentic.jsonl` beside it.
    Raises if the name has no `__` or that file does not exist.
    """
    p = resolve(static_path)
    stem = p.name[:-len(".jsonl")]
    if "__" not in stem:
        raise ValueError(
            f"{p.name} is not a static benchmark (<stem>__<variant>.jsonl), so "
            f"its answer sets cannot be derived; pass --benchmark")
    cand = p.with_name(f"{stem.split('__', 1)[0]}_agentic.jsonl")
    if not cand.exists():
        raise FileNotFoundError(
            f"{p.name} implies answer sets in {cand.name}, which does not "
            f"exist; pass --benchmark")
    return cand


def excluded_ids(paths) -> set[str]:
    """The item ids of the benchmarks a row builder must not draw from (the
    held-out eval), read from each file's `id` field."""
    out: set[str] = set()
    for path in paths:
        with Path(path).open(encoding="utf-8") as f:
            out |= {json.loads(l)["id"] for l in f if l.strip()}
    return out


def assert_no_leak(ids, banned: set[str]) -> None:
    """Abort a build (`SystemExit`) that read a rollout on an excluded item.
    A row builder calls this rather than skipping the rollout."""
    leaked = sum(1 for i in ids if i in banned)
    if leaked:
        raise SystemExit(f"ABORT: {leaked} rollouts are on excluded eval items")


def substrate_of(path: str | Path) -> str | None:
    """The substrate directory a benchmark sits in, or None off the layout."""
    p = Path(path).resolve()
    parent = p.parent
    return parent.name if parent.parent == BENCHMARK_ROOT.resolve() else None

def gold_answers(name_or_path: str | Path) -> dict[str, list[str]]:
    """id -> gold answer set, for a benchmark stem or path. Cached."""
    key = str(name_or_path)
    if key not in _GOLD_CACHE:
        from .schema import answer_set_of
        with resolve(key).open(encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        _GOLD_CACHE[key] = {
            r["id"]: answer_set_of(r.get("meta"), r.get("gold_answer"))
            for r in rows}
    return _GOLD_CACHE[key]
