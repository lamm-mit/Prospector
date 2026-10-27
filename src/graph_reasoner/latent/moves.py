"""Stored episodes -> the next move after every observation, and a knob root's move mix.

The move is decided from the text of the assistant turn that follows an
observation: `answer` (a final answer with no tool call), `follow` (expand a
node named on the page just seen), `backtrack` (expand a node named on an
earlier page or in the question), `fresh` (expand a node seen nowhere),
`find`, `connect`, or `other` (a turn that parses as neither). A page's names
are the endpoints of its edge lines and the nodes a find result lists.

`label_rows` labels the observation rows of one record, for the reads
(`scripts/latent/label_moves.py` writes them beside the reads, where
`readings`, `directions` and `figure` join them by (item, call)). `analyse`
tabulates one knob arm's episodes and `mix` a whole root
(`scripts/latent/mix.py` writes mix.json): the share of calls that follow,
backtrack, expand a fresh node, find or connect; accuracy, mean and median
calls, chain completion, the share answered before the chain was complete,
budget exits, the median lag from completion to the answer. A page is a user
message that starts with the observation header; the schema reminder, the
empty-turn nudge and the forced-answer message leave the "page just seen" as
it was, and the forced final turn is not counted as a call.

Tests: tests/test_latent.py.
"""
from __future__ import annotations

import collections
import json
import os
import re
import statistics as st
from pathlib import Path

import numpy as np

from ..agent.loop import is_answer, parse_tool_call
from ..agent.prompts import FORCED_ANSWER
from ..kg.serialize import normalize, parse_edge_line

__all__ = ["MOVES", "page_nodes", "intent_of", "label_rows", "read_records", "label_extract",
           "write_labels", "analyse", "mix"]

#: The move classes, in the order the figures name them.
MOVES = ("follow", "backtrack", "answer", "fresh", "find", "connect", "other")
#: A find result's line: two spaces, the name, two spaces, the bracketed type and degree.
FOUND_RE = re.compile(r"^  (.+?)  \[.*connections\]$")
OBSERVATION_PREFIX = "OBSERVATION ("
FORCED_PREFIX = FORCED_ANSWER[:24]


def page_nodes(obs: str) -> set[str]:
    """The nodes a page names: both ends of every edge line and every find hit."""
    out: set[str] = set()
    for line in obs.split("\n"):
        e = parse_edge_line(line)
        if e:
            out |= {normalize(e[0]), normalize(e[2])}
        m = FOUND_RE.match(line)
        if m:
            out.add(normalize(m.group(1)))
    return out


def intent_of(text: str, last: set[str], earlier: set[str]) -> str:
    """The move an assistant turn makes, given the names of the page just seen
    and of every earlier page and the question's entity."""
    if is_answer(text):
        return "answer"
    c = parse_tool_call(text)
    if not c:
        return "other"
    tool, args = c
    if tool in ("find", "connect"):
        return tool
    node = normalize(str(args.get("node") or ""))
    if node in last:
        return "follow"
    return "backtrack" if node in earlier else "fresh"


def label_rows(record: dict, rows: list[dict]) -> None:
    """Add `intent` to each observation row of one record, from the turn that
    followed it. A row names the observation's message (`obs_msg`) and the
    turn's (`msg`)."""
    msgs = record["messages"]
    for r in rows:
        last = page_nodes(msgs[r["obs_msg"]]["content"])
        earlier: set[str] = set()
        for i in range(2, r["obs_msg"]):
            if msgs[i]["role"] == "user":
                earlier |= page_nodes(msgs[i]["content"])
        earlier.add(normalize(record.get("anchor") or ""))
        r["intent"] = intent_of(msgs[r["msg"]]["content"], last, earlier)


def read_records(run_dir: Path) -> list[dict]:
    """A stored run's trajectories, in file order."""
    with (Path(run_dir) / "trajectories.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def label_extract(extract_dir: Path, run_dir: Path) -> tuple[list[dict], list[int], np.ndarray]:
    """The observation rows of an extract, their indices in its index.jsonl,
    and their moves."""
    recs = {r["id"]: r for r in read_records(run_dir)}
    rows = [json.loads(line) for line in (Path(extract_dir) / "index.jsonl").open(encoding="utf-8") if line.strip()]
    keep = [i for i, r in enumerate(rows) if r.get("kind") == "observation"]
    rows = [rows[i] for i in keep]
    by_id: dict[str, list[dict]] = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)
    for iid, rs in by_id.items():
        label_rows(recs[iid], rs)
    return rows, keep, np.array([r["intent"] for r in rows])


def write_labels(out: Path, rows: list[dict], keep: list[int], lab: np.ndarray, run_name: str) -> None:
    """The labels file the readings, the directions and the figure join to the
    reads by (item, call)."""
    np.savez(out, keep=np.array(keep), lab=lab, G=np.array([r["G"] for r in rows]), hop=np.array([r["hop"] for r in rows]),
             call=np.array([r["call"] for r in rows]), tool=np.array([r["tool"] for r in rows]),
             tier=np.array([r["tier"] for r in rows]), item=np.array([r["id"] for r in rows]), run=run_name)


def analyse(run_dir: str | Path) -> dict:
    """One arm's episodes -> its move shares and episode-level figures."""
    run_dir = str(run_dir)
    c: collections.Counter = collections.Counter()
    calls, lags = [], []
    n = comp = prem = bud = malformed = 0
    for line in open(os.path.join(run_dir, "trajectories.jsonl"), encoding="utf-8"):
        rec = json.loads(line)
        msgs = rec["messages"]
        n += 1
        calls.append(rec["n_calls"])
        g = rec.get("gold_complete_at_call")
        if g is not None:
            comp += 1
            if rec["status"] == "answered":
                lags.append(rec["n_calls"] - g)
        if rec["status"] == "budget":
            bud += 1
        if rec["status"] == "malformed":
            malformed += 1
        if rec["status"] == "answered" and g is None:
            prem += 1
        earlier = {normalize(rec["anchor"] or "")}
        last: set[str] = set()
        forced = False
        for i, m in enumerate(msgs):
            if m["role"] == "user":
                if i >= 2 and m["content"].startswith(OBSERVATION_PREFIX):
                    earlier |= last
                    last = page_nodes(m["content"])
                elif m["content"].startswith(FORCED_PREFIX):
                    forced = True
                continue
            if m["role"] != "assistant" or forced:
                continue
            text = m["content"]
            if is_answer(text):
                c["answer"] += 1
                continue
            call = parse_tool_call(text)
            if not call:
                c["other"] += 1
                continue
            tool, args = call
            if tool in ("find", "connect"):
                c[tool] += 1
                continue
            node = normalize(str(args.get("node") or ""))
            c["follow" if node in last else "backtrack" if node in earlier else "fresh"] += 1
    res = [json.loads(line) for line in open(os.path.join(run_dir, "results.jsonl"), encoding="utf-8")]
    tot = sum(v for k, v in c.items() if k not in ("answer", "other")) or 1
    ex = c["follow"] + c["backtrack"] + c["fresh"] or 1
    return dict(n=n, acc=st.mean(bool(r["gated_acc"]) for r in res), gate=st.mean(bool(r["gate_pass"]) for r in res),
                calls=st.mean(calls), calls_median=st.median(calls), complete=comp / n,
                lag=st.median(lags) if lags else None, premature=prem / n, budget=bud / n, malformed=malformed / n,
                follow=c["follow"] / tot, backtrack=c["backtrack"] / tot, fresh=c["fresh"] / tot,
                find=c["find"] / tot, connect=c["connect"] / tot, follow_of_expands=c["follow"] / ex,
                calls_total=tot, unparsed_turns=c["other"])


def mix(root: str | Path, log=None) -> dict:
    """Every finished arm under a knob root (`knob-<arm>--<benchmark>/` with a
    summary.json), tabulated: benchmark -> arm -> the row of `analyse` plus the
    arm's kind, setting, layer and direction from its knob.json."""
    root = str(root)
    out: dict = {}
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not (name.startswith("knob-") and os.path.exists(os.path.join(d, "summary.json"))):
            continue
        arm, bench = name[len("knob-"):].split("--", 1)
        knob = json.load(open(os.path.join(d, "knob.json"), encoding="utf-8"))
        row = analyse(d)
        row.update(arm=arm, kind=knob["arm"]["kind"], setting=knob["arm"].get("value", 0.0),
                   layer=knob.get("layer"), direction=knob.get("direction"))
        out.setdefault(bench, {})[arm] = row
        if log:
            log(f"{bench:28s} {arm:9s} acc {row['acc']:.3f} calls {row['calls']:5.1f} backtrack {row['backtrack']:.3f} "
                f"follow|expand {row['follow_of_expands']:.3f} premature {row['premature']:.3f} lag {row['lag']}")
    return out
