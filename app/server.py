"""Replay and live server for the Prospector app.

Replay serves run listings and per-item trajectories from data/runs. The
per-call edge attribution (app/attribution.py) parses the recorded
observation text and is checked against the record's own seen_edges. A cited
edge is marked grounded by set membership in the seen edges, on the scorer's
edge key.

Live (app/live.py) runs the served chain through the mlx: driver on a Mac or
the vllm: driver on a GPU machine and streams one episode as server-sent
events from the rollout loop an eval run uses; the record it writes replays
here afterwards.

The whole-graph map (`_layout_payload`) is also computed here. It is a
picture only: no verdict depends on it.

Run:  python -m uvicorn app.server:app --port 8321
Replay needs the extras ".[app]"; live also needs mlx-lm (.venv-mac on a Mac)
or vLLM (the .venv launch/setup.sh builds on a GPU machine). The server
listens on localhost only; on a remote machine, forward the port with
`ssh -L 8321:localhost:8321 <machine>` and open http://localhost:8321 locally.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.attribution import attribute_turn_edges, edge_key
from app.live import ENGINE as LIVE

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "data" / "runs"
STATIC = Path(__file__).resolve().parent / "static"

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Live mode's engine stops with the server: a vLLM engine runs in a
    process of its own, which would otherwise keep the card's memory."""
    yield
    LIVE.close()


app = FastAPI(title="Prospector: replay + live", lifespan=lifespan)


def _run_dirs() -> list[Path]:
    if not RUNS.exists():
        return []
    return sorted((d for d in RUNS.iterdir()
                   if d.is_dir() and (d / "trajectories.jsonl").exists()),
                  key=lambda d: d.stat().st_mtime, reverse=True)


@app.get("/api/runs")
def list_runs() -> list[dict]:
    out = []
    for d in _run_dirs():
        row: dict = {"name": d.name}
        summ = d / "summary.json"
        if summ.exists():
            try:
                s = json.loads(summ.read_text())
                # `aggregate` is keyed by tier; ALL is the whole benchmark.
                agg = (s.get("aggregate") or {}).get("ALL") or {}
                row.update({
                    "model": s.get("model"),
                    "chat_template": s.get("chat_template"),
                    "backend": s.get("backend"),
                    # gated_acc is the headline metric.
                    "gated_acc": agg.get("gated_acc"),
                    "em_claim": agg.get("em_claim"),
                })
            except (json.JSONDecodeError, OSError):
                pass
        out.append(row)
    return out


def _run_dir(run: str) -> Path:
    d = RUNS / run
    # The run must be a listed directory, not a path; a name the filesystem
    # refuses (too long) is unknown too, not an error.
    try:
        ok = "/" not in run and ".." not in run and (d / "trajectories.jsonl").exists()
    except OSError:
        ok = False
    if not ok:
        raise HTTPException(404, f"unknown run {run!r}")
    return d


@app.get("/api/runs/{run}/items")
def list_items(run: str) -> list[dict]:
    d = _run_dir(run)
    results = {}
    rpath = d / "results.jsonl"
    if rpath.exists():
        with rpath.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    results[r.get("id")] = r
                except json.JSONDecodeError:
                    continue
    out = []
    with (d / "trajectories.jsonl").open() as f:
        for line in f:
            t = json.loads(line)
            r = results.get(t["id"], {})
            out.append({
                "id": t["id"], "tier": t.get("tier"),
                "question": t.get("question"),
                "answer": t.get("answer"),
                "gold_answer": t.get("gold_answer"),
                "n_calls": t.get("n_calls"), "status": t.get("status"),
                # The verdict a results row stores; the page reads it
                # (app.js, verdict).
                "gated_acc": r.get("gated_acc"),
            })
    return out


@app.get("/api/runs/{run}/items/{item_id}")
def get_item(run: str, item_id: str) -> dict:
    d = _run_dir(run)
    with (d / "trajectories.jsonl").open() as f:
        t = next((t for t in map(json.loads, f) if t["id"] == item_id), None)
    if t is not None:
        turn_edges, checks, call_focus = attribute_turn_edges(t)
        seen_keys = {edge_key(e) for e in t.get("seen_edges", [])}
        cited = [{**e, "in_seen": edge_key(e) in seen_keys}
                 for e in t.get("cited_edges", [])]
        t["turn_edges"] = turn_edges
        t["checks"] = checks
        t["call_focus"] = call_focus
        t["cited_edges"] = cited
        # The stored metrics row: the UI shows the run's own verdict.
        rpath = d / "results.jsonl"
        if rpath.exists():
            with rpath.open() as f:
                for rl in f:
                    try:
                        r = json.loads(rl)
                    except json.JSONDecodeError:
                        continue
                    if r.get("id") == item_id:
                        t["result"] = r
                        break
        return t
    raise HTTPException(404, f"no item {item_id!r} in {run!r}")


# -------------------------------------------------------------------- live

def _own_page(request: Request) -> None:
    """Refuse a browser request from any page but this app's own. The two
    routes below start work on this machine, and a page open in the same
    browser, on another site or on another localhost port, could otherwise
    start a load or an episode here, though it could not read the result.
    The app's own page sends `Sec-Fetch-Site: same-origin`, an address typed
    into the browser sends `none`, and a client that is not a browser sends
    no header at all; those pass."""
    site = request.headers.get("sec-fetch-site")
    if site is not None and site not in ("same-origin", "none"):
        raise HTTPException(403, "the live routes serve this app's own page only")


@app.get("/api/live/status")
def live_status() -> dict:
    return LIVE.status()


@app.post("/api/live/load", dependencies=[Depends(_own_page)])
def live_load() -> dict:
    """Warm the model in the background; the UI polls /status until ready."""
    LIVE.load_in_background()
    return LIVE.status()


@app.get("/api/live/questions")
def live_questions() -> list[dict]:
    return LIVE.questions()


@app.get("/api/live/stream", dependencies=[Depends(_own_page)])
async def live_stream(request: Request, question: str = "",
                      item: str | None = None,
                      max_turns: int | None = None) -> StreamingResponse:
    """One episode as server-sent events: status* start (generating delta*
    turn?)* final, then end. A failed episode sends `failure` before `end`;
    the name "error" is taken by EventSource's own connection event. The
    episode runs on a worker thread, and a client that disconnects cancels
    it between tokens."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    cancel = threading.Event()

    def emit(event: str, data: dict) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, (event, data))
        except RuntimeError:          # the event loop is closed (server shutdown)
            pass

    def work() -> None:
        try:
            LIVE.run(question=question, item_id=item or None, emit=emit,
                     cancelled=cancel.is_set, max_turns=max_turns)
        except Exception as exc:      # the UI shows it; the server stays up
            emit("failure", {"message": f"{type(exc).__name__}: {exc}"})
        finally:
            emit("end", {})

    threading.Thread(target=work, name="live-episode", daemon=True).start()

    async def events():
        try:
            while True:
                try:
                    event, data = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        break
                    yield ": keepalive\n\n"
                    continue
                yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                if event == "end":
                    break
        finally:
            cancel.set()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.middleware("http")
async def no_static_cache(request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        resp.headers["Cache-Control"] = "no-cache"
    return resp


app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ------------------------------------------------------------- whole-KG map

CACHE = Path(__file__).resolve().parent / "cache"
_GRAPH_MEM: dict[str, dict] = {}
_FILE_FP: dict[tuple[str, int, int], str] = {}
# A graph within both caps is drawn whole. A larger one is drawn as its
# MAP_MAX_NODES best-connected nodes: the layout uses every edge among them,
# and a fixed-seed sample of MAP_MAX_EDGES of those edges is drawn.
MAP_MAX_NODES = 40_000
MAP_MAX_EDGES = 60_000


def _kb_candidates(run: str | None) -> list[tuple[str, Path]]:
    """(substrate, graph file) pairs a fingerprint may be matched against:
    the run's own graph when its summary names one, the live engine's, then
    every registered substrate's default file. Only files on disk."""
    from graph_reasoner.agent.backend import available_backends, default_kb

    out: list[tuple[str, Path]] = []
    if run:
        try:
            s = json.loads((_run_dir(run) / "summary.json").read_text())
        except (HTTPException, OSError, json.JSONDecodeError):
            s = {}
        if s.get("backend") and s.get("kb_path"):
            out.append((s["backend"], REPO / s["kb_path"]))
    if LIVE.kb_path and LIVE.backend:
        out.append((LIVE.backend, REPO / LIVE.kb_path))
    for kind in available_backends():
        if default_kb(kind):
            out.append((kind, REPO / default_kb(kind)))
    return [c for i, c in enumerate(out) if c[1].is_file() and c not in out[:i]]


def _file_fingerprint(path: Path) -> str:
    """kg.load.file_fingerprint, memoised on (path, size, mtime): a lookup
    may hash every candidate."""
    from graph_reasoner.kg.load import file_fingerprint

    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _FILE_FP:
        _FILE_FP[key] = file_fingerprint(str(path))
    return _FILE_FP[key]


def _backend_for(fingerprint: str, run: str | None):
    """The backend whose own `fingerprint()` is `fingerprint`, or None.

    Every backend's fingerprint starts with its file's hash (SubstrateBase;
    PairIndexedBackend appends `+r<cap>`), so a file is hashed first and
    loaded only on a match, and the loaded backend's fingerprint decides.
    """
    from graph_reasoner.agent.backend import load_backend

    file_part = fingerprint.split("+", 1)[0]
    for kind, path in _kb_candidates(run):
        if _file_fingerprint(path) != file_part:
            continue
        backend = load_backend(kind, str(path))
        if backend.fingerprint() == fingerprint:
            return backend
    return None


def _map_view(G) -> tuple[list[str], list[tuple[int, int]], bool]:
    """The nodes the map lays out, the edges among them, and whether that is
    all of `G`: the whole graph in its own order when it is within both caps,
    else its MAP_MAX_NODES best-connected nodes (by degree, then name). A
    traverse node off the map is still drawn (app.js, traverseEls)."""
    if G.number_of_nodes() <= MAP_MAX_NODES and G.number_of_edges() <= MAP_MAX_EDGES:
        names = list(G.nodes)
        whole = True
    else:
        deg = G.degree
        names = sorted(G.nodes, key=lambda x: (-deg[x], x))[:MAP_MAX_NODES]
        whole = False
    idx = {x: i for i, x in enumerate(names)}
    edges = [(idx[u], idx[v]) for u, v in G.edges()
             if u in idx and v in idx and (whole or u != v)]
    return names, edges, whole


def _layout_payload(G) -> dict:
    """Community-structured layout of the graph the map draws.

    Louvain communities (igraph's `community_multilevel`) are packed as discs
    on an Archimedean spiral, largest first, and each community's induced
    subgraph is force-laid (Fruchterman-Reingold) inside its own disc. Nodes
    carry their community id so the client can draw inter-community edges
    faintly. `G` is the backend's own graph; `_map_view` caps a large one.
    """
    import math
    import random

    import igraph as ig

    names, edges, whole = _map_view(G)
    g = ig.Graph(n=len(names), edges=edges, directed=False)
    g.simplify()
    comms = g.community_multilevel()
    member = comms.membership
    sizes = [len(c) for c in comms]
    radii = [22.0 + 34.0 * math.sqrt(s) for s in sizes]
    # Circle packing for the community discs: the largest at the origin, each
    # next one walks an Archimedean spiral until it fits beside those placed.
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i])
    centers = [None] * len(sizes)
    placed = []
    for ci in order:
        r = radii[ci]
        if not placed:
            centers[ci] = (0.0, 0.0)
            placed.append((0.0, 0.0, r))
            continue
        theta, R = 0.0, 0.0
        while True:
            theta += 0.35
            R += 6.0 * 0.35 / (2 * math.pi)
            x, y = R * math.cos(theta), R * math.sin(theta)
            if all((x - px) ** 2 + (y - py) ** 2 >= (r + pr + 14.0) ** 2
                   for px, py, pr in placed):
                centers[ci] = (x, y)
                placed.append((x, y, r))
                break

    deg = g.degree()
    xs = [0.0] * len(names); ys = [0.0] * len(names)
    for ci, vs in enumerate(comms):
        cx, cy = centers[ci]
        r = radii[ci]
        if len(vs) == 1:
            xs[vs[0]], ys[vs[0]] = cx, cy
            continue
        sub = g.induced_subgraph(vs)
        sl = sub.layout_fruchterman_reingold(niter=400)
        sx = [p[0] for p in sl]; sy = [p[1] for p in sl]
        sspan = max(max(sx) - min(sx), max(sy) - min(sy)) or 1.0
        f = (2.0 * r) / sspan
        for j, v in enumerate(vs):
            xs[v] = cx + (sx[j] - (min(sx) + max(sx)) / 2) * f
            ys[v] = cy + (sy[j] - (min(sy) + max(sy)) / 2) * f

    x0, y0 = min(xs), min(ys)
    span = max(max(xs) - x0, max(ys) - y0) or 1.0
    scale = 4000.0 / span
    nodes = [
        {"id": n,
         "x": round((xs[i] - x0) * scale, 1),
         "y": round((ys[i] - y0) * scale, 1),
         "c": member[i],
         "s": round(2.0 + 2.2 * math.sqrt(deg[i]), 1)}
        for i, n in enumerate(names)
    ]
    if whole:
        return {"n": len(names), "m": len(edges), "n_comms": len(sizes),
                "nodes": nodes, "edges": edges}
    drawn = sorted(random.Random(0).sample(edges, min(len(edges), MAP_MAX_EDGES)))
    return {"n": len(names), "m": len(drawn), "n_comms": len(sizes),
            "nodes": nodes, "edges": drawn, "m_layout": len(edges),
            "n_graph": G.number_of_nodes(), "m_graph": G.number_of_edges()}


@app.get("/api/graph/{fingerprint}")
def get_graph(fingerprint: str, run: str | None = None) -> dict:
    """The map of the graph a trajectory records, matched by the fingerprint
    backends stamp into trajectories and cached under it in app/cache/. `run`
    names the run the trajectory came from, whose summary names its graph
    file. The cache key's suffix is the layout's version; change it when the
    layout or the map caps change."""
    key = f"{fingerprint}-communities-v3"
    if key in _GRAPH_MEM:
        return _GRAPH_MEM[key]
    CACHE.mkdir(exist_ok=True)
    cpath = CACHE / f"graph-{key}.json"
    try:
        cached = cpath.exists()
    except OSError:                     # a fingerprint the filesystem refuses
        raise HTTPException(404, f"no substrate with fingerprint {fingerprint!r}")
    if cached:
        try:
            payload = json.loads(cpath.read_text())
        except ValueError:              # a half-written file: recomputed below
            payload = None
        if payload is not None:
            _GRAPH_MEM[key] = payload
            return payload
    backend = _backend_for(fingerprint, run)
    if backend is None:
        raise HTTPException(404, f"no substrate with fingerprint {fingerprint!r}")
    payload = _layout_payload(backend.digraph())
    # Written beside the cache file and moved into place, so a crash mid-write
    # leaves no truncated file for the next request to read.
    tmp = cpath.with_name(cpath.name + ".partial")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, cpath)
    _GRAPH_MEM[key] = payload
    return payload
