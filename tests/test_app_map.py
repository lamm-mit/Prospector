"""The app's whole-graph map (app/server.py): which graph a trajectory's
fingerprint names, and how much of it is drawn.

The graph is found through the substrate registry and matched on the
fingerprint the backend itself computes, so a fingerprint with a `+r<cap>`
suffix resolves as a bare file hash does. Needs the app extras (fastapi,
python-igraph).
"""
import json

import networkx as nx
import pytest

pytest.importorskip("fastapi", reason="the app extras (.[app]) are not installed")
pytest.importorskip("igraph", reason="the app extras (.[app]) are not installed")

import app.server as srv  # noqa: E402
import graph_reasoner.substrates  # noqa: E402,F401  (registered before the registry is patched)
from graph_reasoner.agent import backend as registry  # noqa: E402


@pytest.fixture
def server(tmp_path, monkeypatch):
    """The server with its runs and cache in tmp_path and only the toy
    substrates registered, so no lookup hashes a release graph."""
    monkeypatch.setattr(srv, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(srv, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(srv, "_GRAPH_MEM", {})
    monkeypatch.setattr(registry, "available_backends", lambda: ["biograph", "triples"])
    monkeypatch.setattr(registry, "_DEFAULT_KB", {})
    return srv


def _run(server, name, backend, kb_path):
    d = server.RUNS / name
    d.mkdir(parents=True)
    (d / "trajectories.jsonl").write_text("")
    (d / "summary.json").write_text(json.dumps({"backend": backend, "kb_path": kb_path}))
    return name


def _biograph_file(tmp_path):
    U = nx.Graph()
    U.add_node("silk", group=3)
    U.add_node("spider", group=3)
    U.add_node("collagen", group=7)
    U.add_edge("silk", "spider", title="is spun by")
    U.add_edge("silk", "collagen", title="is stiffer than")
    p = tmp_path / "toy_bio.graphml"
    nx.write_graphml(U, p)
    return str(p)


def test_a_registered_default_graph_is_found_by_its_fingerprint(server, film_kb, monkeypatch):
    fp = film_kb.fingerprint()
    with pytest.raises(server.HTTPException):
        server.get_graph(fp)                     # no substrate declares the toy file
    monkeypatch.setitem(registry._DEFAULT_KB, "triples", film_kb._path)
    payload = server.get_graph(fp)
    assert payload["n"] == film_kb.digraph().number_of_nodes()
    assert payload["m"] == film_kb.digraph().number_of_edges()
    assert "n_graph" not in payload              # drawn whole


def test_a_capped_fingerprint_resolves_through_the_runs_own_graph(server, tmp_path):
    from graph_reasoner.substrates.biograph import BiographBackend

    path = _biograph_file(tmp_path)
    fp = BiographBackend(path).fingerprint()
    assert fp.endswith("+r160")
    run = _run(server, "bio--toy", "biograph", path)
    with pytest.raises(server.HTTPException):
        server.get_graph(fp)                     # no run, no registered default
    assert server.get_graph(fp, run=run)["n"] == 3
    with pytest.raises(server.HTTPException):    # the bare file hash is not this graph
        server.get_graph(fp.split("+")[0], run=run)


def test_a_graph_over_the_caps_is_drawn_as_its_best_connected_nodes(server, film_kb, monkeypatch):
    monkeypatch.setattr(server, "MAP_MAX_NODES", 3)
    monkeypatch.setattr(server, "MAP_MAX_EDGES", 1)
    G = film_kb.digraph()
    names, edges, whole = server._map_view(G)
    assert not whole
    assert names == ["Film B", "Jane Doe", "Drama"]       # degree, then name
    assert sorted(edges) == [(0, 1), (0, 2)]              # every edge among them
    payload = server._layout_payload(G)
    assert (payload["n"], payload["m"], payload["m_layout"]) == (3, 1, 2)
    assert (payload["n_graph"], payload["m_graph"]) == (G.number_of_nodes(), G.number_of_edges())
