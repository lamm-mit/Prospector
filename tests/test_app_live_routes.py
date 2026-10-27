"""The app's live routes (app/server.py) start work on this machine: a model
load and an episode. A browser request from any page but the app's own, on
another site or on another localhost port, is refused before either starts;
the app's own page (same-origin), an address typed into the browser (none)
and a client that is not a browser (no Sec-Fetch-Site header) go through.
Needs the app extras.
"""
import pytest

pytest.importorskip("fastapi", reason="the app extras (.[app]) are not installed")
pytest.importorskip("igraph", reason="the app extras (.[app]) are not installed")
pytest.importorskip("httpx", reason="fastapi's test client needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

import app.server as srv  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    """The app with the live engine stubbed: a load is counted, never run."""
    loads = []
    monkeypatch.setattr(srv.LIVE, "load_in_background", lambda: loads.append(1))
    monkeypatch.setattr(srv.LIVE, "status", lambda: {"state": "stub"})
    return TestClient(srv.app), loads


def test_a_request_from_another_page_is_refused_before_anything_starts(client):
    c, loads = client
    for site in ("cross-site", "same-site"):
        for method, path in (("post", "/api/live/load"), ("get", "/api/live/stream")):
            r = getattr(c, method)(path, headers={"Sec-Fetch-Site": site})
            assert r.status_code == 403, (site, path, r.text)
    assert loads == []


def test_the_apps_own_page_and_a_plain_client_may_load(client):
    c, loads = client
    assert c.post("/api/live/load", headers={"Sec-Fetch-Site": "same-origin"}).status_code == 200
    assert c.post("/api/live/load", headers={"Sec-Fetch-Site": "none"}).status_code == 200
    assert c.post("/api/live/load").status_code == 200
    assert len(loads) == 3
