"""Tool-layer tests: `agent/tools.py`, plus `agent/backend.py`'s EntityIndex,
`agent/prompts.py`'s templates and `kg/serialize.py`'s edge line.

The seen contract comes first: a tool's rendered edge lines and its returned
edges are the same set. The rest pins what keeps observations reproducible:
three orderings that must not depend on PYTHONHASHSEED (the hub summary's
relations, `find`'s case siblings and `connect`'s paths), the frozen triples
loader, prompt templates without doubled braces, and one edge rendering
shared by every producer.
"""
import json
import random
import re
import subprocess
import sys

import pytest

from graph_reasoner.substrates.triples import TextKBBackend
from graph_reasoner.agent.tools import (
    TOOLS,
    ToolConfig,
    available_tools,
    dispatch,
    tool_docs,
)
from graph_reasoner.kg.serialize import parse_edge_line


@pytest.fixture
def kb(tmp_path):
    """A writer-director pair (parallel relations), a case collision, a hub."""
    lines = [
        "Film A|written_by|Jane Doe",
        "Film A|directed_by|Jane Doe",       # same pair, second label
        "Film A|starred_actors|Ann Poe",
        "Film A|has_genre|Drama",
        "Police|directed_by|Roy Roe",        # collides with the tag below
        "Film B|has_tags|police",
        "Film B|has_genre|Drama",
        "Film C|has_genre|Drama",
        "Film D|has_genre|Drama",
        "Film E|has_genre|Drama",
    ]
    p = tmp_path / "kb.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return TextKBBackend(str(p))


def _lines_as_edges(text):
    out = set()
    for line in text.splitlines():
        edge = parse_edge_line(line.strip())
        if edge:
            out.add(tuple(g.strip() for g in edge))
    return out


def _as_tuples(edges):
    return {(e.source, e.relation, e.target) for e in edges}


# The three branches `expand` can take; the seen contract must hold on each.
# The hub and paged branches return only a page of the node's edges.
CONTRACT_CONFIGS = {
    "default": ToolConfig(),
    # 'Drama' has degree 5. Both knobs are needed: the hub branch pages with
    # `ranked[:page_size]`, so hub_threshold alone withholds nothing.
    "hub": ToolConfig(hub_threshold=3, page_size=2),
    "paged": ToolConfig(page_size=2),         # plain branch, page < full set
}


# The calls that exercise each registered tool's branches, keyed by the
# registry: a tool added to TOOLS fails the test below until it has calls here,
# so no tool can return edges it never rendered unchecked.
CONTRACT_CALLS = {
    "find": [{"name": "Film A"}],
    "expand": [{"node": "Film A"},
               {"node": "Drama"},                             # hub / paged path
               {"node": "Film A", "relation": "written_by"}],
    "connect": [{"a": "Film A", "b": "Ann Poe"},
                {"a": "Film A", "b": "Film C"}],              # no path
}


def test_every_registered_tool_has_contract_calls():
    assert set(CONTRACT_CALLS) == set(TOOLS)


@pytest.mark.parametrize("cfg_name", sorted(CONTRACT_CONFIGS))
def test_seen_contract(kb, cfg_name):
    """Rendered edge lines and returned edges are the same set, exactly: a
    tool neither returns an edge it did not display nor displays one it did
    not return."""
    cfg = CONTRACT_CONFIGS[cfg_name]
    for tool, calls in CONTRACT_CALLS.items():
        for args in calls:
            r = dispatch(kb, cfg, tool, args, rng=random.Random(0))
            # A failed call renders nothing and returns nothing, so the set equality
            # below would hold vacuously. Assert the call worked first.
            assert r.ok, (tool, args, getattr(r, "error", None))
            assert _lines_as_edges(r.text) == _as_tuples(r.edges), (tool, args, r.text)


def test_hub_returns_a_ranked_page_plus_the_summary(kb):
    """Over the hub threshold with no relation filter, `expand` returns the
    relation summary plus a ranked page of edges. The page is seen; the
    withheld edges are reported as a count only."""
    cfg = ToolConfig(hub_threshold=3, page_size=2)
    r = dispatch(kb, cfg, "expand", {"node": "Drama"}, rng=random.Random(0))
    assert r.ok and r.truncated
    assert "has_genre: 5" in r.text              # summary still present
    assert len(r.edges) == 2                     # ranked page, capped
    assert "3 not shown" in r.text               # withheld count is explicit
    # Every returned edge is rendered in the text: the seen contract.
    for e in r.edges:
        assert f"{e.source} --[{e.relation}]--> {e.target}" in r.text
    # ...and the filtered call still reaches everything the page withheld
    # (own config: this asserts relation filtering, not pagination).
    wide = ToolConfig(hub_threshold=3, page_size=40)
    r2 = dispatch(kb, wide, "expand", {"node": "Drama", "relation": "has_genre"},
                  rng=random.Random(0))
    assert len(r2.edges) == 5


def test_pagination_is_complete_and_disjoint(kb):
    """Paging a node yields every edge exactly once — no gaps, no duplicates."""
    cfg = ToolConfig(page_size=2, hub_threshold=100)
    collected, offset = [], 0
    while True:
        r = dispatch(kb, cfg, "expand", {"node": "Drama", "offset": offset},
                     rng=random.Random(offset))
        collected.extend(_as_tuples(r.edges))
        if not r.truncated:
            break
        offset += len(r.edges)
    assert len(collected) == len(set(collected)) == 5


def test_parallel_relations_both_returned(kb):
    """A node pair joined by two relations shows both edges."""
    r = dispatch(kb, ToolConfig(), "expand", {"node": "Film A"}, rng=random.Random(0))
    pair = {rel for src, rel, tgt in _as_tuples(r.edges) if tgt == "Jane Doe"}
    assert pair == {"written_by", "directed_by"}


def test_canonical_prefers_verbatim_over_case_collision(kb):
    """The movie `Police` and the tag `police` are different nodes: an exact
    name resolves to itself, not to a case-folded match."""
    assert kb.canonical("Police") == "Police"
    assert kb.canonical("police") == "police"
    r = dispatch(kb, ToolConfig(), "expand", {"node": "Police"}, rng=random.Random(0))
    assert _as_tuples(r.edges) == {("Police", "directed_by", "Roy Roe")}


def test_find_reports_both_collision_siblings(kb):
    r = dispatch(kb, ToolConfig(), "find", {"name": "police"})
    assert r.ok and r.edges == ()
    assert "Police" in r.text and "police" in r.text


def test_find_fuzzy_and_miss(kb):
    assert "Jane Doe" in dispatch(kb, ToolConfig(), "find", {"name": "jane doe"}).text
    assert "Jane Doe" in dispatch(kb, ToolConfig(), "find", {"name": "Jane Doo"}).text
    miss = dispatch(kb, ToolConfig(), "find", {"name": "Zzzz Qqqq"})
    assert miss.ok and miss.edges == () and "NO MATCH" in miss.text


def test_bad_calls_never_raise(kb):
    cfg = ToolConfig()
    for tool, args in [
        ("nope", {}),                                   # unknown tool
        ("expand", {"node": "Nonexistent Film"}),       # unknown node
        ("expand", {}),                                 # missing required arg
        ("expand", {"node": "Film A", "bogus": 1}),     # unexpected arg
        ("expand", {"node": "Film A", "offset": "abc"}),  # bad type
        ("connect", {"a": "Film A"}),                   # missing required arg
    ]:
        r = dispatch(kb, cfg, tool, args)
        assert not r.ok and r.error and r.edges == ()


def test_no_prompt_template_has_doubled_braces(kb):
    """The prompt templates are filled with `.replace`, not `.format`, so `{{`
    is not an escape and would reach the model verbatim. Every rendering is
    checked: each prompt version's system prompt (over the default tools and
    over find and expand alone), each version's nudges, and the user turn."""
    from graph_reasoner.agent import prompts
    from graph_reasoner.schema import BenchmarkItem

    item = BenchmarkItem(id="easy-0000", tier="easy", question="which film is Drama",
                         gold_answer="Film A", gold_path={"nodes": [], "edges": []},
                         subgraph_edges=[], subgraph_text="")
    filled = {"user": prompts.build_agent_user(item, max_turns=20)}
    for v in prompts._SYSTEMS:
        for tools in (None, ("find", "expand")):
            filled[f"system {v} {tools}"] = prompts.build_agent_system(
                kb, ToolConfig(tools=tools), version=v)
        for name in ("FORCED_ANSWER", "EMPTY_TURN_NUDGE", "SCHEMA_REMINDER"):
            filled[f"{name} {v}"] = prompts.nudge(name, v)
    # The signature is an escaped object opening, `{{"`, not any `}}`: nested
    # JSON legitimately ends `{"args": {...}}`, and SCHEMA_REMINDER does.
    for name, text in filled.items():
        assert '{{"' not in text, f"{name} escapes a JSON object opening"
        assert "{{}}" not in text, f"{name} has an escaped empty object"
        # And no placeholder may survive a fill.
        leftover = re.findall(r"\{[A-Z_]{3,}\}", text)
        assert not leftover, f"{name}: unfilled placeholders {leftover}"


def test_huge_relation_vocab_is_described_not_listed():
    """A relation vocabulary larger than `MAX_VOCAB_IN_PROMPT` is described in
    the tool docs, not listed; `tool_docs` applies the cap itself."""
    from graph_reasoner.agent.tools import MAX_VOCAB_IN_PROMPT, ToolConfig, tool_docs

    class _Fake:
        name = "fake"

        def __init__(self, n):
            self._v = tuple(f"relation number {i}" for i in range(n))

        def relation_vocab(self):
            return self._v

    small = tool_docs(_Fake(MAX_VOCAB_IN_PROMPT), ToolConfig())
    assert "relations in this graph:" in small
    assert "relation number 0" in small

    big = tool_docs(_Fake(MAX_VOCAB_IN_PROMPT + 1), ToolConfig())
    assert "relations in this graph:" not in big
    assert "relation number 0" not in big
    assert "too many to list" in big
    # describing the larger vocabulary takes less text than listing the smaller
    assert len(big) < len(small)


# ------------------------------------------------- reproducibility of observations

def test_hub_relation_summary_is_totally_ordered(tmp_path):
    """Equal-count relations tie-break on name in the hub summary, so the
    observation text does not depend on set-iteration order, which varies with
    PYTHONHASHSEED. The graph is written in neither name nor count order, so
    only the rule can produce the order asserted."""
    lines = ["Hub|zebra|z1", "Hub|middle|m1", "Hub|alpha|a1", "Hub|middle|m2",
             "Hub|beta|b1"]
    p = tmp_path / "kb.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    r = dispatch(TextKBBackend(str(p)), ToolConfig(hub_threshold=3, page_size=2),
                 "expand", {"node": "Hub"}, rng=random.Random(0))
    summary = r.text.split("Showing")[0]
    order = ["middle", "alpha", "beta", "zebra"]
    assert sorted(order, key=summary.index) == order, summary


_CONNECT_SNIPPET = """
import json, random, sys
from graph_reasoner.substrates.triples import TextKBBackend
from graph_reasoner.agent.tools import ToolConfig, dispatch
r = dispatch(TextKBBackend(sys.argv[1]), ToolConfig(), "connect",
             {"a": "Alpha", "b": "Beta"}, rng=random.Random(0))
print(json.dumps({"text": r.text, "edges": [list(e.key()) for e in r.edges]}))
"""


def test_connect_returns_the_same_paths_in_every_process(tmp_path):
    """`find_paths` walks a neighbour set and stops at `connect_limit`, so
    which paths come back would follow set order, which varies with
    PYTHONHASHSEED, unless the walk is ordered. Two processes with different
    hash seeds must return the same text and the same edges; a repeat inside
    one process shares its hash seed and could not tell."""
    lines = [f"Alpha|links|Mid{i}" for i in range(10)] + [f"Mid{i}|links|Beta" for i in range(10)]
    p = tmp_path / "fan.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    outs = []
    for hashseed in ("1", "424242"):
        proc = subprocess.run(
            [sys.executable, "-c", _CONNECT_SNIPPET, str(p)],
            capture_output=True, text=True, timeout=60,
            env={"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin"})
        assert proc.returncode == 0, proc.stderr
        outs.append(json.loads(proc.stdout))
    assert outs[0]["edges"], "connect returned no path; the fixture is broken"
    assert outs[0] == outs[1], "connect's paths depend on the hash seed"


def test_the_triples_loader_keeps_the_first_relation_and_every_label(tmp_path):
    """`kg/triples.py` is frozen: the DiGraph keeps the first-seen relation of
    a parallel pair, which the fixtures' expected observations rest on, and
    the relation index keeps every label in file order."""
    import networkx as nx

    from graph_reasoner.kg.triples import load_triples_kb, load_triples_relations

    p = tmp_path / "kb.txt"
    p.write_text("Film A|written_by|Jane Doe\nFilm A|directed_by|Jane Doe\n", encoding="utf-8")
    G = load_triples_kb(str(p))
    assert type(G) is nx.DiGraph
    assert G.number_of_edges() == 1
    assert G.edges["Film A", "Jane Doe"]["relation"] == "written_by"
    assert load_triples_relations(str(p)).relations("Film A", "Jane Doe") == ("written_by", "directed_by")


def test_find_ranks_case_siblings_by_raw_name():
    """Case-collision siblings with equal match ratios rank by raw name,
    capitals first: `normalize` makes `Dry Etching` and `dry etching` compare
    equal, so the raw name is the final sort key and degree does not break the
    tie. Every served policy was trained under this order."""
    from graph_reasoner.agent.backend import EntityIndex

    deg = {"dry etching": 126, "Dry Etching": 66}
    idx = EntityIndex(deg, rank=lambda n: deg.get(n, 0))
    assert idx.search("dry etch")[:2] == ["Dry Etching", "dry etching"]


def test_every_producer_renders_an_edge_the_same_way():
    """Every producer of an edge line renders it identically; the producers
    are compared to each other, not each to a string literal. The policy cites
    edges from these lines and the gate checks citations by set membership."""
    from graph_reasoner.agent.backend import SeenEdge
    from graph_reasoner.agent.tools import render_edge
    from graph_reasoner.kg.serialize import render_edge_line, serialize_subgraph
    from graph_reasoner.select.prompts import edges_of

    import networkx as nx

    core = render_edge_line("A", "rel", "B")
    edge_obj = SeenEdge("A", "rel", "B")
    edge_dct = [{"source": "A", "relation": "rel", "target": "B"}]

    # the tool layer, which renders what the policy reads
    assert render_edge(edge_obj) == core

    # the static arms' subgraph_text
    g = nx.DiGraph()
    g.add_edge("A", "B", relation="rel")
    text, _ = serialize_subgraph(g, rng=random.Random(0))
    assert text == core

    # the verifier prompt: same core, four-space indent
    assert edges_of(edge_dct) == "    " + core


def test_tool_allowlist_hides_and_refuses(kb):
    """A withheld tool is neither advertised nor callable, and the find hint
    that points at expand disappears when expand is withheld."""
    cfg = ToolConfig(tools=("find", "expand"))
    assert available_tools(cfg) == ("find", "expand")
    assert available_tools(ToolConfig()) == ("find", "expand", "connect")
    docs = tool_docs(kb, cfg, relation_vocab=False)
    assert "expand(" in docs and "connect(" not in docs
    r = dispatch(kb, cfg, "connect", {"a": "Film A", "b": "Jane Doe"})
    assert not r.ok and "unknown tool" in r.error and "connect" not in r.error.split("Available:")[1]
    r = dispatch(kb, ToolConfig(tools=("find", "connect")), "find", {"name": "Film A"})
    assert r.ok and "use expand()" not in r.text
    r = dispatch(kb, ToolConfig(), "find", {"name": "Film A"})
    assert r.ok and "use expand()" in r.text
    with pytest.raises(ValueError):
        available_tools(ToolConfig(tools=("find", "teleport")))



def test_hub_relation_summary_is_capped_but_the_ranking_is_not(tmp_path):
    """`hub_summary_limit` cuts the per-relation summary, never the edge page,
    and a summary under the limit lists every relation with no trailer."""
    rels = ["contains", "affects", "used in", "found in", "part of", "binds"]
    lines = []
    for i, r in enumerate(rels):
        for j in range(len(rels) - i):          # contains: 6 edges ... binds: 1
            lines.append(f"Hub|{r}|node {r} {j}")
    p = tmp_path / "kb.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    kb = TextKBBackend(str(p))

    capped = ToolConfig(hub_threshold=3, page_size=2, hub_summary_limit=4)
    r = dispatch(kb, capped, "expand", {"node": "Hub"}, rng=random.Random(0))
    assert r.ok and r.truncated and len(r.edges) == 2
    for kept in ("contains: 6", "affects: 5", "used in: 4", "found in: 3"):
        assert kept in r.text
    assert "part of" not in r.text.split("Showing")[0]   # cut from the summary
    assert "(2 more relations not listed)" in r.text
    assert "19 not shown" in r.text                        # the edge count is untouched

    # Under the limit every relation is listed with no trailer, so the default
    # limit and a huge one render the same text.
    plain = ToolConfig(hub_threshold=3, page_size=2)
    r0 = dispatch(kb, plain, "expand", {"node": "Hub"}, rng=random.Random(0))
    huge = ToolConfig(hub_threshold=3, page_size=2, hub_summary_limit=10**9)
    r1 = dispatch(kb, huge, "expand", {"node": "Hub"}, rng=random.Random(0))
    assert r0.text == r1.text
    assert "more relations not listed" not in r0.text
    assert all(f"{x}:" in r0.text for x in rels)


def test_int_arguments_are_coerced_by_the_spec(kb):
    """`ToolSpec.int_args` drives the coercion, and the error text the model
    reads on a bad value is the one every served policy was trained on."""
    r = dispatch(kb, ToolConfig(), "expand", {"node": "Film A", "offset": "abc"})
    assert r.error == 'expand "offset" must be an integer.'
    r = dispatch(kb, ToolConfig(), "connect", {"a": "Film A", "b": "Jane Doe", "max_hops": "x"})
    assert r.error == 'connect "max_hops" must be an integer.'
    assert dispatch(kb, ToolConfig(page_size=2), "expand",
                    {"node": "Film A", "offset": "2"}).ok


def test_rendered_lines_parse_back():
    """The edge line and the observation header are each written by one
    renderer and read back by one parser (the app's attribution, the detour
    builder, a resumed episode); the pair must round-trip."""
    from graph_reasoner.agent.prompts import parse_observation_header, render_error, render_observation
    from graph_reasoner.agent.tools import ToolResult
    from graph_reasoner.kg.serialize import render_edge_line

    assert parse_edge_line(render_edge_line("A b", "rel x", "C")) == ("A b", "rel x", "C")
    assert parse_edge_line("  rel: 3") is None
    obs = render_observation(ToolResult(True, render_edge_line("A", "r", "B")), call_no=3,
                             max_turns=20, tool="expand", n_new=1, n_seen=4)
    assert parse_observation_header(obs) == (3, 20, "expand")
    assert parse_observation_header(render_error("bad", call_no=4, max_turns=20)) == (4, 20, "error")
    assert parse_observation_header("You have no tool calls left.") is None
