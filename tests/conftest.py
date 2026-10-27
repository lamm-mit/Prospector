"""Shared test helpers and fixtures.

`script()` finds a CLI under scripts/ by basename, so a test that loads a
script by path does not hardcode its folder.

The film graph is the toy the loop, metric, resume and live-driver tests drive:
five edges, one question whose intermediate (Jane Doe) and answer (Film B)
must be discovered, and one real edge no scripted episode fetches
(`Secret Film`), so citing it is a hallucination. `tool_call` and
`final_answer` write the text protocol a policy emits.

`shipped_tokenizer` is the Qwen3.5 tokenizer under the `plain_chatml` template
every training config installs; the mask and sampling tests run against it.
"""
import functools
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def script(name: str) -> pathlib.Path:
    """The one file under scripts/ with this basename."""
    hits = sorted(ROOT.glob(f"scripts/**/{name}"))
    if not hits:
        raise FileNotFoundError(f"{name} not found under scripts/")
    if len(hits) > 1:
        raise AssertionError(f"{name} is ambiguous: {[str(h) for h in hits]}")
    return hits[0]


def all_scripts() -> list[pathlib.Path]:
    return sorted(ROOT.glob("scripts/**/*.py"))


# ------------------------------------------------------------ the film graph

FILM_EDGES = [
    "Film A|written_by|Jane Doe",
    "Film A|directed_by|Jane Doe",
    "Film B|written_by|Jane Doe",
    "Film B|has_genre|Drama",
    "Secret Film|directed_by|Hidden Person",   # a real edge, never fetched
]

#: The two edges the gold path runs through, as a policy cites them.
FILM_GOLD_EDGES = [{"source": "Film A", "relation": "written_by", "target": "Jane Doe"},
                   {"source": "Film B", "relation": "written_by", "target": "Jane Doe"}]


@pytest.fixture
def film_kb(tmp_path):
    from graph_reasoner.substrates.triples import TextKBBackend

    p = tmp_path / "kb.txt"
    p.write_text("\n".join(FILM_EDGES) + "\n", encoding="utf-8")
    return TextKBBackend(str(p))


@pytest.fixture
def film_item():
    """The question names only the anchor; the intermediate and the answer
    must be discovered."""
    from graph_reasoner.schema import BenchmarkItem

    return BenchmarkItem(
        id="medium-0000", tier="medium",
        question="which movies share a writer with Film A",
        gold_answer="Film B",
        gold_path={"nodes": ["Film A", "Jane Doe", "Film B"], "edges": FILM_GOLD_EDGES},
        subgraph_edges=FILM_GOLD_EDGES[:1],
        subgraph_text="Film A --[written_by]--> Jane Doe",
        meta={"anchor": "Film A", "kb_answer_set": ["Film B"], "answer_type": "movie"},
    )


def tool_call(tool: str, args: dict) -> str:
    return f'<tool_call>{json.dumps({"tool": tool, "args": args})}</tool_call>'


def final_answer(text: str, edges: list[dict]) -> str:
    return f'<graph_json>{json.dumps({"nodes": [], "edges": edges})}</graph_json>\nANSWER: {text}'


def run_turns(kb, item, turns: list[str], cfg=None):
    """One episode driven through the rollout loop by a fixed list of
    assistant turns, then finalized: the scripted stand-in for a model."""
    from graph_reasoner.agent.loop import EpisodeConfig, advance, finalize, start_episode

    cfg = cfg or EpisodeConfig()
    ep = start_episode(item, kb, cfg)
    for text in turns:
        if not advance(ep, text, kb, cfg):
            break
    finalize(ep)
    return ep


# ------------------------------------------------------------ stubs and runs

@pytest.fixture
def stub_local_hf(monkeypatch):
    """The hf: seam replaced by one that answers every conversation at once.
    Returns a log of how many engines were built and how wide each batch was."""
    log = {"built": 0, "widths": []}
    reply = final_answer("Jane Doe", [])

    class Stub:
        def __init__(self, *a, **k):
            log["built"] += 1

        def generate_chat(self, conversations, **kw):
            log["widths"].append(len(conversations))
            return [reply] * len(conversations)

    monkeypatch.setattr("graph_reasoner.eval.local_model.LocalHFModel", Stub)
    return log


def write_run(root: pathlib.Path, name: str, results: list[dict],
              trajectories: list[dict]) -> pathlib.Path:
    """A run directory as the eval writes it: results.jsonl beside
    trajectories.jsonl, one JSON row per line."""
    d = root / name
    d.mkdir(parents=True)
    for fname, rows in (("results.jsonl", results), ("trajectories.jsonl", trajectories)):
        (d / fname).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return d


# ------------------------------------------------------------ the tokenizer

SHIPPED_TOKENIZER = "Qwen/Qwen3.5-0.8B"
TOKENIZER_ABSENT = (f"the shipped tokenizer ({SHIPPED_TOKENIZER}) is not cached "
                    "or transformers is not installed")


@functools.cache
def load_shipped_tokenizer():
    """The Qwen3.5 tokenizer with `plain_chatml` installed, from the local
    cache only, or None when transformers is absent (ImportError) or the
    tokenizer is not in the cache (OSError). Any other failure to load, and
    an error in the template override, fails the test instead of skipping."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(SHIPPED_TOKENIZER, local_files_only=True)
    except (ImportError, OSError):
        return None
    from graph_reasoner.train.tokenize import override_chat_template

    return override_chat_template(tok, "plain_chatml", model=SHIPPED_TOKENIZER)


@pytest.fixture(scope="session")
def shipped_tokenizer():
    """Loaded once per session; the test skips when it is unavailable. Callers
    must not mutate it: a test that changes the template loads its own."""
    tok = load_shipped_tokenizer()
    if tok is None:
        pytest.skip(TOKENIZER_ABSENT)
    return tok


def unmasked_text(target_tokens: list[int], weights: list[int], tokenizer) -> str:
    """Decode exactly the tokens that carry loss. An OBSERVATION block in the
    result means the mask is wrong."""
    return tokenizer.decode([t for t, w in zip(target_tokens, weights) if w])


@pytest.fixture(autouse=True)
def _no_api_keys(monkeypatch):
    """Remove the API keys from every test's environment, and so from any
    subprocess it starts, so an unstubbed call cannot spend credit; and
    OPENAI_BASE_URL, so the provider route depends on the model id alone."""
    for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
