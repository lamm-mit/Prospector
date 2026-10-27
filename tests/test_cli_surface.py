"""Every script's command line, and the README's commands against it.

Each script's `--help` exits 0 (which also imports it), every
`python scripts/...` and `prospector <command>` line in the README's code
blocks names a script or a command that exists and only flags it declares,
and every config, benchmark and launch script the README names exists.

The helps run once per session, in parallel, from the repository root. The
commands come from README.md and launch/README.md, the GPU machine's guide.
"""
import functools
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from conftest import ROOT, all_scripts

SCRIPTS = [p.relative_to(ROOT).as_posix() for p in all_scripts()]
README = "\n".join((ROOT / f).read_text(encoding="utf-8")
                   for f in ("README.md", "launch/README.md"))

#: Extras a script may need at import; without them its --help is skipped, not
#: failed, so the fast suite runs on a `.[dev]` install.
OPTIONAL = {"torch", "transformers", "trl", "peft", "vllm", "mlx", "mlx_lm",
            "fastapi", "uvicorn", "igraph", "datasets", "accelerate", "safetensors"}


@functools.cache
def _helps() -> dict[str, tuple[int, str]]:
    def run(rel):
        p = subprocess.run([sys.executable, rel, "--help"], capture_output=True,
                           text=True, cwd=ROOT, timeout=120)
        return rel, (p.returncode, p.stdout + p.stderr)

    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(pool.map(run, SCRIPTS))


def _help_or_skip(rel: str) -> str:
    rc, out = _helps()[rel]
    if rc != 0:
        missing = re.search(r"No module named '([\w.]+)'", out)
        if missing and missing.group(1).split(".")[0] in OPTIONAL:
            pytest.skip(f"{rel} needs {missing.group(1)}, an optional extra")
    assert rc == 0, f"{rel} --help exited {rc}:\n{out[-2000:]}"
    return out


def test_the_scripts_were_found():
    assert len(SCRIPTS) > 20, SCRIPTS


@pytest.mark.parametrize("rel", SCRIPTS)
def test_every_script_answers_help(rel):
    out = _help_or_skip(rel)
    assert out.strip(), f"{rel} --help printed nothing"


def _readme_commands() -> list[tuple[str, list[str]]]:
    """(script or `prospector <command>`, flags) for every such line in a
    code block, with backslash continuations joined and trailing comments
    dropped."""
    out = []
    for block in re.findall(r"```[a-z]*\n(.*?)```", README, flags=re.S):
        for line in re.sub(r"\\\n\s*", " ", block).splitlines():
            line = re.sub(r"\s+#\s.*$", "", line)
            m = re.search(r"(?:^|\s)(?:\S*/)?python3?\s+(scripts/\S+\.py)(.*)", line)
            if m:
                out.append((m.group(1), re.findall(r"(?<!\S)(--[a-z][a-z0-9-]*)", m.group(2))))
            m = re.search(r"(?:^|\s)prospector\s+([a-z]+)(.*)", line)
            if m:
                out.append((f"prospector {m.group(1)}", re.findall(r"(?<!\S)(--[a-z][a-z0-9-]*)", m.group(2))))
    return out


@functools.cache
def _prospector_help(command: str) -> str:
    p = subprocess.run([sys.executable, "-m", "graph_reasoner.recipe", command, "--help"],
                       capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert p.returncode == 0, f"prospector {command} --help exited {p.returncode}:\n{p.stdout + p.stderr}"
    return p.stdout + p.stderr


COMMANDS = _readme_commands()


def test_the_readme_commands_were_found():
    assert len(COMMANDS) > 15, COMMANDS


@pytest.mark.parametrize("script, flags", COMMANDS,
                         ids=[f"{s.split('/')[-1]}-{i}" for i, (s, _) in enumerate(COMMANDS)])
def test_every_readme_command_names_a_script_and_its_flags(script, flags):
    if script.startswith("prospector "):
        out = _prospector_help(script.split()[1])
    else:
        assert script in SCRIPTS, f"README runs {script}, which is not in scripts/"
        out = _help_or_skip(script)
    unknown = [f for f in flags if not re.search(rf"(?<![\w-]){re.escape(f)}(?![\w-])", out)]
    assert not unknown, f"README passes {unknown} to {script}, whose --help does not declare them"


# The files the README names that a clone does not hold: the two baselines',
# which the README's own build commands write just before they are used.
BUILT_BY_THE_README = {"data/benchmarks/tsmc/tsmc_core__closed_book.jsonl",
                       "data/benchmarks/tsmc/tsmc_core__khop400anchor.jsonl"}


@pytest.mark.parametrize("pattern", [
    r"scripts/[\w/.-]+\.py", r"configs/[\w/.-]+\.yaml", r"launch/[\w.-]+\.sh",
    r"data/benchmarks/[\w/.-]+\.jsonl",
])
def test_every_file_the_readme_names_exists(pattern):
    named = sorted(set(re.findall(pattern, README)))
    assert named, f"the README names no {pattern}: the pattern has gone stale"
    missing = [n for n in named if n not in BUILT_BY_THE_README and not (ROOT / n).exists()]
    assert not missing, f"the README names files that do not exist: {missing}"
