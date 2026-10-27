"""Import, name-resolution and layering checks over the whole tree.

Every module imports, every registered backend constructs and every tracked
benchmark name resolves. Also checked: a script loaded by path exists, a
function's global names are bound, a substrate does not import the eval layer,
shared training defaults come from one table, the package does not import
`launch/`, train's shared modules import no heavy dependency, and every
declared config key is read.
"""
import importlib
import pathlib
import pkgutil
import re
import subprocess
import sys

import pytest

import graph_reasoner
from conftest import ROOT
from graph_reasoner import benchmarks

# The tracked benchmarks: each graph's evaluation questions and the training
# questions. A stored run is named `<tag>--<benchmark>` and can be rescored
# only while that name resolves. The baselines' files are built, not tracked.
SHIPPED_BENCHMARKS = [
    "tsmc_core_agentic", "biograph_core_agentic", "pfas_core_agentic",
    "matprop_core_agentic", "tsmc_train_agentic_4x",
]


def test_every_module_in_the_package_imports():
    """Catches a bad relative-import depth or a renamed module, which
    otherwise shows only when the code path that touches it runs."""
    failed = []
    swept = 0
    for mod in pkgutil.walk_packages(graph_reasoner.__path__,
                                     prefix="graph_reasoner."):
        swept += 1
        try:
            importlib.import_module(mod.name)
        except Exception as exc:                      # noqa: BLE001
            failed.append(f"{mod.name}: {type(exc).__name__}: {exc}")
    # A walk that yields nothing passes every assertion below it.
    assert swept > 20, f"the package sweep saw only {swept} modules"
    assert not failed, "modules that do not import:\n  " + "\n  ".join(failed)


def test_every_registered_backend_can_be_constructed_or_fails_honestly():
    """A registered backend constructs, or fails only because the file is
    missing; a broken one would otherwise fail mid-eval."""
    from graph_reasoner.agent.backend import available_backends, load_backend
    assert available_backends(), "no backends registered — nothing was checked"
    for name in available_backends():
        try:
            load_backend(name, "/nonexistent/path/for/this/test")
        except Exception as exc:                      # noqa: BLE001
            # A missing file is the expected failure. A missing module or a
            # NameError means the backend is broken, not the path.
            assert not isinstance(exc, (ModuleNotFoundError, ImportError, NameError)), \
                f"backend {name!r} is registered but broken: {type(exc).__name__}: {exc}"


@pytest.mark.parametrize("name", SHIPPED_BENCHMARKS)
def test_shipped_benchmark_names_resolve(name):
    """Each name resolves to its own graph's directory. Resolution searches
    the substrate directories in order, so a name reused on two graphs would
    resolve to the first."""
    path = benchmarks.resolve(name)
    assert path.exists() and path.parent.name == name.split("_")[0], path


def test_an_unknown_benchmark_raises_and_says_where_it_looked():
    """An unknown name fails at the lookup, listing the paths searched,
    instead of returning a path that does not exist."""
    with pytest.raises(FileNotFoundError) as e:
        benchmarks.resolve("no_such_benchmark_xyz")
    assert "looked in" in str(e.value)


def test_every_dynamically_loaded_script_exists():
    """Every script loaded by path exists.

    `importlib.util.spec_from_file_location` is a reference the import graph
    cannot see. The check is by basename, because the path expression cannot
    be evaluated statically; the targets must exist in src/, scripts/ or app/.
    """
    targets = [ROOT / "src", ROOT / "scripts", ROOT / "app"]
    present = {p.name for r in targets for p in r.rglob("*.py")}
    missing = []
    resolved = 0
    for root in [*targets, ROOT / "tests"]:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            # A fixed window, not a `\(...\)` match: in the usual argument,
            # `Path(__file__).parent / "x.py"`, the first `)` precedes the filename.
            for m in re.finditer(r"spec_from_file_location\(", text):
                window = text[m.end():m.end() + 300]
                for target in re.findall(r'"([\w.-]+\.py)"', window):
                    resolved += 1
                    if target not in present:
                        missing.append(f"{path}: loads {target!r}, which no "
                                       f"file in src/ scripts/ app/ provides")
    # A scan that matches nothing proves nothing: the window regex must still
    # find the path-loads the tests make.
    assert resolved > 0, "no spec_from_file_location target was extracted"
    assert not missing, ("a script is loaded by path but does not exist:\n  "
                         + "\n  ".join(missing))


def test_every_function_body_resolves_its_global_names():
    """No function references a global name that its module never binds.
    Such a module imports cleanly and raises `NameError` only when the
    function runs, so the import sweep cannot see it."""
    import builtins
    import symtable

    # Bound by the interpreter, so never module-level symbols.
    dunders = {"__file__", "__name__", "__doc__", "__package__", "__spec__",
               "__loader__", "__builtins__", "__debug__"}
    offenders, scanned = [], 0
    roots = [ROOT / "src", ROOT / "scripts", ROOT / "app"]
    for path in sorted(p for r in roots for p in r.rglob("*.py")):
        scanned += 1
        top = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
        module_names = {s.get_name() for s in top.get_symbols()} | dunders

        def walk(table, _path=path):
            for child in table.get_children():
                if child.get_type() == "function":
                    for sym in child.get_symbols():
                        name = sym.get_name()
                        if (sym.is_global() and sym.is_referenced()
                                and name not in module_names
                                and not hasattr(builtins, name)):
                            offenders.append(
                                f"{_path.relative_to(ROOT)}: {child.get_name()}() "
                                f"references {name!r}, bound nowhere at module scope")
                walk(child, _path)

        walk(top)
    assert scanned > 80, f"scanned {scanned} files under {[str(r) for r in roots]}"
    assert not offenders, ("a function references a global that does not exist — "
                           "it raises NameError on its first call:\n  "
                           + "\n  ".join(offenders))


def test_importing_a_substrate_does_not_pull_in_eval():
    """Importing a substrate does not import the eval layer: a graph backend
    must not need the code that scores it. The probe runs in a fresh
    interpreter, because an earlier test may already have imported `eval`."""
    probe = (
        "import sys, graph_reasoner.substrates.tsmc.backend as _;"
        "bad = sorted(m for m in sys.modules if m.startswith('graph_reasoner.eval'));"
        "print('|'.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    leaked = [m for m in out.stdout.strip().split("|") if m]
    assert not leaked, ("importing a substrate loaded the eval layer: "
                        + ", ".join(leaked))


def test_no_shared_default_is_written_as_a_bare_literal():
    """A `config.get(k, D)` in the training modules whose key is in `SHARED`
    draws its default from `SHARED`, so a shared default is written once.
    The run settings the stage files set explicitly (lr, grad_accum,
    max_seq_len) are not in `SHARED`."""
    import ast

    from graph_reasoner.train.defaults import SHARED

    files = sorted((ROOT / "src/graph_reasoner/train").rglob("*.py"))
    assert len(files) >= 6, f"only {len(files)} training modules scanned"

    offenders = []
    for path in files:
        if path.name == "defaults.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and len(node.args) == 2
                    and isinstance(node.args[0], ast.Constant)):
                continue
            key = node.args[0].value
            if key not in SHARED:
                continue
            # The default must read from the table, not restate its value.
            if "SHARED" not in ast.unparse(node.args[1]):
                offenders.append(
                    f"{path.relative_to(ROOT)}:{node.lineno} "
                    f"{key}={ast.unparse(node.args[1])} "
                    f"(use SHARED[{key!r}])")
    assert not offenders, (
        "shared defaults restated as literals:\n  " + "\n  ".join(offenders))


def test_nothing_in_the_package_imports_launch():
    """The package never imports `launch/`, which holds the GPU machine's
    setup (setup.sh and the pins). Only imports are checked; a string that
    names a file there is fine.
    """
    import ast

    files = sorted((ROOT / "src/graph_reasoner").rglob("*.py"))
    assert len(files) > 25, f"only {len(files)} package modules scanned"

    offenders = []
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                if n == "launch" or n.startswith("launch."):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} imports {n}")
    assert not offenders, "package imports launch/:\n  " + "\n  ".join(offenders)


def test_train_shared_modules_stay_cheap_to_import():
    """Importing `graph_reasoner.train`'s shared modules, every module of the
    package but the two trainers and the vLLM checkpoint loader, loads no
    torch, trl, peft, transformers, openai, httpx or vllm, so the objective's
    tests run CPU-only in seconds. The list is enumerated from the package,
    so a module added to it is covered the day it is added."""
    import graph_reasoner.train as train
    heavy = {"sft", "grpo", "vllm_ckpt"}
    shared = sorted(m.name for m in pkgutil.iter_modules(train.__path__) if m.name not in heavy)
    assert {"config", "defaults", "generate", "prefixes", "reward", "rollout", "transcripts"} <= set(shared)
    probe = (
        "import sys; sys.path.insert(0, 'src');"
        "import graph_reasoner.train, " + ", ".join(f"graph_reasoner.train.{m}" for m in shared) + ","
        " graph_reasoner.recipe, graph_reasoner.recipe.cli;"
        "bad = sorted({m.split('.')[0] for m in sys.modules} &"
        " {'torch','trl','peft','transformers','openai','httpx','vllm'});"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", \
        f"train's shared modules pulled heavy deps at import: {out.stdout.strip()}"


def test_no_code_outside_the_substrates_names_a_graph():
    """No string constant in `src/` (outside `substrates/`) or in `scripts/`
    names a registered graph, outside docstrings: a graph is a parameter
    (`--graph`), declared by its substrate. `recipe/names.py` is the one exception,
    for the two tokens the released names spell as nothing (the default graph
    and the transfer graph's token)."""
    import ast

    from graph_reasoner.agent.backend import available_backends

    names = re.compile(r"\b(" + "|".join(map(re.escape, available_backends())) + r")\b")
    allowed = {ROOT / "src/graph_reasoner/recipe/names.py"}
    offenders, scanned = [], 0
    for root in (ROOT / "src/graph_reasoner", ROOT / "scripts"):
        for path in sorted(root.rglob("*.py")):
            if "substrates" in path.parts or path in allowed:
                continue
            scanned += 1
            tree = ast.parse(path.read_text(encoding="utf-8"))
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    first = node.body[0] if node.body else None
                    if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                            and isinstance(first.value.value, str)):
                        docstrings.add(id(first.value))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                        and id(node) not in docstrings and names.search(node.value)):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.value[:80]!r}")
    assert scanned > 60, f"scanned {scanned} files"
    assert not offenders, ("code names a graph; take it from --graph or the substrate:\n  "
                           + "\n  ".join(offenders))


def test_every_declared_config_key_is_actually_read():
    """Every key a trainer's allowlist declares is read somewhere; a declared,
    unread key would validate and do nothing. Keys read in `train/rollout.py`
    count for both trainers, including its `EPISODE_KEYS` and `TOOL_KEYS`."""
    import ast

    def keys_read(rel: str) -> set[str]:
        out: set[str] = set()
        for node in ast.walk(ast.parse((ROOT / rel).read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in ("config", "cfg")
                    and node.args and isinstance(node.args[0], ast.Constant)):
                out.add(node.args[0].value)
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                    and node.value.id in ("config", "cfg")
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)):
                out.add(node.slice.value)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "_require" and len(node.args) > 1
                    and isinstance(node.args[1], ast.Constant)):
                out.add(node.args[1].value)
        return out

    from graph_reasoner.train.rollout import EPISODE_KEYS, TOOL_KEYS

    shared = keys_read("src/graph_reasoner/train/rollout.py") | set(EPISODE_KEYS) | set(TOOL_KEYS)
    from graph_reasoner.train.grpo import KNOWN_CONFIG_KEYS as GRPO_KEYS
    from graph_reasoner.train.sft import SFT_CONFIG_KEYS

    cases = [
        ("train/grpo.py", GRPO_KEYS, "KNOWN_CONFIG_KEYS"),
        ("train/sft.py", SFT_CONFIG_KEYS, "SFT_CONFIG_KEYS"),
    ]
    for rel, declared, name in cases:
        read = keys_read(f"src/graph_reasoner/{rel}") | shared
        unread = sorted(set(declared) - read)
        assert not unread, (
            f"{rel}: {name} declares {unread}, which nothing reads — a config "
            f"setting one would validate clean and do nothing")


def test_the_retired_metric_names_are_gone():
    """The headline is `gated_acc`; its old name and the two retired fields,
    a gated accuracy under the loose rule and the loose rule itself, name
    nothing in the code, the tests, the configurations or the documents."""
    retired = ("gated_" + "v2", "gated_" + "success", "em_" + "loose", "false_" + "abstention")
    roots = [ROOT / d for d in ("src", "scripts", "app", "tests", "configs", "launch")]
    files = [p for r in roots for p in r.rglob("*") if p.is_file() and p.suffix in (".py", ".js", ".md", ".yaml", ".html", ".css")]
    files += [ROOT / "README.md"]
    me = pathlib.Path(__file__).resolve()
    hits = [(p.relative_to(ROOT).as_posix(), name) for p in files if p.resolve() != me
            for name in retired if name in p.read_text(encoding="utf-8", errors="replace")]
    assert not hits, hits
