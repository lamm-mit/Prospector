"""One folder per knowledge graph. Adding a KG is adding a folder.

Each subpackage implements `agent.backend.GraphBackend` and calls
`register_backend` at import time. Importing this package imports every
subpackage whose name does not start with an underscore, so a new folder (with
its `__init__.py`) needs no edit here. Outside its folder a graph adds only
data and configuration: the graph file under `data/raw/`, a question config
under `configs/benchmark/`, and its questions under `data/benchmarks/<name>/`.

A folder needs a backend module and, if `bench/qa_gen.py` will write its
questions, a `qa_style.py` holding a `TEACHER_SYSTEM` string; the generator
has no default persona.
"""
import importlib
import pkgutil

# Imported for the side effect: each subpackage registers its backend. The
# underscored modules are shared code, not substrates.
__all__ = sorted(m.name for m in pkgutil.iter_modules(__path__)
                 if m.ispkg and not m.name.startswith("_"))
for _name in __all__:
    importlib.import_module(f"{__name__}.{_name}")
