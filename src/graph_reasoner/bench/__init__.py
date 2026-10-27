"""Benchmark construction: gold paths -> subgraphs -> questions.

Produces `data/benchmarks/<substrate>/`; the CLIs are `scripts/bench/`.

Not to be confused with `traj/`, which goes the other way (rollouts ->
training rows) and shares no code with this package.
"""
from .qa_gen import build_benchmark, offline_qa

__all__ = ["build_benchmark", "offline_qa"]
