"""The agent layer: the rollout loop, the three tools and the gate.

Exports the graph-access protocol and the backend registry, never a concrete
graph. Resolve a graph by name with `load_backend`.
"""
from .backend import GraphBackend, SeenEdge, available_backends, load_backend

__all__ = ["GraphBackend", "SeenEdge", "available_backends", "load_backend"]
