"""Substrate-agnostic graph machinery: load a KG, sample gold paths and their
noisy subgraphs, search paths, and normalize an edge the one canonical way.

A concrete KG is a `substrates/<name>/` package; nothing here imports one, or
`agent/`.
"""
from .load import classify_nodes, file_fingerprint, load_kg
from .sample import GoldPath, PathEdge, build_subgraph, sample_gold_path, undirected_neighbors
from .serialize import edge_key, normalize, parse_edge_line, render_edge_line, serialize_subgraph

__all__ = [
    "load_kg", "classify_nodes", "file_fingerprint",
    "GoldPath", "PathEdge", "sample_gold_path", "build_subgraph", "undirected_neighbors",
    "edge_key", "normalize", "render_edge_line", "parse_edge_line", "serialize_subgraph",
]
