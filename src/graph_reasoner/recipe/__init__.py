"""The recipe: its names (`names`), derived from (graph, size, tag), and the
runner that trains any size on any registered graph. The names module's
public names are re-exported here, so `from graph_reasoner.recipe import Layout`
is the import; the runner's live in `plan`, `run`, `teach`, `verify` and `latent`.
"""
from .names import (BAND_KEY, BAND_SEEDS, BASE_MODELS, DEFAULT_GRAPH, DEFAULT_TAG, G2_ROLL_SEEDS,  # noqa: F401
                    GATE_SEEDS, MANIFEST, MANIFEST_SCHEMA, POOL_SEEDS, PREFIX_BAND_SEED,
                    TRAINING, TRANSFER_TOKENS, Graph, Layout, Spec, Training, add_spec_arguments,
                    load_hyperparameters, read_manifest, sha256_of, smoke_config, spec_from_args,
                    training_config, write_manifest)

__all__ = ["BAND_KEY", "BAND_SEEDS", "BASE_MODELS", "DEFAULT_GRAPH", "DEFAULT_TAG", "G2_ROLL_SEEDS",
           "GATE_SEEDS", "MANIFEST", "MANIFEST_SCHEMA", "POOL_SEEDS", "PREFIX_BAND_SEED", "TRAINING",
           "TRANSFER_TOKENS", "Graph", "Layout", "Spec", "Training", "add_spec_arguments",
           "load_hyperparameters", "read_manifest", "sha256_of", "smoke_config", "spec_from_args",
           "training_config", "write_manifest"]
