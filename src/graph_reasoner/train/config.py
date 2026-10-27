"""Config checks shared by the SFT and GRPO trainers.

Each trainer owns the set of keys it reads, and `check_config_keys` refuses
any other. No torch, trl or peft is imported here: `refuse_broken_cuda` takes
the torch module as an argument.
"""
from __future__ import annotations


def require(config: dict, key: str) -> str:
    """The config's value for `key` as a string; SystemExit when it is unset.

    For `backend` and `kb`, which have no default: a default graph would
    train on the wrong graph without failing.
    """
    # Function-scoped, so importing this module does not load the backend
    # registry.
    from ..agent.backend import available_backends

    value = config.get(key)
    if not value:
        raise SystemExit(
            f"config must set `{key}` — there is no default substrate; "
            f"registered: {', '.join(available_backends())}")
    return str(value)


def check_config_keys(config: dict, known: set[str] | frozenset[str], *,
                      known_name: str = "KNOWN_CONFIG_KEYS") -> None:
    """Refuse a config with a key this trainer does not read.

    Keys are read with `config.get`, so an unrecognised one would be ignored
    and the run would train something other than what its config says.
    `known_name` is the constant the message tells the reader to edit.
    """
    strays = sorted(set(config) - set(known))
    if strays:
        raise SystemExit(
            f"unknown config key(s): {', '.join(strays)} — the trainer reads "
            f"none of these, so the config describes a mechanism that will not "
            f"run. Fix the key or add it to {known_name} together with the "
            f"code that reads it.")


def check_hyperparameter_keys(config: dict, *, known: set[str] | frozenset[str],
                              resolved: set[str] | frozenset[str], known_name: str,
                              path: str = "the config") -> None:
    """Refuse a hyperparameter file that carries a key the recipe derives
    (`resolved`: the paths, the base model, the band's rollouts) or one the
    trainer does not read. A stage's file holds hyperparameters only; the
    recipe adds the rest from --graph, --size, --tag and --stage and records
    it in the run's manifest.
    """
    derived = sorted(set(config) & set(resolved))
    for band in ("informative_band", "prefix_band"):
        if isinstance(config.get(band), dict) and "rollouts" in config[band]:
            derived.append(f"{band}.rollouts")
    if derived:
        raise SystemExit(
            f"{path} carries {', '.join(derived)}, which the recipe derives from "
            f"--graph/--size/--tag/--stage and records in the run's run.json; a "
            f"hyperparameter file does not name it.")
    check_config_keys(config, set(known) - set(resolved), known_name=known_name)


def refuse_broken_cuda(torch) -> None:
    """Refuse to train when a GPU was allocated but torch cannot see it.

    CUDA_VISIBLE_DEVICES set while `cuda.is_available()` is False means CUDA
    failed to initialise, and torch would train on CPU without a word. A
    CPU-only machine leaves the variable unset and passes. Both trainers
    call this before loading a model.
    """
    import os

    if os.environ.get("CUDA_VISIBLE_DEVICES") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA_VISIBLE_DEVICES is set but torch.cuda.is_available() is "
            "False: CUDA failed to initialise on this machine; refusing to "
            "train on CPU.")
