"""What made an artifact: the provenance record written beside it.

A script that generates an artifact builds the record with `block` (the tool,
its arguments, the git state, and the source runs with the models that
produced them) and writes it with `write_sidecar` to `<out>.provenance.json`.
A run's summary records `git_state` and `env_state`.

No timestamp is recorded by default, so two rebuilds from the same inputs
give the same record. Pass `extra={"built_at": ...}` if an artifact needs one.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

PROVENANCE_SCHEMA = "provenance_v1"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(("git", *args), capture_output=True, text=True,
                             timeout=5, cwd=Path(__file__).resolve().parents[2])
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_state() -> dict:
    """The commit an artifact was built at, and whether the tree was dirty.

    `dirty: true` means the code that ran is not the code at that SHA, so the
    artifact is not reproducible from the commit alone. A dirty build is
    recorded, not refused. Files under `data/` are left out: they are what a
    build reads and writes (a sidecar it just wrote among them), not code.
    """
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--", ".", ":(exclude)data")
    return {
        "sha": sha,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


#: The stack an inference or training run ran under, by distribution name.
_ENV_PACKAGES = ("graph-reasoner", "torch", "transformers", "trl", "peft", "vllm",
                 "mlx-lm", "flash-linear-attention", "causal-conv1d")


def env_state() -> dict:
    """The interpreter and the installed version of each package an eval or a
    trainer can run on, None where it is absent. Recorded beside `git_state`
    in a run's summary."""
    import platform
    from importlib import metadata

    out: dict = {"python": platform.python_version()}
    for name in _ENV_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def describe_run(run_dir: str | Path) -> dict:
    """What a run directory says about itself, read from its own summary.json.

    Nothing is inferred from the directory's name, which can be stale.
    """
    d = Path(run_dir)
    info: dict = {"run": d.name, "path": str(d)}
    try:
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    except Exception:
        info["summary"] = "unreadable"
        return info
    for key in ("model", "adapter", "backend",
                "chat_template", "kg_fingerprint", "benchmark"):
        if s.get(key) not in (None, ""):
            info[key] = s[key]
    policy = s.get("policy") or {}
    for key in ("max_turns", "relation_vocab", "temperature", "n"):
        if policy.get(key) is not None:
            info[key] = policy[key]
    # The keys that decide whether two runs' timings are comparable: the driver
    # behind `wall_ms`, the requests in flight, and the disk-cache hits.
    timing = s.get("timing") or {}
    for key in ("driver", "concurrency", "cache_hits"):
        if timing.get(key) is not None:
            info[key] = timing[key]
    return info


def teacher_label(runs: list[dict]) -> str:
    """One short string naming the model that produced these runs: its name,
    `mixed:<a>,<b>` for several, `unknown` for none. It is what belongs in a
    row's `source` field."""
    models = sorted({str(r.get("model", "")) for r in runs if r.get("model")})
    if not models:
        return "unknown"
    if len(models) == 1:
        return models[0]
    return "mixed:" + ",".join(models)


def block(tool: str, argv: list[str] | None = None, *,
          source_runs: list[str | Path] | None = None,
          extra: dict | None = None) -> dict:
    """The provenance record to attach to a generated artifact."""
    runs = [describe_run(r) for r in (source_runs or [])]
    prov = {
        "schema": PROVENANCE_SCHEMA,
        "tool": tool,
        "git": git_state(),
    }
    if argv is not None:
        prov["argv"] = list(argv)
    if runs:
        prov["source_runs"] = runs
        prov["teacher"] = teacher_label(runs)
    if extra:
        prov.update(extra)
    return prov


def sidecar_path(out_path: str | Path) -> Path:
    return Path(str(out_path) + ".provenance.json")


def write_sidecar(out_path: str | Path, prov: dict) -> Path:
    """Write `<out>.provenance.json` beside the artifact and return its path.

    The sidecar is a separate file so that it can be committed where the
    artifact is not, as under `data/traj/`.
    """
    p = sidecar_path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    return p
