"""Reads and moves at one layer -> the steering directions and their dose units.

From the fitting graph's reads at a layer: the follow axis (the follow class
mean minus the backtrack class mean over expand pages), the answer axis (the
answer mean minus the follow-or-backtrack mean), `n_random` random unit
vectors (draw k from seed 1000 k + layer, named random, random2, ...), and
each of the follow and answer axes with the other's component projected out
(follow_perp, answer_perp, with the cosine removed and the gap's share of the
parent's). Every direction is a unit vector in the raw residual space; its
dose unit is the gap between the two class means along it.

`scripts/latent/directions.py` writes directions_L<layer>.npz (one array per
direction) and directions_L<layer>.json (the gaps, class sizes, AUCs and the
dose unit as a share of the median residual norm), the files `steer` reads.
Tests: tests/test_latent.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .moves import label_rows, read_records
from .readings import auroc

__all__ = ["unit", "gap_along", "perp_direction", "build", "per_layer_files"]


def unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def gap_along(d: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """The distance between the two classes' means along `d`."""
    return float(A.mean(axis=0) @ d - B.mean(axis=0) @ d)


def perp_direction(d: np.ndarray, other: np.ndarray) -> np.ndarray:
    """The unit direction `d` with its component along the unit direction
    `other` removed and the result renormalised. The class gap along it is
    the gap along `d` times sqrt(1 - cos^2) of the two directions."""
    v = d - float(d @ other) * other
    return v / max(float(np.linalg.norm(v)), 1e-12)


def build(extract_dir: Path, run_dir: Path, layers: list[int], n_random: int = 1, log=None) -> tuple[dict, dict]:
    """The directions at each layer: arrays keyed `<name>_L<layer>`, and the
    record of their gaps, class sizes and AUCs keyed by layer."""
    extract_dir = Path(extract_dir)
    records = {r["id"]: r for r in read_records(run_dir)}
    rows = [json.loads(line) for line in (extract_dir / "index.jsonl").open(encoding="utf-8") if line.strip()]
    obs = [r for r in rows if r.get("kind") == "observation"]
    by_id: dict[str, list[dict]] = {}
    for r in obs:
        by_id.setdefault(r["id"], []).append(r)
    for iid, rs in by_id.items():
        label_rows(records[iid], rs)
    vec: dict[str, np.ndarray] = {}
    meta: dict = {"source": str(extract_dir), "layers": {}}
    for layer in layers:
        H = np.load(extract_dir / f"L{layer}.npy").astype(np.float64)[[r["row"] for r in obs]]
        exp = np.array([r["tool"] == "expand" for r in obs])
        intent = np.array([r["intent"] for r in obs])
        resid = float(np.median(np.linalg.norm(H, axis=1)))
        m: dict = {"resid_norm_median": resid}
        pairs = {"answer": (H[exp & (intent == "answer")], H[exp & np.isin(intent, ["follow", "backtrack"])]),
                 "follow": (H[exp & (intent == "follow")], H[exp & (intent == "backtrack")])}
        units: dict[str, np.ndarray] = {}
        for name, (X1, X0) in pairs.items():
            d = unit(X1.mean(0) - X0.mean(0))
            gap = gap_along(d, X1, X0)
            units[name] = d
            vec[f"{name}_L{layer}"] = d.astype(np.float32)
            y = np.r_[np.ones(len(X1), bool), np.zeros(len(X0), bool)]
            m[name] = {"gap": gap, "n": [int(len(X1)), int(len(X0))],
                       "auroc": auroc(np.concatenate([X1, X0]) @ d, y), "dose1_over_resid": gap / resid}
        # Each axis with the other's component projected out, at the gap along
        # the projected direction: an arm along the raw follow axis at dose d is
        # the follow_perp arm at dose d plus the answer component the raw axis
        # carries.
        for name, other in (("follow", "answer"), ("answer", "follow")):
            X1, X0 = pairs[name]
            d = perp_direction(units[name], units[other])
            gap = gap_along(d, X1, X0)
            vec[f"{name}_perp_L{layer}"] = d.astype(np.float32)
            y = np.r_[np.ones(len(X1), bool), np.zeros(len(X0), bool)]
            m[f"{name}_perp"] = {"gap": gap, "removed": other, "cos_removed": float(units[name] @ units[other]),
                                 "gap_over_parent": gap / m[name]["gap"], "n": m[name]["n"],
                                 "auroc": auroc(np.concatenate([X1, X0]) @ d, y), "dose1_over_resid": gap / resid}
        for k in range(1, n_random + 1):
            key = "random" if k == 1 else f"random{k}"
            vec[f"{key}_L{layer}"] = unit(np.random.default_rng(1000 * k + layer).normal(size=H.shape[1])).astype(np.float32)
        m["cos_answer_follow"] = float(vec[f"answer_L{layer}"] @ vec[f"follow_L{layer}"])
        meta["layers"][str(layer)] = m
        if log:
            log(f"L{layer}: follow gap {m['follow']['gap']:.2f} ({m['follow']['dose1_over_resid']:.0%} of the residual, "
                f"AUROC {m['follow']['auroc']:.3f}, n {m['follow']['n']}), answer gap {m['answer']['gap']:.2f} "
                f"({m['answer']['dose1_over_resid']:.0%}, AUROC {m['answer']['auroc']:.3f})")
    return vec, meta


def per_layer_files(vec: dict, meta: dict, layer: int) -> tuple[dict, dict]:
    """One layer's files: unsuffixed keys, and `random` carrying the answer
    axis's gap (the `rnd` arms' unit; the `rndf` arms take the follow axis's)."""
    m = meta["layers"][str(layer)]
    out_vec = {k[: -len(f"_L{layer}")]: v for k, v in vec.items() if k.endswith(f"_L{layer}")}
    out_meta = {"layer": layer, "resid_norm_median": m["resid_norm_median"], "follow": m["follow"], "answer": m["answer"],
                "random": {"gap": m["answer"]["gap"]}}
    for key in ("follow_perp", "answer_perp"):
        out_meta[key] = m[key]
    return out_vec, out_meta
