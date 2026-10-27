"""Reads, moves and knob mixes -> the data of the manuscript's latent-state figure.

Panel A: the three moves in the plane of their two Fisher axes, fitted on the
fitting graph's standardised reads at a layer, with the unseen graph's reads
projected onto the same plane (`panel_a_data`). Panel C: every steering arm's
move share, early answers, calls and accuracy against its dose as the mean and
spread over the reads the knob roots hold (`agg`), with each axis's random
null pooled over its draws (`null_of`, `agg_draws`). Panel B is the stored
depth profile, which the caller reads itself.

The paper repository's `figures/extract_data.py` imports this module and
passes the reads, the labels, the layers and the roots; nothing here names a
graph, a run or a root. The figure itself is drawn there. Tests:
tests/test_latent.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .readings import READINGS, labels_join, lda_plane

__all__ = ["CLASSES", "READINGS", "AXES", "MIN_READS", "panel_a_data", "load_mix", "agg", "agg_draws", "null_of"]

#: The three moves of the plane, in drawing order.
CLASSES = ("follow", "backtrack", "answer")
#: The two steering axes of panel C: the arm kind, the mix field the axis
#: moves, and the random-draw kinds at that axis's dose unit, pooled when
#: more than one ran.
AXES = {"follow": {"kind": "fol", "field": "follow_of_expands", "nulls": ("rndf", "rndf2", "rndf3")},
        "answer": {"kind": "ans", "field": "premature", "nulls": ("rnd", "rnd2", "rnd3")}}
#: An arm with fewer reads is not drawn.
MIN_READS = 2


def _plane_rows(reads: Path, labels: Path) -> dict:
    L = labels_join(Path(reads) / "index.jsonl", labels)
    sel = (L["tool"] == "expand") & np.isin(L["lab"], CLASSES)
    return dict(idx=L["idx"][sel], lab=L["lab"][sel])


def panel_a_data(fit_reads: Path, fit_labels: Path, test_reads: Path, test_labels: Path, layers) -> dict:
    """layer -> {"fit": (coordinates, moves), "test": (coordinates, moves),
    "lim": the square box around both, "arrows": one dose along each axis as
    the displacement between class centroids in the plane}."""
    fit_reads, test_reads = Path(fit_reads), Path(test_reads)
    lh, lb = _plane_rows(fit_reads, fit_labels), _plane_rows(test_reads, test_labels)
    out = {}
    for layer in layers:
        Xh = np.load(fit_reads / f"L{layer}.npy")[lh["idx"]].astype(np.float64)
        Xb = np.load(test_reads / f"L{layer}.npy")[lb["idx"]].astype(np.float64)
        mu, sd = Xh.mean(0), Xh.std(0) + 1e-6
        Zh, Zb = (Xh - mu) / sd, (Xb - mu) / sd
        A = lda_plane(Zh, lh["lab"], CLASSES)
        c0, s0 = (Zh @ A).mean(0), (Zh @ A).std(0)
        proj = lambda Z: ((Z @ A) - c0) / s0  # noqa: E731
        d = {"fit": (proj(Zh), lh["lab"]), "test": (proj(Zb), lb["lab"])}
        allc = np.concatenate([d["fit"][0], d["test"][0]])
        lo, hi = allc.min(0), allc.max(0)
        half = (hi - lo).max() * 0.53
        d["lim"] = tuple((float(c - half), float(c + half)) for c in (lo + hi) / 2)
        # One dose along a steering axis is the raw class-mean difference, so in
        # the plane it is the displacement from one class centroid to the other:
        # backtrack to follow, continue to answer.
        Ch = proj(Zh)
        cen = {c: Ch[lh["lab"] == c].mean(0) for c in CLASSES}
        cen["continue"] = Ch[lh["lab"] != "answer"].mean(0)
        d["arrows"] = {"follow": (cen["backtrack"], cen["follow"]), "answer": (cen["continue"], cen["answer"])}
        out[layer] = d
    return out


def load_mix(roots) -> dict:
    """benchmark -> arm -> the rows of every root that holds the arm, in root order."""
    out: dict = {}
    for root in roots:
        path = os.path.join(str(root), "mix.json")
        if not os.path.exists(path):
            continue
        for bench, arms in json.load(open(path, encoding="utf-8")).items():
            for arm, row in arms.items():
                out.setdefault(bench, {}).setdefault(arm, []).append(row)
    return out


def agg(mix, bench, kind, field, min_reads: int = MIN_READS, zero_at: float = 0.0):
    """settings, mean, spread, n over the reads of every arm of `kind` with at
    least `min_reads` reads (plus dose zero, placed at setting `zero_at`),
    sorted by setting. The spread is the standard deviation at three reads or
    more, half the range at two, zero at one."""
    arms = [(arm, rows) for arm, rows in mix.get(bench, {}).items()
            if rows[0]["kind"] in (kind, "dose0") and len(rows) >= min_reads]
    setting = lambda rows: zero_at if rows[0]["kind"] == "dose0" else float(rows[0]["setting"])  # noqa: E731
    arms.sort(key=lambda ar: setting(ar[1]))
    x = np.array([setting(rows) for _a, rows in arms])
    vals = [np.array([r[field] for r in rows], float) for _a, rows in arms]
    mean = np.array([v.mean() for v in vals])
    spread = np.array([v.std(ddof=1) if len(v) >= 3 else (v.max() - v.min()) / 2 if len(v) == 2 else 0.0 for v in vals])
    n = np.array([len(v) for v in vals])
    return x, mean, spread, n


def agg_draws(mix, bench, kinds, field):
    """The null pooled over random draws: each draw's mean over its reads
    first, then the mean and spread across the draws at the settings every
    draw ran, so a draw with more reads does not dominate; n is the number of
    draws."""
    per = []
    for kind in kinds:
        x, m, _s, _n = agg(mix, bench, kind, field, min_reads=1)
        per.append({float(xi): mi for xi, mi in zip(x, m) if xi != 0})
    xs = sorted(set.intersection(*[set(d) for d in per]))
    M = np.array([[d[xi] for xi in xs] for d in per])
    spread = M.std(0, ddof=1) if len(per) >= 3 else (M.max(0) - M.min(0)) / 2
    return np.array(xs), M.mean(0), spread, np.full(len(xs), len(per))


def null_of(mix, bench, kinds, axis_kind, gaps):
    """A function of the field giving the null's settings, means, spreads and
    n for one axis: the draws in `kinds` pooled when at least two ran at
    `bench`, one of them alone otherwise, or, when none ran at this axis's
    unit, the answer-unit draw rescaled to it; dose 0 is never part of it."""
    ran = [k for k in kinds if any(r[0]["kind"] == k for r in mix.get(bench, {}).values())]
    if len(ran) >= 2:
        return lambda field: agg_draws(mix, bench, ran, field)
    kind, unit = (ran[0], 1.0) if ran else ("rnd", gaps["random"]["gap"] / gaps[{"fol": "follow", "ans": "answer"}[axis_kind]]["gap"])

    def one(field):
        x, m, sd, n = agg(mix, bench, kind, field)
        k = x != 0
        return x[k] * unit, m[k], sd[k], n[k]
    return one
