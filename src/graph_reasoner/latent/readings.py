"""Reads and moves -> the next-move readings: their AUC at every read point, and the plane of the three moves.

A reading is a linear direction fitted on one graph's reads (the fitting
graph) and scored held out there, by item-grouped five-fold cross-validation,
and zero-shot on a second graph (the unseen graph), whose reads are
standardised with the fitting graph's statistics and only projected. The
direction at each read point is the shrinkage-whitened class-mean difference
(lambda 0.5) in the space standardised by the fitting graph's observation
rows. Read point 0 is the embedding output and n the output of decoder layer
n. The two readings are the manuscript's: following against backtracking on
expand pages, and answering against continuing (following or backtracking) on
expand pages.

`profile` writes the profile `scripts/latent/depth_profile.py` stores as
depth_profile.json; `lda_plane` gives the two Fisher axes of the three moves
for the figure's panel A (`figure.panel_a_data`). Tests: tests/test_latent.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["READINGS", "auroc", "grouped_folds", "shrunk_inv", "labels_join", "profile", "lda_plane"]

#: The readings: key (the stored profile's name), short name, the positive
#: class, the classes the reading is scored over. All on expand pages.
READINGS = (
    ("follow vs backtrack (expand pages)", "follow vs backtrack", "follow", ("follow", "backtrack")),
    ("answer vs continue (expand pages)", "answer vs continue", "answer", ("follow", "backtrack", "answer")),
)


def auroc(scores, labels) -> float:
    """The rank statistic, ties at mid-rank; nan when one class is empty."""
    labels = np.asarray(labels).astype(bool)
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    x = np.asarray(scores, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    _u, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    r = np.bincount(inv, weights=ranks)[inv] / counts[inv]
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def grouped_folds(groups, k: int, seed: int) -> list[np.ndarray]:
    """`k` folds of row indices with every group (item) in one fold, the
    groups shuffled by `seed`."""
    items = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    fold_of = {it: i % k for i, it in enumerate(items)}
    g = np.asarray([fold_of[x] for x in groups])
    return [np.nonzero(g == i)[0] for i in range(k)]


def shrunk_inv(Z: np.ndarray, lam: float = 0.5):
    """The inverse of the rows' covariance shrunk `lam` of the way to its
    scaled identity; None when the covariance is zero."""
    Zc = Z - Z.mean(0)
    S = Zc.T @ Zc / max(len(Z) - 1, 1)
    d = S.shape[0]
    t = np.trace(S)
    if t == 0:
        return None
    return np.linalg.inv((1 - lam) * S + lam * t / d * np.eye(d))


def labels_join(index_path: Path, labels_path: Path) -> dict:
    """The labels file joined to the reads' index by (item, call): the row
    index of every labelled observation in the reads, with its move, tool and
    item, and the gold-arrival flags when the file carries them."""
    L = np.load(labels_path)
    rows = [json.loads(line) for line in Path(index_path).open(encoding="utf-8") if line.strip()]
    pos = {(r["id"], r.get("call")): i for i, r in enumerate(rows) if r.get("kind") == "observation"}
    idx = np.array([pos[(i, int(c))] for i, c in zip(L["item"], L["call"])])
    out = dict(idx=idx, lab=L["lab"], tool=L["tool"], item=L["item"])
    for k in ("G", "hop"):
        if k in L.files:
            out[k] = L[k]
    return out


def _targets(lb: dict) -> dict:
    exp = lb["tool"] == "expand"
    return {key: (exp & np.isin(lb["lab"], classes), lb["lab"] == positive)
            for key, _name, positive, classes in READINGS}


def profile(fit_reads: Path, fit_labels: Path, test_reads: Path, test_labels: Path,
            layers: list[int], names: tuple[str, str] = ("home", "bio"), log=None) -> dict:
    """reading -> {names[0]: [AUC per layer, held out on the fitting graph],
    names[1]: [AUC per layer, zero-shot on the unseen graph]}."""
    fit_reads, test_reads = Path(fit_reads), Path(test_reads)
    lh = labels_join(fit_reads / "index.jsonl", fit_labels)
    lb = labels_join(test_reads / "index.jsonl", test_labels)
    th, tb = _targets(lh), _targets(lb)
    out = {k: {names[0]: [], names[1]: []} for k, *_ in READINGS}
    folds_cache: dict = {}
    for layer in layers:
        Xh = np.load(fit_reads / f"L{layer}.npy")[lh["idx"]].astype(np.float64)
        Xb = np.load(test_reads / f"L{layer}.npy")[lb["idx"]].astype(np.float64)
        mu, sd = Xh.mean(0), Xh.std(0) + 1e-6
        Zh, Zb = (Xh - mu) / sd, (Xb - mu) / sd
        for k, *_ in READINGS:
            mh, yh = th[k]
            mb, yb = tb[k]
            Z, y, items = Zh[mh], yh[mh], lh["item"][mh]
            if k not in folds_cache:
                folds_cache[k] = grouped_folds(items, 5, 0)
            s = np.zeros(len(y))
            for te in folds_cache[k]:
                tr = np.setdiff1d(np.arange(len(y)), te)
                W = shrunk_inv(Z[tr])
                diff = Z[tr][y[tr]].mean(0) - Z[tr][~y[tr]].mean(0)
                d = diff if W is None else W @ diff
                s[te] = Z[te] @ d
            out[k][names[0]].append(auroc(s, y))
            W = shrunk_inv(Z)
            diff = Z[y].mean(0) - Z[~y].mean(0)
            d = diff if W is None else W @ diff
            out[k][names[1]].append(auroc(Zb[mb] @ d, yb[mb]))
        if log:
            log(f"L{layer:>2}: " + "  ".join(f"{name} {out[k][names[0]][-1]:.3f}/{out[k][names[1]][-1]:.3f}"
                                           for k, name, *_ in READINGS))
    return out


def lda_plane(Z: np.ndarray, lab: np.ndarray, classes: tuple[str, ...], lam: float = 0.5) -> np.ndarray:
    """The two Fisher axes of `classes` in the standardised space, the columns
    of a (d, 2) matrix: the pooled within-class covariance shrunk `lam` of the
    way to its scaled identity, the whitened class means weighted by the
    square root of their sizes, and their two leading right singular vectors."""
    mus = {c: Z[lab == c].mean(0) for c in classes}
    R = np.concatenate([Z[lab == c] - mus[c] for c in classes])
    Sw = R.T @ R / max(len(Z) - len(classes), 1)
    d = Sw.shape[0]
    Sw = (1 - lam) * Sw + lam * np.trace(Sw) / d * np.eye(d)
    w, V = np.linalg.eigh(Sw)
    Wh = V @ np.diag(w ** -0.5) @ V.T
    M = np.stack([mus[c] for c in classes])
    n = np.array([(lab == c).sum() for c in classes], float)
    gm = (n[:, None] * M).sum(0) / n.sum()
    _u, _s, Vt = np.linalg.svd(np.sqrt(n)[:, None] * ((M - gm) @ Wh), full_matrices=False)
    return Wh @ Vt[:2].T
