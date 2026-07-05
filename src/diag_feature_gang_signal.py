"""Diagnostic: do node features X carry gang signal usable for *coarsening*?

Two questions, kept separate:

1. NODE-LEVEL signal -- are alert (gang) nodes' features distinguishable from
   the rest at all?  (Cohen's d per feature + a single LDA-margin separability.)
   This is necessary but not sufficient for coarsening.

2. BOUNDARY signal (the coarsening-relevant one) -- coarsening merges *adjacent*
   nodes, so a gang survives iff its features differ from the non-gang neighbours
   it would otherwise be merged with.  We compare, per gang, the feature distance
   on INTERNAL edges (both endpoints in the gang) vs BOUNDARY edges (one endpoint
   outside), and contrast that against size-matched random connected sets (the
   null).  If boundary contrast >> internal contrast for real gangs but not for
   the null, features mark gang boundaries and g(A) X W can sharpen coarsening.

Run:
    /Users/javada/miniconda3/envs/FedStruct/bin/python -m src.diag_feature_gang_signal \
        --experiment tutorial_demo16
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

from src.experiment_utils import load_and_preprocess_data
from src.run_elliptic_gang_conductance import random_connected_set
from scipy.sparse import csr_matrix


def nodes_of(p) -> np.ndarray:
    nd = p.nodes
    if torch.is_tensor(nd):
        return nd.detach().cpu().numpy().astype(np.int64)
    return np.asarray(list(nd), dtype=np.int64)


def standardize(X: np.ndarray) -> np.ndarray:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd
    return np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)


def cohens_d(Xs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    a = Xs[mask]
    b = Xs[~mask]
    pooled = np.sqrt(0.5 * (a.var(axis=0) + b.var(axis=0)) + 1e-12)
    return (a.mean(axis=0) - b.mean(axis=0)) / pooled


def lda_margin_auc(Xs: np.ndarray, mask: np.ndarray, ridge: float = 1e-2) -> float:
    """Node-level ridge-LDA separability of gang vs non-gang -> AUC of the 1-D score."""
    a = Xs[mask]
    b = Xs[~mask]
    mu = a.mean(0) - b.mean(0)
    cov = np.cov(Xs.T) + ridge * np.eye(Xs.shape[1])
    w = np.linalg.solve(cov, mu)
    s = Xs @ w
    sa, sb = s[mask], s[~mask]
    # AUC via rank statistic
    order = np.argsort(np.concatenate([sa, sb]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    r_a = ranks[: len(sa)].sum()
    auc = (r_a - len(sa) * (len(sa) + 1) / 2) / (len(sa) * len(sb))
    return float(auc)


def edge_contrast(Xs, edges, in_set):
    """Mean L2 feature distance on internal vs boundary edges of a node set."""
    u, v = edges
    su, sv = in_set[u], in_set[v]
    internal = su & sv
    boundary = su ^ sv  # exactly one endpoint inside
    d = np.linalg.norm(Xs[u] - Xs[v], axis=1)
    di = d[internal].mean() if internal.any() else np.nan
    db = d[boundary].mean() if boundary.any() else np.nan
    return di, db, int(internal.sum()), int(boundary.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="tutorial_demo16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--null-samples", type=int, default=200)
    args = ap.parse_args()

    root = Path.cwd() / "experiments" / args.experiment
    graph, alert_train, normal_train, alert_test, normal_test = load_and_preprocess_data(
        data_dir=root / "config",
        patterns_dir=root,
        train_ratio=0.25,
        to_undirected=True,
        remove_overlaps=False,
        device=torch.device("cpu"),
        seed=args.seed,
    )
    alerts = alert_train + alert_test
    X = graph.x.detach().cpu().numpy().astype(np.float64)
    n, f = X.shape
    Xs = standardize(X)
    edges = graph.edge_index.detach().cpu().numpy()

    alert_nodes = np.unique(np.concatenate([nodes_of(p) for p in alerts]))
    mask = np.zeros(n, dtype=bool)
    mask[alert_nodes] = True

    print("=" * 72)
    print(f"experiment={args.experiment}  nodes={n}  features={f}  "
          f"alert_nodes={mask.sum()} ({mask.mean():.1%})  alert_gangs={len(alerts)}")
    print("=" * 72)

    # ---- 1. node-level signal -------------------------------------------------
    d = cohens_d(Xs, mask)
    order = np.argsort(-np.abs(d))
    auc = lda_margin_auc(Xs, mask)
    print("\n[1] NODE-LEVEL signal (gang node vs rest)")
    print(f"    ridge-LDA node separability AUC = {auc:.3f}  (0.5=no signal)")
    print(f"    mean |Cohen d| = {np.abs(d).mean():.3f}   max |Cohen d| = {np.abs(d).max():.3f}")
    cols = getattr(graph, "feature_names", None)
    print("    top-8 discriminative feature indices (|d|):")
    for j in order[:8]:
        name = cols[j] if cols is not None and j < len(cols) else f"feat[{j}]"
        print(f"        {name:<40} d={d[j]:+.3f}")

    # ---- 2. boundary contrast (coarsening-relevant) ---------------------------
    A = csr_matrix(
        (np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(n, n)
    )
    A = A + A.T
    A.data[:] = 1.0
    rng = np.random.default_rng(args.seed)

    gang_ratios = []
    for p in alerts:
        s = nodes_of(p)
        in_set = np.zeros(n, dtype=bool)
        in_set[s] = True
        di, db, ni, nb = edge_contrast(Xs, edges, in_set)
        if np.isfinite(di) and np.isfinite(db) and di > 1e-9:
            gang_ratios.append(db / di)
    gang_ratios = np.array(gang_ratios)

    null_ratios = []
    sizes = [p.num_nodes for p in alerts]
    for _ in range(args.null_samples):
        size = int(rng.choice(sizes))
        s = random_connected_set(A, size, rng)
        in_set = np.zeros(n, dtype=bool)
        in_set[np.array(s)] = True
        di, db, ni, nb = edge_contrast(Xs, edges, in_set)
        if np.isfinite(di) and np.isfinite(db) and di > 1e-9:
            null_ratios.append(db / di)
    null_ratios = np.array(null_ratios)

    print("\n[2] BOUNDARY contrast  (boundary edge feat-dist / internal edge feat-dist)")
    print(f"    real gangs : median {np.median(gang_ratios):.3f}  mean {gang_ratios.mean():.3f}  "
          f"(n={len(gang_ratios)})")
    print(f"    random null: median {np.median(null_ratios):.3f}  mean {null_ratios.mean():.3f}  "
          f"(n={len(null_ratios)})")
    frac = (gang_ratios > 1.0).mean()
    print(f"    fraction of gangs with boundary>internal contrast = {frac:.1%}")
    # one-sided: are gang ratios stochastically larger than null?
    if len(gang_ratios) and len(null_ratios):
        bigger = np.mean(gang_ratios[:, None] > null_ratios[None, :])
        print(f"    P(gang ratio > null ratio) = {bigger:.3f}  (0.5=no boundary signal)")

    # ---- 3. boundary contrast on the DISCRIMINATIVE direction -----------------
    # The gang signal lives in ~1 direction (feat[6]); raw L2 over 71 dims buries
    # it.  Project onto the ridge-LDA direction w and ask: does that single
    # gang-ness coordinate jump across the gang boundary?  This is exactly the
    # coordinate W would learn, so it predicts whether g(A)XW helps coarsening.
    cov = np.cov(Xs.T) + 1e-2 * np.eye(f)
    w = np.linalg.solve(cov, Xs[mask].mean(0) - Xs[~mask].mean(0))
    w /= np.linalg.norm(w) + 1e-12
    s = Xs @ w  # (n,) node gang-ness score

    def lda_gap(in_set):
        """mean score inside the set minus mean score on its 1-hop outside boundary."""
        nbr = A[in_set].nonzero()[1]
        outside = np.unique(nbr[~in_set[nbr]])
        if len(outside) == 0 or in_set.sum() == 0:
            return np.nan
        return s[in_set].mean() - s[outside].mean()

    gang_gaps, null_gaps = [], []
    for p in alerts:
        in_set = np.zeros(n, dtype=bool)
        in_set[nodes_of(p)] = True
        g = lda_gap(in_set)
        if np.isfinite(g):
            gang_gaps.append(g)
    for _ in range(args.null_samples):
        in_set = np.zeros(n, dtype=bool)
        in_set[np.array(random_connected_set(A, int(rng.choice(sizes)), rng))] = True
        g = lda_gap(in_set)
        if np.isfinite(g):
            null_gaps.append(g)
    gang_gaps, null_gaps = np.array(gang_gaps), np.array(null_gaps)
    sd_all = s.std() + 1e-12
    print("\n[3] BOUNDARY gap on the LDA gang-ness direction "
          "(mean score inside - mean score on 1-hop boundary), in score-std units")
    print(f"    real gangs : median {np.median(gang_gaps)/sd_all:+.3f}  "
          f"mean {gang_gaps.mean()/sd_all:+.3f}  (n={len(gang_gaps)})")
    print(f"    random null: median {np.median(null_gaps)/sd_all:+.3f}  "
          f"mean {null_gaps.mean()/sd_all:+.3f}  (n={len(null_gaps)})")
    if len(gang_gaps) and len(null_gaps):
        bigger = np.mean(np.abs(gang_gaps)[:, None] > np.abs(null_gaps)[None, :])
        print(f"    P(|gang gap| > |null gap|) = {bigger:.3f}  (0.5=no boundary signal)")

    print("\nRead: AUC>0.5 in [1] means features know gang membership at node level. "
          "[3] is the coarsening-relevant test: a large gang-vs-boundary gap on the "
          "discriminative direction means g(A)XW (with W~the LDA direction) preserves "
          "a coordinate that separates gangs from their neighbours -> sharper coarsening.")


if __name__ == "__main__":
    main()
