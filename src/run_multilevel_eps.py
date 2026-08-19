"""Multi-level Ward coarsening with a fresh ``epsilon`` budget per level.

The best-train-F1 variant converged at level 1 (a single Ward tree already spans the
whole hierarchy, so level 1 takes the global optimum and later levels merge nothing).
Giving each level a *bounded* budget instead produces a genuine multi-level
trajectory: every level contracts as far as ``--eps-per-level`` allows, rebuilds the
tree on the reduced graph, and recurses.

Two reduction regimes are compared, since the coarse graph's weighting decides what
the next level's metric can see:

* ``weighted``   -- ``W_c = S^T W S`` with the diagonal kept: off-diagonal entries are
  the **cut** between supernodes, the diagonal is a supernode's **internal** weight
  (``d_s = d_u + d_v``, volume preserving).  Scored with the self-loop-aware
  ``L_sym = I - D^{-1/2} W D^{-1/2}`` (:func:`_weighted_normalized_laplacian`), which
  reads that diagonal.
* ``unweighted`` -- the same contraction but the weights are **dropped**: the diagonal
  is discarded and every surviving connection is set to 1, so the coarse graph is a
  plain unweighted graph.  Scored with the renormalization-trick
  :func:`_normalized_laplacian`.

Per level we report the level's own RSA distortion (against the *current* graph, the
quantity the budget gates) and the cumulative exact RSA against the *original* graph,
alongside detection over all gangs.

Run::

    python -m src.run_multilevel_eps --day-start 24 --transfer-days 10 \
        --eps-per-level 0.5 --max-levels 6 --out results/multilevel_eps
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering

from src.analyze_missed_gangs import cached_wallet_features
from src.collective_detector import CollectiveBankDetector, DetectorConfig, GraphData
from src.loukas_sgc_detection import (
    _exact_rsa_epsilon,
    _l_orthonormalize,
    _normalized_laplacian,
    _reduce_adjacency,
    _reduce_basis,
    _screened_metric,
    _weighted_normalized_laplacian,
    evaluate_loukas_patterns,
)
from src.run_collective_bank_detection import _labels_from_tree
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import build_torch_graph, make_patterns


def metric_for(W, mode, tau):
    """Screened metric matching the reduction regime."""

    base = _weighted_normalized_laplacian(W) if mode == "weighted" else _normalized_laplacian(W)
    return _screened_metric(base, tau)


def reduce_for(W, groups, mode):
    """Contract; keep cut/internal weights, or drop weights and keep connections."""

    if mode == "weighted":
        return _reduce_adjacency(W, groups, keep_self_loops=True).coalesce()
    Wc = _reduce_adjacency(W, groups, keep_self_loops=False).coalesce()
    idx = Wc.indices()
    ones = torch.ones(idx.shape[1], dtype=Wc.dtype, device=Wc.device)
    return torch.sparse_coo_tensor(idx, ones, Wc.shape).coalesce()  # binary


def _ward_children(W, A):
    n = int(W.shape[0])
    idx = W.coalesce().indices().cpu().numpy()
    off = idx[0] != idx[1]
    conn = csr_matrix((np.ones(int(off.sum())), (idx[0][off], idx[1][off])), shape=(n, n))
    model = AgglomerativeClustering(n_clusters=2, linkage="ward", connectivity=conn,
                                    compute_full_tree=True).fit(A)
    return np.asarray(model.children_)


def multilevel_eps(data, basis, gangs, *, mode, tau, threshold, max_levels, num_cuts,
                   eps_per_level, metric0, a0):
    """Each level contracts as far as ``eps_per_level`` allows, then recurses."""

    W = data.adjacency.coalesce()
    B = basis
    orig_to_cur = torch.arange(data.num_nodes, dtype=torch.long)
    records = []
    for level in range(1, max_levels + 1):
        n_cur = int(W.shape[0])
        if n_cur <= 3:
            break
        try:
            metric_l = metric_for(W, mode, tau)
            a_l = _l_orthonormalize(B, metric_l)  # target rows of the CURRENT graph
            if a_l.shape[1] == 0:
                break
            children = _ward_children(W, a_l.detach().cpu().numpy())
        except Exception as exc:
            print(f"    level {level} ({mode}): failed ({type(exc).__name__}: {exc})")
            break

        # walk fine->coarse; this level's own RSA distortion is monotone, so stop at
        # the coarsest cut still inside the per-level budget
        ks = np.unique(np.round(np.geomspace(2, n_cur - 1, max(num_cuts, 2))).astype(int))
        ks = ks[(ks >= 2) & (ks <= n_cur - 1)][::-1]
        best = None
        for k in ks.tolist():
            labels = torch.from_numpy(_labels_from_tree(children, n_cur, int(k)))
            e = _exact_rsa_epsilon(a_l, metric_l, labels)
            if e <= eps_per_level:
                best = (e, k, labels)
            else:
                break
        if best is None:
            break
        lvl_eps, k_best, groups = best
        n_new = int(groups.max()) + 1
        if n_new >= n_cur:
            break

        orig_to_cur = groups[orig_to_cur]
        res, _ = evaluate_loukas_patterns(gangs, orig_to_cur, data.y, threshold=threshold)
        cum = _exact_rsa_epsilon(a0, metric0, orig_to_cur)
        records.append({
            "mode": mode, "level": level, "n_before": n_cur, "n_after": n_new,
            "level_eps": lvl_eps, "cum_eps": cum,
            "detection": float(np.mean([r.detected for r in res])),
            "recall": float(np.mean([r.recall for r in res])),
            "precision": float(np.mean([r.precision for r in res])),
            "f1": float(np.mean([r.f1 for r in res])),
        })
        W = reduce_for(W, groups, mode)
        B = _reduce_basis(B, groups)
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--day-start", type=int, default=24)
    ap.add_argument("--transfer-days", type=int, default=10)
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--degree", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--conf-weight", type=float, default=10.0)
    ap.add_argument("--eps-per-level", type=float, default=0.5)
    ap.add_argument("--max-levels", type=int, default=6)
    ap.add_argument("--num-cuts", type=int, default=80)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/multilevel_eps", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print("=== caching days ===")
    days, cols = {}, None
    for k in range(args.transfer_days + 1):
        day = args.day_start + k
        A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, day, day)
        if cols is None:
            X, cols = cached_wallet_features(args.data_dir, nodes_df, day, return_columns=True)
        else:
            X = cached_wallet_features(args.data_dir, nodes_df, day, keep_columns=cols)
        g = build_torch_graph(A_w, A_unw, cls, X, weighted=False)
        sets = connected_components_sets(A_unw, np.where(cls == 1)[0], args.min_gang_size)
        if not sets:
            continue
        days[day] = {"data": GraphData.from_graph(g),
                     "gangs": make_patterns(sets, "alert", "gang", "g")}
        print(f"  day {day}: N={days[day]['data'].num_nodes:,} gangs={len(days[day]['gangs'])}")

    from src.run_elliptic_gang_detection import split_train_test
    D0 = days[args.day_start]
    tr, _ = split_train_test(D0["gangs"], args.train_ratio, np.random.default_rng(args.seed))
    cfg = DetectorConfig(degree=args.degree, tau=args.tau, epochs=args.epochs,
                         conf_weight=args.conf_weight, conf_reduce="mean",
                         optimizer="riemannian", coarsen_target="bank",
                         threshold=args.threshold, seed=args.seed)
    det = CollectiveBankDetector(cfg)
    print(f"\n=== training day {args.day_start} ({len(tr)} train gangs) ===")
    det.fit(D0["data"], tr)
    print(f"  lambda_min -> {det.fit_info_['objective']:.4g}")

    rows = []
    for mode in ("weighted", "unweighted"):
        print(f"\n########## mode = {mode} (eps/level = {args.eps_per_level}) ##########")
        for day, D in days.items():
            data, gangs = D["data"], D["gangs"]
            basis = det.target_subspace(data, gangs)
            metric0 = metric_for(data.adjacency, mode, args.tau)
            try:
                a0 = _l_orthonormalize(basis, metric0)
            except ValueError:
                continue
            recs = multilevel_eps(data, basis, gangs, mode=mode, tau=args.tau,
                                  threshold=args.threshold, max_levels=args.max_levels,
                                  num_cuts=args.num_cuts, eps_per_level=args.eps_per_level,
                                  metric0=metric0, a0=a0)
            for r in recs:
                rows.append(r | {"day": day})
            traj = " -> ".join(f"L{r['level']}:{r['n_after']:,}({r['detection']:.0%})" for r in recs)
            print(f"  day {day}: N={data.num_nodes:,}  {traj}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "multilevel_eps_per_day_level.csv", index=False)

    print("\n" + "=" * 100)
    print(f"MULTI-LEVEL WARD, fresh eps={args.eps_per_level} budget PER LEVEL: "
          "weighted (cut+internal) vs unweighted (connections only)")
    print("=" * 100)
    for mode in ("weighted", "unweighted"):
        sub = df[df["mode"] == mode]
        if not len(sub):
            continue
        agg = sub.groupby("level").agg(days=("day", "nunique"),
                                       med_n_before=("n_before", "median"),
                                       med_n_after=("n_after", "median"),
                                       lvl_eps=("level_eps", "mean"),
                                       cum_eps=("cum_eps", "mean"),
                                       detect=("detection", "mean"),
                                       recall=("recall", "mean"),
                                       precision=("precision", "mean"),
                                       f1=("f1", "mean"))
        print(f"\n--- {mode} ---")
        print(agg.round(3).to_string())
    print("\n=== best level per mode (mean over days) ===")
    for mode in ("weighted", "unweighted"):
        sub = df[df["mode"] == mode]
        if not len(sub):
            continue
        a = sub.groupby("level").agg(detect=("detection", "mean"), f1=("f1", "mean"),
                                     prec=("precision", "mean"))
        bi = a.detect.idxmax()
        print(f"  {mode:<11} best detection at level {bi}: {a.detect[bi]:.1%} "
              f"(f1={a.f1[bi]:.3f}, prec={a.prec[bi]:.3f})   "
              f"best f1 at level {a.f1.idxmax()}: {a.f1.max():.3f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for mode, c in [("weighted", "tab:blue"), ("unweighted", "tab:orange")]:
        sub = df[df["mode"] == mode]
        if not len(sub):
            continue
        a = sub.groupby("level").agg(d=("detection", "mean"), f=("f1", "mean"),
                                     p=("precision", "mean"), n=("n_after", "median"))
        a1.plot(a.index, a.d, "o-", color=c, label=f"{mode} detection")
        a1.plot(a.index, a.f, "s--", color=c, alpha=0.6, label=f"{mode} f1")
        a2.plot(a.index, a.n, "o-", color=c, label=f"{mode} median n_coarse")
    a1.set_xlabel("level"); a1.set_ylabel("rate"); a1.grid(alpha=0.3); a1.legend(fontsize=8)
    a1.set_title(f"Detection / F1 per level (eps={args.eps_per_level} per level)")
    a2.set_xlabel("level"); a2.set_ylabel("median supernodes"); a2.set_yscale("log")
    a2.grid(alpha=0.3); a2.legend(fontsize=8); a2.set_title("Graph size per level")
    fig.tight_layout(); fig.savefig(args.out / "multilevel_eps.png", dpi=150); plt.close(fig)
    print(f"\nCSV + plot -> {args.out}")


if __name__ == "__main__":
    main()
