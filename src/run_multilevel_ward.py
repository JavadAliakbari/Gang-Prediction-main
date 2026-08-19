"""Multi-level Ward coarsening, each level cut at the best *training* F1.

At every level we build the contiguity-constrained Ward tree on the current
(weighted) graph, cut it where the **training** gangs' mean F1 peaks, contract, and
recurse.  Detection over *all* gangs is reported at every level.

The reduction is the Laplacian-consistent, volume-preserving Loukas form

    W_c = S^T W S       (S = 0/1 assignment matrix)

so that, exactly as requested:

* ``W_c[a, b]`` (a != b) is the **cut** between supernodes ``a`` and ``b``, and
* ``W_c[a, a]`` is supernode ``a``'s **internal** weight (each internal undirected
  edge counted twice), giving ``d_s = d_u + d_v`` automatically.

That diagonal is only meaningful to a metric that reads it, so every level uses the
self-loop-aware ``L_sym = I - D^{-1/2} W D^{-1/2}`` (:func:`_weighted_normalized_laplacian`)
rather than the renormalization-trick Laplacian, which would discard it.

Two subtleties handled here:

* **Composition.** At level >= 2 the Ward labels index *supernodes*, but the gangs
  index *original* nodes.  Every cut is therefore composed through the running
  ``original -> current`` map before it is scored, so both the best-F1 choice and
  the reported detection always refer to original nodes.
* **Honest split.** The cut is chosen on the **training** gangs only; detection is
  reported over all gangs.

Run::

    python -m src.run_multilevel_ward --day-start 24 --transfer-days 10 \
        --max-levels 6 --out results/multilevel
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
    _reduce_adjacency,
    _reduce_basis,
    _screened_metric,
    _weighted_normalized_laplacian,
    evaluate_loukas_patterns,
)
from src.run_collective_bank_detection import _labels_from_tree
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import build_torch_graph, make_patterns, split_train_test


def _ward_children(W, basis, tau):
    """Full Ward merge tree on the M_tau-orthonormal rows of the current graph."""

    metric = _screened_metric(_weighted_normalized_laplacian(W), tau)
    A = _l_orthonormalize(basis, metric).detach().cpu().numpy()
    if A.shape[1] == 0:
        raise ValueError("degenerate target subspace")
    n = int(W.shape[0])
    idx = W.coalesce().indices().cpu().numpy()
    off = idx[0] != idx[1]  # connectivity must exclude the internal self-loops
    conn = csr_matrix((np.ones(int(off.sum())), (idx[0][off], idx[1][off])), shape=(n, n))
    model = AgglomerativeClustering(n_clusters=2, linkage="ward", connectivity=conn,
                                    compute_full_tree=True).fit(A)
    return np.asarray(model.children_), A


def multilevel_ward(data, basis, gangs, train_gangs, *, tau, threshold, max_levels,
                    num_cuts, metric0, a0):
    """Coarsen level by level, each cut at the best training F1; score every level."""

    W = data.adjacency.coalesce()
    B = basis
    n0 = data.num_nodes
    orig_to_cur = torch.arange(n0, dtype=torch.long)
    records = []

    for level in range(1, max_levels + 1):
        n_cur = int(W.shape[0])
        if n_cur <= 3:
            break
        try:
            children, _ = _ward_children(W, B, tau)
        except Exception as exc:
            print(f"    level {level}: ward failed ({exc})")
            break

        # baseline: the train F1 of the CURRENT state (i.e. "coarsen no further").
        # A single Ward tree already spans the whole hierarchy, so once a level has
        # taken the global F1 optimum no later level can beat this -- without the
        # check below the loop would grind out 1-merge no-ops forever.
        res0, _ = evaluate_loukas_patterns(train_gangs, orig_to_cur, data.y,
                                           threshold=threshold)
        f1_now = float(np.mean([r.f1 for r in res0])) if res0 else 0.0

        # walk cuts fine->coarse; compose to original nodes; keep the best TRAIN F1
        ks = np.unique(np.round(np.geomspace(2, n_cur - 1, max(num_cuts, 2))).astype(int))
        ks = ks[(ks >= 2) & (ks <= n_cur - 1)][::-1]
        best = None
        for k in ks.tolist():
            labels = torch.from_numpy(_labels_from_tree(children, n_cur, int(k)))
            composed = labels[orig_to_cur]
            res, _ = evaluate_loukas_patterns(train_gangs, composed, data.y,
                                              threshold=threshold)
            f1 = float(np.mean([r.f1 for r in res])) if res else 0.0
            if best is None or f1 > best[0]:
                best = (f1, k, labels)
        if best is None:
            break
        train_f1, k_best, groups = best
        n_new = int(groups.max()) + 1
        if n_new >= n_cur:  # no contraction available -> converged
            break
        if level > 1 and train_f1 <= f1_now + 1e-9:
            print(f"    level {level}: converged (no cut beats train F1 {f1_now:.3f})")
            break

        orig_to_cur = groups[orig_to_cur]
        res, _ = evaluate_loukas_patterns(gangs, orig_to_cur, data.y, threshold=threshold)
        eps = _exact_rsa_epsilon(a0, metric0, orig_to_cur)
        rec = {
            "level": level, "n_before": n_cur, "n_after": n_new,
            "train_f1": train_f1, "epsilon": eps,
            "detection": float(np.mean([r.detected for r in res])),
            "recall": float(np.mean([r.recall for r in res])),
            "precision": float(np.mean([r.precision for r in res])),
            "f1": float(np.mean([r.f1 for r in res])),
            "per_gang": [(gangs[i].num_nodes, int(res[i].detected)) for i in range(len(gangs))],
        }
        records.append(rec)

        # volume-preserving reduction: off-diag = cut, diag = internal weight
        W = _reduce_adjacency(W, groups, keep_self_loops=True).coalesce()
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
    ap.add_argument("--max-levels", type=int, default=6)
    ap.add_argument("--num-cuts", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/multilevel", type=Path)
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

    train_day = args.day_start
    D0 = days[train_day]
    tr, _ = split_train_test(D0["gangs"], args.train_ratio, np.random.default_rng(args.seed))
    cfg = DetectorConfig(degree=args.degree, tau=args.tau, epochs=args.epochs,
                         conf_weight=args.conf_weight, conf_reduce="mean",
                         optimizer="riemannian", coarsen_target="bank",
                         threshold=args.threshold, seed=args.seed)
    det = CollectiveBankDetector(cfg)
    print(f"\n=== training day {train_day} ({len(tr)} train gangs) ===")
    det.fit(D0["data"], tr)
    print(f"  lambda_min -> {det.fit_info_['objective']:.4g}")

    rows = []
    for day, D in days.items():
        data, gangs = D["data"], D["gangs"]
        # this day's own training split (cut chosen on train gangs only)
        tr_d, _ = split_train_test(gangs, args.train_ratio, np.random.default_rng(args.seed))
        basis = det.target_subspace(data, gangs)
        metric0 = _screened_metric(_weighted_normalized_laplacian(data.adjacency), args.tau)
        try:
            a0 = _l_orthonormalize(basis, metric0)
        except ValueError:
            print(f"  day {day}: degenerate basis, skipped")
            continue
        recs = multilevel_ward(data, basis, gangs, tr_d, tau=args.tau,
                               threshold=args.threshold, max_levels=args.max_levels,
                               num_cuts=args.num_cuts, metric0=metric0, a0=a0)
        print(f"\n  day {day} (N={data.num_nodes:,}, {len(gangs)} gangs)")
        print(f"    {'lvl':>4}{'n_before':>10}{'n_after':>9}{'train_f1':>10}{'eps':>8}"
              f"{'detect':>9}{'recall':>8}{'prec':>8}{'f1':>7}")
        for r in recs:
            print(f"    {r['level']:>4}{r['n_before']:>10,}{r['n_after']:>9,}"
                  f"{r['train_f1']:>10.3f}{r['epsilon']:>8.3f}{r['detection']:>9.1%}"
                  f"{r['recall']:>8.3f}{r['precision']:>8.3f}{r['f1']:>7.3f}")
            rows.append({k: v for k, v in r.items() if k != "per_gang"} | {"day": day})
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "multilevel_per_day_level.csv", index=False)

    print("\n" + "=" * 92)
    print("MULTI-LEVEL WARD (each level cut at best TRAINING F1; W_c = S^T W S, "
          "diag = internal, off-diag = cut)")
    print("=" * 92)
    agg = df.groupby("level").agg(days=("day", "nunique"),
                                  med_n_before=("n_before", "median"),
                                  med_n_after=("n_after", "median"),
                                  train_f1=("train_f1", "mean"), eps=("epsilon", "mean"),
                                  detect=("detection", "mean"), recall=("recall", "mean"),
                                  precision=("precision", "mean"), f1=("f1", "mean"))
    print(agg.round(3).to_string())
    agg.to_csv(args.out / "multilevel_summary.csv")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for day, sub in df.groupby("day"):
        a1.plot(sub.level, sub.detection, "o-", alpha=0.4, lw=1)
    a1.plot(agg.index, agg.detect, "ko-", lw=2.5, label="mean over days")
    a1.set_xlabel("coarsening level"); a1.set_ylabel("detection rate")
    a1.set_title("Detection per Ward level (each cut at best train F1)")
    a1.legend(); a1.grid(alpha=0.3)
    a2.plot(agg.index, agg.precision, "s-", label="precision")
    a2.plot(agg.index, agg.recall, "^-", label="recall")
    a2.plot(agg.index, agg.f1, "d-", label="f1")
    a2.plot(agg.index, agg.eps, ":", color="gray", label="RSA ε")
    a2.set_xlabel("coarsening level"); a2.grid(alpha=0.3); a2.legend()
    a2.set_title("Quality vs level")
    fig.tight_layout(); fig.savefig(args.out / "multilevel.png", dpi=150); plt.close(fig)
    print(f"\nCSV + plot -> {args.out}")


if __name__ == "__main__":
    main()
