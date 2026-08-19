"""Ward mass ablation on Elliptic++: cardinality vs self-loop weight vs volume.

Standard Ward weights a merge by the **cardinality** ``|C1||C2|/(|C1|+|C2|)``.  On a
weighted coarse graph that ignores what the reduction actually tracks, so we compare
three notions of a supernode's mass, everything else held fixed (same frozen filter,
same target subspace, same label-free ``epsilon <= --epsilon`` cut, same scoring):

* ``cardinality`` -- standard Ward (baseline; verified to reproduce sklearn exactly)
* ``selfloop``    -- ``W_ii + 1``: mass = the internal weight the supernode absorbed
* ``volume``      -- ``W_ii + cut_i + 1``: mass = total volume

The coarse graph preserves cut (off-diagonal) and density (diagonal) via
``W_c = S^T W S``, and is scored in the self-loop-aware ``L_sym``.

Run::

    python -m src.run_mass_sweep --day-start 24 --transfer-days 10 --out results/mass_sweep
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.analyze_missed_gangs import cached_wallet_features
from src.collective_detector import CollectiveBankDetector, DetectorConfig, GraphData
from src.loukas_sgc_detection import (
    _exact_rsa_epsilon,
    _l_orthonormalize,
    _screened_metric,
    _weighted_normalized_laplacian,
    evaluate_loukas_patterns,
)
from src.run_collective_bank_detection import _labels_from_tree
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import build_torch_graph, make_patterns, split_train_test
from src.weighted_ward import weighted_ward_tree


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
    ap.add_argument("--epsilon", type=float, default=0.5)
    ap.add_argument("--num-cuts", type=int, default=120)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--floor", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/mass_sweep", type=Path)
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
                     "gangs": make_patterns(sets, "alert", "gang", "g"),
                     "sizes": [len(s) for s in sets]}
        print(f"  day {day}: N={days[day]['data'].num_nodes:,} gangs={len(days[day]['gangs'])}")

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
    for mode in ("cardinality", "selfloop", "volume"):
        print(f"\n########## mass = {mode} ##########")
        for day, D in days.items():
            data, gangs = D["data"], D["gangs"]
            basis = det.target_subspace(data, gangs)
            metric = _screened_metric(_weighted_normalized_laplacian(data.adjacency), args.tau)
            try:
                A = _l_orthonormalize(basis, metric)
            except ValueError:
                continue
            t0 = time.time()
            children = weighted_ward_tree(data.adjacency, A, mass_mode=mode, floor=args.floor)
            n = data.num_nodes
            # coarsest cut inside the label-free epsilon budget (eps is monotone)
            ks = np.unique(np.round(np.geomspace(2, n - 1, args.num_cuts)).astype(int))
            ks = ks[(ks >= 2) & (ks <= n - 1)][::-1]
            best = None
            for k in ks.tolist():
                labels = torch.from_numpy(_labels_from_tree(children, n, int(k)))
                e = _exact_rsa_epsilon(A, metric, labels)
                if e <= args.epsilon:
                    best = (e, k, labels)
                else:
                    break
            if best is None:
                continue
            eps, k_best, labels = best
            res, _ = evaluate_loukas_patterns(gangs, labels, data.y, threshold=args.threshold)
            n_coarse = int(labels.max()) + 1
            for gi in range(len(gangs)):
                rows.append({"mass": mode, "day": day, "size": D["sizes"][gi],
                             "n_coarse": n_coarse, "eps": eps,
                             "detected": int(res[gi].detected), "recall": res[gi].recall,
                             "precision": res[gi].precision, "f1": res[gi].f1})
            dr = float(np.mean([r.detected for r in res]))
            print(f"  day {day}: n_coarse={n_coarse:,} eps={eps:.3f} detect={dr:.1%} "
                  f"({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "mass_sweep_per_gang.csv", index=False)

    print("\n" + "=" * 96)
    print(f"WARD MASS ABLATION (label-free epsilon<={args.epsilon} cut; cut+density-preserving W_c)")
    print("=" * 96)
    print(f"{'mass':<14}{'detect':>9}{'fan(>=30)':>11}{'precision':>11}{'recall':>9}"
          f"{'f1':>8}{'med n_coarse':>14}{'mean eps':>10}")
    print("-" * 96)
    S = []
    for mode in ("cardinality", "selfloop", "volume"):
        s = df[df["mass"] == mode]
        if not len(s):
            continue
        fan = s[s["size"] >= 30]
        rec = {"mass": mode, "detect": s.detected.mean(),
               "fan": fan.detected.mean() if len(fan) else np.nan,
               "precision": s.precision.mean(), "recall": s.recall.mean(),
               "f1": s.f1.mean(), "n_coarse": s.n_coarse.median(), "eps": s.eps.mean()}
        S.append(rec)
        print(f"{mode:<14}{rec['detect']:>9.1%}{rec['fan']:>11.1%}{rec['precision']:>11.3f}"
              f"{rec['recall']:>9.3f}{rec['f1']:>8.3f}{rec['n_coarse']:>14,.0f}{rec['eps']:>10.3f}")
    pd.DataFrame(S).to_csv(args.out / "mass_sweep_summary.csv", index=False)

    print("\n=== per-day detection ===")
    piv = df.pivot_table(index="day", columns="mass", values="detected", aggfunc="mean")
    print(piv.round(3).to_string())
    base = "cardinality"
    for m in ("selfloop", "volume"):
        if m in piv.columns and base in piv.columns:
            print(f"  {m} beats {base} on {(piv[m] > piv[base]).sum()} days, "
                  f"loses {(piv[m] < piv[base]).sum()}, ties {(piv[m] == piv[base]).sum()}")

    x = np.arange(len(S))
    Sd = pd.DataFrame(S)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, Sd.detect, 0.4, label="detection", color="tab:blue")
    ax.bar(x + 0.2, Sd.precision, 0.4, label="precision", color="tab:green")
    ax.set_xticks(x); ax.set_xticklabels(Sd["mass"]); ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3); ax.legend()
    ax.set_title(f"Ward mass ablation (ε≤{args.epsilon}, cut+density-preserving reduction)")
    fig.tight_layout(); fig.savefig(args.out / "mass_sweep.png", dpi=150); plt.close(fig)
    print(f"\nCSV + plot -> {args.out}")


if __name__ == "__main__":
    main()
