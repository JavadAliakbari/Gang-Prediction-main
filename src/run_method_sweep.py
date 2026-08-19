"""Compare coarsening *candidate families* on the Elliptic++ transfer days.

The tau- and epsilon-sweeps both dead-ended: fans stay at 0% detection for every
tau, and the only epsilon that collapses them (1.0) destroys precision (0.20) and
overall detection (10%).  Prop 6.14 explains why: contraction cost scales as
``sqrt((lam_max+tau)/(lam_2(S)+tau))`` -- a fan/tree has internal Fiedler value
``lam_2 ~ 0``, so contracting it *pairwise* (edges / ward) needs ~|S| sequential
merges, each charging distortion.  No global budget can serve both fans and dense
gangs.

The lever left is the candidate **family**, not the budget:

* ``neighborhood`` contracts ``{i} u N(i)`` -- a hub and *all* its spokes in ONE
  step, charging the cost once.  That is exactly the fan geometry.
* ``star`` runs a hub-priority star pre-pass (``{hub} u {spokes}`` with spoke degree
  ``<= leaf_degree``, capped at ``max_star_size``) then edge matching.  NB the
  defaults (``leaf_degree=1`` = pure leaves only, ``max_star_size=64``) are far too
  restrictive for Elliptic fans, so we sweep them.
* ``capped`` interpolates: sets of up to ``max_contraction_size`` at a time.

Reported per family, directly comparable to the epsilon-sweep table: overall
detection, fan detection (size>=30), sparse detection (density<0.15), precision.

Run::

    python -m src.run_method_sweep --day-start 24 --transfer-days 10 \
        --out results/method_sweep
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

from src.collective_detector import CollectiveBankDetector, DetectorConfig
from src.loukas_sgc_detection import evaluate_loukas_patterns, loukas_coarsen_pytorch
from src.run_elliptic_gang_detection import split_train_test
from src.run_tau_sweep import load_days

# name -> loukas_coarsen_pytorch kwargs (None marks the ward-tree baseline)
FAMILIES = [
    ("ward-tree (baseline)", None),
    ("edges", dict(method="edges")),
    ("neighborhood", dict(method="neighborhood")),
    ("star ld=1 ms=512", dict(method="star", leaf_degree=1, max_star_size=512)),
    ("star ld=3 ms=512", dict(method="star", leaf_degree=3, max_star_size=512)),
    ("capped c=16", dict(method="capped", max_contraction_size=16)),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--day-start", type=int, default=24)
    ap.add_argument("--transfer-days", type=int, default=10)
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--feature-mode", choices=["wallet", "random"], default="wallet")
    ap.add_argument("--random-width", type=int, default=64)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--degree", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--conf-weight", type=float, default=10.0)
    ap.add_argument("--epsilon", type=float, default=0.5, help="same budget for all families")
    ap.add_argument("--reduction", type=float, default=0.3)
    ap.add_argument("--max-levels", type=int, default=10)
    ap.add_argument("--ward-num-cuts", type=int, default=150)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/method_sweep", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print("=== caching days ===")
    days, _ = load_days(args)
    train_day = args.day_start

    # --- train ONCE (the coarsening family does not affect the filter) ------
    print(f"\n=== training day {train_day} (tau={args.tau}) ===")
    tr, _ = split_train_test(days[train_day]["gangs"], args.train_ratio,
                             np.random.default_rng(args.seed))
    cfg = DetectorConfig(degree=args.degree, tau=args.tau, epochs=args.epochs,
                         conf_weight=args.conf_weight, conf_reduce="mean",
                         optimizer="riemannian", coarsen_target="bank",
                         coarsening_method="ward-tree", ward_stop="epsilon",
                         epsilon=args.epsilon, reduction=args.reduction,
                         max_levels=args.max_levels, ward_num_cuts=args.ward_num_cuts,
                         threshold=args.threshold, seed=args.seed)
    det = CollectiveBankDetector(cfg)
    det.fit(days[train_day]["data"], tr)
    print(f"  lambda_min -> {det.fit_info_['objective']:.4g}")

    # --- per day: basis once (family-independent), then every family --------
    rows = []
    for day, D in days.items():
        data, gangs = D["data"], D["gangs"]
        basis = det.target_subspace(data, gangs)
        for name, kw in FAMILIES:
            try:
                if kw is None:
                    co, _ = det.coarsen(data, basis, gangs)
                else:
                    co = loukas_coarsen_pytorch(
                        data.adjacency, basis, laplacian=cfg.coarsening_laplacian,
                        tau=cfg.tau, max_levels=args.max_levels,
                        reduction=args.reduction, epsilon=args.epsilon,
                        epsilon_ramp_levels=cfg.epsilon_ramp_levels, **kw)
                res, _ = evaluate_loukas_patterns(gangs, co.node_to_supernode, data.y,
                                                  threshold=args.threshold)
            except Exception as exc:
                print(f"  day {day} {name}: FAILED ({type(exc).__name__}: {exc})")
                continue
            for gi in range(len(gangs)):
                rows.append({
                    "family": name, "day": day, "size": D["struct"][gi]["size"],
                    "density": D["struct"][gi]["density"],
                    "n_coarse": int(co.n_coarse),
                    "detected": int(res[gi].detected), "recall": res[gi].recall,
                    "precision": res[gi].precision, "f1": res[gi].f1,
                })
        print(f"  day {day} done")

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "method_sweep_per_gang.csv", index=False)

    S = []
    for name, _ in FAMILIES:
        d = df[df.family == name]
        if not len(d):
            continue
        fan, sparse = d[d["size"] >= 30], d[d["density"] < 0.15]
        S.append({"family": name, "detect": d.detected.mean(),
                  "fan_detect": fan.detected.mean() if len(fan) else np.nan,
                  "sparse_detect": sparse.detected.mean() if len(sparse) else np.nan,
                  "precision": d.precision.mean(), "recall": d.recall.mean(),
                  "f1": d.f1.mean(), "median_n_coarse": d.n_coarse.median()})
    S = pd.DataFrame(S)
    S.to_csv(args.out / "method_sweep_summary.csv", index=False)

    print("\n" + "=" * 100)
    print(f"COARSENING FAMILY SWEEP  (epsilon={args.epsilon} for all; can a one-step "
          "hub+spokes family collapse the fans?)")
    print("=" * 100)
    print(f"{'family':<22}{'detect':>9}{'fan(>=30)':>11}{'sparse':>9}{'precision':>11}"
          f"{'recall':>9}{'f1':>8}{'n_coarse':>11}")
    print("-" * 100)
    for _, r in S.iterrows():
        print(f"{r.family:<22}{r.detect:>9.1%}{r.fan_detect:>11.1%}{r.sparse_detect:>9.1%}"
              f"{r.precision:>11.3f}{r.recall:>9.3f}{r.f1:>8.3f}{r.median_n_coarse:>11.0f}")

    x = np.arange(len(S))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - 0.3, S.detect, 0.2, label="detection (all)", color="tab:blue")
    ax.bar(x - 0.1, S.fan_detect, 0.2, label="fan detection (size≥30)", color="tab:red")
    ax.bar(x + 0.1, S.sparse_detect, 0.2, label="sparse (density<0.15)", color="tab:orange")
    ax.bar(x + 0.3, S.precision, 0.2, label="mean precision", color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels(S.family, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"Coarsening candidate family vs fan detection (ε={args.epsilon})")
    fig.tight_layout()
    fig.savefig(args.out / "method_sweep.png", dpi=150)
    plt.close(fig)
    print(f"\nCSV + plot -> {args.out}")


if __name__ == "__main__":
    main()
