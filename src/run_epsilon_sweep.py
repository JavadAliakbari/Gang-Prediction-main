"""Sweep the RSA distortion budget ``epsilon`` over the Elliptic++ transfer days.

The tau-sweep showed screening is *not* the lever: fan gangs stay at 0% detection
for every tau, while their *capture* is already the highest of any size bucket.
That points at the **contraction cost** (Prop 6.14) rather than capture: a sprawling
low-density gang is expensive to collapse into one supernode, so the epsilon-budget
Ward cut simply refuses the merge.  This sweep tests that directly.

``epsilon`` affects only *where the Ward tree is cut* -- not the learned filter and
not the tree itself.  So we train **once**, build each day's tree **once** (walking
it out to the largest epsilon), and then re-cut the stored trajectory at every
epsilon.  For each budget we report:

* ``detect``    -- overall detection rate over all gangs / days
* ``fan``       -- detection restricted to fan gangs (size >= 30: the systematic
                   failures; median density 0.02, starness 2.5)
* ``sparse``    -- detection restricted to sparse gangs (density < 0.15), the more
                   principled proxy for the same population
* ``precision`` -- mean per-gang precision: the cost of over-coarsening
* ``n_coarse``  -- median supernode count (how far the budget actually coarsens)

Run::

    python -m src.run_epsilon_sweep --day-start 24 --transfer-days 10 \
        --epsilons 0.3,0.5,0.75,1,1.5,2,3 --out results/eps_sweep
"""

from __future__ import annotations

import argparse
import json
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
from src.loukas_sgc_detection import evaluate_loukas_patterns
from src.run_collective_bank_detection import ward_tree_coarsen
from src.run_elliptic_gang_detection import split_train_test
from src.run_tau_sweep import load_days


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
    ap.add_argument("--tau", type=float, default=0.5, help="tau-sweep optimum")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--conf-weight", type=float, default=10.0)
    ap.add_argument("--epsilons", default="0.3,0.5,0.75,1,1.5,2,3")
    ap.add_argument("--ward-num-cuts", type=int, default=150)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/eps_sweep", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    epsilons = [float(e) for e in args.epsilons.split(",")]
    max_eps = max(epsilons)

    print("=== caching days ===")
    days, _ = load_days(args)
    train_day = args.day_start

    # --- train ONCE (epsilon does not affect the filter) --------------------
    print(f"\n=== training day {train_day} (tau={args.tau}) ===")
    tr, _ = split_train_test(days[train_day]["gangs"], args.train_ratio,
                             np.random.default_rng(args.seed))
    cfg = DetectorConfig(degree=args.degree, tau=args.tau, epochs=args.epochs,
                         conf_weight=args.conf_weight, conf_reduce="mean",
                         optimizer="riemannian", coarsen_target="bank",
                         coarsening_method="ward-tree", ward_stop="epsilon",
                         epsilon=max_eps, ward_num_cuts=args.ward_num_cuts,
                         threshold=args.threshold, seed=args.seed)
    det = CollectiveBankDetector(cfg)
    det.fit(days[train_day]["data"], tr)
    print(f"  lambda_min -> {det.fit_info_['objective']:.4g}")

    # --- per day: build the tree ONCE, re-cut at every epsilon --------------
    rows = []
    for day, D in days.items():
        data, gangs = D["data"], D["gangs"]
        basis = det.target_subspace(data, gangs)
        # walk the tree out to the LARGEST budget; trajectory holds every cut
        _, traj = ward_tree_coarsen(
            data.adjacency, basis, gangs, data.y, tau=args.tau,
            laplacian="symmetric", threshold=args.threshold, stop="epsilon",
            epsilon_budget=max_eps, num_cuts=args.ward_num_cuts,
        )
        for eps in epsilons:
            feasible = [t for t in traj if t["epsilon"] <= eps]
            entry = feasible[-1] if feasible else traj[0]  # coarsest within budget
            n2s = torch.from_numpy(entry["labels"]).to(data.y.device)
            res, _ = evaluate_loukas_patterns(gangs, n2s, data.y,
                                              threshold=args.threshold)
            for gi in range(len(gangs)):
                rows.append({
                    "eps": eps, "day": day, "size": D["struct"][gi]["size"],
                    "density": D["struct"][gi]["density"],
                    "mbar1": float(D["mbar1"][gi]), "Phi": float(D["phi"][gi]),
                    "n_coarse": entry["n_coarse"], "eps_actual": entry["epsilon"],
                    "detected": int(res[gi].detected), "recall": res[gi].recall,
                    "precision": res[gi].precision, "f1": res[gi].f1,
                })
        print(f"  day {day}: tree walked ({len(traj)} cuts), re-cut at {len(epsilons)} budgets")

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "eps_sweep_per_gang.csv", index=False)

    # --- summarize ----------------------------------------------------------
    S = []
    for eps in epsilons:
        d = df[df.eps == eps]
        fan = d[d["size"] >= 30]
        sparse = d[d["density"] < 0.15]
        S.append({
            "eps": eps,
            "detect": float(d.detected.mean()),
            "fan_detect": float(fan.detected.mean()) if len(fan) else float("nan"),
            "n_fan": int(len(fan)),
            "sparse_detect": float(sparse.detected.mean()) if len(sparse) else float("nan"),
            "n_sparse": int(len(sparse)),
            "precision": float(d.precision.mean()),
            "recall": float(d.recall.mean()),
            "f1": float(d.f1.mean()),
            "median_n_coarse": float(d.n_coarse.median()),
        })
    S = pd.DataFrame(S)
    S.to_csv(args.out / "eps_sweep_summary.csv", index=False)

    print("\n" + "=" * 96)
    print("EPSILON SWEEP  (does a bigger contraction budget let the fans collapse?)")
    print("=" * 96)
    print(f"{'eps':>6}{'detect':>9}{'fan(>=30)':>11}{'sparse':>9}{'precision':>11}"
          f"{'recall':>9}{'f1':>8}{'n_coarse':>11}")
    print("-" * 96)
    for _, r in S.iterrows():
        print(f"{r.eps:>6.2f}{r.detect:>9.1%}{r.fan_detect:>11.1%}{r.sparse_detect:>9.1%}"
              f"{r.precision:>11.3f}{r.recall:>9.3f}{r.f1:>8.3f}{r.median_n_coarse:>11.0f}")
    print(f"\nfan gangs pooled: {S.n_fan.iloc[0]}   sparse gangs pooled: {S.n_sparse.iloc[0]}")

    # --- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(S.eps, S.detect, "o-", label="detection (all gangs)", color="tab:blue")
    ax.plot(S.eps, S.fan_detect, "s-", label="fan detection (size≥30)", color="tab:red")
    ax.plot(S.eps, S.sparse_detect, "^--", label="sparse detection (density<0.15)",
            color="tab:orange")
    ax.plot(S.eps, S.precision, "d:", label="mean precision (cost)", color="tab:green")
    ax.set_xlabel(r"RSA distortion budget $\varepsilon$")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax2 = ax.twinx()
    ax2.plot(S.eps, S.median_n_coarse, "v-.", color="gray", alpha=0.6,
             label="median n_coarse")
    ax2.set_ylabel("median supernodes", color="gray")
    ax2.set_yscale("log")
    ax.set_title("Detection vs contraction budget: do the fans ever collapse?")
    fig.tight_layout()
    fig.savefig(args.out / "eps_sweep.png", dpi=150)
    plt.close(fig)
    (args.out / "summary.json").write_text(S.to_json(orient="records", indent=2) + "\n")
    print(f"\nCSV + plot -> {args.out}")


if __name__ == "__main__":
    main()
