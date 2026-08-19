"""Does dropping lambda_min's cross-gang separation term help?  (+ CE combinations)

``lambda_min(Gamma)`` asks for two things at once: every gang **captured**, and the
gangs **mutually independent** (Definition 7.1 -- it is the hardest gang's capture
*after discounting alignment with the others*).  It also carries the capacity wall
``lambda_min = 0`` whenever ``m > d`` (Theorem 7.3).

But the coarsener only ever merges **adjacent** nodes, so two spatially disjoint
gangs cannot land in one supernode however aligned their embeddings are -- Prop 8.5:
*"copies do not merge ... repetition is thus harmless for detection"*, and the
capacity threshold *"constrains only embedding-based resolution"*.  So the cross-gang
separation lambda_min buys is largely **already free**, and paying for it costs
capacity.  The separation that genuinely matters -- from the gang's own
neighbourhood -- is the confusability ``chi``, a separate term.

Arms (all with the same confusability penalty, all transferred frozen with the
label-free epsilon Ward cut):

    lambda_min      capture + cross-gang separation      (baseline)
    trace           mean_j Gamma_jj  -- pure capture, no capacity wall
    softmin_diag    worst gang's Gamma_jj -- pure capture, worst-case

each optionally combined with the joint CE head (``label_weight``).

Run::

    python -m src.run_capture_objective_sweep --day-start 24 --transfer-days 10 \
        --out results/capobj
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

from src.analyze_missed_gangs import cached_wallet_features
from src.collective_detector import CollectiveBankDetector, DetectorConfig, GraphData
from src.loukas_sgc_detection import evaluate_loukas_patterns
from src.run_collective_bank_detection import build_node_split
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import build_torch_graph, make_patterns, split_train_test

# (name, capture_objective, label_weight)
ARMS = [
    ("lambda_min", "lambda_min", 0.0),
    ("trace", "trace", 0.0),
    ("softmin_diag", "softmin_diag", 0.0),
    ("lambda_min + CE", "lambda_min", 100.0),
    ("trace + CE", "trace", 100.0),
    ("softmin_diag + CE", "softmin_diag", 100.0),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--day-start", type=int, default=24)
    ap.add_argument("--transfer-days", type=int, default=10)
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--neg-per-pos", type=float, default=1.0)
    ap.add_argument("--degree", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--conf-weight", type=float, default=10.0)
    ap.add_argument("--epsilon", type=float, default=0.5)
    ap.add_argument("--ward-num-cuts", type=int, default=150)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/capobj", type=Path)
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
    d0, gangs0 = D0["data"], D0["gangs"]
    tr, te = split_train_test(gangs0, args.train_ratio, np.random.default_rng(args.seed))
    y_node, train_idx, _ = build_node_split(
        tr, te, d0.num_nodes, (d0.y == 1), neg_per_pos=args.neg_per_pos, seed=args.seed)
    print(f"\n  {len(tr)} train gangs | {len(train_idx):,} labelled nodes")

    rows, summ = [], []
    for name, obj, lab_w in ARMS:
        print(f"\n########## {name}  (capture_objective={obj}, label_weight={lab_w}) ##########")
        cfg = DetectorConfig(degree=args.degree, tau=args.tau, epochs=args.epochs,
                             conf_weight=args.conf_weight, conf_reduce="mean",
                             optimizer="riemannian", coarsen_target="bank",
                             capture_objective=obj, label_weight=lab_w,
                             coarsening_method="ward-tree", ward_stop="epsilon",
                             epsilon=args.epsilon, ward_num_cuts=args.ward_num_cuts,
                             threshold=args.threshold, seed=args.seed)
        det = CollectiveBankDetector(cfg)
        det.fit(d0, tr, label_y=y_node, label_idx=train_idx)
        cap = det.capture(d0, tr)
        print(f"  train-day: lam_min={cap['lambda_min_gamma']:.4g}  "
              f"mean_cap={cap['mean_capture']:.4f}  min_cap={cap['min_capture']:.4f}  "
              f"chi={det.fit_info_.get('confusability', float('nan')):.4g}")
        for day, D in days.items():
            data, gangs = D["data"], D["gangs"]
            basis = det.target_subspace(data, gangs)
            co, _ = det.coarsen(data, basis, gangs)
            res, _ = evaluate_loukas_patterns(gangs, co.node_to_supernode, data.y,
                                              threshold=args.threshold)
            dcap = det.capture(data, gangs)
            for gi in range(len(gangs)):
                rows.append({"arm": name, "day": day, "size": D["sizes"][gi],
                             "capture": float(dcap["per_gang_capture"][gi]),
                             "detected": int(res[gi].detected), "recall": res[gi].recall,
                             "precision": res[gi].precision, "f1": res[gi].f1})
        sub = pd.DataFrame([r for r in rows if r["arm"] == name])
        fan = sub[sub["size"] >= 30]
        rec = {"arm": name, "obj": obj, "lab_w": lab_w,
               "lam_min_train": cap["lambda_min_gamma"],
               "mean_cap_train": cap["mean_capture"],
               "transfer_capture": sub.capture.median(),
               "detect": sub.detected.mean(),
               "fan": fan.detected.mean() if len(fan) else np.nan,
               "precision": sub.precision.mean(), "recall": sub.recall.mean(),
               "f1": sub.f1.mean()}
        summ.append(rec)
        print(f"  -> TRANSFER detect={rec['detect']:.1%} prec={rec['precision']:.3f} "
              f"f1={rec['f1']:.3f} median_capture={rec['transfer_capture']:.4f}")

    df = pd.DataFrame(rows); df.to_csv(args.out / "capobj_per_gang.csv", index=False)
    S = pd.DataFrame(summ); S.to_csv(args.out / "capobj_summary.csv", index=False)

    print("\n" + "=" * 112)
    print("CAPTURE OBJECTIVE SWEEP  (is lambda_min's cross-gang separation term worth paying for?)")
    print("=" * 112)
    print(f"{'arm':<20}{'lam_min(tr)':>13}{'mean_cap(tr)':>14}{'xfer_cap':>10}"
          f"{'detect':>9}{'fan':>7}{'precision':>11}{'recall':>9}{'f1':>7}")
    print("-" * 112)
    for _, r in S.iterrows():
        print(f"{r.arm:<20}{r.lam_min_train:>13.4g}{r.mean_cap_train:>14.4f}"
              f"{r.transfer_capture:>10.4f}{r.detect:>9.1%}{r.fan:>7.1%}"
              f"{r.precision:>11.3f}{r.recall:>9.3f}{r.f1:>7.3f}")

    # paired McNemar vs the lambda_min baseline
    df["gid"] = df.day.astype(str) + "_" + df.groupby(["arm", "day"]).cumcount().astype(str)
    P = df.pivot_table(index="gid", columns="arm", values="detected", aggfunc="first")
    from scipy.stats import binomtest
    base = "lambda_min"
    print(f"\npaired McNemar vs {base} (n={len(P)} gangs):")
    print(f"  {'arm':<20}{'gained':>8}{'lost':>7}{'net':>6}{'p':>9}")
    for arm in P.columns:
        if arm == base:
            continue
        b = int(((P[base] == 0) & (P[arm] == 1)).sum())
        c = int(((P[base] == 1) & (P[arm] == 0)).sum())
        p = binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
        print(f"  {arm:<20}{b:>8}{c:>7}{b - c:>6}{p:>9.3f}")

    x = np.arange(len(S))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, S.detect, 0.4, label="detection", color="tab:blue")
    ax.bar(x + 0.2, S.precision, 0.4, label="precision", color="tab:green")
    ax.set_xticks(x); ax.set_xticklabels(S.arm, rotation=18, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3); ax.legend()
    ax.set_title("Capture objective: is lambda_min's separation term worth its capacity?")
    fig.tight_layout(); fig.savefig(args.out / "capobj.png", dpi=150); plt.close(fig)
    print(f"\nCSV + plot -> {args.out}")


if __name__ == "__main__":
    main()
