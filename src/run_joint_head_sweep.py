"""Joint label-head training: does node supervision improve gang detection?

Adds a supervised head on the learned embedding and trains it *jointly* with the
filter bank (this is :func:`fit_joint_bank_head`, already in the codebase but never
wired into the Elliptic path)::

    loss = -lambda_min(Gamma(Z))  +  label_weight * CE(head(Z), y)

with ``head = nn.Linear(d, 2)``.  The label gradient flows into **both** the head and
the per-channel filter coefficients, so the one learned ``Z`` has to serve the
coarsening *and* node prediction.

Why this is worth testing here: the unsupervised collective objective is extremely
weak on Elliptic (lambda_min ~ 2e-4 -- it only has ~18 gang indicators to learn
from), while node labels give **thousands** of supervised nodes.  The untrained
control already showed the filter matters (+19 pts), so a stronger training signal
might matter more.

The risk is real too: capture wants ``Z`` *constant on each gang* (a collective,
group property), while CE wants ``Z`` *separable illicit-vs-licit* (a per-node
property).  Those are different objectives and may fight.

``label_weight = 0`` reproduces the unsupervised bank through the identical code
path, so it is an exact A/B baseline.  Theta is then frozen and transferred to every
day with the label-free epsilon Ward cut.

Run::

    python -m src.run_joint_head_sweep --day-start 24 --transfer-days 10 \
        --label-weights 0,0.1,1,10,100 --out results/joint_head
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
from src.run_collective_bank_detection import build_node_split, fit_joint_bank_head
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import build_torch_graph, make_patterns, split_train_test


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
    ap.add_argument("--learning-rate", type=float, default=0.02)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--label-weights", default="0,0.1,1,10,100")
    ap.add_argument("--epsilon", type=float, default=0.5)
    ap.add_argument("--ward-num-cuts", type=int, default=150)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/joint_head", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    betas = [float(b) for b in args.label_weights.split(",")]

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
    y_node, train_idx, test_idx = build_node_split(
        tr, te, d0.num_nodes, (d0.y == 1), neg_per_pos=args.neg_per_pos, seed=args.seed)
    print(f"\n  node split: {len(train_idx):,} train nodes "
          f"({int((y_node[train_idx] == 1).sum()):,} illicit), {len(test_idx):,} test nodes")
    print(f"  gang split: {len(tr)} train gangs, {len(te)} test gangs")

    rows, summ = [], []
    for beta in betas:
        print(f"\n########## label_weight (beta) = {beta} ##########")
        theta, _, head_metrics, lam = fit_joint_bank_head(
            d0.a_hat, d0.adjacency, d0.X, y_node, tr, train_idx, test_idx,
            degree=args.degree, epochs=args.epochs, learning_rate=args.learning_rate,
            ridge=args.ridge, tau=args.tau, label_weight=beta, seed=args.seed,
            basis="chebyshev")
        print(f"  lambda_min(final)={lam:.4g}   head node-level: "
              f"acc={head_metrics['accuracy']:.3f} prec={head_metrics['precision']:.3f} "
              f"rec={head_metrics['recall']:.3f} auc={head_metrics['auc']:.3f}")

        # freeze theta, transfer to every day with the label-free epsilon Ward cut
        cfg = DetectorConfig(degree=args.degree, tau=args.tau, coarsen_target="bank",
                             coarsening_method="ward-tree", ward_stop="epsilon",
                             epsilon=args.epsilon, ward_num_cuts=args.ward_num_cuts,
                             ridge=args.ridge, threshold=args.threshold, seed=args.seed)
        det = CollectiveBankDetector(cfg)
        det.theta_ = theta
        det.fit_info_ = None
        for day, D in days.items():
            data, gangs = D["data"], D["gangs"]
            basis = det.target_subspace(data, gangs)
            co, _ = det.coarsen(data, basis, gangs)
            res, _ = evaluate_loukas_patterns(gangs, co.node_to_supernode, data.y,
                                              threshold=args.threshold)
            for gi in range(len(gangs)):
                rows.append({"beta": beta, "day": day, "size": D["sizes"][gi],
                             "n_coarse": int(co.n_coarse),
                             "detected": int(res[gi].detected), "recall": res[gi].recall,
                             "precision": res[gi].precision, "f1": res[gi].f1})
        sub = pd.DataFrame([r for r in rows if r["beta"] == beta])
        fan = sub[sub["size"] >= 30]
        rec = {"beta": beta, "lam_min": lam, "head_auc": head_metrics["auc"],
               "head_f1_prec": head_metrics["precision"], "head_recall": head_metrics["recall"],
               "detect": sub.detected.mean(),
               "fan": fan.detected.mean() if len(fan) else np.nan,
               "precision": sub.precision.mean(), "recall": sub.recall.mean(),
               "f1": sub.f1.mean(), "n_coarse": sub.n_coarse.median()}
        summ.append(rec)
        print(f"  -> TRANSFER detect={rec['detect']:.1%} fan={rec['fan']:.1%} "
              f"prec={rec['precision']:.3f} f1={rec['f1']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "joint_head_per_gang.csv", index=False)
    S = pd.DataFrame(summ)
    S.to_csv(args.out / "joint_head_summary.csv", index=False)

    print("\n" + "=" * 104)
    print("JOINT LABEL-HEAD SWEEP   loss = -lambda_min(Gamma) + beta * CE(head(Z), y)")
    print("=" * 104)
    print(f"{'beta':>8}{'lam_min':>11}{'head_auc':>10}{'detect':>9}{'fan(>=30)':>11}"
          f"{'precision':>11}{'recall':>9}{'f1':>8}{'med n_coarse':>14}")
    print("-" * 104)
    for _, r in S.iterrows():
        print(f"{r.beta:>8.4g}{r.lam_min:>11.4g}{r.head_auc:>10.3f}{r.detect:>9.1%}"
              f"{r.fan:>11.1%}{r.precision:>11.3f}{r.recall:>9.3f}{r.f1:>8.3f}"
              f"{r.n_coarse:>14,.0f}")
    b0 = S[S.beta == 0]
    if len(b0):
        base = float(b0.detect.iloc[0])
        best = S.loc[S.detect.idxmax()]
        print(f"\n  baseline (beta=0): {base:.1%}   best: beta={best.beta:g} at "
              f"{best.detect:.1%}  ({best.detect - base:+.1%})")
    print("\n=== per-day detection by beta ===")
    print(df.pivot_table(index="day", columns="beta", values="detected",
                         aggfunc="mean").round(3).to_string())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(S.beta, S.detect, "o-", label="gang detection (transfer)")
    a1.plot(S.beta, S.precision, "s--", label="precision")
    a1.plot(S.beta, S.f1, "d:", label="f1")
    a1.set_xscale("symlog"); a1.set_xlabel("label weight β"); a1.set_ylim(0, 1)
    a1.grid(alpha=0.3); a1.legend(); a1.set_title("Does node supervision help detection?")
    a2.plot(S.beta, S.lam_min, "o-", color="tab:red", label="λ_min (capture)")
    a2.set_xscale("symlog"); a2.set_yscale("log"); a2.set_ylabel("λ_min", color="tab:red")
    a2b = a2.twinx(); a2b.plot(S.beta, S.head_auc, "s-", color="tab:blue", label="head AUC")
    a2b.set_ylabel("head node AUC", color="tab:blue")
    a2.set_xlabel("label weight β"); a2.grid(alpha=0.3)
    a2.set_title("The two objectives: capture vs node separability")
    fig.tight_layout(); fig.savefig(args.out / "joint_head.png", dpi=150); plt.close(fig)
    print(f"\nCSV + plot -> {args.out}")


if __name__ == "__main__":
    main()
