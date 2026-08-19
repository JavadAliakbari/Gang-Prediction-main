"""beta = 0 vs moderate vs beta -> inf, plus WHY cross-entropy fights lambda_min.

Arms of ``loss = -capture_weight * lambda_min(Gamma) + label_weight * CE(head(Z), y)``:

    capture-only  (cap=1, lab=0)     the pure detect-all bank         [beta = 0]
    joint         (cap=1, lab=1..1e3)
    CE-only       (cap=0, lab=1)     purely supervised embedding      [beta -> inf]

``capture_weight=0`` makes the ``beta -> inf`` endpoint *exact* rather than a huge-
label_weight hack that would blow up the effective learning rate.

**The mechanism test.**  CE has one illicit class, so it pulls every gang toward one
common logit direction.  But Definition 7.1 makes ``lambda_min(Gamma)`` the capture
of the hardest gang *after discounting alignment with the others* -- the off-diagonal
of ``Gamma`` **is** the gang-gang overlap, and Prop 7.5 bounds ``lambda_min`` by that
coupling.  So the prediction is specific:

    as beta grows, mean |off-diagonal(Gamma)| (gang-gang alignment) should RISE
    while mean diagonal(Gamma) (per-gang capture C_S) stays healthy,

i.e. lambda_min collapses from *alignment*, not from lost capture.  If instead the
diagonal collapses too, my explanation is wrong and CE is simply destroying capture.

Run::

    python -m src.run_beta_extremes --day-start 24 --transfer-days 10 --out results/beta_ext
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
from src.run_collective_bank_detection import (
    _basis_stack,
    _collective_gamma,
    _filtered_bank,
    _train_gang_m_vhat,
    build_node_split,
    fit_joint_bank_head,
)
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import build_torch_graph, make_patterns, split_train_test

# (name, capture_weight, label_weight)
ARMS = [
    ("beta=0 (capture only)", 1.0, 0.0),
    ("beta=1", 1.0, 1.0),
    ("beta=10", 1.0, 10.0),
    ("beta=100", 1.0, 100.0),
    ("beta=1000", 1.0, 1000.0),
    ("beta=inf (CE only)", 0.0, 1.0),
]


def gamma_structure(data, theta, train_gangs, tau, ridge, degree):
    """Diagnose Gamma: per-gang capture (diagonal) vs gang-gang alignment (off-diag)."""

    Z = _filtered_bank(_basis_stack(data.a_hat, data.X, degree, "chebyshev"), theta)
    m_vhat = _train_gang_m_vhat(data.a_hat, data.adjacency, train_gangs, tau)
    G = _collective_gamma(data.a_hat, Z, m_vhat, ridge, tau).detach()
    m = G.shape[0]
    diag = torch.diagonal(G).clamp(0, 1)
    off = G - torch.diag(torch.diagonal(G))
    mean_off = float(off.abs().sum() / max(m * (m - 1), 1))
    # alignment relative to capture scale: how much of the Gram is off-diagonal mass
    ratio = mean_off / max(float(diag.mean()), 1e-12)
    return (float(diag.mean()), float(diag.min()), mean_off, ratio,
            float(torch.linalg.eigvalsh(G)[0]))


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
    ap.add_argument("--epsilon", type=float, default=0.5)
    ap.add_argument("--ward-num-cuts", type=int, default=150)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/beta_ext", type=Path)
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
    y_node, train_idx, test_idx = build_node_split(
        tr, te, d0.num_nodes, (d0.y == 1), neg_per_pos=args.neg_per_pos, seed=args.seed)
    print(f"\n  {len(tr)} train gangs | {len(train_idx):,} labelled train nodes")

    rows, summ = [], []
    for name, cap_w, lab_w in ARMS:
        print(f"\n########## {name}  (capture_weight={cap_w}, label_weight={lab_w}) ##########")
        theta, _, hm, lam = fit_joint_bank_head(
            d0.a_hat, d0.adjacency, d0.X, y_node, tr, train_idx, test_idx,
            degree=args.degree, epochs=args.epochs, learning_rate=args.learning_rate,
            ridge=args.ridge, tau=args.tau, label_weight=lab_w, seed=args.seed,
            basis="chebyshev", capture_weight=cap_w)
        cap_mean, cap_min, off, ratio, lam_chk = gamma_structure(
            d0, theta, tr, args.tau, args.ridge, args.degree)
        print(f"  lambda_min={lam_chk:.4g}  Gamma diag(mean cap)={cap_mean:.4f} "
              f"min={cap_min:.4f}  |offdiag|={off:.4f}  off/diag={ratio:.3f}  "
              f"head_auc={hm['auc']:.3f}")

        cfg = DetectorConfig(degree=args.degree, tau=args.tau, coarsen_target="bank",
                             coarsening_method="ward-tree", ward_stop="epsilon",
                             epsilon=args.epsilon, ward_num_cuts=args.ward_num_cuts,
                             ridge=args.ridge, threshold=args.threshold, seed=args.seed)
        det = CollectiveBankDetector(cfg)
        det.theta_, det.fit_info_ = theta, None
        for day, D in days.items():
            data, gangs = D["data"], D["gangs"]
            basis = det.target_subspace(data, gangs)
            co, _ = det.coarsen(data, basis, gangs)
            res, _ = evaluate_loukas_patterns(gangs, co.node_to_supernode, data.y,
                                              threshold=args.threshold)
            for gi in range(len(gangs)):
                rows.append({"arm": name, "day": day, "size": D["sizes"][gi],
                             "detected": int(res[gi].detected), "recall": res[gi].recall,
                             "precision": res[gi].precision, "f1": res[gi].f1,
                             "n_coarse": int(co.n_coarse)})
        sub = pd.DataFrame([r for r in rows if r["arm"] == name])
        fan = sub[sub["size"] >= 30]
        rec = {"arm": name, "cap_w": cap_w, "lab_w": lab_w, "lam_min": lam_chk,
               "cap_mean": cap_mean, "cap_min": cap_min, "offdiag": off,
               "off_over_diag": ratio, "head_auc": hm["auc"],
               "detect": sub.detected.mean(),
               "fan": fan.detected.mean() if len(fan) else np.nan,
               "precision": sub.precision.mean(), "recall": sub.recall.mean(),
               "f1": sub.f1.mean()}
        summ.append(rec)
        print(f"  -> TRANSFER detect={rec['detect']:.1%} prec={rec['precision']:.3f} "
              f"f1={rec['f1']:.3f}")

    df = pd.DataFrame(rows); df.to_csv(args.out / "beta_ext_per_gang.csv", index=False)
    S = pd.DataFrame(summ); S.to_csv(args.out / "beta_ext_summary.csv", index=False)

    print("\n" + "=" * 112)
    print("BETA EXTREMES + WHY CE FIGHTS LAMBDA_MIN")
    print("=" * 112)
    print(f"{'arm':<22}{'lam_min':>10}{'cap_mean':>10}{'cap_min':>9}{'|offdiag|':>11}"
          f"{'off/diag':>10}{'head_auc':>10}{'detect':>9}{'prec':>8}{'f1':>7}")
    print("-" * 112)
    for _, r in S.iterrows():
        print(f"{r.arm:<22}{r.lam_min:>10.4g}{r.cap_mean:>10.4f}{r.cap_min:>9.4f}"
              f"{r.offdiag:>11.4f}{r.off_over_diag:>10.3f}{r.head_auc:>10.3f}"
              f"{r.detect:>9.1%}{r.precision:>8.3f}{r.f1:>7.3f}")
    print("\nMECHANISM: if CE works by ALIGNING gangs onto one class direction, then as")
    print("beta grows |offdiag| / off_over_diag should RISE while cap_mean stays healthy.")

    print("\n=== per-day detection ===")
    print(df.pivot_table(index="day", columns="arm", values="detected",
                         aggfunc="mean").round(3).to_string())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(S))
    a1.bar(x - 0.2, S.detect, 0.4, label="detection", color="tab:blue")
    a1.bar(x + 0.2, S.precision, 0.4, label="precision", color="tab:green")
    a1.set_xticks(x); a1.set_xticklabels(S.arm, rotation=20, ha="right", fontsize=7)
    a1.set_ylim(0, 1.05); a1.grid(axis="y", alpha=0.3); a1.legend()
    a1.set_title("Transfer detection across the beta range")
    a2.plot(x, S.cap_mean, "o-", label="Γ diagonal = per-gang capture")
    a2.plot(x, S.offdiag, "s-", label="Γ |off-diagonal| = gang-gang alignment")
    a2.plot(x, S.lam_min, "^-", label="λ_min")
    a2.set_yscale("log"); a2.set_xticks(x)
    a2.set_xticklabels(S.arm, rotation=20, ha="right", fontsize=7)
    a2.grid(alpha=0.3); a2.legend(fontsize=8)
    a2.set_title("Why: does CE align the gangs onto one direction?")
    fig.tight_layout(); fig.savefig(args.out / "beta_ext.png", dpi=150); plt.close(fig)
    print(f"\nCSV + plot -> {args.out}")


if __name__ == "__main__":
    main()
