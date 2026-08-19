"""TAU-SWEEP: does the screened metric remove the fan/star (m_bar1) bottleneck?

The pooled transfer analysis (:mod:`src.analyze_missed_gangs`) showed the misses are
governed by the boundary-edge mean ``m_bar1``: gangs with high ``m_bar1`` (sparse
fan/star components) are never detected, and each day's detection rate tracks that
day's median ``m_bar1`` at ``r = -0.84``.  That is exactly the paper's structural
failure mode -- and the paper's claimed cure is the screened metric:

    m_bar1^tau  =  Phi (m_bar1 + tau) / (Phi + tau)   ->   Phi     as tau grows,

"so a single estimable parameter makes capture conductance-only, uniformly over
motif types".  If that holds, then as ``tau`` grows the detection outcome should
*stop* depending on ``m_bar1`` and depend only on ``Phi``.

This script trains one bank per ``tau`` on ``--day-start``, freezes it, applies it to
every day in the transfer window (all graphs cached once), and reports per ``tau``:

* ``AUC(m_bar1)``  -- should move toward 0.5 (no m_bar1 dependence) if screening works
* ``r(day detection rate, day median m_bar1)`` -- should collapse toward 0
* detection rate overall and for the sparse/fan gangs (size>=30, density<0.15)
* ``lambda_min`` and median capture (does screening also lift capture?)

Run::

    python -m src.run_tau_sweep --day-start 24 --transfer-days 10 \
        --taus 0,0.5,1,2,5,10 --out results/tau_sweep
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
from sklearn.metrics import roc_auc_score

from src.analyze_missed_gangs import (
    cached_wallet_features,
    gang_moments,
    gang_structure,
)
from src.collective_detector import CollectiveBankDetector, DetectorConfig, GraphData
from src.loukas_sgc_detection import evaluate_loukas_patterns
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import (
    build_torch_graph,
    make_patterns,
    split_train_test,
)
from src.run_elliptic_modular import random_structural_features


def load_days(args):
    """Build + cache every day's graph, gangs and tau-independent statistics once."""

    days = {}
    cols = None
    for k in range(args.transfer_days + 1):
        day = args.day_start + k
        A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, day, day)
        if args.feature_mode == "wallet":
            if cols is None:
                X, cols = cached_wallet_features(
                    args.data_dir, nodes_df, day, return_columns=True
                )
            else:
                X = cached_wallet_features(
                    args.data_dir, nodes_df, day, keep_columns=cols
                )
        else:
            X = random_structural_features(
                int(A_unw.shape[0]), args.random_width, args.seed + day
            )
        g = build_torch_graph(A_w, A_unw, cls, X, weighted=False)
        gsets = connected_components_sets(
            A_unw, np.where(cls == 1)[0], args.min_gang_size
        )
        if not gsets:
            continue
        gangs = make_patterns(gsets, "alert", "gang", "g")
        data = GraphData.from_graph(g)
        gang_of = torch.full((data.num_nodes,), -1, dtype=torch.long)
        for gi, S in enumerate(gsets):
            gang_of[torch.as_tensor(list(S), dtype=torch.long)] = gi
        # tau-independent structural stats (L-moments, density, star-ness)
        phi, mbar1 = gang_moments(data.a_hat, data.adjacency, gangs)
        struct = gang_structure(g.edge_index, gang_of, len(gangs), data.num_nodes)
        days[day] = {
            "data": data,
            "gangs": gangs,
            "phi": phi.numpy(),
            "mbar1": mbar1.numpy(),
            "struct": struct,
        }
        print(f"  cached day {day}: N={data.num_nodes:,} gangs={len(gangs)}")
    return days, cols


def run_tau(tau, days, args, train_day):
    """Train on ``train_day`` at this ``tau``, apply frozen to every cached day."""

    d0 = days[train_day]
    tr, _ = split_train_test(
        d0["gangs"], args.train_ratio, np.random.default_rng(args.seed)
    )
    cfg = DetectorConfig(
        degree=args.degree,
        tau=tau,
        epochs=args.epochs,
        conf_weight=args.conf_weight,
        conf_reduce="mean",
        optimizer="riemannian",
        coarsen_target="bank",
        coarsening_method="ward-tree",
        ward_stop="epsilon",
        epsilon=args.epsilon,
        ward_num_cuts=args.ward_num_cuts,
        threshold=args.threshold,
        seed=args.seed,
    )
    det = CollectiveBankDetector(cfg)
    det.fit(d0["data"], tr)
    rows = []
    for day, D in days.items():
        data, gangs = D["data"], D["gangs"]
        basis = det.target_subspace(data, gangs)
        coarsening, _ = det.coarsen(data, basis, gangs)
        res, _ = evaluate_loukas_patterns(
            gangs, coarsening.node_to_supernode, data.y, threshold=cfg.threshold
        )
        cap = det.capture(data, gangs)["per_gang_capture"]
        for gi in range(len(gangs)):
            phi, mb = float(D["phi"][gi]), float(D["mbar1"][gi])
            rows.append(
                {
                    "tau": tau,
                    "day": day,
                    "size": D["struct"][gi]["size"],
                    "Phi": phi,
                    "mbar1": mb,
                    # screened energy mean (eq. 6): interpolates m_bar1 -> Phi as tau grows
                    "mbar1_tau": (
                        phi * (mb + tau) / (phi + tau) if (phi + tau) > 0 else mb
                    ),
                    "capture": float(cap[gi]),
                    "density": D["struct"][gi]["density"],
                    "starness": D["struct"][gi]["starness"],
                    "detected": int(res[gi].detected),
                    "recall": res[gi].recall,
                    "precision": res[gi].precision,
                }
            )
    return rows, float(det.fit_info_["objective"])


def summarize(df: pd.DataFrame, lam: float, tau: float) -> dict:
    per_day = df.groupby("day").agg(
        det_rate=("detected", "mean"), med_mbar1=("mbar1", "median")
    )
    r = (
        float(per_day["det_rate"].corr(per_day["med_mbar1"]))
        if len(per_day) > 2
        else np.nan
    )

    def auc(col):
        try:
            return float(roc_auc_score(df.detected, df[col]))
        except Exception:
            return float("nan")

    fan = df[(df["size"] >= 30)]
    sparse = df[df.density < 0.15]
    return {
        "tau": tau,
        "lambda_min": lam,
        "detection_rate": float(df.detected.mean()),
        "n_gangs": int(len(df)),
        "auc_mbar1": auc("mbar1"),
        "auc_mbar1_tau": auc("mbar1_tau"),
        "auc_Phi": auc("Phi"),
        "auc_density": auc("density"),
        "r_day_mbar1": r,
        "det_rate_size30plus": float(fan.detected.mean()) if len(fan) else np.nan,
        "n_size30plus": int(len(fan)),
        "det_rate_sparse": float(sparse.detected.mean()) if len(sparse) else np.nan,
        "median_capture": float(df.capture.median()),
    }


def plot_sweep(S: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    t = S.tau
    axes[0].plot(t, S.detection_rate, "o-", label="all gangs")
    axes[0].plot(
        t, S.det_rate_size30plus, "s--", color="tab:red", label="size≥30 (fans)"
    )
    axes[0].plot(t, S.det_rate_sparse, "^--", color="tab:orange", label="density<0.15")
    axes[0].set_xlabel("τ")
    axes[0].set_ylabel("detection rate")
    axes[0].set_title("Detection vs screening τ")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(-0.02, 1.0)

    axes[1].axhline(0.5, color="k", ls=":", lw=1, label="0.5 = no dependence")
    axes[1].plot(t, S.auc_mbar1, "o-", color="tab:purple", label=r"AUC($\bar m_1$)")
    axes[1].plot(t, S.auc_Phi, "s-", color="tab:blue", label=r"AUC($\Phi$)")
    axes[1].set_xlabel("τ")
    axes[1].set_ylabel("AUC for predicting detection")
    axes[1].set_title(r"Does detection stop depending on $\bar m_1$?")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].axhline(0, color="k", ls=":", lw=1)
    axes[2].plot(t, S.r_day_mbar1, "o-", color="tab:red")
    axes[2].set_xlabel("τ")
    axes[2].set_ylabel(r"$r$(day detection, day median $\bar m_1$)")
    axes[2].set_title("Per-day swing explained by $\\bar m_1$\n(→ 0 = cured)")
    axes[2].grid(alpha=0.3)

    ax2 = axes[3]
    ax2.plot(t, S.median_capture, "o-", color="tab:green", label="median capture $C_S$")
    ax2.set_xlabel("τ")
    ax2.set_ylabel("median capture", color="tab:green")
    ax3 = ax2.twinx()
    ax3.plot(t, S.lambda_min, "s--", color="tab:gray", label=r"$\lambda_{\min}$")
    ax3.set_ylabel(r"$\lambda_{\min}(\Gamma)$", color="tab:gray")
    ax2.set_title("Capture vs τ")
    ax2.grid(alpha=0.3)
    fig.suptitle(
        "TAU-SWEEP: does screening remove the fan/star bottleneck?", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--day-start", type=int, default=24)
    ap.add_argument("--transfer-days", type=int, default=10)
    ap.add_argument("--taus", default="0,0.5,1,2,5,10")
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--feature-mode", choices=["wallet", "random"], default="wallet")
    ap.add_argument("--random-width", type=int, default=64)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--degree", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--conf-weight", type=float, default=10.0)
    ap.add_argument("--epsilon", type=float, default=0.5)
    ap.add_argument("--ward-num-cuts", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/tau_sweep", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    taus = [float(x) for x in str(args.taus).split(",") if x != ""]
    print(
        f"=== caching days {args.day_start}..{args.day_start + args.transfer_days} ==="
    )
    days, _ = load_days(args)

    all_rows, summ = [], []
    for tau in taus:
        print(f"\n=== tau = {tau:g} ===")
        rows, lam = run_tau(tau, days, args, args.day_start)
        df = pd.DataFrame(rows)
        all_rows.extend(rows)
        s = summarize(df, lam, tau)
        summ.append(s)
        print(
            f"  lambda_min={lam:.4g}  detection={s['detection_rate']:.1%}  "
            f"AUC(mbar1)={s['auc_mbar1']:.2f}  r_day={s['r_day_mbar1']:+.2f}  "
            f"fans(size≥30)={s['det_rate_size30plus']:.1%}  cap={s['median_capture']:.4f}"
        )

    S = pd.DataFrame(summ)
    pd.DataFrame(all_rows).to_csv(args.out / "tau_sweep_per_gang.csv", index=False)
    S.to_csv(args.out / "tau_sweep_summary.csv", index=False)
    plot_sweep(S, args.out / "tau_sweep.png")

    print("\n" + "=" * 96)
    print(
        "TAU-SWEEP  (theory: as tau grows, m_bar1^tau -> Phi, so AUC(m_bar1) -> 0.5 and r_day -> 0)"
    )
    print("=" * 96)
    print(
        f"{'tau':>6}{'lam_min':>10}{'detect':>9}{'AUC(mbar1)':>12}{'AUC(Phi)':>10}"
        f"{'r_day(mbar1)':>14}{'fans>=30':>10}{'sparse':>9}{'capture':>10}"
    )
    print("-" * 96)
    for _, r in S.iterrows():
        print(
            f"{r.tau:>6g}{r.lambda_min:>10.3g}{r.detection_rate:>9.1%}{r.auc_mbar1:>12.2f}"
            f"{r.auc_Phi:>10.2f}{r.r_day_mbar1:>14.2f}{r.det_rate_size30plus:>10.1%}"
            f"{r.det_rate_sparse:>9.1%}{r.median_capture:>10.4f}"
        )
    (args.out / "summary.json").write_text(
        json.dumps(summ, indent=2, default=str) + "\n"
    )
    print(f"\nCSV + plot -> {args.out}")


if __name__ == "__main__":
    main()
