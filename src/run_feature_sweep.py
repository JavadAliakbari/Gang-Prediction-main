"""Feature/capture sweep + the untrained-filter control, on the Elliptic++ days.

Two questions, crossed into one experiment matrix:

1. **Does feature reach set the capture floor?**  Theorem 6.5: the bank can only
   re-weight bands its features occupy (``max C_S -> nu^tau(reach(X))``), so if the
   55-dim wallet features do not reach the gang modes, no amount of filter learning
   helps.  Arms: ``wallet`` (55) / ``random-N`` (isotropic range finder) /
   ``wallet+struct`` (concat).

2. **Does the training do anything at all?**  For every feature mode we also run an
   **untrained** control: the same pipeline with ``Theta`` left at its uniform
   initialization (a flat filter over all K+1 hops) -- no gradient steps.  If the
   untrained arm matches the trained arm, the learned filter is contributing
   nothing and the detections are riding on trivial small-gang collapse.

Everything downstream is identical (frozen filter -> target subspace -> label-free
epsilon-budget Ward cut -> score), so the arms are directly comparable to the
tau/epsilon/family sweep tables.

Run::

    python -m src.run_feature_sweep --day-start 24 --transfer-days 10 \
        --out results/feature_sweep
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


def load_days_multifeature(args):
    """Cache each day's graph, gangs and *all three* feature variants once."""

    days, cols = {}, None
    for k in range(args.transfer_days + 1):
        day = args.day_start + k
        A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, day, day)
        gsets = connected_components_sets(
            A_unw, np.where(cls == 1)[0], args.min_gang_size
        )
        if not gsets:
            continue
        if cols is None:
            Xw, cols = cached_wallet_features(
                args.data_dir, nodes_df, day, return_columns=True
            )
        else:
            Xw = cached_wallet_features(args.data_dir, nodes_df, day, keep_columns=cols)
        n = int(A_unw.shape[0])
        Xr = random_structural_features(n, args.random_width, args.seed)
        Xb = torch.cat(
            [Xw, random_structural_features(n, args.struct_width, args.seed)], 1
        )
        feats = {"wallet": Xw, "random": Xr, "both": Xb}
        # graph operators are feature-independent: build once, swap X per arm
        g = build_torch_graph(A_w, A_unw, cls, Xw, weighted=False)
        base = GraphData.from_graph(g)
        gangs = make_patterns(gsets, "alert", "gang", "g")
        gang_of = torch.full((base.num_nodes,), -1, dtype=torch.long)
        for gi, S in enumerate(gsets):
            gang_of[torch.as_tensor(list(S), dtype=torch.long)] = gi
        phi, mbar1 = gang_moments(base.a_hat, base.adjacency, gangs)
        struct = gang_structure(g.edge_index, gang_of, len(gangs), base.num_nodes)
        days[day] = {
            "base": base,
            "feats": feats,
            "gangs": gangs,
            "struct": struct,
            "phi": phi.numpy(),
            "mbar1": mbar1.numpy(),
        }
        print(
            f"  cached day {day}: N={base.num_nodes:,} gangs={len(gangs)} "
            f"dims wallet={Xw.shape[1]} random={Xr.shape[1]} both={Xb.shape[1]}"
        )
    return days


def data_for(D, mode):
    """GraphData for this day with the arm's feature matrix swapped in."""

    b = D["base"]
    X = D["feats"][mode].to(dtype=b.a_hat.dtype)
    return GraphData(
        edge_index=b.edge_index, a_hat=b.a_hat, adjacency=b.adjacency, X=X, y=b.y
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--day-start", type=int, default=24)
    ap.add_argument("--transfer-days", type=int, default=10)
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--random-width", type=int, default=128)
    ap.add_argument(
        "--struct-width", type=int, default=64, help="structural cols in wallet+struct"
    )
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--degree", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--conf-weight", type=float, default=10.0)
    ap.add_argument("--epsilon", type=float, default=0.5)
    ap.add_argument("--ward-num-cuts", type=int, default=150)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/feature_sweep", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print("=== caching days (3 feature variants each) ===")
    days = load_days_multifeature(args)
    train_day = args.day_start
    D0 = days[train_day]
    tr, _ = split_train_test(
        D0["gangs"], args.train_ratio, np.random.default_rng(args.seed)
    )

    arms = [(m, t) for m in ("wallet", "random", "both") for t in (False, True)]
    rows, summary = [], []
    for mode, trained in arms:
        tag = f"{mode}{'(trained)' if trained else '(UNTRAINED)'}"
        cfg = DetectorConfig(
            degree=args.degree,
            tau=args.tau,
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
        d0 = data_for(D0, mode)
        print(f"\n=== arm {tag} (d={d0.feature_dim}) ===")
        if trained:
            det.fit(d0, tr)
            lam_i = det.fit_info_["init_objective"]
            lam_f = det.fit_info_["objective"]
        else:
            # uniform init: a flat filter over all K+1 hops, zero gradient steps
            th = torch.ones(cfg.degree + 1, d0.feature_dim, dtype=d0.X.dtype)
            det.theta_ = th / th.norm(dim=0, keepdim=True)
            det.fit_info_ = None
            lam_i = lam_f = float(det.capture(d0, tr)["lambda_min_gamma"])
        print(f"  lambda_min {lam_i:.4g} -> {lam_f:.4g}")

        for day, D in days.items():
            data, gangs = data_for(D, mode), D["gangs"]
            basis = det.target_subspace(data, gangs)
            co, _ = det.coarsen(data, basis, gangs)
            res, _ = evaluate_loukas_patterns(
                gangs, co.node_to_supernode, data.y, threshold=args.threshold
            )
            cap = det.capture(data, gangs)["per_gang_capture"]
            for gi in range(len(gangs)):
                rows.append(
                    {
                        "mode": mode,
                        "trained": int(trained),
                        "arm": tag,
                        "day": day,
                        "size": D["struct"][gi]["size"],
                        "density": D["struct"][gi]["density"],
                        "capture": float(cap[gi]),
                        "detected": int(res[gi].detected),
                        "recall": res[gi].recall,
                        "precision": res[gi].precision,
                        "f1": res[gi].f1,
                        "n_coarse": int(co.n_coarse),
                    }
                )
        d = pd.DataFrame([r for r in rows if r["arm"] == tag])
        fan, sp = d[d["size"] >= 30], d[d["density"] < 0.15]
        summary.append(
            {
                "arm": tag,
                "mode": mode,
                "trained": int(trained),
                "d": d0.feature_dim,
                "lam_init": lam_i,
                "lam_final": lam_f,
                "capture": float(d.capture.median()),
                "detect": float(d.detected.mean()),
                "fan_detect": float(fan.detected.mean()) if len(fan) else np.nan,
                "sparse_detect": float(sp.detected.mean()) if len(sp) else np.nan,
                "precision": float(d.precision.mean()),
                "f1": float(d.f1.mean()),
            }
        )
        s = summary[-1]
        print(
            f"  -> detect={s['detect']:.1%} fan={s['fan_detect']:.1%} "
            f"prec={s['precision']:.3f} capture={s['capture']:.4f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "feature_sweep_per_gang.csv", index=False)
    S = pd.DataFrame(summary)
    S.to_csv(args.out / "feature_sweep_summary.csv", index=False)

    print("\n" + "=" * 104)
    print("FEATURE / CAPTURE SWEEP  +  UNTRAINED CONTROL")
    print("=" * 104)
    print(
        f"{'arm':<22}{'d':>5}{'lam_min':>11}{'capture':>10}{'detect':>9}"
        f"{'fan(>=30)':>11}{'sparse':>9}{'precision':>11}{'f1':>8}"
    )
    print("-" * 104)
    for _, r in S.iterrows():
        print(
            f"{r.arm:<22}{r.d:>5.0f}{r.lam_final:>11.4g}{r.capture:>10.4f}"
            f"{r.detect:>9.1%}{r.fan_detect:>11.1%}{r.sparse_detect:>9.1%}"
            f"{r.precision:>11.3f}{r.f1:>8.3f}"
        )

    print("\nTRAINED vs UNTRAINED (does the learning buy anything?)")
    for mode in ("wallet", "random", "both"):
        a = S[(S["mode"] == mode) & (S.trained == 0)].iloc[0]
        b = S[(S["mode"] == mode) & (S.trained == 1)].iloc[0]
        print(
            f"  {mode:<8} detect {a.detect:.1%} -> {b.detect:.1%} "
            f"({b.detect - a.detect:+.1%})   capture {a.capture:.4f} -> {b.capture:.4f}"
            f"   lam_min {a.lam_final:.3g} -> {b.lam_final:.3g}"
        )

    x = np.arange(len(S))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - 0.2, S.detect, 0.4, label="detection", color="tab:blue")
    ax.bar(x + 0.2, S.precision, 0.4, label="precision", color="tab:green")
    for xi, (_, r) in zip(x, S.iterrows()):
        ax.text(
            xi, 1.02, f"λ={r.lam_final:.1e}\nC={r.capture:.3f}", ha="center", fontsize=7
        )
    ax.set_xticks(x)
    ax.set_xticklabels(S.arm, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("rate")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_title("Feature reach + untrained control (λ_min, capture annotated)")
    fig.tight_layout()
    fig.savefig(args.out / "feature_sweep.png", dpi=150)
    plt.close(fig)
    print(f"\nCSV + plot -> {args.out}")


if __name__ == "__main__":
    main()
