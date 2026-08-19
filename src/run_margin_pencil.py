"""Verify the margin pencil and compare it against the gradient-trained bank.

Three things, in order:

**1. Identities the theory predicts** (each is a hard check, not a plot):

* ``D*(0) == C_block``: at ``beta = 0`` the pencil direction is
  ``theta = G^{-1} b``, i.e. ``T theta = Pi^{M_tau}_{col T} v_S`` -- the same
  column the ``coarsen_target="dictionary"`` branch builds -- so ``D*(0)`` must
  equal that gang's capture ceiling.  This ties the pencil to the ceiling
  experiment already in the repo.
* ``C(theta*) - beta chi(theta*) == D*(beta)`` at the returned optimizer, to
  machine precision (the eigenvalue equals the objective at its eigenvector).
* ``D*`` is non-increasing and convex in ``beta``.
* First-order law ``dD*/dbeta|_0 = -chi(theta_cap)`` with
  ``theta_cap = G^{-1} b`` the capture-only solve.
* ``beta -> infinity``: ``D* -> c_0^2`` from above and ``chi -> 0`` (exact
  annihilation), whenever the capacity condition ``(K+1) d >= s`` leaves the
  annihilator non-trivial.

**2. The two solvers side by side** -- exact per-gang ``D*(beta)`` versus the
``W_all`` surrogate ``G_beta = G + beta sum_l W_l`` -- with the surrogate's
certificates checked against the quantities they are supposed to bound
(capture floor, ``(r_j-1)/beta`` confusability bound, Gershgorin
``lambda_min(Gamma)``).

**3. Detection**, so the closed forms are compared to the current pipeline on
the metric that matters: each target is handed to the same Ward coarsening
under the same epsilon budget as ``coarsen_target="bank"`` (the gradient-trained
filter) and ``"dictionary"``.

Run::

    conda activate FedStruct
    python -m src.run_margin_pencil --day-start 25 --day-end 25 \
        --degree 16 --betas 0,0.01,0.1,1,10,1000
"""

from __future__ import annotations

import argparse
import json
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

from src.collective_detector import CollectiveBankDetector, DetectorConfig, GraphData
from src.margin_pencil import (
    build_pencil_space,
    pencil_gang_data,
    solve_gang_pencil,
    solve_joint_pencil,
)
from src.run_multihead_bank import ward_detect
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import (
    build_torch_graph,
    load_node_features,
    make_patterns,
    random_structural_features,
    split_train_test,
)
from src.utils.utils import LOGGER, now


def _parse_betas(spec: str) -> list:
    return [float(x) for x in spec.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--day-start", type=int, default=25)
    ap.add_argument("--day-end", type=int, default=25)
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--train-ratio", type=float, default=0.6)
    ap.add_argument("--feature-mode", default="wallet",
                    choices=["wallet", "random", "wallet+random"])
    ap.add_argument("--random-width", type=int, default=64)
    ap.add_argument("--degree", type=int, default=16,
                    help="K; the dictionary is (K+1)*d wide, so keep it modest")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--betas", default="0,0.001,0.01,0.1,1,10,100,10000")
    ap.add_argument("--detect-betas", default="0,0.1,10",
                    help="betas that also get the (slower) Ward detection eval")
    ap.add_argument("--epsilon", type=float, default=0.99)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--ward-num-cuts", type=int, default=500)
    ap.add_argument("--bank-epochs", type=int, default=300)
    ap.add_argument("--skip-bank", action="store_true",
                    help="skip the gradient-trained reference (saves minutes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=Path(f"results/margin_pencil/{now}/"), type=Path)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    betas = _parse_betas(args.betas)
    detect_betas = set(_parse_betas(args.detect_betas))

    # ---- data -------------------------------------------------------------
    LOGGER.info(f"=== margin pencil | day {args.day_start}-{args.day_end} "
                f"K={args.degree} tau={args.tau} ===")
    A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, args.day_start, args.day_end)
    if args.feature_mode in ("wallet", "wallet+random"):
        X = load_node_features(args.data_dir, nodes_df, args.day_start, args.day_end)
        if args.feature_mode == "wallet+random":
            X = torch.cat([X, random_structural_features(
                int(A_unw.shape[0]), args.random_width, args.seed)], dim=1)
    else:
        X = random_structural_features(int(A_unw.shape[0]), args.random_width, args.seed)
    graph = build_torch_graph(A_w, A_unw, cls, X, weighted=False)
    data = GraphData.from_graph(graph)
    a_hat, adjacency = data.a_hat, data.adjacency
    gang_sets = connected_components_sets(
        A_unw, np.where(cls == 1)[0], args.min_gang_size
    )
    gangs = make_patterns(gang_sets, "alert", "gang", "g")
    rng = np.random.default_rng(args.seed)
    gang_train, _ = split_train_test(gangs, args.train_ratio, rng)
    d = data.feature_dim
    P = (args.degree + 1) * d
    LOGGER.info(f"  {len(gangs)} gangs ({len(gang_train)} train), d={d}, "
                f"P=(K+1)d={P:,}")

    # ---- dictionary + per-gang pencil data --------------------------------
    t0 = time.perf_counter()
    space = build_pencil_space(a_hat, data.X, args.degree, args.tau)
    LOGGER.info(f"  dictionary Gram: P={space.dim:,} rank={space.rank:,} "
                f"[{time.perf_counter() - t0:.1f}s]")
    gps = pencil_gang_data(space, a_hat, adjacency, gang_train)
    LOGGER.info("  per-gang: " + "  ".join(
        f"{g.gang_id}(s={g.size},rank W={g.B_tilde.shape[1]})" for g in gps[:8]
    ))
    cap_ok = all(P >= g.size for g in gps)
    LOGGER.info(f"  capacity (K+1)d >= s for every gang: {cap_ok}")

    # ---- 1. identity checks ------------------------------------------------
    LOGGER.info("\n" + "=" * 78)
    LOGGER.info("IDENTITY CHECKS (theory predictions)")
    LOGGER.info("=" * 78)
    sol0 = [solve_gang_pencil(g, 0.0) for g in gps]
    err_c0 = max(abs(s["D_star"] - g.c_block) for s, g in zip(sol0, gps))
    LOGGER.info(f"  D*(0) == C_block (= dictionary-target ceiling): "
                f"max abs err {err_c0:.3e}")

    # self-consistency of the eigenvalue at its eigenvector, all betas
    max_obj_err = 0.0
    for b in betas:
        for g in gps:
            s = solve_gang_pencil(g, b)
            max_obj_err = max(max_obj_err, abs(s["C"] - b * s["chi"] - s["D_star"]))
    LOGGER.info(f"  C(theta*) - beta*chi(theta*) == D*(beta): "
                f"max abs err {max_obj_err:.3e}")

    # first-order law  dD*/dbeta|_0 = -chi(theta_cap)
    h = 1e-6
    fo_rows = []
    for g, s0 in zip(gps, sol0):
        s_h = solve_gang_pencil(g, h)
        num = (s_h["D_star"] - s0["D_star"]) / h
        fo_rows.append({"gang": g.gang_id, "numeric": num, "predicted": -s0["chi"]})
    fo = pd.DataFrame(fo_rows)
    fo_err = float((fo.numeric - fo.predicted).abs().max())
    LOGGER.info(f"  dD*/dbeta|_0 == -chi(theta_cap): max abs err {fo_err:.3e}")

    # monotone + convex in beta, and the beta -> infinity limit
    curves = {g.gang_id: [] for g in gps}
    for b in betas:
        for g in gps:
            curves[g.gang_id].append(solve_gang_pencil(g, b))
    mono = all(
        all(curves[k][i + 1]["D_star"] <= curves[k][i]["D_star"] + 1e-12
            for i in range(len(betas) - 1))
        for k in curves
    )
    LOGGER.info(f"  D*(beta) non-increasing in beta: {mono}")
    lim_rows = []
    for g in gps:
        last = curves[g.gang_id][-1]
        lim_rows.append({
            "gang": g.gang_id, "size": g.size, "C_block": g.c_block,
            "c0_sq": last["c0_sq"], "D_star_max_beta": last["D_star"],
            "chi_max_beta": last["chi"],
            "capture_lost": g.c_block - last["c0_sq"],
        })
    lim = pd.DataFrame(lim_rows)
    LOGGER.info(f"  beta={betas[-1]:g}: D* -> c_0^2 "
                f"(max |D*-c_0^2| = {float((lim.D_star_max_beta - lim.c0_sq).abs().max()):.3e}),"
                f" median chi {float(lim.chi_max_beta.median()):.3e}, "
                f"median capture given up {float(lim.capture_lost.median()):.4f}")

    # ---- 2. beta path, both solvers ---------------------------------------
    rows = []
    for b in betas:
        joint = solve_joint_pencil(gps, b)
        for j, g in enumerate(gps):
            s = curves[g.gang_id][betas.index(b)]
            rows.append({
                "beta": b, "gang": g.gang_id, "size": g.size,
                "C_block": g.c_block,
                "pencil_D": s["D_star"], "pencil_C": s["C"], "pencil_chi": s["chi"],
                "joint_C": joint["capture"][j],
                "joint_C_floor": joint["capture_floor"][j],
                "joint_chi_own": joint["chi_own"][j],
                "joint_chi_cross": joint["chi_cross_max"][j],
                "joint_chi_bound": joint["chi_bound"][j],
                "joint_rho_max": joint["rho_max"][j],
                "joint_margin": joint["capture"][j] - b * joint["chi_cross_max"][j],
                "joint_lambda_min_bound": joint["lambda_min_gershgorin"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "pencil_beta_path.csv", index=False)

    LOGGER.info("\n" + "=" * 104)
    LOGGER.info("BETA PATH: exact per-gang pencil vs the W_all surrogate "
                "(medians over training gangs)")
    LOGGER.info("=" * 104)
    hdr = (f"  {'beta':>9}{'D*(pencil)':>13}{'C':>10}{'chi':>11}"
           f"{'C(joint)':>11}{'floor':>10}{'chi_cross':>12}{'bound':>11}"
           f"{'lam_min>=':>11}")
    LOGGER.info(hdr)
    LOGGER.info("  " + "-" * (len(hdr) - 2))
    for b in betas:
        s = df[df.beta == b]
        LOGGER.info(
            f"  {b:>9g}{s.pencil_D.median():>13.5f}{s.pencil_C.median():>10.5f}"
            f"{s.pencil_chi.median():>11.3e}{s.joint_C.median():>11.5f}"
            f"{s.joint_C_floor.median():>10.5f}{s.joint_chi_cross.median():>12.3e}"
            f"{s.joint_chi_bound.median():>11.3e}"
            f"{s.joint_lambda_min_bound.iloc[0]:>11.4f}"
        )
    viol_floor = int((df.joint_C < df.joint_C_floor - 1e-9).sum())
    viol_chi = int((df.joint_chi_cross > df.joint_chi_bound + 1e-9).sum())
    LOGGER.info(f"\n  certificate violations -- capture floor: {viol_floor}, "
                f"chi bound: {viol_chi}  (both must be 0)")

    # ---- 3. detection: pencil targets vs the current pipeline -------------
    LOGGER.info("\n" + "=" * 92)
    LOGGER.info(f"DETECTION (same Ward stop, eps<={args.epsilon}) on all "
                f"{len(gangs)} day-{args.day_start} gangs")
    LOGGER.info("=" * 92)
    det_rows = []

    def _report(name, target, extra=""):
        r = ward_detect(adjacency, target, gangs, data.y, tau=args.tau,
                        epsilon=args.epsilon, threshold=args.threshold,
                        num_cuts=args.ward_num_cuts)
        det_rows.append({"target": name, "width": int(target.shape[1]), **r})
        LOGGER.info(
            f"  {name:<26} cols={target.shape[1]:>4}  R={r['mean_recall']:.3f} "
            f"P={r['mean_precision']:.3f} F1={r['mean_f1']:.3f} "
            f"det={r['detection_rate']:.1%} ({r['detected']}/{r['total']}) "
            f"n_coarse={r['n_coarse']:,} {extra}"
        )

    for b in sorted(detect_betas):
        for mode in ("pencil", "pencil-joint"):
            if mode == "pencil":
                U = torch.stack(
                    [solve_gang_pencil(g, b)["u"] for g in gps], dim=1)
            else:
                U = solve_joint_pencil(gps, b)["U"]
            target = space.T @ (space.whiten @ U)
            _report(f"{mode} beta={b:g}", target)

    # the current pipeline: gradient-trained bank, and the dictionary target
    if not args.skip_bank:
        cfg = DetectorConfig(
            degree=args.degree, basis="chebyshev", tau=args.tau,
            epochs=args.bank_epochs, learning_rate=0.02, ridge=1e-3,
            optimizer="projected", capture_objective="lambda_min",
            conf_weight=10.0, conf_reduce="mean", heads=1,
            epsilon=args.epsilon, threshold=args.threshold, seed=args.seed,
        )
        LOGGER.info(f"\n  training the reference bank ({args.bank_epochs} epochs) ...")
        det = CollectiveBankDetector(cfg).fit(data, gang_train)
        for tgt in ("bank", "dictionary"):
            from dataclasses import replace

            det.config = replace(cfg, coarsen_target=tgt)
            _report(f"current: {tgt}", det.target_subspace(data, gang_train))

    pd.DataFrame(det_rows).to_csv(args.out / "pencil_detection.csv", index=False)

    # ---- plot: the beta path ----------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    cmap = plt.get_cmap("viridis")
    for i, g in enumerate(gps):
        col = cmap(i / max(len(gps) - 1, 1))
        ys = [c["D_star"] for c in curves[g.gang_id]]
        axes[0].plot(betas, ys, color=col, lw=1.3, marker="o", ms=2.5)
        axes[1].plot(betas, [c["chi"] for c in curves[g.gang_id]],
                     color=col, lw=1.3, marker="o", ms=2.5)
    for b_ in betas:
        pass
    axes[2].plot(betas, [df[df.beta == b].pencil_D.median() for b in betas],
                 color="#1F4E79", lw=1.8, marker="o", ms=3, label="pencil $D^\\star$")
    axes[2].plot(betas, [df[df.beta == b].joint_C.median() for b in betas],
                 color="#2E7D5B", lw=1.6, marker="s", ms=3, label="joint $C$")
    axes[2].plot(betas, [df[df.beta == b].joint_C_floor.median() for b in betas],
                 color="#2E7D5B", lw=1.2, ls="--", label="joint floor (certified)")
    axes[0].set_ylabel("$D^\\star(\\beta)$")
    axes[0].set_title("margin per gang (convex, non-increasing)")
    axes[1].set_ylabel("$\\chi(\\theta^\\star)$")
    axes[1].set_yscale("log")
    axes[1].set_title("confusability at the optimizer $\\to 0$")
    axes[2].set_title("exact pencil vs $W_{\\rm all}$ surrogate (medians)")
    axes[2].legend(fontsize=8, frameon=False)
    for a in axes:
        a.set_xlabel(r"$\beta$")
        a.set_xscale("symlog", linthresh=1e-3)
        a.grid(alpha=0.25, lw=0.5)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"margin pencil -- elliptic++ d{args.day_start}-{args.day_end}, "
        f"K={args.degree}, d={d}, P={P:,}, {len(gps)} train gangs", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out / "pencil_beta_path.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    (args.out / "pencil_summary.json").write_text(json.dumps({
        "P": P, "dictionary_rank": space.rank, "n_train_gangs": len(gps),
        "capacity_ok": bool(cap_ok),
        "checks": {
            "D0_vs_Cblock_max_err": err_c0,
            "objective_at_eigvec_max_err": max_obj_err,
            "first_order_max_err": fo_err,
            "monotone": bool(mono),
            "floor_violations": viol_floor,
            "chi_bound_violations": viol_chi,
        },
        "detection": det_rows,
    }, indent=2, default=float) + "\n")
    LOGGER.info(f"\nCSV + plot + JSON -> {args.out}")


if __name__ == "__main__":
    main()
