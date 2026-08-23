"""Compare the ``A_eq`` signal-to-confusion pencil with the max-min objective.

The collective target is currently chosen by the max-min certificate
``lambda_min(Gamma) - beta * max_j chi_j``.  That expression is the right thing
to *report*, but a poor thing to *optimize*: ``lambda_min`` is identically zero
once ``m > d``, both spectral extrema are nonsmooth at multiplicities, a single
noisy gang dominates every update, and there is no closed form.  The alternative
scored here replaces it with one generalized eigenproblem

    A_eq w_k = lambda_k (H + rho I) w_k,
    A_eq = Bbar (Bbar^T Bbar + alpha I)^{-1} Bbar^T,
    H    = sum_j omega_j  bar H_j / (|S_j| - 1),

whose top-``d`` eigenspace maximizes ``tr(W^T A_eq W)`` subject to
``W^T (H + rho I) W = I`` -- captured group energy per unit internal
fluctuation, a regularized Fisher discriminant on the whitened dictionary
(:mod:`src.margin_pencil`).

Every solver is judged on *identical* ground:

* the same graph, the same train/test gang split, the same coarsener and budget;
* the same certificate, computed on the ``(N, d)`` target subspace the coarsener
  is actually handed (:func:`src.margin_pencil.basis_subspace_report`), so the
  gradient bank and the two closed forms are read off the same statistic;
* the same downstream detection metrics, on the training graph and on frozen
  transfer days.

Run::

    conda activate FedStruct
    python -m src.run_elliptic_aeq_comparison --train-days 26 --transfer-days 3
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
from src.margin_pencil import basis_subspace_report, community_patterns
from src.run_collective_bank_detection import _m_apply, _train_gang_m_vhat
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import (
    build_torch_graph,
    load_node_features,
    make_patterns,
    random_structural_features,
    split_train_test,
)
from src.utils.utils import LOGGER, now

# name -> DetectorConfig overrides.  Everything not listed is shared, so the
# only thing that varies down the table is how the target subspace was chosen.
# ``_communities`` is a runner option, not a DetectorConfig field: it asks for
# that many LABEL-FREE community indicators to be added to A_eq's signal
# operator.  A_eq's rank is the number of signal groups, so it is simultaneously
# what lets the target exceed m columns and what stops col(A_eq) from being
# exactly the training gangs' indicator span.
CONFIGS = [
    ("gradient lambda_min", dict(collective_solver="gradient")),
    (
        "gradient lambda_min + chi",
        dict(collective_solver="gradient", conf_weight=1.0),
    ),
    ("closed-form beta=0", dict(collective_solver="closed-form", pencil_beta=0.0)),
    ("closed-form beta=1", dict(collective_solver="closed-form", pencil_beta=1.0)),
    ("A_eq d=m rho=1e-3", dict(collective_solver="aeq", aeq_rho=1e-3)),
    ("A_eq d=m rho=1e-1", dict(collective_solver="aeq", aeq_rho=1e-1)),
    ("A_eq d=m rho=1", dict(collective_solver="aeq", aeq_rho=1.0)),
    ("A_eq d=m rho=10", dict(collective_solver="aeq", aeq_rho=10.0)),
    (
        "A_eq d=m rho=1 rewt",
        dict(collective_solver="aeq", aeq_rho=1.0, aeq_reweight_iters=5,
             aeq_kappa=50.0),
    ),
    # signal enrichment: same solver, wider Bbar
    ("A_eq d=m +comm rho=1",
     dict(collective_solver="aeq", aeq_rho=1.0, _communities=128)),
    ("A_eq d=119 +comm rho=1e-1",
     dict(collective_solver="aeq", aeq_rho=1e-1, aeq_width=119, _communities=128)),
    ("A_eq d=119 +comm rho=1",
     dict(collective_solver="aeq", aeq_rho=1.0, aeq_width=119, _communities=128)),
    ("A_eq d=119 +comm rho=10",
     dict(collective_solver="aeq", aeq_rho=10.0, aeq_width=119, _communities=128)),
    ("A_eq d=119 +comm rho=1 rewt",
     dict(collective_solver="aeq", aeq_rho=1.0, aeq_width=119, _communities=128,
          aeq_reweight_iters=5, aeq_kappa=50.0)),
]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _load_day(window, args, feature_columns=None):
    """One ``(lo, hi)`` window as ``GraphData`` + its gangs + the feature columns."""

    lo, hi = window
    A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, lo, hi)
    cols = feature_columns
    if args.feature_mode in ("wallet", "wallet+random"):
        if feature_columns is None:
            Xfeat, cols = load_node_features(
                args.data_dir, nodes_df, lo, hi, return_columns=True
            )
        else:
            Xfeat = load_node_features(
                args.data_dir, nodes_df, lo, hi, keep_columns=feature_columns
            )
        if args.feature_mode == "wallet+random":
            Xfeat = torch.cat(
                [Xfeat, random_structural_features(
                    int(A_unw.shape[0]), args.random_width, args.seed + lo)],
                dim=1,
            )
    else:
        Xfeat = random_structural_features(
            int(A_unw.shape[0]), args.random_width, args.seed + lo
        )
    graph = build_torch_graph(A_w, A_unw, cls, Xfeat, weighted=False)
    gang_sets = connected_components_sets(
        A_unw, np.where(cls == 1)[0], args.min_gang_size
    )
    gangs = make_patterns(gang_sets, "alert", "gang", "g")
    return GraphData.from_graph(graph), gangs, cols


# --------------------------------------------------------------------------- #
# width matching
# --------------------------------------------------------------------------- #
def _truncation(Z, a_hat, adjacency, patterns, tau, width, mode):
    """A ``(d0, width)`` mixing matrix ``R`` narrowing a target to ``width`` columns.

    The solvers do not natively produce targets of the same width: the gradient
    bank is ``heads * d_features`` wide, while both pencil solvers are capped at
    ``m`` (``A_eq`` has rank ``<= m``, and the closed form has exactly one column
    per group).  Comparing a 119-column target with a 16-column one confounds
    the objective with the target budget, so every target is narrowed to the
    same ``width`` by the same rule before it reaches the coarsener.

    ``mode="energy"``  keeps the ``width`` directions of largest ``M_tau`` energy
    (label-free: top eigenvectors of ``G_Z = Z^T M_tau Z``).
    ``mode="capture"`` keeps the ``width``-dimensional subspace of ``span(Z)``
    maximizing ``tr(Gamma)`` on the training groups -- the most charitable
    narrowing available, since it discards only what the collective objective
    itself scores as worthless.

    ``R`` is returned rather than applied, because it is a *coefficient* mixing:
    the frozen transfer days reuse the same ``R`` on their own ``Z``, so nothing
    about a held-out day ever enters the truncation.
    """

    d0 = int(Z.shape[1])
    if width <= 0 or d0 <= width:
        return None
    MZ = _m_apply(a_hat, Z, tau)
    G = Z.T @ MZ
    G = 0.5 * (G + G.T)
    ev, evec = torch.linalg.eigh(G)
    if mode == "energy":
        idx = torch.argsort(ev, descending=True)[:width]
        return evec[:, idx]
    keep = ev > ev.max().clamp_min(1e-300) * 1e-12
    Wh = evec[:, keep] / ev[keep].sqrt().unsqueeze(0)  # (d0, r): G_Z-whitening
    m_vhat = _train_gang_m_vhat(a_hat, adjacency, patterns, tau).to(Z.dtype)
    C = Wh.T @ (Z.T @ m_vhat)  # (r, m); Gamma = C^T C so tr(Gamma) = ||C||_F^2
    U, _, _ = torch.linalg.svd(C, full_matrices=False)
    return Wh @ U[:, :width]


# --------------------------------------------------------------------------- #
# one configuration
# --------------------------------------------------------------------------- #
def _run_config(name, overrides, base_kwargs, data, gang_train, gang_test, gangs,
                transfer, args, community_cache) -> dict:
    overrides = dict(overrides)
    n_comm = int(overrides.pop("_communities", 0))
    cfg = DetectorConfig(**{**base_kwargs, **overrides})
    det = CollectiveBankDetector(cfg)

    signal_patterns = None
    if n_comm > 0:
        if n_comm not in community_cache:
            community_cache[n_comm] = community_patterns(
                data.a_hat, data.X, n_clusters=n_comm, seed=args.seed
            )
        signal_patterns = community_cache[n_comm]

    t0 = time.perf_counter()
    det.fit([("train", data, gang_train, gang_test)],
            signal_patterns=signal_patterns)
    fit_seconds = time.perf_counter() - t0

    basis = det.target_subspace(data, gang_train)
    native_width = int(basis.shape[1])
    R = _truncation(basis, data.a_hat, data.adjacency, gang_train, cfg.tau,
                    args.target_width, args.target_truncate)
    if R is not None:
        basis = basis @ R
    cert = basis_subspace_report(
        data.a_hat, data.adjacency, basis, gang_train,
        tau=cfg.tau, beta=args.certificate_beta,
    )
    coarsening, _ = det.coarsen(data, basis, gang_train)
    rep = det.evaluate(
        data, coarsening, {"train": gang_train, "test": gang_test, "all": gangs}
    )

    row = {
        "config": name,
        "solver": cfg.collective_solver,
        "fit_seconds": fit_seconds,
        "n_signal_groups": len(gang_train) + (len(signal_patterns or [])),
        "native_width": native_width,
        "width": cert["width"],
        "lambda_min_Gamma": cert["lambda_min_Gamma"],
        "chi_max": cert["chi_max"],
        "margin": cert["margin"],
        "capture_mean": cert["capture_mean"],
        "capture_min": cert["capture_min"],
        "trace_Gamma": cert["trace_Gamma"],
        "n_coarse": int(coarsening.n_coarse),
        "epsilon": float(coarsening.epsilon),
    }
    for split in ("train", "test", "all"):
        r = rep.get(split)
        if r is None:
            continue
        row[f"{split}_recall"] = r["mean_recall"]
        row[f"{split}_precision"] = r["mean_precision"]
        row[f"{split}_f1"] = r["mean_f1"]
        row[f"{split}_detection"] = r["detection_rate"]
        row[f"{split}_detected"] = r["detected"]
        row[f"{split}_total"] = r["total"]

    # frozen transfer: same filter/target coefficients, each later day's own graph
    t_rows = []
    for day, (t_data, t_gangs) in transfer.items():
        if not t_gangs:
            continue
        t_basis = det.target_subspace(t_data, t_gangs)
        if R is not None:  # the SAME frozen mixing: no day-k labels are used
            t_basis = t_basis @ R
        t_co, _ = det.coarsen(t_data, t_basis, t_gangs)
        t_rep = det.evaluate(t_data, t_co, {"all": t_gangs})["all"]
        t_cert = basis_subspace_report(
            t_data.a_hat, t_data.adjacency, t_basis, t_gangs,
            tau=cfg.tau, beta=args.certificate_beta,
        )
        t_rows.append({
            "day": day, "n_gangs": len(t_gangs),
            "recall": t_rep["mean_recall"], "precision": t_rep["mean_precision"],
            "f1": t_rep["mean_f1"], "detection": t_rep["detection_rate"],
            "detected": t_rep["detected"], "total": t_rep["total"],
            "lambda_min_Gamma": t_cert["lambda_min_Gamma"],
            "chi_max": t_cert["chi_max"], "margin": t_cert["margin"],
        })
    if t_rows:
        T = pd.DataFrame(t_rows)
        row.update({
            "transfer_recall": T.recall.mean(),
            "transfer_precision": T.precision.mean(),
            "transfer_f1": T.f1.mean(),
            "transfer_detection": T.detection.mean(),
            "transfer_detected": int(T.detected.sum()),
            "transfer_total": int(T.total.sum()),
            "transfer_margin": T.margin.mean(),
        })

    extra = {"transfer_rows": t_rows, "chi_per_gang": cert["chi"],
             "capture_per_gang": cert["capture"]}
    if cfg.collective_solver == "aeq":
        extra["eigenvalues"] = det.fit_info_["eigenvalues"]
        extra["rho_used"] = det.fit_info_["rho_used"]
        extra["reweight_history"] = det.fit_info_["reweight_history"]
    return row, extra


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--train-days", default="26",
                    help="'26' = one day; '24-26' merges the window into one graph")
    ap.add_argument("--transfer-days", type=int, default=3)
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--train-ratio", type=float, default=0.6)
    ap.add_argument("--feature-mode",
                    choices=["wallet", "random", "wallet+random"],
                    default="wallet+random")
    ap.add_argument("--random-width", type=int, default=64)
    ap.add_argument("--degree", type=int, default=32)
    ap.add_argument("--basis", default="chebyshev")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--ridge", type=float, default=1e-5)
    ap.add_argument("--optimizer", default="projected")
    ap.add_argument("--coarsening-method", default="ward-tree")
    ap.add_argument("--reduction", type=float, default=0.3)
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--max-levels", type=int, default=10)
    ap.add_argument("--ward-stop", choices=["epsilon", "f1"], default="f1")
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument(
        "--target-width",
        type=int,
        default=0,
        help="narrow EVERY solver's target to this many columns before the "
        "coarsener sees it (0 = each solver's native width).  The bank is "
        "heads*d_features wide and the pencil solvers are capped at m, so "
        "without this the objective is confounded with the target budget.",
    )
    ap.add_argument(
        "--target-truncate",
        choices=["capture", "energy"],
        default="capture",
        help="how --target-width narrows a target: 'capture' keeps the subspace "
        "maximizing tr(Gamma) on the TRAIN groups (most charitable -- it drops "
        "only what the collective objective already scores as worthless); "
        "'energy' keeps the largest-M_tau-energy directions (label-free). Either "
        "way the mixing is frozen and reused on the transfer days.",
    )
    ap.add_argument("--certificate-beta", type=float, default=1.0,
                    help="beta used ONLY to score the reported margin "
                         "lambda_min(Gamma) - beta*max_j chi_j")
    ap.add_argument("--only", default="",
                    help="comma-separated substrings; run only matching configs")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=f"results/aeq_comparison/{now}/", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    lo, hi = (args.train_days.split("-") + [args.train_days.split("-")[-1]])[:2]
    window = (int(lo), int(hi))
    LOGGER.info(f"=== A_eq vs max-min objective | Elliptic++ days {window} ===")

    data, gangs, feature_columns = _load_day(window, args)
    rng = np.random.default_rng(args.seed)
    gang_train, gang_test = split_train_test(gangs, args.train_ratio, rng)
    LOGGER.info(
        f"  N={data.num_nodes:,}  d={data.feature_dim}  gangs={len(gangs)} "
        f"(train {len(gang_train)} / test {len(gang_test)})"
    )
    if len(gang_train) > data.feature_dim:
        LOGGER.info("  NOTE: m > d -- lambda_min(Gamma) is at the capacity wall")

    transfer = {}
    for k in range(1, args.transfer_days + 1):
        day = window[1] + k
        t_data, t_gangs, _ = _load_day((day, day), args, feature_columns)
        if t_data.feature_dim != data.feature_dim or not t_gangs:
            LOGGER.info(f"  transfer day {day} skipped")
            continue
        transfer[day] = (t_data, t_gangs)
    LOGGER.info(f"  transfer days: {sorted(transfer)}")

    base_kwargs = dict(
        degree=args.degree, basis=args.basis, tau=args.tau, epochs=args.epochs,
        learning_rate=args.learning_rate, ridge=args.ridge,
        optimizer=args.optimizer, coarsening_method=args.coarsening_method,
        reduction=args.reduction, epsilon=args.epsilon,
        max_levels=args.max_levels, ward_stop=args.ward_stop,
        threshold=args.threshold, seed=args.seed,
    )

    picks = [p.strip() for p in args.only.split(",") if p.strip()]
    rows, extras, community_cache = [], {}, {}
    for name, overrides in CONFIGS:
        if picks and not any(p in name for p in picks):
            continue
        LOGGER.info(f"\n--- {name} ---")
        row, extra = _run_config(name, overrides, base_kwargs, data, gang_train,
                                 gang_test, gangs, transfer, args,
                                 community_cache)
        rows.append(row)
        extras[name] = extra
        LOGGER.info(
            f"  fit {row['fit_seconds']:.1f}s  d={row['width']}  "
            f"lambda_min={row['lambda_min_Gamma']:.4g}  chi={row['chi_max']:.4g}  "
            f"margin={row['margin']:.4g}  |  all F1={row['all_f1']:.3f} "
            f"det={row['all_detection']:.1%}"
        )
        pd.DataFrame(rows).to_csv(args.out / "aeq_comparison.csv", index=False)

    D = pd.DataFrame(rows)
    D.to_csv(args.out / "aeq_comparison.csv", index=False)
    (args.out / "aeq_comparison.json").write_text(
        json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                    "rows": rows, "extras": extras}, indent=2, default=str) + "\n"
    )

    # ---- tables ----------------------------------------------------------- #
    LOGGER.info("\n" + "=" * 108)
    LOGGER.info(
        "CERTIFICATE ON THE TARGET SUBSPACE (train gangs; identical for every solver)"
        + (f"  [width matched to {args.target_width} via '{args.target_truncate}']"
           if args.target_width > 0 else "  [each solver's native width]")
    )
    LOGGER.info("=" * 108)
    h = (f"  {'config':<28}{'fit s':>8}{'d':>5}{'sig':>6}{'lam_min(G)':>13}{'max chi':>11}"
         f"{'margin':>12}{'mean cap':>11}{'min cap':>10}{'tr(G)':>9}")
    LOGGER.info(h)
    LOGGER.info("  " + "-" * (len(h) - 2))
    for r in rows:
        LOGGER.info(
            f"  {r['config']:<28}{r['fit_seconds']:>8.1f}{r['width']:>5}"
            f"{r['n_signal_groups']:>6}{r['lambda_min_Gamma']:>13.5g}"
            f"{r['chi_max']:>11.4g}"
            f"{r['margin']:>12.5g}{r['capture_mean']:>11.4g}"
            f"{r['capture_min']:>10.4g}{r['trace_Gamma']:>9.4g}"
        )

    LOGGER.info("\n" + "=" * 108)
    LOGGER.info("DETECTION (same coarsener + budget; 'transfer' = frozen target, later days)")
    LOGGER.info("=" * 108)
    h = (f"  {'config':<28}{'train F1':>10}{'test F1':>9}{'all F1':>9}"
         f"{'all rec':>9}{'all prec':>10}{'all det':>9}{'det/tot':>10}"
         f"{'trn F1':>9}{'trn det':>9}")
    LOGGER.info(h)
    LOGGER.info("  " + "-" * (len(h) - 2))
    for r in rows:
        LOGGER.info(
            f"  {r['config']:<28}{r['train_f1']:>10.3f}{r['test_f1']:>9.3f}"
            f"{r['all_f1']:>9.3f}{r['all_recall']:>9.3f}"
            f"{r['all_precision']:>10.3f}{r['all_detection']:>9.1%}"
            f"{r['all_detected']:>5}/{r['all_total']:<4}"
            f"{r.get('transfer_f1', float('nan')):>9.3f}"
            f"{r.get('transfer_detection', float('nan')):>9.1%}"
        )

    # ---- plot ------------------------------------------------------------- #
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    y = np.arange(len(D))
    axes[0].barh(y, D.margin, color="tab:blue")
    axes[0].set_title(f"certificate margin\nlam_min(Gamma) - {args.certificate_beta:g}*max chi")
    axes[1].barh(y, D.all_f1, color="tab:green")
    axes[1].set_title("F1, all gangs (training graph)")
    axes[2].barh(y, D.get("transfer_detection", pd.Series(np.zeros(len(D)))),
                 color="tab:orange")
    axes[2].set_title("detection rate, frozen transfer days")
    for ax in axes:
        ax.set_yticks(y)
        ax.set_yticklabels(D.config, fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "aeq_comparison.png", dpi=150)
    LOGGER.info(f"\nCSV + JSON + plot -> {args.out}")


if __name__ == "__main__":
    main()
