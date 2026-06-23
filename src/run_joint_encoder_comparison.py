"""Joint (theta, W) linear encoder vs structural and raw-feature baselines.

This compares three *linear* encoders on the same graph, all coarsened with the
identical Loukas RSA procedure to the same reduction, so the only thing that
changes is the target subspace ``R = span(Z)``:

    structural   :  Z = g_theta(A_hat) Omega           (random range finder)
    raw-feature  :  Z = g_theta(A_hat) X               (all node features)
    joint        :  Z = [g_theta(A_hat) Omega | g_theta(A_hat) X W]

The joint encoder learns both the spectral filter ``theta`` and a small feature
map ``W in R^{f x d}`` by gradient on ``lambda_min(G(theta, W))`` -- the
learnable-feature-metric (linear GNN) upgrade of the fixed ``Sigma_X = X X.T``.
It *augments* the structural channel with the learned feature channel rather
than replacing it (the cure for "features worse than random").

Each encoder is judged on the two jobs it must serve, kept separate:

* coarsening (structural)  -- post-coarsening alert recall / precision;
* classification (label)   -- held-out alert AUC from a ridge-LDA discriminant
  fit on that encoder's node embedding (a separate head, never the energy).

Run, e.g.::

    python -m src.GangPrediction.run_joint_encoder_comparison \
        --experiment tutorial_demo16 --degree 8 --structural-width 32 \
        --embed-dim 8 --reduction 0.7 --threshold 0.51
"""

from __future__ import annotations

import os
import argparse
import json
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.utils import *
from src.experiment_utils import load_and_preprocess_data
from src.loukas_sgc_detection import (
    _orthonormal_range,
    build_joint_subspace,
    build_sgc_subspace,
    evaluate_loukas_patterns,
    graph_operators,
    loukas_coarsen_pytorch,
)
from src.sgc_detection import (
    _roc_auc,
    apply_feature_channel,
    apply_graph_filter,
    fit_collective_sgc,
    fit_feature_discriminant,
    fit_joint_encoder,
    score_feature_patterns,
)


def _classifier_auc(
    normalized: torch.Tensor,
    train_patterns: list,
    test_patterns: list,
    node_embedding: torch.Tensor,
    *,
    ridge: float,
) -> tuple[float, float]:
    """Ridge-LDA head on a node embedding: returns (train_auc, held-out test_auc)."""

    discriminant = fit_feature_discriminant(
        normalized,
        train_patterns,
        features=node_embedding,
        degree=0,
        head="mean",
        ridge=ridge,
    )
    scores = score_feature_patterns(
        normalized, test_patterns, discriminant, features=node_embedding
    )
    alert = [float(s) for s, p in zip(scores, test_patterns) if p.label == "alert"]
    normal = [float(s) for s, p in zip(scores, test_patterns) if p.label != "alert"]
    return discriminant.train_auc, _roc_auc(alert, normal)


def _coarsen_and_score(
    name: str,
    *,
    coarsen_basis: torch.Tensor,
    classify_embedding: torch.Tensor,
    adjacency: torch.Tensor,
    normalized: torch.Tensor,
    node_labels: torch.Tensor,
    clf_train: list,
    clf_test: list,
    eval_patterns: list,
    args: argparse.Namespace,
) -> dict:
    """Coarsen with ``coarsen_basis`` and classify with ``classify_embedding``."""

    coarsening = loukas_coarsen_pytorch(
        adjacency,
        coarsen_basis,
        reduction=args.reduction,
        epsilon=args.epsilon,
        max_levels=args.max_levels,
        method=args.coarsening_method,
        max_contraction_size=args.max_contraction_size,
        leaf_degree=args.leaf_degree,
        min_spokes=args.min_spokes,
        kmeans_iters=args.kmeans_iters,
        kmeans_seed=seed,
        max_cluster_size=args.linkage_max_size,
    )
    _, by_label = evaluate_loukas_patterns(
        eval_patterns,
        coarsening.node_to_supernode,
        node_labels,
        threshold=args.threshold,
    )
    train_auc, test_auc = _classifier_auc(
        normalized, clf_train, clf_test, classify_embedding, ridge=args.feature_ridge
    )
    alert = by_label.get("alert", {})
    normal = by_label.get("normal", {})
    return {
        "encoder": name,
        "basis_dim": int(coarsen_basis.shape[1]),
        "n_coarse": coarsening.n_coarse,
        "n_levels": len(coarsening.sigmas),
        "reduction": coarsening.reduction,
        "epsilon": coarsening.epsilon,
        "alert_detection_rate": alert.get("detection_rate"),
        "alert_mean_recall": alert.get("mean_recall"),
        "alert_mean_precision": alert.get("mean_precision"),
        "alert_by_type": alert.get("by_pattern_type", {}),
        "normal_detection_rate": normal.get("detection_rate"),
        "normal_mean_recall": normal.get("mean_recall"),
        "normal_mean_precision": normal.get("mean_precision"),
        "normal_by_type": normal.get("by_pattern_type", {}),
        "classifier_train_auc": train_auc,
        "classifier_test_auc": test_auc,
    }


def _save_comparison_plot(rows: list, output: Path, *, experiment: str = "") -> None:
    """Six-panel comparison figure.

    Panels:
    1  Alert coarsening quality (recall / precision / detection rate)
    2  Normal coarsening quality (recall / precision / detection rate)
    3  Per-type alert detection rate heatmap (pattern-type x encoder)
    4  Per-type alert mean recall heatmap
    5  Held-out vs train classifier AUC (generalisation gap)
    6  Coarsening diagnostics (n_coarse, epsilon, levels)
    """

    names = [row["encoder"] for row in rows]
    n = len(names)
    x = np.arange(n)
    w = 0.25  # bar width for 3-metric groups

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    suptitle = (
        f"Linear encoder comparison — {experiment}"
        if experiment
        else "Linear encoder comparison"
    )
    fig.suptitle(suptitle, fontsize=12, y=1.01)
    colors = ["tab:green", "tab:orange", "tab:blue"]

    # ── panel 1: alert coarsening ────────────────────────────────────────────
    ax = axes[0, 0]
    alert_recall = [row["alert_mean_recall"] or 0.0 for row in rows]
    alert_prec = [row["alert_mean_precision"] or 0.0 for row in rows]
    alert_det = [row["alert_detection_rate"] or 0.0 for row in rows]
    for offset, vals, label, color in zip(
        [-w, 0, w],
        [alert_recall, alert_prec, alert_det],
        ["mean recall", "mean precision", "detection rate"],
        colors,
    ):
        bars = ax.bar(x + offset, vals, w * 0.95, color=color, label=label)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.12)
    ax.set_title("Alert: coarsening quality")
    ax.set_ylabel("score")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── panel 2: normal coarsening ───────────────────────────────────────────
    ax = axes[0, 1]
    norm_recall = [row["normal_mean_recall"] or 0.0 for row in rows]
    norm_prec = [row["normal_mean_precision"] or 0.0 for row in rows]
    norm_det = [row["normal_detection_rate"] or 0.0 for row in rows]
    for offset, vals, label, color in zip(
        [-w, 0, w],
        [norm_recall, norm_prec, norm_det],
        ["mean recall", "mean precision", "detection rate"],
        colors,
    ):
        bars = ax.bar(x + offset, vals, w * 0.95, color=color, label=label)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.12)
    ax.set_title("Normal: coarsening quality")
    ax.set_ylabel("score")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── panel 3: per-type alert detection rate heatmap ───────────────────────
    ax = axes[0, 2]
    all_alert_types = sorted({t for row in rows for t in row["alert_by_type"]})
    if all_alert_types:
        det_matrix = np.array(
            [
                [
                    row["alert_by_type"].get(t, {}).get("detection_rate", 0.0)
                    for t in all_alert_types
                ]
                for row in rows
            ]
        )  # (n_encoders, n_types)
        im = ax.imshow(det_matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(all_alert_types)))
        ax.set_xticklabels(all_alert_types, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=9)
        for i in range(n):
            for j, t in enumerate(all_alert_types):
                v = det_matrix[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if 0.3 < v < 0.8 else "white",
                )
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    ax.set_title("Alert detection rate by type")

    # ── panel 4: per-type alert mean recall heatmap ──────────────────────────
    ax = axes[1, 0]
    if all_alert_types:
        recall_matrix = np.array(
            [
                [
                    row["alert_by_type"].get(t, {}).get("mean_recall", 0.0)
                    for t in all_alert_types
                ]
                for row in rows
            ]
        )
        im2 = ax.imshow(recall_matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(all_alert_types)))
        ax.set_xticklabels(all_alert_types, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=9)
        for i in range(n):
            for j, t in enumerate(all_alert_types):
                v = recall_matrix[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if v < 0.7 else "white",
                )
        fig.colorbar(im2, ax=ax, fraction=0.04, pad=0.04)
    ax.set_title("Alert mean recall by type")

    # ── panel 5: classifier AUC – train vs held-out ──────────────────────────
    ax = axes[1, 1]
    train_auc = [row["classifier_train_auc"] or 0.0 for row in rows]
    test_auc = [row["classifier_test_auc"] or 0.0 for row in rows]
    ax.bar(x - w / 2, train_auc, w, color="tab:purple", alpha=0.85, label="train AUC")
    ax.bar(x + w / 2, test_auc, w, color="tab:cyan", alpha=0.85, label="held-out AUC")
    ax.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.6)
    for i, (tr, te) in enumerate(zip(train_auc, test_auc)):
        ax.text(i - w / 2, tr + 0.01, f"{tr:.2f}", ha="center", fontsize=7)
        ax.text(i + w / 2, te + 0.01, f"{te:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.12)
    ax.set_title("Classifier AUC (train vs held-out)")
    ax.set_ylabel("AUC")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── panel 6: coarsening diagnostics ─────────────────────────────────────
    ax = axes[1, 2]
    n_coarse = [row["n_coarse"] for row in rows]
    epsilons = [row["epsilon"] for row in rows]
    n_levels = [row["n_levels"] for row in rows]
    n_orig = rows[0]["n_coarse"] / max(1 - rows[0]["reduction"], 1e-9)
    ax2 = ax.twinx()
    bars = ax.bar(x, n_coarse, 0.5, color="tab:brown", alpha=0.7, label="n_coarse")
    ax.axhline(
        n_orig * (1 - rows[0]["reduction"]),
        color="grey",
        ls="--",
        lw=1,
        alpha=0.5,
        label="target n_coarse",
    )
    ax2.plot(x, epsilons, "o-", color="tab:red", label="epsilon (RSA)")
    for i, (nc, eps_val, nl) in enumerate(zip(n_coarse, epsilons, n_levels)):
        ax.text(i, nc + n_orig * 0.005, str(nc), ha="center", fontsize=8)
        ax2.text(
            i,
            eps_val + max(epsilons) * 0.02,
            f"ε={eps_val:.1f}\n{nl}lvl",
            ha="center",
            fontsize=7,
            color="tab:red",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Coarsening diagnostics")
    ax.set_ylabel("n_coarse")
    ax2.set_ylabel("epsilon (RSA bound)", color="tab:red")
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="tutorial_demo5")
    parser.add_argument("--train-ratio", type=float, default=0.25)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--structural-width", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-3, help="encoder W ridge")
    parser.add_argument(
        "--feature-ridge", type=float, default=1e-2, help="classifier LDA ridge"
    )
    parser.add_argument(
        "--label-weight",
        type=float,
        default=1.0,
        help="weight of the supervised LDA-margin term in the joint encoder "
        "objective (0 = pure lambda_min; >0 trades coarsening for label "
        "separability)",
    )
    parser.add_argument(
        "--label-ridge",
        type=float,
        default=1e-2,
        help="ridge for the joint encoder's supervised LDA-margin term",
    )
    parser.add_argument(
        "--per-hop-features",
        action="store_true",
        default=False,
        help="give the joint encoder one feature map W_k per propagation depth "
        "(linear filterbank sum_k A_hat^k X W_k) instead of a single shared W",
    )
    parser.add_argument(
        "--retention-mode",
        choices=["auto", "pattern", "channel"],
        default="auto",
        help="lambda_min Gram side when #patterns m exceeds the channel rank r: "
        "'pattern' keeps the m x m Gram (theta frozen once m>r); 'auto' "
        "(default) switches to the r x r channel Gram so theta/W keep a "
        "gradient; 'channel' always uses the channel Gram",
    )
    parser.add_argument(
        "--retention-reduce",
        choices=["min", "softmin", "mean"],
        default="softmin",
        help="how the retention spectrum is reduced: 'min' (strict lambda_min, "
        "default); 'softmin' (smooth eigenvalue-weighted trace "
        "-tau*logsumexp(-lambda/tau), tau=--retention-temp*lambda_max); 'mean' "
        "(trace/energy limit)",
    )
    parser.add_argument(
        "--retention-temp",
        type=float,
        default=0.5,
        help="softmin temperature as a fraction of lambda_max (smaller -> closer "
        "to strict min, larger -> closer to mean)",
    )
    parser.add_argument("--reduction", type=float, default=0.7)
    parser.add_argument("--epsilon", type=float, default=float("inf"))
    parser.add_argument("--max-levels", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.51)
    parser.add_argument(
        "--coarsening-method",
        choices=["edges", "neighborhood", "capped", "star", "kmeans", "linkage"],
        default="edges",
        help="local-variation candidate family: 'edges' (1 pair, conservative); "
        "'neighborhood' ({i}uN(i), aggressive); 'capped' (in-between, sets <= "
        "--max-contraction-size); 'star' (hub+spokes pre-pass for fan patterns); "
        "'kmeans' (global subspace clustering + connectivity, non-greedy); "
        "'linkage' (single-linkage union-find on the cost graph, connected by "
        "construction)",
    )
    parser.add_argument(
        "--kmeans-iters",
        type=int,
        default=10,
        help="Lloyd iterations for --coarsening-method kmeans",
    )
    parser.add_argument(
        "--linkage-max-size",
        type=int,
        default=8,
        help="supernode size cap for --coarsening-method linkage (curbs single-"
        "linkage chaining; 0 = uncapped). Smaller -> better fans, larger -> "
        "better overall",
    )
    parser.add_argument(
        "--max-contraction-size",
        type=int,
        default=4,
        help="cap on contraction-set size for --coarsening-method capped "
        "(2 = edges, large -> neighborhood)",
    )
    parser.add_argument(
        "--leaf-degree",
        type=int,
        default=1,
        help="for --coarsening-method star: a spoke is a neighbor with "
        "combinatorial degree <= this (raise to catch fan_in sources with "
        "extra edges, at the cost of more collateral on dense motifs)",
    )
    parser.add_argument(
        "--min-spokes",
        type=int,
        default=4,
        help="for --coarsening-method star: a hub must have at least this many "
        "spokes to fire (raise to restrict to large, genuine fans)",
    )
    # parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-normal-train",
        action="store_true",
        default=True,
        help="add normal patterns to the lambda_min retention target",
    )
    parser.add_argument("--remove-overlaps", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    experiment_root = Path.cwd() / "experiments" / args.experiment
    graph, alert_train, normal_train, alert_test, normal_test = (
        load_and_preprocess_data(
            data_dir=experiment_root / "config",
            patterns_dir=experiment_root,
            train_ratio=args.train_ratio,
            to_undirected=True,
            remove_overlaps=args.remove_overlaps,
            device=torch.device(args.device),
            seed=seed,
        )
    )
    if getattr(graph, "x", None) is None:
        raise ValueError("joint encoder comparison requires node features (graph.x)")

    normalized, adjacency = graph_operators(graph)
    X = graph.x.to(device=normalized.device, dtype=normalized.dtype)
    total_width = args.structural_width + args.embed_dim

    retain = alert_train + normal_train if args.include_normal_train else alert_train
    if not retain:
        raise ValueError("no patterns available for the lambda_min retention target")
    clf_train = alert_train + normal_train
    clf_test = alert_test + normal_test
    eval_patterns = alert_test + normal_test

    common = dict(
        degree=args.degree, epochs=args.epochs, learning_rate=args.learning_rate
    )

    # --- structural encoder: theta on the structural Gram (Sigma_X = I) ---
    structural_fit = fit_collective_sgc(
        normalized,
        retain,
        features=None,
        mode="lambda_min",
        retention_mode=args.retention_mode,
        retention_reduce=args.retention_reduce,
        retention_temp=args.retention_temp,
        **common,
    )
    generator = torch.Generator(device=normalized.device)
    generator.manual_seed(seed)
    omega = torch.randn(
        normalized.shape[0],
        total_width,
        dtype=normalized.dtype,
        device=normalized.device,
        generator=generator,
    )
    structural_basis = build_sgc_subspace(
        normalized, structural_fit.theta, None, width=total_width, seed=seed
    )
    structural_embed = apply_graph_filter(normalized, omega, structural_fit.theta)

    # --- raw-feature encoder: theta on the feature-aware Gram (Sigma_X = X X.T) ---
    feature_fit = fit_collective_sgc(
        normalized,
        retain,
        features=X,
        mode="lambda_min",
        retention_mode=args.retention_mode,
        retention_reduce=args.retention_reduce,
        retention_temp=args.retention_temp,
        **common,
    )
    feature_basis = build_sgc_subspace(
        normalized, feature_fit.theta, X, width=total_width, seed=seed
    )
    feature_embed = apply_graph_filter(normalized, X, feature_fit.theta)

    # --- joint encoder: learn (theta, W) on lambda_min(G(theta, W)) ---
    joint = fit_joint_encoder(
        normalized,
        retain,
        features=X,
        degree=args.degree,
        embed_dim=args.embed_dim,
        structural_width=args.structural_width,
        ridge=args.ridge,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=seed,
        label_patterns=clf_train,
        label_weight=args.label_weight,
        label_ridge=args.label_ridge,
        per_hop_features=args.per_hop_features,
        retention_mode=args.retention_mode,
        retention_reduce=args.retention_reduce,
        retention_temp=args.retention_temp,
    )
    joint_basis = build_joint_subspace(
        normalized,
        joint.theta,
        X,
        joint.feature_map,
        structural_width=args.structural_width,
        seed=seed,
        per_hop=joint.per_hop_features,
    )
    joint_embed = apply_feature_channel(
        normalized, X, joint.feature_map, joint.theta, per_hop=joint.per_hop_features
    )

    encoders = [
        ("structural", structural_basis, structural_embed),
        ("raw-feature", feature_basis, feature_embed),
        ("joint", joint_basis, joint_embed),
    ]
    rows = [
        _coarsen_and_score(
            name,
            coarsen_basis=basis,
            classify_embedding=embed,
            adjacency=adjacency,
            normalized=normalized,
            node_labels=graph.y,
            clf_train=clf_train,
            clf_test=clf_test,
            eval_patterns=eval_patterns,
            args=args,
        )
        for name, basis, embed in encoders
    ]

    out_dir = Path(args.output) if args.output else Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / "joint_encoder_comparison.json"
    plot_out = out_dir / "joint_encoder_comparison.png"
    json_out.write_text(
        json.dumps(
            {
                "experiment": args.experiment,
                "degree": args.degree,
                "structural_width": args.structural_width,
                "embed_dim": joint.embed_dim,
                "target_dim": total_width,
                "reduction": args.reduction,
                "joint_encoder_objective_lambda_min": joint.objective,
                "joint_encoder_vanilla_objective": joint.vanilla_objective,
                "joint_encoder_label_weight": joint.label_weight,
                "joint_encoder_label_separation": joint.label_separation,
                "joint_encoder_combined_objective": joint.combined_objective,
                "joint_encoder_per_hop_features": joint.per_hop_features,
                "retention_mode": args.retention_mode,
                "retention_reduce": args.retention_reduce,
                "retention_temp": args.retention_temp,
                "retention_side": {
                    "structural": structural_fit.retention_side,
                    "raw-feature": feature_fit.retention_side,
                    "joint": joint.retention_side,
                },
                "encoders": rows,
            },
            indent=2,
        )
        + "\n"
    )
    _save_comparison_plot(rows, plot_out, experiment=args.experiment)

    LOGGER.info("\nLinear encoder comparison  (Z = g_theta(A_hat) [.])")
    LOGGER.info(
        f"  experiment={args.experiment}  degree={args.degree}  "
        f"target_dim~{total_width}  reduction={args.reduction:.0%}  "
        f"method={args.coarsening_method}"
    )
    feature_channel = (
        "per-hop W_k (filterbank)" if joint.per_hop_features else "shared W"
    )
    LOGGER.info(
        f"  joint encoder lambda_min(G(theta,W)): {joint.objective:.6g} "
        f"(init {joint.vanilla_objective:.6g})  feature-channel={feature_channel}"
    )
    reduce_desc = args.retention_reduce + (
        f"(tau={args.retention_temp:g}*lambda_max)"
        if args.retention_reduce == "softmin"
        else ""
    )
    LOGGER.info(
        f"  retention_mode={args.retention_mode}  reduce={reduce_desc}  Gram side -> "
        f"structural:{structural_fit.retention_side}  "
        f"raw-feature:{feature_fit.retention_side}  joint:{joint.retention_side}"
    )
    if joint.label_weight > 0:
        LOGGER.info(
            f"  joint encoder label supervision: weight={joint.label_weight:g}  "
            f"LDA-margin={joint.label_separation:.6g}  "
            f"combined={joint.combined_objective:.6g}"
        )
    header = (
        f"\n  {'encoder':<12} {'dim':>4} {'n_coarse':>9} "
        f"{'alert_recall':>13} {'alert_prec':>11} {'alert_det':>10} "
        f"{'clf_AUC(test)':>14} {'clf_AUC(train)':>15}"
    )
    LOGGER.info(header)
    LOGGER.info("  " + "-" * (len(header) - 3))
    for row in rows:
        LOGGER.info(
            f"  {row['encoder']:<12} {row['basis_dim']:>4} {row['n_coarse']:>9} "
            f"{(row['alert_mean_recall'] or 0):>13.3f} "
            f"{(row['alert_mean_precision'] or 0):>11.3f} "
            f"{(row['alert_detection_rate'] or 0):>10.1%} "
            f"{(row['classifier_test_auc'] or 0):>14.3f} "
            f"{(row['classifier_train_auc'] or 0):>15.3f}"
        )

    # Per-type alert detection rate
    all_alert_types = sorted({t for row in rows for t in row["alert_by_type"]})
    if all_alert_types:
        LOGGER.info("\n  Alert detection rate by pattern type:")
        type_header = f"  {'encoder':<12}  " + "".join(
            f"{t:>14}" for t in all_alert_types
        )
        LOGGER.info(type_header)
        LOGGER.info("  " + "-" * (len(type_header) - 2))
        for row in rows:
            cells = "".join(
                f"{row['alert_by_type'].get(t, {}).get('detection_rate', 0.0):>14.1%}"
                for t in all_alert_types
            )
            LOGGER.info(f"  {row['encoder']:<12}  {cells}")

        LOGGER.info("\n  Alert mean recall by pattern type:")
        LOGGER.info(type_header)
        LOGGER.info("  " + "-" * (len(type_header) - 2))
        for row in rows:
            cells = "".join(
                f"{row['alert_by_type'].get(t, {}).get('mean_recall', 0.0):>14.3f}"
                for t in all_alert_types
            )
            LOGGER.info(f"  {row['encoder']:<12}  {cells}")

    LOGGER.info(
        "\nRead: 'alert_recall/prec/det' is the STRUCTURAL coarsening quality "
        "(theta/W's job); 'clf_AUC(test)' is the held-out alert classifier on "
        "each encoder's embedding (the label head). The structural embedding "
        "carries no node features so its AUC sits near 0.5; the feature and "
        "joint embeddings read the label. The joint encoder augments structure "
        "with the learned feature channel, so it keeps coarsening quality while "
        "lifting the classifier above the structural baseline."
    )
    LOGGER.info(f"\nJSON report:  {json_out}")
    LOGGER.info(f"Plot:         {plot_out}")


if __name__ == "__main__":
    main()
