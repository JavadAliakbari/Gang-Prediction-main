"""Train SGC theta, build R=span(g_theta(A_hat)X), then run Loukas RSA.

The default trains theta on alerts only.  Pass ``--include-normal-train`` to
fit the collective Eq. (48) objective on both training families instead.
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.experiment_utils import load_and_preprocess_data
from src.loukas_sgc_detection import (
    build_sgc_subspace,
    evaluate_loukas_patterns,
    graph_operators,
    loukas_coarsen_pytorch,
    save_loukas_report,
)
from src.sgc_detection import (
    _roc_auc,
    fit_collective_sgc,
    fit_feature_discriminant,
    score_feature_patterns,
    score_pattern_energies,
)
from src.utils.utils import *


def _save_detection_plot(
    detections: list,
    by_label: dict,
    coarsening,
    *,
    threshold: float,
    experiment: str,
    output: Path,
) -> None:
    """Save a 2-panel diagnostic plot: detection rate by pattern type + PR scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Loukas RSA detection  |  {experiment}  |  "
        f"threshold={threshold}  |  "
        f"{coarsening.n_original}→{coarsening.n_coarse} nodes",
        fontsize=12,
    )

    # Left: alert detection rate per pattern type
    ax = axes[0]
    by_type = by_label.get("alert", {}).get("by_pattern_type", {})
    types = sorted(by_type)
    rates = [by_type[t]["detection_rate"] for t in types]
    colors = ["tab:green" if r > threshold else "tab:red" for r in rates]
    ax.bar(types, rates, color=colors)
    ax.axhline(threshold, ls="--", c="k", lw=1, label=f"threshold={threshold}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("detection rate")
    ax.set_title("Alert detection rate by pattern type")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Right: recall vs precision scatter, coloured by label
    ax = axes[1]
    label_colors = {"alert": "tab:blue", "normal": "tab:orange"}
    for lbl, col in label_colors.items():
        pts = [d for d in detections if d.label == lbl]
        if not pts:
            continue
        rec = [d.recall for d in pts]
        prec = [d.precision for d in pts]
        ax.scatter(rec, prec, s=25, alpha=0.7, color=col, label=lbl)
    ax.axvline(threshold, ls="--", c="k", lw=1)
    ax.axhline(threshold, ls="--", c="k", lw=1)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Per-pattern recall vs precision\n(top-right = detected)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _save_score_histogram(
    alert_scores: list,
    normal_scores: list,
    auc: float,
    *,
    title: str,
    output: Path,
) -> None:
    """Histogram of alert vs normal feature-energy scores with the ROC AUC."""
    fig, ax = plt.subplots(figsize=(8, 5))
    alert = np.asarray(alert_scores, dtype=float)
    normal = np.asarray(normal_scores, dtype=float)
    combined = np.concatenate([a for a in (alert, normal) if a.size])
    if combined.size:
        bins = np.linspace(combined.min(), combined.max(), 40)
    else:
        bins = 40
    if normal.size:
        ax.hist(
            normal,
            bins=bins,
            alpha=0.6,
            density=True,
            color="tab:orange",
            label=f"normal (n={normal.size})",
        )
    if alert.size:
        ax.hist(
            alert,
            bins=bins,
            alpha=0.6,
            density=True,
            color="tab:blue",
            label=f"alert (n={alert.size})",
        )
    auc_text = "n/a" if auc is None or np.isnan(auc) else f"{auc:.3f}"
    ax.set_title(f"{title}\nalert-vs-normal separation AUC = {auc_text}")
    ax.set_xlabel(r"retained feature energy  $\theta^\top M_j \theta$")
    ax.set_ylabel("density")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="tutorial_demo16")
    parser.add_argument("--train-ratio", type=float, default=0.25)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--subspace-width", type=int, default=64)
    parser.add_argument("--reduction", type=float, default=0.7)
    parser.add_argument("--epsilon", type=float, default=100)
    parser.add_argument(
        "--coarsening-method",
        choices=["edges", "neighborhood"],
        default="edges",
        help=(
            "edges: contract single edges (Loukas Algorithm 2, edge family); "
            "neighborhood: contract a vertex with its neighbors "
            "C_i={i}uN(i) (neighborhood family)"
        ),
    )
    parser.add_argument("--max-levels", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.51)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-normal-train", action="store_true")
    parser.add_argument(
        "--feature-mode",
        choices=["node", "gaussian"],
        default="node",
        help="node: Z=g(A)X and G uses Sigma_X=XX^T; gaussian: isotropic Sigma_X=I",
    )
    parser.add_argument(
        "--theta-objective",
        choices=["lambda_min", "fisher", "discriminative"],
        default="discriminative",
        help=(
            "lambda_min: max retention (Eq. 48); fisher: alert-vs-normal LDA "
            "(closed form); discriminative: gradient-trained soft Fisher ratio "
            "of per-pattern energies"
        ),
    )
    parser.add_argument("--remove-overlaps", action="store_true")
    parser.add_argument(
        "--feature-head",
        choices=["none", "mean", "joint"],
        default="joint",
        help=(
            "alert classifier on node features (separate from the spectral "
            "energy): none disables it; mean = ridge-LDA on size-normalized "
            "mean features; joint = ridge-LDA on the (K+1)xf propagated stack"
        ),
    )
    parser.add_argument("--feature-ridge", type=float, default=1e-2)
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
        )
    )
    # The discriminative objectives (fisher / discriminative) need both classes.
    if args.theta_objective in ("fisher", "discriminative"):
        train_patterns = alert_train + normal_train
    else:
        train_patterns = (
            alert_train + normal_train if args.include_normal_train else alert_train
        )
    if not train_patterns:
        raise ValueError("no training patterns available for Eq. (48) theta fitting")

    use_node_features = args.feature_mode == "node"
    fit_features = graph.x if use_node_features else None

    normalized, adjacency = graph_operators(graph)
    fit = fit_collective_sgc(
        normalized,
        train_patterns,
        degree=args.degree,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        features=fit_features,
        mode=args.theta_objective,
    )
    basis = build_sgc_subspace(
        normalized,
        fit.theta,
        graph.x if use_node_features else None,
        width=args.subspace_width,
        seed=args.seed,
    )
    coarsening = loukas_coarsen_pytorch(
        adjacency,
        basis,
        reduction=args.reduction,
        epsilon=args.epsilon,
        max_levels=args.max_levels,
        method=args.coarsening_method,
    )
    detections, by_label = evaluate_loukas_patterns(
        alert_test + normal_test,
        coarsening.node_to_supernode,
        graph.y,
        threshold=args.threshold,
    )

    out_dir = Path(args.output) if args.output else Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / "loukas_sgc_detection.json"
    plot_out = out_dir / "loukas_sgc_detection.png"
    hist_out = out_dir / "theta_score_histogram.png"
    feature_hist_out = out_dir / "feature_lda_score_histogram.png"
    save_loukas_report(
        json_out,
        fit,
        coarsening,
        detections,
        by_label,
        subspace_width=basis.shape[1],
        threshold=args.threshold,
    )
    _save_detection_plot(
        detections,
        by_label,
        coarsening,
        threshold=args.threshold,
        experiment=args.experiment,
        output=plot_out,
    )

    # Held-out alert-vs-normal feature-energy separation under the fitted theta.
    test_patterns = alert_test + normal_test
    test_energies = score_pattern_energies(
        normalized, test_patterns, fit.theta, degree=args.degree, features=fit_features
    )
    test_alert = [
        float(e) for e, p in zip(test_energies, test_patterns) if p.label == "alert"
    ]
    test_normal = [
        float(e) for e, p in zip(test_energies, test_patterns) if p.label != "alert"
    ]
    test_auc = _roc_auc(test_alert, test_normal)
    _save_score_histogram(
        test_alert,
        test_normal,
        test_auc,
        title=(
            f"{args.experiment}  |  feature-mode={args.feature_mode}  |  "
            f"objective={args.theta_objective}  (held-out test)"
        ),
        output=hist_out,
    )

    # --- Feature-space alert classifier (separate from theta / coarsening) ---
    # The scalar energy theta.T M_j theta traces over the feature covariance and
    # cannot pick a discriminative feature direction; it is also size-confounded.
    # So we read the label with a dedicated feature head and report ITS held-out
    # AUC alongside the energy.  theta stays a purely structural coarsening tool.
    feature_train_auc = None
    feature_test_auc = None
    if args.feature_head != "none":
        if getattr(graph, "x", None) is None:
            raise ValueError("--feature-head requires node features (graph.x)")
        feature_train = alert_train + normal_train
        feature_test = alert_test + normal_test
        discriminant = fit_feature_discriminant(
            normalized,
            feature_train,
            features=graph.x,
            degree=args.degree,
            head=args.feature_head,
            ridge=args.feature_ridge,
        )
        feature_train_auc = discriminant.train_auc
        feature_scores = score_feature_patterns(
            normalized, feature_test, discriminant, features=graph.x
        )
        feature_alert = [
            float(s) for s, p in zip(feature_scores, feature_test) if p.label == "alert"
        ]
        feature_normal = [
            float(s) for s, p in zip(feature_scores, feature_test) if p.label != "alert"
        ]
        feature_test_auc = _roc_auc(feature_alert, feature_normal)
        _save_score_histogram(
            feature_alert,
            feature_normal,
            feature_test_auc,
            title=(
                f"{args.experiment}  |  feature-LDA head={args.feature_head}  "
                f"(held-out test)"
            ),
            output=feature_hist_out,
        )

    print("\nSGC target + Loukas RSA pattern detection")
    print(
        f"  theta trained on: {len(train_patterns)} patterns ({sorted(set(fit.train_labels))})"
    )
    print(f"  feature mode: {args.feature_mode}  |  objective: {args.theta_objective}")
    print(f"  theta objective lambda_min(G): {fit.objective:.6g}")
    if fit.separation_ratio is not None:
        ratio_name = (
            "fisher separation ratio (S+/S-)"
            if args.theta_objective == "fisher"
            else "discriminative score ratio"
        )
        print(f"  {ratio_name}: {fit.separation_ratio:.6g}")
    if fit.auc is not None:
        print(f"  train decision-score AUC (margin s_j, thresholded): {fit.auc:.4f}")
    print(f"  held-out feature-energy AUC (theta^T M_j theta): {test_auc:.4f}")
    if feature_test_auc is not None:
        print(
            f"  held-out feature-LDA AUC (w^T z_j, head={args.feature_head}): "
            f"{feature_test_auc:.4f}  (train {feature_train_auc:.4f})"
        )
    print(f"  R=span(Z) dimension: {basis.shape[1]}")
    print(
        f"  coarsening ({args.coarsening_method}): "
        f"{coarsening.n_original} -> {coarsening.n_coarse} "
        f"(reduction={coarsening.reduction:.1%}, epsilon={coarsening.epsilon:.4g})"
    )
    print(f"  detection: recall > {args.threshold} and precision > {args.threshold}")
    for label in ("alert", "normal"):
        if label not in by_label:
            continue
        metrics = by_label[label]
        print(
            f"\n{label}: {int(metrics['detected'])}/{int(metrics['total'])} "
            f"({metrics['detection_rate']:.1%}); "
            f"mean recall={metrics['mean_recall']:.3f}, "
            f"mean precision={metrics['mean_precision']:.3f}"
        )
        for pattern_type, values in metrics["by_pattern_type"].items():
            print(
                f"  {pattern_type}: {int(values['detected'])}/{int(values['total'])} "
                f"({values['detection_rate']:.1%}), "
                f"R={values['mean_recall']:.3f}, P={values['mean_precision']:.3f}"
            )
    print(f"\nJSON report:  {json_out}")
    print(f"Plot:         {plot_out}")
    print(f"Score hist:   {hist_out}")
    if feature_test_auc is not None:
        print(f"Feature hist: {feature_hist_out}")

    # Honest read of the two jobs:
    #  * theta / span(g_theta(A)X) is a STRUCTURAL coarsening tool -- judge it by
    #    post-coarsening recall/precision above, not by reading the label.
    #  * the feature-LDA head is what actually classifies alert-vs-normal; on
    #    AMLGenTex that is largely a node-feature problem, so the held-out
    #    feature-LDA AUC (above) is the number to trust for detection.
    print(
        "\nSummary: use theta for coarsening (structural recall/precision) and the "
        "feature-LDA head for alert classification (held-out feature-LDA AUC). "
        "The spectral energy is not a label classifier and should not be read as one."
    )


if __name__ == "__main__":
    main()
