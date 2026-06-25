"""Compare two Loukas RSA target subspaces on planted-gang detection.

Two coarsening targets are run on the *same* graph and the *same* test
patterns, then their Pattern-model recall/precision are compared:

* ``sgc``       -- ``R = span(g_theta(A_hat) X)`` with ``theta`` fit by Eq. (48);
* ``laplacian`` -- ``R = span(U_K)``, the bottom-``K`` eigenvectors of ``L=D-W``.

The SGC subspace is tuned to the planted patterns; the Laplacian subspace is the
classical learning-free spectral baseline.  Useful diagnostics are written to a
PNG so both algorithms can be verified to coarsen the graph and detect gangs.
"""

from __future__ import annotations

import os
import argparse
import math
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

from src.utils.utils import *
from src.experiment_utils import load_and_preprocess_data
from src.loukas_sgc_detection import (
    build_laplacian_subspace,
    build_sgc_subspace,
    evaluate_loukas_patterns,
    graph_operators,
    loukas_coarsen_pytorch,
)
from src.sgc_detection import fit_collective_sgc
from src.utils.utils import save_path as _utils_save_path


def _alert_rate_by_type(by_label: dict) -> tuple[list[str], list[float]]:
    info = by_label.get("alert", {}).get("by_pattern_type", {})
    types = sorted(info)
    rates = [info[name]["detection_rate"] for name in types]
    return types, rates


def _mean_metric(by_label: dict, label: str, key: str) -> float:
    return float(by_label.get(label, {}).get(key, 0.0))


def _plot_comparison(
    methods: dict,
    *,
    threshold: float,
    experiment: str,
    output: Path,
) -> None:
    """Render a 2x2 diagnostic figure comparing both target subspaces."""

    colors = {"sgc": "tab:blue", "laplacian": "tab:orange"}
    names = {
        "sgc": "SGC  R=span(g_theta(A)X)",
        "laplacian": "Laplacian  R=span(U_K)",
    }
    order = ["sgc", "laplacian"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Loukas RSA target-subspace comparison  |  {experiment}  |  "
        f"detect rule: recall & precision > {threshold}",
        fontsize=13,
    )

    # (a) Alert detection rate by pattern type --------------------------------
    ax = axes[0, 0]
    all_types: list[str] = sorted(
        {
            ptype
            for method in methods.values()
            for ptype in _alert_rate_by_type(method["by_label"])[0]
        }
    )
    x = np.arange(len(all_types))
    bar_w = 0.38
    for offset, key in zip((-bar_w / 2, bar_w / 2), order):
        info = methods[key]["by_label"].get("alert", {}).get("by_pattern_type", {})
        rates = [info.get(ptype, {}).get("detection_rate", 0.0) for ptype in all_types]
        ax.bar(x + offset, rates, bar_w, label=names[key], color=colors[key])
    ax.set_xticks(x)
    ax.set_xticklabels(all_types, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("alert detection rate")
    ax.set_title("(a) Alert detection rate by pattern type")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (b) Mean recall / precision per label -----------------------------------
    ax = axes[0, 1]
    groups = [
        ("alert", "recall"),
        ("alert", "precision"),
        ("normal", "recall"),
        ("normal", "precision"),
    ]
    labels = [f"{lbl}\n{met}" for lbl, met in groups]
    x = np.arange(len(groups))
    for offset, key in zip((-bar_w / 2, bar_w / 2), order):
        vals = [
            _mean_metric(methods[key]["by_label"], lbl, f"mean_{met}")
            for lbl, met in groups
        ]
        ax.bar(x + offset, vals, bar_w, label=names[key], color=colors[key])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(threshold, ls="--", c="k", lw=1, label=f"threshold={threshold}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean value")
    ax.set_title("(b) Mean recall / precision by label")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (c) Per-alert recall vs precision scatter -------------------------------
    ax = axes[1, 0]
    for key in order:
        alerts = [d for d in methods[key]["detections"] if d.label == "alert"]
        rec = [d.recall for d in alerts]
        prec = [d.precision for d in alerts]
        ax.scatter(rec, prec, s=28, alpha=0.7, color=colors[key], label=names[key])
    ax.axvline(threshold, ls="--", c="k", lw=1)
    ax.axhline(threshold, ls="--", c="k", lw=1)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("(c) Per-alert recall vs precision\n(top-right quadrant = detected)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) Coarsening level sizes ----------------------------------------------
    ax = axes[1, 1]
    for key in order:
        sizes = methods[key]["coarsening"].sizes
        ax.plot(
            range(len(sizes)), sizes, marker="o", color=colors[key], label=names[key]
        )
    ax.set_xlabel("coarsening level")
    ax.set_ylabel("number of nodes")
    ax.set_title("(d) Multilevel coarsening sizes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="tutorial_demo16")
    parser.add_argument("--train-ratio", type=float, default=0.25)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--subspace-width", type=int, default=40)
    parser.add_argument("--reduction", type=float, default=0.7)
    parser.add_argument("--epsilon", type=float, default=math.inf)
    parser.add_argument("--max-levels", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.51)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-normal-train", action="store_true")
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
        )
    )
    train_patterns = (
        alert_train + normal_train if args.include_normal_train else alert_train
    )
    if not train_patterns:
        raise ValueError("no training patterns available for Eq. (48) theta fitting")
    test_patterns = alert_test + normal_test

    normalized, adjacency = graph_operators(graph)

    # Eq. (48): fit theta, then build the SGC target subspace span(g_theta(A)X).
    fit = fit_collective_sgc(
        normalized,
        train_patterns,
        degree=args.degree,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    sgc_basis = build_sgc_subspace(
        normalized, fit.theta, graph.x, width=args.subspace_width, seed=args.seed
    )

    # Baseline: span(U_K), the bottom-K combinatorial Laplacian eigenvectors.
    laplacian_basis = build_laplacian_subspace(adjacency, width=args.subspace_width)

    methods = {}
    for key, basis in (("sgc", sgc_basis), ("laplacian", laplacian_basis)):
        coarsening = loukas_coarsen_pytorch(
            adjacency,
            basis,
            reduction=args.reduction,
            epsilon=args.epsilon,
            max_levels=args.max_levels,
        )
        detections, by_label = evaluate_loukas_patterns(
            test_patterns,
            coarsening.node_to_supernode,
            graph.y,
            threshold=args.threshold,
        )
        methods[key] = {
            "basis_dim": basis.shape[1],
            "coarsening": coarsening,
            "detections": detections,
            "by_label": by_label,
        }

    # ---- console comparison -------------------------------------------------
    LOGGER.info("\nTarget-subspace comparison (Loukas RSA + Pattern-model detection)")
    LOGGER.info(f"  experiment: {args.experiment}")
    LOGGER.info(
        f"  theta objective lambda_min(G): {fit.objective:.6g} "
        f"(vanilla SGC: {fit.vanilla_sgc_objective:.6g})"
    )
    LOGGER.info(
        f"  detection rule: recall > {args.threshold} and precision > {args.threshold}\n"
    )

    header = f"  {'metric':28s} {'SGC span(gX)':>16s} {'Laplacian span(U_K)':>22s}"
    LOGGER.info(header)
    LOGGER.info("  " + "-" * (len(header) - 2))

    def _row(name: str, sgc_val, lap_val, pct: bool = False) -> None:
        fmt = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:.4g}")
        LOGGER.info(f"  {name:28s} {fmt(sgc_val):>16s} {fmt(lap_val):>22s}")

    _row(
        "R=span dimension",
        methods["sgc"]["basis_dim"],
        methods["laplacian"]["basis_dim"],
    )
    _row(
        "coarse nodes",
        methods["sgc"]["coarsening"].n_coarse,
        methods["laplacian"]["coarsening"].n_coarse,
    )
    _row(
        "reduction",
        methods["sgc"]["coarsening"].reduction,
        methods["laplacian"]["coarsening"].reduction,
        pct=True,
    )
    _row(
        "epsilon (RSA bound)",
        methods["sgc"]["coarsening"].epsilon,
        methods["laplacian"]["coarsening"].epsilon,
    )
    for label in ("alert", "normal"):
        _row(
            f"{label} detection rate",
            _mean_metric(methods["sgc"]["by_label"], label, "detection_rate"),
            _mean_metric(methods["laplacian"]["by_label"], label, "detection_rate"),
            pct=True,
        )
        _row(
            f"{label} mean recall",
            _mean_metric(methods["sgc"]["by_label"], label, "mean_recall"),
            _mean_metric(methods["laplacian"]["by_label"], label, "mean_recall"),
        )
        _row(
            f"{label} mean precision",
            _mean_metric(methods["sgc"]["by_label"], label, "mean_precision"),
            _mean_metric(methods["laplacian"]["by_label"], label, "mean_precision"),
        )

    output = args.output or Path(_utils_save_path) / "subspace_comparison.png"
    _plot_comparison(
        methods,
        threshold=args.threshold,
        experiment=args.experiment,
        output=output,
    )

    # JSON summary
    json_out = output.with_suffix(".json")
    import json, dataclasses

    summary = {
        "experiment": args.experiment,
        "theta_objective_lambda_min_G": fit.objective,
        "vanilla_sgc_objective": fit.vanilla_sgc_objective,
        "detection_threshold": args.threshold,
    }
    for key in ("sgc", "laplacian"):
        c = methods[key]["coarsening"]
        summary[key] = {
            "basis_dim": methods[key]["basis_dim"],
            "n_original": c.n_original,
            "n_coarse": c.n_coarse,
            "reduction": c.reduction,
            "epsilon": c.epsilon,
            "by_label": methods[key]["by_label"],
            "patterns": [dataclasses.asdict(d) for d in methods[key]["detections"]],
        }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2) + "\n")

    LOGGER.info(f"\nDiagnostic figure: {output}")
    LOGGER.info(f"JSON summary:      {json_out}")


if __name__ == "__main__":
    main()
