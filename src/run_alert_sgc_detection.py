"""Train Eq. (48) SGC only on alert patterns, then score both test families.

This is intentionally separate from ``run_sgc_detection``.  It learns theta
and calibrates the threshold from ``alert_train`` only.  ``alert_test`` measures
the alert detection rate; ``normal_test`` measures normal-pattern activation
under an alert-only detector (the false-positive rate if alerts are the target).

Example:
    python -m src.run_alert_sgc_detection --experiment tutorial_demo16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.experiment_utils import load_and_preprocess_data
from src.sgc_detection import plot_diagnostics, save_report, train_and_detect


def _print_summary(by_label: dict) -> None:
    for label in ("alert", "normal"):
        if label not in by_label:
            continue
        values = by_label[label]
        heading = "alert detection rate" if label == "alert" else "normal-pattern activation rate"
        print(
            f"\nHeld-out {heading}: {int(values['detected'])}/{int(values['total'])} "
            f"({values['detection_rate']:.1%})"
        )
        for pattern_type, type_values in values["by_pattern_type"].items():
            print(
                f"  {pattern_type}: {int(type_values['detected'])}/{int(type_values['total'])} "
                f"({type_values['detection_rate']:.1%})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="tutorial_demo16")
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--threshold-quantile", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--remove-overlaps", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--plots-dir", type=Path, default=None)
    args = parser.parse_args()

    experiment_root = Path.cwd() / "experiments" / args.experiment
    graph, alert_train, _, alert_test, normal_test = load_and_preprocess_data(
        data_dir=experiment_root / "config",
        patterns_dir=experiment_root,
        train_ratio=args.train_ratio,
        to_undirected=True,
        remove_overlaps=args.remove_overlaps,
        device=torch.device(args.device),
    )
    if not alert_train:
        raise ValueError("alert_train is empty; Eq. (48) alert-only training cannot proceed")

    # Critical split boundary: normal patterns never enter theta fitting or
    # threshold calibration.  They are held out exclusively for evaluation.
    fit, detections, by_label = train_and_detect(
        graph,
        train_patterns=alert_train,
        test_patterns=alert_test + normal_test,
        degree=args.degree,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        threshold_quantile=args.threshold_quantile,
    )

    output = args.output or experiment_root / "results" / "alert_only_sgc_detection.json"
    save_report(output, fit, detections, by_label)
    plot_paths = plot_diagnostics(
        args.plots_dir or output.parent / "alert_only_sgc_plots",
        fit,
        detections,
        by_label,
    )

    print("\nAlert-only collective trainable SGC (Eq. 48)")
    print(f"  training alert patterns: {len(alert_train)}")
    print(f"  objective lambda_min(G): {fit.objective:.6g}")
    print(f"  vanilla-SGC lambda_min(G): {fit.vanilla_sgc_objective:.6g}")
    print(f"  learned theta: {fit.theta.detach().cpu().tolist()}")
    print(f"  alert-trained threshold: {fit.train_threshold:.6g}")
    _print_summary(by_label)
    print(f"\nJSON report: {output}")
    print("Plots:")
    for name, path in plot_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
