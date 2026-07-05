"""Compare SGC vs APPNP vs 2-layer GCN as the coarsening/classification encoder.

All three are trained on the *same* supervised node-classification objective and
evaluated identically, so the only thing that varies is the propagation:

    sgc    :  H = (A_hat^K X) W            (linear, fixed K-hop diffusion)
    appnp  :  H = PPR_alpha(A_hat) (X W)   (linear, personalized-PageRank teleport)
    gcn2   :  H = A_hat relu(A_hat X W0)   (2-layer, nonlinear)

For each, the learned node embedding ``H`` gives the Loukas coarsening target
``R = span(H)`` (scored by post-coarsening alert recall/precision) and feeds a
ridge-LDA head (scored by held-out alert AUC) -- the same two jobs and the same
metrics as ``run_joint_encoder_comparison``.

Run, e.g.::

    python -m src.run_propagation_comparison --experiment tutorial_demo16 \
        --degree 8 --embed-dim 16 --alpha 0.1 --reduction 0.7 --coarsening-method star
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import torch

from src.utils.utils import *  # noqa: F401,F403  (provides global `seed`)
from src.experiment_utils import load_and_preprocess_data
from src.loukas_sgc_detection import (
    _orthonormal_range,
    evaluate_loukas_patterns,
    graph_operators,
    loukas_coarsen_pytorch,
)
from src.sgc_detection import (
    _roc_auc,
    fit_feature_discriminant,
    score_feature_patterns,
)
from src.propagation_encoders import (
    APPNPEncoder,
    GCN2Encoder,
    SGCEncoder,
    fit_encoder,
    fit_retention_encoder,
)
from src.utils.utils import *


def _classifier_auc(normalized, train_patterns, test_patterns, embedding, *, ridge):
    discriminant = fit_feature_discriminant(
        normalized,
        train_patterns,
        features=embedding,
        degree=0,
        head="mean",
        ridge=ridge,
    )
    scores = score_feature_patterns(
        normalized, test_patterns, discriminant, features=embedding
    )
    alert = [float(s) for s, p in zip(scores, test_patterns) if p.label == "alert"]
    normal = [float(s) for s, p in zip(scores, test_patterns) if p.label != "alert"]
    return discriminant.train_auc, _roc_auc(alert, normal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="tutorial_demo16")
    parser.add_argument("--train-ratio", type=float, default=0.25)
    parser.add_argument("--degree", type=int, default=8, help="propagation hops K")
    parser.add_argument(
        "--embed-dim", type=int, default=16, help="node embedding dimension"
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="APPNP teleport")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--feature-ridge", type=float, default=1e-2)
    parser.add_argument(
        "--objective",
        choices=["supervised", "retention"],
        default="retention",
        help="how the encoder params are trained: 'supervised' (class-weighted CE "
        "on node labels) or 'retention' (unsupervised soft-min lambda_min(G) on "
        "the retain patterns -- the SGC coarsening objective with the filter "
        "swapped for APPNP/GCN)",
    )
    parser.add_argument(
        "--include-normal-train",
        action="store_true",
        default=True,
        help="retention objective: add normal patterns to the retain set",
    )
    parser.add_argument(
        "--ridge", type=float, default=1e-3, help="retention Gram ridge"
    )
    parser.add_argument(
        "--retention-mode", choices=["auto", "pattern", "channel"], default="auto"
    )
    parser.add_argument(
        "--retention-reduce", choices=["min", "softmin", "mean"], default="softmin"
    )
    parser.add_argument("--retention-temp", type=float, default=0.1)
    parser.add_argument("--reduction", type=float, default=0.7)
    parser.add_argument("--epsilon", type=float, default=float("inf"))
    parser.add_argument("--max-levels", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.51)
    parser.add_argument(
        "--coarsening-method",
        choices=["edges", "neighborhood", "capped", "star", "kmeans", "linkage"],
        default="edges",
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
        raise ValueError("propagation comparison requires node features (graph.x)")

    normalized, adjacency = graph_operators(graph)
    X = graph.x.to(device=normalized.device, dtype=normalized.dtype)
    y = graph.y.to(normalized.device)
    train_idx = graph.train_idx.to(normalized.device)
    feature_dim = X.shape[1]
    num_classes = int(y.max().item()) + 1

    clf_train = alert_train + normal_train
    clf_test = alert_test + normal_test
    eval_patterns = alert_test + normal_test

    retention = args.objective == "retention"
    retain = alert_train + (normal_train if args.include_normal_train else [])
    builders = {
        "sgc": lambda: SGCEncoder(
            feature_dim, args.embed_dim, num_classes, args.degree
        ),
        "appnp": lambda: APPNPEncoder(
            feature_dim,
            args.embed_dim,
            num_classes,
            args.degree,
            args.alpha,
            learn_alpha=retention,
        ),
        "gcn2": lambda: GCN2Encoder(feature_dim, args.embed_dim, num_classes),
    }

    rows = []
    for name, builder in builders.items():
        torch.manual_seed(seed)
        encoder = builder()
        if retention:
            embedding, _ = fit_retention_encoder(
                encoder,
                normalized,
                X,
                retain,
                ridge=args.ridge,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                retention_mode=args.retention_mode,
                retention_reduce=args.retention_reduce,
                retention_temp=args.retention_temp,
                seed=seed,
            )
        else:
            embedding = fit_encoder(
                encoder,
                normalized,
                X,
                y,
                train_idx,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=seed,
            )
        basis = _orthonormal_range(embedding)
        coarsening = loukas_coarsen_pytorch(
            adjacency,
            basis,
            reduction=args.reduction,
            epsilon=args.epsilon,
            max_levels=args.max_levels,
            method=args.coarsening_method,
        )
        _, by_label = evaluate_loukas_patterns(
            eval_patterns,
            coarsening.node_to_supernode,
            graph.y,
            threshold=args.threshold,
        )
        train_auc, test_auc = _classifier_auc(
            normalized, clf_train, clf_test, embedding, ridge=args.feature_ridge
        )
        alert = by_label.get("alert", {})
        rows.append(
            {
                "encoder": name,
                "basis_dim": int(basis.shape[1]),
                "n_coarse": coarsening.n_coarse,
                "epsilon": coarsening.epsilon,
                "alert_mean_recall": alert.get("mean_recall"),
                "alert_mean_precision": alert.get("mean_precision"),
                "alert_detection_rate": alert.get("detection_rate"),
                "alert_by_type": alert.get("by_pattern_type", {}),
                "classifier_train_auc": train_auc,
                "classifier_test_auc": test_auc,
            }
        )

    out_dir = Path(args.output) if args.output else Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / "propagation_comparison.json"
    json_out.write_text(
        json.dumps(
            {
                "experiment": args.experiment,
                "objective": args.objective,
                "degree": args.degree,
                "embed_dim": args.embed_dim,
                "alpha": args.alpha,
                "reduction": args.reduction,
                "coarsening_method": args.coarsening_method,
                "retention_reduce": args.retention_reduce,
                "encoders": rows,
            },
            indent=2,
        )
        + "\n"
    )

    LOGGER.info(
        "\nPropagation-backbone comparison  (H -> R=span(H) for coarsening; ridge-LDA head)"
    )
    LOGGER.info(
        f"  experiment={args.experiment}  K={args.degree}  embed_dim={args.embed_dim}  "
        f"alpha={args.alpha}  reduction={args.reduction:.0%}  method={args.coarsening_method}"
    )
    header = (
        f"  {'encoder':<8} {'dim':>4} {'n_coarse':>9} {'alert_recall':>13} "
        f"{'alert_prec':>11} {'alert_det':>10} {'clf_AUC(test)':>14} {'clf_AUC(train)':>15}"
    )
    LOGGER.info(header)
    LOGGER.info("  " + "-" * (len(header) - 3))
    for row in rows:
        LOGGER.info(
            f"  {row['encoder']:<8} {row['basis_dim']:>4} {row['n_coarse']:>9} "
            f"{(row['alert_mean_recall'] or 0):>13.3f} "
            f"{(row['alert_mean_precision'] or 0):>11.3f} "
            f"{(row['alert_detection_rate'] or 0):>10.1%} "
            f"{(row['classifier_test_auc'] or 0):>14.3f} "
            f"{(row['classifier_train_auc'] or 0):>15.3f}"
        )

    all_types = sorted({t for row in rows for t in row["alert_by_type"]})
    if all_types:
        LOGGER.info("\n  Alert detection rate by pattern type:")
        type_header = f"  {'encoder':<8}  " + "".join(f"{t:>14}" for t in all_types)
        LOGGER.info(type_header)
        LOGGER.info("  " + "-" * (len(type_header) - 2))
        for row in rows:
            cells = "".join(
                f"{row['alert_by_type'].get(t, {}).get('detection_rate', 0.0):>14.1%}"
                for t in all_types
            )
            LOGGER.info(f"  {row['encoder']:<8}  {cells}")

    LOGGER.info(f"\nJSON report:  {json_out}")


if __name__ == "__main__":
    main()
