"""Benchmark the contrastive-ratio discriminator against fisher/discriminative.

Implements the worst-case / margin objective

    theta* = argmax_theta  lambda_min(G_+) / lambda_max(G_-),
    G_pm   = V_pm^T g_theta(A_hat) X X^T g_theta(A_hat) V_pm   (feature-aware),

added to ``fit_collective_sgc`` as ``mode="contrastive_ratio"``.  Numerator keeps
every positive (no gang missed); denominator suppresses every negative (no
look-alike passed); the scale-free ratio > 1 means a hard worst-case separation.

We benchmark held-out alert-vs-normal AUC on the Elliptic++ gang setup
(alert = illicit connected-component gangs, normal = licit components) -- the
structurally-similar two-class task where retention alone (``lambda_min``) cannot
discriminate.  All discriminative modes see both classes at train time; AUC is
measured on the held-out split only.

Run::

    python -m src.run_contrastive_ratio_benchmark --day-start 24 --day-end 26
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import (
    build_torch_graph,
    load_node_features,
    make_patterns,
    split_train_test,
)
from src.experiment_utils import load_and_preprocess_data
from src.loukas_sgc_detection import graph_operators
from src.sgc_detection import _roc_auc, detect_patterns, fit_collective_sgc


def _load_elliptic(args, rng):
    """Elliptic++ gangs (alert) vs licit components (normal)."""

    A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, args.day_start, args.day_end)
    X = load_node_features(args.data_dir, nodes_df, args.day_start, args.day_end)
    graph = build_torch_graph(A_w, A_unw, cls, X, weighted=False)
    illicit = np.where(cls == 1)[0]
    licit = np.where(cls == 2)[0]
    gang_sets = connected_components_sets(A_unw, illicit, args.min_gang_size)
    licit_sets = sorted(
        connected_components_sets(A_unw, licit, args.min_gang_size),
        key=len,
        reverse=True,
    )[: args.max_normal_patterns]
    gangs = make_patterns(gang_sets, "alert", "gang", "g")
    normals = make_patterns(licit_sets, "normal", "normal", "n")
    gang_tr, gang_te = split_train_test(gangs, args.train_ratio, rng)
    norm_tr, norm_te = split_train_test(normals, args.train_ratio, rng)
    return graph, gang_tr + norm_tr, gang_te + norm_te, gang_tr, norm_tr, gang_te, norm_te


def _load_amlgentex(args):
    """AMLGenTex synthetic alert-vs-normal patterns (the intended task)."""

    root = Path.cwd() / "experiments" / args.experiment
    graph, alert_tr, normal_tr, alert_te, normal_te = load_and_preprocess_data(
        data_dir=root / "config",
        patterns_dir=root,
        train_ratio=args.train_ratio,
        to_undirected=True,
        remove_overlaps=False,
        device=torch.device("cpu"),
        seed=args.seed,
    )
    return (
        graph,
        alert_tr + normal_tr,
        alert_te + normal_te,
        alert_tr,
        normal_tr,
        alert_te,
        normal_te,
    )


def heldout_auc(adjacency, test_patterns, fit, features):
    """Held-out alert-vs-normal AUC from the margin scores of `detect_patterns`."""

    detections, _ = detect_patterns(adjacency, test_patterns, fit, features=features)
    alert = [d.score for d in detections if d.label == "alert"]
    normal = [d.score for d in detections if d.label == "normal"]
    return _roc_auc(alert, normal) if alert and normal else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", default=None,
                    help="AMLGenTex experiment name (alert-vs-normal); if unset, "
                    "use the Elliptic++ gang-vs-licit task")
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--day-start", type=int, default=24)
    ap.add_argument("--day-end", type=int, default=26)
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--max-normal-patterns", type=int, default=80)
    ap.add_argument("--train-ratio", type=float, default=0.5)
    ap.add_argument("--degree", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=0.01)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--retention-temp", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.experiment:
        print(f"=== contrastive-ratio benchmark | AMLGenTex {args.experiment} ===")
        graph, train, test, a_tr, n_tr, a_te, n_te = _load_amlgentex(args)
    else:
        print(
            f"=== contrastive-ratio benchmark | Elliptic++ days "
            f"{args.day_start}-{args.day_end} ==="
        )
        graph, train, test, a_tr, n_tr, a_te, n_te = _load_elliptic(args, rng)
    if getattr(graph, "x", None) is None:
        raise ValueError("benchmark requires node features (graph.x)")
    print(
        f"  train: {len(a_tr)} alert + {len(n_tr)} normal | "
        f"test: {len(a_te)} alert + {len(n_te)} normal\n"
    )

    normalized, _ = graph_operators(graph)
    Xf = graph.x.to(device=normalized.device, dtype=normalized.dtype)
    common = dict(
        degree=args.degree,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        features=Xf,
        ridge=args.ridge,
        retention_temp=args.retention_temp,
    )

    modes = ["lambda_min", "fisher", "discriminative", "contrastive_ratio"]
    print(
        f"{'mode':<18} {'train_AUC':>10} {'heldout_AUC':>12} "
        f"{'sep_ratio':>10} {'retain_floor':>13}"
    )
    print("-" * 66)
    rows = []
    for mode in modes:
        fit = fit_collective_sgc(normalized, train, mode=mode, **common)
        h_auc = heldout_auc(normalized, test, fit, Xf)
        sep = fit.separation_ratio
        rows.append((mode, fit.auc, h_auc, sep, fit.objective))
        print(
            f"{mode:<18} {(fit.auc or float('nan')):>10.3f} {h_auc:>12.3f} "
            f"{(sep if sep is not None else float('nan')):>10.3f} "
            f"{fit.objective:>13.4g}"
        )

    print(
        "\nRead: 'heldout_AUC' is alert-vs-normal separability of the learned "
        "theta on the held-out split (the discriminator's real job). 'sep_ratio' "
        "for contrastive_ratio is lambda_min(G+)/lambda_max(G-) at the optimum "
        "(>1 = worst-case separable). 'lambda_min(G+)' is the retention floor "
        "(shared scale). lambda_min retains but does not discriminate; the "
        "contrastive ratio is the robust/margin sibling of fisher."
    )


if __name__ == "__main__":
    main()
