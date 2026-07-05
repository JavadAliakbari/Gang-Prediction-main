"""Request 2: turn the feature gang-classifier into *conductance* for coarsening.

The feature classifier is excellent at saying *which nodes are gang nodes*
(node AUC ~0.94).  This converts that node score ``s(v)`` into an edge
reweighting of the graph the coarsener contracts on, without touching the
(structural) target subspace:

    w'_{ij} = w_{ij} * (1 + beta * |s_i - s_j|)            (sign=+, "repel")

Boundary edges (large gang-score gap) get *heavier*, so the Loukas
local-variation cost ``trace(R^T L_C R)`` of any contraction that straddles a
gang boundary rises and the coarsener avoids merging a gang into its
neighbours -- the classifier knowledge enters as conductance, on top of the
structural basis.  ``sign=-`` (``w' = w / (1+beta|ds|)``, "attract") is the
opposite hypothesis (make within-gang edges cheap); we test both.

Run:
    /Users/javada/miniconda3/envs/FedStruct/bin/python -m src.run_feature_conductance \
        --experiment tutorial_demo16
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

from src.experiment_utils import load_and_preprocess_data
from src.loukas_sgc_detection import (
    build_sgc_subspace,
    evaluate_loukas_patterns,
    graph_operators,
    loukas_coarsen_pytorch,
)
from src.sgc_detection import fit_collective_sgc, fit_node_discriminant_map


def reweight_adjacency(adjacency: torch.Tensor, score: torch.Tensor, beta: float, sign: int):
    """Scale each edge weight by a function of its endpoints' gang-score gap."""
    idx = adjacency.indices()
    val = adjacency.values()
    gap = (score[idx[0]] - score[idx[1]]).abs()
    factor = 1.0 + beta * gap
    new_val = val * factor if sign > 0 else val / factor
    return torch.sparse_coo_tensor(idx, new_val, adjacency.shape).coalesce()


def score_detection(adjacency, basis, eval_patterns, node_labels, args, seed):
    coarsening = loukas_coarsen_pytorch(
        adjacency, basis, reduction=args.reduction, epsilon=float("inf"),
        max_levels=args.max_levels, method=args.coarsening_method,
        max_cluster_size=args.linkage_max_size, kmeans_seed=seed,
    )
    _, by_label = evaluate_loukas_patterns(
        eval_patterns, coarsening.node_to_supernode, node_labels, threshold=args.threshold
    )
    a = by_label.get("alert", {})
    return (a.get("detection_rate"), a.get("mean_recall"), a.get("mean_precision"),
            coarsening.n_coarse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="tutorial_demo16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--degree", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=0.005)
    ap.add_argument("--total-width", type=int, default=96)
    ap.add_argument("--reduction", type=float, default=0.7)
    ap.add_argument("--max-levels", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--coarsening-method", default="edges")
    ap.add_argument("--linkage-max-size", type=int, default=4)
    ap.add_argument("--betas", default="0.5,1,2,4,8")
    args = ap.parse_args()

    root = Path.cwd() / "experiments" / args.experiment
    graph, alert_train, normal_train, alert_test, normal_test = load_and_preprocess_data(
        data_dir=root / "config", patterns_dir=root, train_ratio=0.25,
        to_undirected=True, remove_overlaps=False, device=torch.device("cpu"),
        seed=args.seed,
    )
    normalized, adjacency = graph_operators(graph)
    X = graph.x.to(dtype=normalized.dtype)
    eval_patterns = alert_test + normal_test

    # structural target basis (Sigma_X = I), held fixed across all conductances.
    fit = fit_collective_sgc(normalized, alert_train + normal_train, features=None,
                             mode="lambda_min", degree=args.degree, epochs=args.epochs,
                             learning_rate=args.learning_rate,
                             retention_reduce="softmin", retention_temp=0.5)
    basis = build_sgc_subspace(normalized, fit.theta, None, width=args.total_width,
                               seed=args.seed)

    # node gang-score from the top feature discriminant direction.
    W = fit_node_discriminant_map(X, alert_train, embed_dim=1, ridge=1e-2)
    score = (X @ W[:, :1]).squeeze(1)
    score = (score - score.mean()) / score.std().clamp_min(1e-9)

    print("=" * 64)
    print(f"experiment={args.experiment}  feature-conductance on structural basis")
    print("=" * 64)
    det, rec, prec, nc = score_detection(adjacency, basis, eval_patterns, graph.y, args, args.seed)
    print(f"  baseline (no reweight)   det={det:.1%}  recall={rec:.3f}  prec={prec:.3f}  n_coarse={nc}")
    for sign, tag in ((+1, "repel boundary"), (-1, "attract within")):
        for beta in [float(b) for b in args.betas.split(",")]:
            A2 = reweight_adjacency(adjacency, score, beta, sign)
            det, rec, prec, nc = score_detection(A2, basis, eval_patterns, graph.y, args, args.seed)
            print(f"  sign={tag:<15} beta={beta:<4}  det={det:.1%}  "
                  f"recall={rec:.3f}  prec={prec:.3f}  n_coarse={nc}")


if __name__ == "__main__":
    main()
