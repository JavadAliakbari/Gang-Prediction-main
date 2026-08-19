"""Analysis + plots of the *learned* coarsening on the Elliptic++ Actors graph.

Runs the modular :class:`CollectiveBankDetector`, then measures and visualizes
what the learned target subspace buys the coarsening.  The unifying finding is
that **a gang's conductance controls everything**: low-conductance (tight) gangs
get near-constant target rows, hence cheap internal edges, hence they collapse
into one supernode and are detected; the few sprawling high-conductance illicit
components do not.  Four deliverables:

1. **Per-gang precision / recall** (large gangs first).
2. **Edge-cost geometry.** The coarsener charges each edge the RSA local variation
   ``c(u,v) = ||A[u]-A[v]||^2`` on the ``M_tau``-orthonormal target rows ``A``.  We
   show *per gang* that internal edges are cheaper than that gang's boundary edges
   (a scatter below the diagonal), the aggregate split, and a node-link drawing of
   a tight gang with edges colored by cost (cheap interior, costly boundary).
3. **Supernode conductance** ``Phi(S)=cut(S)/vol(S)`` -- gang supernodes vs
   background -- and the conductance -> detection link that explains (1)-(2).
4. **Non-gang (licit) precision / recall** and the false-collapse rate.

Run::

    python -m src.analyze_elliptic_coarsening --day-start 24 --day-end 24 \
        --feature-mode random --random-width 128 --epochs 400 \
        --out results/elliptic_analysis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

from src.analyze_elliptic_coarsening import analyze_coarsening
from src.pattern_models import make_patterns
from src.collective_detector import CollectiveBankDetector, DetectorConfig, GraphData
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import (
    build_torch_graph,
    load_node_features,
    split_train_test,
)
from src.run_elliptic_modular import random_structural_features


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--day-start", type=int, default=24)
    ap.add_argument("--day-end", type=int, default=26)
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--feature-mode", choices=["wallet", "random"], default="random")
    ap.add_argument("--random-width", type=int, default=128)
    ap.add_argument("--max-normal-patterns", type=int, default=120)
    ap.add_argument("--train-ratio", type=float, default=0.5)
    ap.add_argument("--degree", type=int, default=12)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--conf-weight", type=float, default=5.0)
    ap.add_argument("--conf-delta", type=float, default=0.0)
    ap.add_argument("--coarsen-target", choices=["bank", "indicators"], default="bank")
    ap.add_argument("--coarsening-method", default="ward-tree")
    ap.add_argument("--ward-stop", choices=["epsilon", "f1"], default="f1")
    ap.add_argument("--epsilon", type=float, default=0.8)
    ap.add_argument("--ward-num-cuts", type=int, default=150)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument(
        "--num-gang-graphs",
        type=int,
        default=2,
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/elliptic_analysis", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(
        f"=== Elliptic++ coarsening analysis | days {args.day_start}-{args.day_end} ==="
    )
    A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, args.day_start, args.day_end)
    if args.feature_mode == "wallet":
        Xfeat = load_node_features(
            args.data_dir, nodes_df, args.day_start, args.day_end
        )
    else:
        Xfeat = random_structural_features(
            int(A_unw.shape[0]), args.random_width, args.seed
        )
    graph = build_torch_graph(A_w, A_unw, cls, Xfeat, weighted=False)

    gang_sets = connected_components_sets(
        A_unw, np.where(cls == 1)[0], args.min_gang_size
    )
    gangs = make_patterns(gang_sets, "alert", "gang", "g")
    licit_sets = connected_components_sets(
        A_unw, np.where(cls == 2)[0], args.min_gang_size
    )
    licit_sets = sorted(licit_sets, key=len, reverse=True)[: args.max_normal_patterns]
    normals = make_patterns(licit_sets, "normal", "normal", "n")
    print(
        f"  gangs={len(gangs)} (sizes {sorted((p.num_nodes for p in gangs), reverse=True)[:8]}...)  normals={len(normals)}"
    )

    rng = np.random.default_rng(args.seed)
    gang_train, gang_test = split_train_test(gangs, args.train_ratio, rng)

    data = GraphData.from_graph(graph)
    cfg = DetectorConfig(
        degree=args.degree,
        tau=args.tau,
        epochs=args.epochs,
        conf_weight=args.conf_weight,
        conf_reduce="mean",
        conf_delta=args.conf_delta,
        optimizer="riemannian",
        coarsen_target=args.coarsen_target,
        coarsening_method=args.coarsening_method,
        ward_stop=args.ward_stop,
        epsilon=args.epsilon,
        ward_num_cuts=args.ward_num_cuts,
        threshold=args.threshold,
        seed=args.seed,
    )
    det = CollectiveBankDetector(cfg)
    print(
        f"  fitting (K={cfg.degree}, tau={cfg.tau}, beta={cfg.conf_weight}, target={cfg.coarsen_target}) …"
    )
    det.fit(data, gang_train)
    basis = det.target_subspace(data, gang_train)
    coarsening, _ = det.coarsen(data, basis, gang_train)
    n2s = coarsening.node_to_supernode
    print(
        f"    lambda_min {det.fit_info_['init_objective']:.4g} -> {det.fit_info_['objective']:.4g}"
        f"   N={coarsening.n_original:,} -> {coarsening.n_coarse:,}  epsilon={coarsening.epsilon:.4g}"
    )

    analyze_coarsening(
        data,
        basis,
        gangs,
        det,
        gang_train,
        args,
        cfg,
        coarsening,
        n2s,
        normals,
    )


if __name__ == "__main__":
    main()
