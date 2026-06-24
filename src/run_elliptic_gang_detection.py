"""Detect illicit *gangs* in Elliptic++ Actors by spectral graph coarsening.

This wires the two existing pipelines together on a real AML graph:

* ``run_elliptic_gang_conductance.build_graph`` reads the Elliptic++ Actors
  (wallet-address) dataset for a chosen day window and returns the transaction
  graph plus class labels.  The illicit *gangs* are the connected components
  (size >= 2) of the illicit-induced subgraph -- exactly the low-conductance,
  densely connected motifs studied in "Graph Coarsening for Gang Detection".

* ``run_joint_encoder_comparison`` compares three *linear* encoders, each used
  as the target subspace ``R = span(Z)`` for the identical Loukas RSA coarsening
  (Loukas Algorithm 1).  A gang is **detected** when the coarsening collapses it
  into (mostly) a single super-node: recall = (largest share of the gang landing
  in one super-node)/|gang| and precision = (that share)/|super-node|, with a
  gang counted detected when both exceed ``--threshold``.  This is the
  "each gang becomes one super-node" criterion from the slides.

Per the request we **omit the Laplacian baseline** -- the bottom-K eigenvectors
of L need a full eigendecomposition, which is too expensive on a ~50k-node
graph.  We keep the three filter-based encoders, all of which avoid any dense
eigendecomposition:

    structural   :  Z = g_theta(A_hat) Omega            (random range finder)
    raw-feature  :  Z = g_theta(A_hat) X                 (wallet features)
    joint        :  Z = [g_theta(A_hat) Omega | g_theta(A_hat) X W]

Days 24-26 carry the most illicit accounts (550 + 1099 + 714 = 2363) while the
graph stays a manageable ~49k nodes, so they are the default window.

Run::

    conda activate FedStruct
    python -m src.run_elliptic_gang_detection \
        --day-start 24 --day-end 26 --coarsening-method edges --epsilon 0.3

Option 2 (default): multi-level ``edges`` local-variation coarsening governed by
a finite ``--epsilon`` RSA cost gate.  Each level contracts a cheapest-first
matching (per-level cap = 2), so a gang collapses hierarchically over ~log2(|gang|)
levels -- reaching even the large gangs that a single capped pass cannot -- while
the epsilon gate refuses the expensive gang-boundary merges that would contaminate
a super-node.  The gate scores edges by the spectral cost of the encoder subspace
only, so the *decision* never sees the gang labels; precision/recall are computed
afterwards for reporting.  ``--epsilon`` is the single precision<->recall knob
(smaller = purer/fewer, larger = more complete/contaminated).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import torch
from scipy.sparse import triu

from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.pattern_models import create_pattern
from src.loukas_sgc_detection import (
    build_joint_subspace,
    build_sgc_subspace,
    evaluate_loukas_patterns,
    graph_operators,
    loukas_coarsen_pytorch,
)
from src.sgc_detection import (
    apply_feature_channel,
    apply_graph_filter,
    fit_collective_sgc,
    fit_joint_encoder,
)

# ---------------------------------------------------------------------------
# Feature matrix aligned to the build_graph node ordering
# ---------------------------------------------------------------------------

_NON_FEATURE_COLS = {"address", "Time step", "class", "Time_step"}


def load_node_features(
    data_dir: Path, nodes_df: pd.DataFrame, day_start: int, day_end: int
) -> torch.Tensor:
    """Standardised wallet-feature matrix X aligned to ``nodes_df`` row order.

    ``build_graph`` assigns node index i to ``nodes_df['address'].iloc[i]``; we
    reindex the feature rows by that address order so X[i] is node i's features.
    """

    feat = pd.read_csv(data_dir / "wallets_features.csv", dtype={"address": str})
    feat = feat[feat["Time step"].between(day_start, day_end)]
    feat = feat.drop_duplicates(subset=["address"], keep="last").set_index("address")
    feat = feat.reindex(nodes_df["address"].values)  # align to node index order

    num = feat.drop(columns=[c for c in _NON_FEATURE_COLS if c in feat.columns])
    num = num.select_dtypes(include=[np.number])
    X = num.to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # z-score (drop zero-variance columns so they don't blow up)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    keep = sd > 1e-12
    X = (X[:, keep] - mu[keep]) / sd[keep]
    print(f"  Feature matrix X: {X.shape[0]:,} x {X.shape[1]} (standardised)")
    return torch.from_numpy(X).to(torch.float64)


# ---------------------------------------------------------------------------
# Build the torch graph + patterns
# ---------------------------------------------------------------------------


def build_torch_graph(A_w, A_unw, cls, X, weighted: bool):
    """Lightweight graph object exposing the attributes the encoders need."""

    src_mat = A_w if weighted else A_unw
    up = triu(src_mat, k=1).tocoo()  # one direction; operators re-symmetrise
    edge_index = torch.tensor(np.vstack([up.row, up.col]), dtype=torch.long)
    edge_weight = torch.tensor(up.data, dtype=torch.float64)
    y = torch.tensor((cls == 1).astype(np.int64))  # 1 = illicit, 0 = otherwise
    graph = SimpleNamespace(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=int(A_unw.shape[0]),
        x=X.to(torch.float64),
        y=y,
    )
    return graph


def make_patterns(sets, label, pattern_type, prefix):
    """Wrap node-index arrays as Pattern objects with the given label."""

    return [
        create_pattern(
            pattern_id=f"{prefix}{i}",
            nodes=[int(v) for v in S],
            pattern_type=pattern_type,
            label=label,  # 'alert' (gang) or 'normal'; the encoders key on this
        )
        for i, S in enumerate(sets)
    ]


def split_train_test(patterns, train_ratio, rng):
    idx = np.arange(len(patterns))
    rng.shuffle(idx)
    cut = int(round(train_ratio * len(patterns)))
    train = [patterns[i] for i in idx[:cut]]
    test = [patterns[i] for i in idx[cut:]]
    return train, test


# ---------------------------------------------------------------------------
# Encoder construction + coarsening
# ---------------------------------------------------------------------------


def coarsen_and_count(name, basis, *, adjacency, node_labels, eval_patterns, args):
    """Coarsen with R=span(basis) and count detected gangs / normals."""

    coarsening = loukas_coarsen_pytorch(
        adjacency,
        basis,
        reduction=args.reduction,
        epsilon=args.epsilon,  # option 2: finite -> label-free RSA cost gate
        max_levels=args.max_levels,
        method=args.coarsening_method,
        max_contraction_size=args.max_contraction_size,
        max_cluster_size=args.linkage_max_size,
    )
    _, by_label = evaluate_loukas_patterns(
        eval_patterns,
        coarsening.node_to_supernode,
        node_labels,
        threshold=args.threshold,
    )
    gang = by_label.get("alert", {})
    normal = by_label.get("normal", {})
    return {
        "encoder": name,
        "basis_dim": int(basis.shape[1]),
        "n_original": coarsening.n_original,
        "n_coarse": coarsening.n_coarse,
        "n_levels": len(coarsening.sigmas),
        "epsilon_used": args.epsilon,
        "epsilon_actual": coarsening.epsilon,  # cumulative RSA bound spent
        "reduction_actual": 1.0 - coarsening.n_coarse / coarsening.n_original,
        "gangs_detected": int(gang.get("detected", 0)),
        "gangs_total": int(gang.get("total", 0)),
        "gang_detection_rate": gang.get("detection_rate", 0.0),
        "gang_mean_recall": gang.get("mean_recall", 0.0),
        "gang_mean_precision": gang.get("mean_precision", 0.0),
        "normal_detected": int(normal.get("detected", 0)),
        "normal_total": int(normal.get("total", 0)),
        "normal_detection_rate": normal.get("detection_rate", 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--day-start", type=int, default=24)
    ap.add_argument("--day-end", type=int, default=26)
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument(
        "--weighted",
        action="store_true",
        default=False,
        help="use transaction multiplicity as edge weight (else 0/1)",
    )
    ap.add_argument("--train-ratio", type=float, default=0.5)
    ap.add_argument(
        "--max-normal-patterns",
        type=int,
        default=80,
        help="cap on licit components used as the 'normal' class",
    )
    # encoder / coarsening hyper-parameters (mirror run_joint_encoder_comparison)
    ap.add_argument("--degree", type=int, default=16)
    ap.add_argument("--structural-width", type=int, default=32)
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=0.01)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument(
        "--reduction",
        type=float,
        default=0.99,
        help="count-based stop (1 - n_coarse/N). In option 2 keep this "
        "high so the *epsilon* RSA cost gate governs the stop instead",
    )
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument(
        "--epsilon",
        type=float,
        default=5.0,
        help="OPTION 2 -- label-free RSA cost budget (prod_l (1+sigma_l) - 1). "
        "A merge is refused once it would push the cumulative spectral error past "
        "this bound, so cheap gang-internal edges contract while expensive "
        "gang-boundary edges are deferred. This is the precision knob and uses "
        "ONLY the encoder subspace, never the gang labels. inf = no gate",
    )
    ap.add_argument(
        "--max-levels",
        type=int,
        default=20,
        help="option 2 collapses gangs hierarchically over many levels "
        "(a size-k gang needs ~log2(k) edge-matching levels)",
    )
    ap.add_argument(
        "--coarsening-method",
        choices=["edges", "neighborhood", "capped", "star", "kmeans", "linkage"],
        default="edges",
        help="local-variation candidate family. 'edges' (default, option 2) is "
        "canonical Loukas Algorithm 2: one cheapest-first matching per level, so "
        "per-level cap = 2 and gangs collapse multiplicatively across levels under "
        "the epsilon gate. 'capped' raises the per-level cap to "
        "--max-contraction-size; 'linkage' is the fast one-pass single-linkage "
        "variant (use with --linkage-max-size, not epsilon)",
    )
    ap.add_argument(
        "--max-contraction-size",
        type=int,
        default=4,
        help="per-level contraction-set cap for --coarsening-method capped "
        "(2 = edges, larger -> fewer levels but coarser per-level steps)",
    )
    ap.add_argument(
        "--linkage-max-size",
        type=int,
        default=2,
        help="super-node size cap for --coarsening-method linkage (curbs single-"
        "linkage chaining so a collapsed gang stays pure; 0 = uncapped)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/elliptic_gang_detection", type=Path)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    print(f"=== Elliptic++ gang detection | days {args.day_start}-{args.day_end} ===")
    A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, args.day_start, args.day_end)
    X = load_node_features(args.data_dir, nodes_df, args.day_start, args.day_end)
    graph = build_torch_graph(A_w, A_unw, cls, X, weighted=args.weighted)

    # --- gangs (alert) and a licit 'normal' contrast set ---
    illicit_idx = np.where(cls == 1)[0]
    licit_idx = np.where(cls == 2)[0]
    gang_sets = connected_components_sets(A_unw, illicit_idx, args.min_gang_size)
    licit_sets = connected_components_sets(A_unw, licit_idx, args.min_gang_size)
    # cap the normal class to the largest licit components for a balanced retain
    licit_sets = sorted(licit_sets, key=len, reverse=True)[: args.max_normal_patterns]

    gangs = make_patterns(gang_sets, "alert", "gang", "g")
    normals = make_patterns(licit_sets, "normal", "normal", "n")
    print(
        f"\n  Gangs (illicit CC>= {args.min_gang_size}): {len(gangs)}  | "
        f"normal licit components used: {len(normals)}"
    )
    print(f"  Gang sizes: {sorted((p.num_nodes for p in gangs), reverse=True)}")

    gang_train, gang_test = split_train_test(gangs, args.train_ratio, rng)
    normal_train, normal_test = split_train_test(normals, args.train_ratio, rng)
    retain = gang_train + normal_train
    # headline metric: detection over *all* gangs (plus held-out normals for FP).
    eval_patterns = gangs + normal_test
    print(
        f"  retain (train) patterns: {len(gang_train)} gangs + "
        f"{len(normal_train)} normal; evaluating on all {len(gangs)} gangs"
    )

    normalized, adjacency = graph_operators(graph)
    Xf = graph.x.to(device=normalized.device, dtype=normalized.dtype)
    total_width = args.structural_width + args.embed_dim
    common = dict(
        degree=args.degree, epochs=args.epochs, learning_rate=args.learning_rate
    )

    print("\n  Fitting structural encoder (theta on structural Gram) …")
    structural_fit = fit_collective_sgc(
        normalized, retain, features=None, mode="lambda_min", **common
    )
    structural_basis = build_sgc_subspace(
        normalized, structural_fit.theta, None, width=total_width, seed=args.seed
    )

    print("  Fitting raw-feature encoder (theta on feature-aware Gram) …")
    feature_fit = fit_collective_sgc(
        normalized, retain, features=Xf, mode="lambda_min", **common
    )
    feature_basis = build_sgc_subspace(
        normalized, feature_fit.theta, Xf, width=total_width, seed=args.seed
    )

    print("  Fitting joint encoder (theta, W) on lambda_min(G(theta,W)) …")
    joint = fit_joint_encoder(
        normalized,
        retain,
        features=Xf,
        degree=args.degree,
        embed_dim=args.embed_dim,
        structural_width=args.structural_width,
        ridge=args.ridge,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        label_patterns=gang_train + normal_train,
        label_weight=1.0,
    )
    joint_basis = build_joint_subspace(
        normalized,
        joint.theta,
        Xf,
        joint.feature_map,
        structural_width=args.structural_width,
        seed=args.seed,
        per_hop=joint.per_hop_features,
    )

    encoders = [
        ("structural", structural_basis),
        ("raw-feature", feature_basis),
        ("joint", joint_basis),
    ]
    eps_desc = "inf" if args.epsilon == float("inf") else f"{args.epsilon:g}"
    print(
        f"\n  Coarsening (method={args.coarsening_method}, epsilon={eps_desc}, "
        f"max_levels={args.max_levels}, reduction-cap={args.reduction:.0%}) "
        f"and counting detected gangs …"
    )
    rows = [
        coarsen_and_count(
            name,
            basis,
            adjacency=adjacency,
            node_labels=graph.y,
            eval_patterns=eval_patterns,
            args=args,
        )
        for name, basis in encoders
    ]

    # ---- report -----------------------------------------------------------
    out_json = args.out / f"gang_detection_d{args.day_start}-{args.day_end}.json"
    out_json.write_text(
        json.dumps(
            {
                "day_start": args.day_start,
                "day_end": args.day_end,
                "weighted": args.weighted,
                "n_nodes": graph.num_nodes,
                "n_edges": int(graph.edge_index.shape[1]),
                "n_gangs": len(gangs),
                "reduction": args.reduction,
                "epsilon": args.epsilon,
                "max_levels": args.max_levels,
                "coarsening_method": args.coarsening_method,
                "threshold": args.threshold,
                "encoders": rows,
            },
            indent=2,
        )
        + "\n"
    )

    print("\n" + "=" * 78)
    print(
        f"GANG DETECTION — Elliptic++ days {args.day_start}-{args.day_end}  "
        f"(N={graph.num_nodes:,}, E={graph.edge_index.shape[1]:,}, "
        f"{len(gangs)} gangs, reduction={args.reduction:.0%})"
    )
    print("=" * 78)
    hdr = (
        f"{'encoder':<12} {'n_coarse':>9} {'lvl':>4} {'eps':>6} "
        f"{'gangs_detected':>15} {'det_rate':>9} {'mean_recall':>12} "
        f"{'mean_prec':>10} {'normal_FP':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['encoder']:<12} {r['n_coarse']:>9,} {r['n_levels']:>4} "
            f"{r['epsilon_actual']:>6.2f} "
            f"{r['gangs_detected']:>7}/{r['gangs_total']:<7} "
            f"{r['gang_detection_rate']:>8.1%} "
            f"{r['gang_mean_recall']:>12.3f} {r['gang_mean_precision']:>10.3f} "
            f"{r['normal_detected']:>4}/{r['normal_total']:<5}"
        )
    print(
        "\nRead: 'gangs_detected' = illicit connected-component motifs that "
        "collapse into (mostly) a single super-node under the coarsening "
        "(recall>thr AND precision>thr). 'eps' is the cumulative RSA cost actually "
        "spent (the label-free stop). 'normal_FP' = licit components that also "
        "collapse (lower is better). Precision/recall are reported for evaluation "
        "only; the coarsening decision uses epsilon, never the gang labels."
    )
    print(f"\nJSON report: {out_json}")


if __name__ == "__main__":
    main()
