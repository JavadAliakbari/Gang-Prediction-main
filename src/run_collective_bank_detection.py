"""Controlled collective learnable-filter (``L_sym``) gang-detection experiment.

This is an end-to-end, *controlled* implementation of the collective detect-all
objective of Section "Collective Detection":

    Theta* = argmax_{||theta^(a)||=1 for all a}  lambda_min(Gamma(Theta)),
    Gamma(Theta) = Vhat^T L Z (Z^T L Z)^+ Z^T L Vhat  in R^{m x m},
    Z[:, a] = g_{theta^(a)}(A_hat) x_a = sum_k Theta_{ka} A_hat^k x_a  (one filter
                                                                       per channel),

with everything measured in the *symmetric normalized* metric ``L = I - A_hat``
(the analysis switches from the plain ``l2`` norm to the ``L_sym`` seminorm), and
``Vhat`` the ``L``-normalized *degree-weighted* gang indicators
``v_S = D_tilde^{1/2} 1_S / sqrt(vol(S))`` (so ``||v_S||_L^2 = Phi = cut/vol``).

Pipeline (the seven requested steps):

1. build a random (Erdos-Renyi) background graph with ``--num-nodes`` nodes;
2. plant ``--num-motifs`` dense motifs of a chosen ``--motif-type``
   (``clique`` / ``cycle`` / ``star`` / ``random``, the last with a tunable
   ``--motif-density``);
3. hold out a fraction, training on ``--train-ratio`` of the motifs;
4. learn the per-channel filter bank ``Theta`` by ascending
   ``lambda_min(Gamma(Theta))`` on the *training* motifs only;
5. form the embedding ``Z = g_Theta(A_hat) X`` and the target subspace
   ``R = span(Z)`` from the learned filters;
6. hand ``R`` to the Loukas RSA coarsening (reusing the existing
   :func:`loukas_coarsen_pytorch`);
7. report post-coarsening recall / precision / detection rate (reusing
   :func:`evaluate_loukas_patterns`), broken out over train / test / all motifs.

Note on the coarsening metric.  The Loukas coarsening historically measured RSA
distortion in the *combinatorial* Laplacian ``L = D - W``; the algorithm above is
derived in the *symmetric normalized* ``L = I - A_hat``.  Both are now exposed
through ``--coarsening-laplacian`` (default ``symmetric`` to match the algorithm).

Run, e.g.::

    python -m src.run_collective_bank_detection \
        --motif-type clique --num-motifs 10 --num-nodes 2000 \
        --train-ratio 0.4 --degree 10 --feature-dim 64 --reduction 0.85
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from itertools import combinations
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data

from src.utils.utils import *
from src.sgc_detection import propagation_stack
from src.loukas_sgc_detection import (
    _degrees,
    _orthonormal_range,
    evaluate_loukas_patterns,
    graph_operators,
    loukas_coarsen_pytorch,
)
from src.pattern_models import create_pattern


# --------------------------------------------------------------------------- #
# 1-2.  synthetic random graph with planted dense motifs
# --------------------------------------------------------------------------- #
def _motif_edges(
    nodes: list[int],
    motif_type: str,
    *,
    density: float = 1.0,
    rng: "np.random.Generator | None" = None,
) -> list[tuple[int, int]]:
    """Return the undirected edge list of one planted motif on ``nodes``.

    ``motif_type="random"`` plants an Erdos-Renyi ``G(s, density)`` subgraph: a
    random spanning path is always included first (so the motif is connected /
    a valid single gang), then extra random pairs are added until the edge count
    reaches ``round(density * s(s-1)/2)``.  ``density=1.0`` reproduces the clique.
    """

    s = len(nodes)
    if motif_type == "clique":
        return [(int(u), int(v)) for u, v in combinations(nodes, 2)]
    if motif_type == "cycle":
        return [(int(nodes[i]), int(nodes[(i + 1) % s])) for i in range(s)]
    if motif_type == "star":
        hub = int(nodes[0])
        return [(hub, int(nodes[i])) for i in range(1, s)]
    if motif_type == "random":
        if rng is None:
            raise ValueError("random motif requires a numpy Generator via `rng`")
        if not 0.0 <= density <= 1.0:
            raise ValueError("--motif-density must be in [0, 1]")
        perm = [int(nodes[i]) for i in rng.permutation(s)]
        motif: set[tuple[int, int]] = set()
        for i in range(s - 1):  # spanning path -> guaranteed connected
            a, b = perm[i], perm[i + 1]
            motif.add((min(a, b), max(a, b)))
        max_edges = s * (s - 1) // 2
        target = max(len(motif), int(round(float(density) * max_edges)))
        pairs = [
            (int(u), int(v)) for u, v in combinations(sorted(int(n) for n in nodes), 2)
        ]
        for idx in rng.permutation(len(pairs)):
            if len(motif) >= target:
                break
            u, v = pairs[idx]
            motif.add((min(u, v), max(u, v)))
        return sorted(motif)
    raise ValueError("motif_type must be 'clique', 'cycle', 'star', or 'random'")


def build_synthetic_graph(
    *,
    num_nodes: int,
    num_motifs: int,
    motif_type: str,
    motif_size: int,
    avg_degree: float,
    feature_dim: int,
    rng_seed: int,
    motif_density: float = 1.0,
) -> tuple[Data, list]:
    """Erdos-Renyi background + ``num_motifs`` disjoint planted motifs.

    Returns the graph (``x``, ``edge_index``, ``y``) and the list of
    :class:`Pattern` objects (label ``"alert"``) for evaluation.  Node labels
    ``y`` mark every motif node as class 1 so the coarsening evaluation can pool
    pseudo-labels.
    """

    if num_motifs * motif_size > num_nodes:
        raise ValueError("num_motifs * motif_size exceeds num_nodes")

    rng = np.random.default_rng(rng_seed)

    # --- disjoint node blocks for the motifs ---------------------------------
    perm = rng.permutation(num_nodes)
    motif_nodes = perm[: num_motifs * motif_size].reshape(num_motifs, motif_size)

    edges: set[tuple[int, int]] = set()

    # --- Erdos-Renyi background ---------------------------------------------
    n_background = int(num_nodes * avg_degree / 2)
    src = rng.integers(0, num_nodes, size=n_background)
    dst = rng.integers(0, num_nodes, size=n_background)
    for u, v in zip(src.tolist(), dst.tolist()):
        if u != v:
            edges.add((min(u, v), max(u, v)))

    # --- planted motifs ------------------------------------------------------
    patterns = []
    y = np.zeros(num_nodes, dtype=np.int64)
    for m in range(num_motifs):
        nodes = motif_nodes[m].tolist()
        for u, v in _motif_edges(nodes, motif_type, density=motif_density, rng=rng):
            edges.add((min(u, v), max(u, v)))
        y[nodes] = 1
        patterns.append(
            create_pattern(f"{motif_type}_{m}", nodes, motif_type, label="alert")
        )

    # --- undirected edge_index ----------------------------------------------
    edge_array = np.array(sorted(edges), dtype=np.int64).T  # (2, E)
    edge_index = torch.from_numpy(
        np.concatenate([edge_array, edge_array[::-1]], axis=1)
    ).long()

    # --- isotropic node features (the reachability channel of the theory) ----
    gen = torch.Generator().manual_seed(rng_seed)
    X = torch.randn(num_nodes, feature_dim, dtype=torch.float64, generator=gen)
    X = (X - X.mean(0, keepdim=True)) / X.std(0, keepdim=True).clamp_min(1e-8)

    graph = Data(
        x=X,
        edge_index=edge_index,
        y=torch.from_numpy(y),
        num_nodes=num_nodes,
    )
    return graph, patterns


# --------------------------------------------------------------------------- #
# 3-4.  collective L_sym filter-bank learning
# --------------------------------------------------------------------------- #
def degree_weighted_indicators(adjacency: torch.Tensor, patterns: list) -> torch.Tensor:
    """Degree-weighted indicators ``v_S = D_tilde^{1/2} 1_S / sqrt(vol(S))``.

    ``D_tilde = D + I`` matches the self-loop renormalization of ``A_hat``, so
    ``||v_S||_L^2 = Phi(S)`` under ``L = I - A_hat``.
    """

    n = adjacency.shape[0]
    dtype, device = adjacency.dtype, adjacency.device
    d_tilde = _degrees(adjacency) + 1.0  # self-loop augmented degree
    columns = []
    for pattern in patterns:
        nodes = torch.as_tensor(pattern.node_indices, dtype=torch.long, device=device)
        column = torch.zeros(n, dtype=dtype, device=device)
        column[nodes] = d_tilde[nodes].sqrt()
        column = column / d_tilde[nodes].sum().clamp_min(torch.finfo(dtype).eps).sqrt()
        columns.append(column)
    return torch.stack(columns, dim=1)  # (N, m)


def _l_apply(a_hat: torch.Tensor, signals: torch.Tensor) -> torch.Tensor:
    """Apply ``L = I - A_hat`` to dense ``signals`` (columns are graph signals)."""

    return signals - torch.sparse.mm(a_hat, signals)


def _filtered_bank(propagated: list[torch.Tensor], theta: torch.Tensor) -> torch.Tensor:
    """Per-channel filter bank ``Z[:, a] = sum_k theta[k, a] (A_hat^k X)[:, a]``.

    ``propagated[k]`` is ``A_hat^k X`` of shape ``(N, d)`` and ``theta`` is
    ``(K+1, d)``; each hop scales every channel by its own coefficient.
    """

    return sum(propagated[k] * theta[k].unsqueeze(0) for k in range(theta.shape[0]))


def _collective_gamma(
    a_hat: torch.Tensor,
    Z: torch.Tensor,
    l_vhat: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """Collective ``L``-Gram ``Gamma = Vhat^T L Z (Z^T L Z)^+ Z^T L Vhat``.

    ``l_vhat = L Vhat`` is precomputed (independent of ``theta``).  A small
    ``ridge`` stabilizes the pseudo-inverse of the ``d x d`` channel Gram.
    """

    l_z = _l_apply(a_hat, Z)  # L Z            (N, d)
    g_z = Z.T @ l_z  # Z^T L Z               (d, d)
    g_z = 0.5 * (g_z + g_z.T)
    m = Z.T @ l_vhat  # Z^T L Vhat            (d, m)
    eye = torch.eye(g_z.shape[0], dtype=g_z.dtype, device=g_z.device)
    g_inv_m = torch.linalg.solve(g_z + ridge * eye, m)  # (d, m)
    gamma = m.T @ g_inv_m  # (m, m)
    return 0.5 * (gamma + gamma.T)


def fit_collective_bank(
    a_hat: torch.Tensor,
    adjacency: torch.Tensor,
    train_patterns: list,
    X: torch.Tensor,
    *,
    degree: int,
    epochs: int,
    learning_rate: float,
    ridge: float,
    fit_seed: int,
) -> dict:
    """Ascend ``lambda_min(Gamma(Theta))`` over the per-channel unit spheres.

    Returns the learned ``theta`` (``(K+1, d)``, unit columns) plus the initial
    and final objective and the optimization history.
    """

    if len(train_patterns) > X.shape[1]:
        LOGGER.warning(
            f"  capacity: m_train={len(train_patterns)} > d={X.shape[1]}; "
            "lambda_min(Gamma) is 0 by Theorem (Capacity threshold) -- raise "
            "--feature-dim or lower --num-motifs / --train-ratio."
        )

    dtype, device = a_hat.dtype, a_hat.device
    V = degree_weighted_indicators(adjacency, train_patterns)  # (N, m)
    l_v = _l_apply(a_hat, V)
    phi = (V * l_v).sum(0).clamp_min(torch.finfo(dtype).eps)  # Phi_j = ||v_j||_L^2
    l_vhat = l_v / phi.sqrt().unsqueeze(0)  # L Vhat = L v_j / sqrt(Phi_j)

    propagated = propagation_stack(a_hat, X, degree)  # [A_hat^k X], k=0..K

    torch.manual_seed(fit_seed)
    raw = torch.nn.Parameter(
        torch.ones(degree + 1, X.shape[1], dtype=dtype, device=device)
    )

    def _unit(theta_raw: torch.Tensor) -> torch.Tensor:
        return theta_raw / theta_raw.norm(dim=0, keepdim=True).clamp_min(
            torch.finfo(dtype).eps
        )

    with torch.no_grad():
        Z0 = _filtered_bank(propagated, _unit(raw))
        init_obj = float(
            torch.linalg.eigvalsh(_collective_gamma(a_hat, Z0, l_vhat, ridge))[0]
        )

    optimizer = torch.optim.Adam((raw,), lr=learning_rate)
    best_theta = _unit(raw).detach().clone()
    best_obj = init_obj
    history: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        theta = _unit(raw)
        Z = _filtered_bank(propagated, theta)
        gamma = _collective_gamma(a_hat, Z, l_vhat, ridge)
        lam_min = torch.linalg.eigvalsh(gamma)[0]
        (-lam_min).backward()
        optimizer.step()

        value = float(lam_min.detach())
        history.append(value)
        if value > best_obj:
            best_obj = value
            best_theta = _unit(raw).detach().clone()

    return {
        "theta": best_theta,
        "init_objective": init_obj,
        "objective": best_obj,
        "history": history,
    }


def build_bank_subspace(
    a_hat: torch.Tensor, X: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """Embedding ``Z = g_Theta(A_hat) X`` and orthonormal basis of ``R = span(Z)``."""

    propagated = propagation_stack(a_hat, X, theta.shape[0] - 1)
    Z = _filtered_bank(propagated, theta)
    return _orthonormal_range(Z)


def retained_energy(
    a_hat: torch.Tensor,
    adjacency: torch.Tensor,
    patterns: list,
    X: torch.Tensor,
    theta: torch.Tensor,
    ridge: float,
) -> dict:
    """Per-gang retained ``L``-energy ``C_S = Gamma_jj`` and the collective margin."""

    V = degree_weighted_indicators(adjacency, patterns)
    l_v = _l_apply(a_hat, V)
    phi = (V * l_v).sum(0).clamp_min(torch.finfo(a_hat.dtype).eps)
    l_vhat = l_v / phi.sqrt().unsqueeze(0)
    propagated = propagation_stack(a_hat, X, theta.shape[0] - 1)
    Z = _filtered_bank(propagated, theta)
    gamma = _collective_gamma(a_hat, Z, l_vhat, ridge)
    diag = torch.diagonal(gamma).clamp(0.0, 1.0)
    return {
        "per_gang_capture": [float(v) for v in diag],
        "min_capture": float(diag.min()),
        "mean_capture": float(diag.mean()),
        "lambda_min_gamma": float(torch.linalg.eigvalsh(gamma)[0]),
    }


# --------------------------------------------------------------------------- #
# 6-7.  coarsen + evaluate
# --------------------------------------------------------------------------- #
def _alert_metrics(patterns: list, node_to_supernode, node_labels, threshold: float):
    """Run the Loukas pattern evaluation and pull out the alert-class summary."""

    _, by_label = evaluate_loukas_patterns(
        patterns, node_to_supernode, node_labels, threshold=threshold
    )
    return by_label.get("alert", {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # graph / motifs
    parser.add_argument("--num-nodes", type=int, default=10000)
    parser.add_argument("--num-motifs", type=int, default=50)
    parser.add_argument(
        "--motif-type",
        choices=["clique", "cycle", "star", "random"],
        default="random",
    )
    parser.add_argument(
        "--motif-density",
        type=float,
        default=0.5,
        help="edge density for --motif-type random (fraction of s(s-1)/2 possible "
        "edges; a spanning path is always added so the motif stays connected)",
    )
    parser.add_argument("--motif-size", type=int, default=25)
    parser.add_argument("--avg-degree", type=float, default=5.0)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.4)
    # filter-bank learning
    parser.add_argument("--degree", type=int, default=15, help="polynomial degree K")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument(
        "--max-levels",
        type=int,
        default=10000,
        help="option 2 collapses gangs hierarchically over many levels "
        "(a size-k gang needs ~log2(k) edge-matching levels)",
    )
    # coarsening
    parser.add_argument("--reduction", type=float, default=0.95)
    parser.add_argument(
        "--epsilon",
        type=float,
        # default=None,
        default=5.0,
        help="target RSA distortion budget prod_l(1+sigma_l)-1; when set it drives "
        "the coarsening (contract as much as possible until this bound is hit) and "
        "OVERRIDES --reduction",
    )
    parser.add_argument(
        "--epsilon-ramp-levels",
        type=int,
        default=1,
        help="optional: ration the --epsilon budget as a linear ramp over this many "
        "levels instead of offering it all at level 0 (only used with --epsilon)",
    )
    parser.add_argument(
        "--coarsening-method",
        choices=["edges", "neighborhood", "capped", "star", "kmeans", "linkage"],
        default="edges",
    )
    parser.add_argument(
        "--coarsening-laplacian",
        choices=["symmetric", "combinatorial"],
        default="symmetric",
        help="RSA metric for coarsening: 'symmetric' (L = I - A_hat, matches the "
        "algorithm, default) or 'combinatorial' (L = D - W, the legacy metric)",
    )
    parser.add_argument("--threshold", type=float, default=0.51)
    parser.add_argument("--seed", type=int, default=seed)
    path = f"results/collective_bank_detection/{now}/"
    parser.add_argument("--output", type=Path, default=path)
    args = parser.parse_args()

    # 1-2. build the controlled graph -----------------------------------------
    graph, patterns = build_synthetic_graph(
        num_nodes=args.num_nodes,
        num_motifs=args.num_motifs,
        motif_type=args.motif_type,
        motif_size=args.motif_size,
        avg_degree=args.avg_degree,
        feature_dim=args.feature_dim,
        rng_seed=args.seed,
        motif_density=args.motif_density,
    )
    normalized, adjacency = graph_operators(graph)  # A_hat (sym-norm) and raw W
    X = graph.x.to(device=normalized.device, dtype=normalized.dtype)

    # 3. train / test split of the motifs -------------------------------------
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(patterns))
    n_train = max(1, int(round(args.train_ratio * len(patterns))))
    train_patterns = [patterns[i] for i in order[:n_train]]
    test_patterns = [patterns[i] for i in order[n_train:]]

    LOGGER.info("\nCollective learnable filter-bank detection (L_sym metric)")
    _motif_desc = f"{args.motif_type}(size {args.motif_size}" + (
        f", density {args.motif_density:.2f})" if args.motif_type == "random" else ")"
    )
    # epsilon (RSA distortion budget) overrides the reduction-rate stopping rule
    if args.epsilon is not None:
        _budget_desc = f"epsilon<={args.epsilon:g}"
    else:
        _budget_desc = f"reduction={args.reduction:.0%}"
    LOGGER.info(
        f"  graph: N={args.num_nodes}  motifs={args.num_motifs}x{_motif_desc}"
        f"  avg_degree={args.avg_degree}  d={args.feature_dim}"
    )
    LOGGER.info(
        f"  split: train={len(train_patterns)}  test={len(test_patterns)}  "
        f"K={args.degree}  {_budget_desc}  "
        f"coarsening={args.coarsening_method}/{args.coarsening_laplacian}"
    )

    # 4. learn the filter bank on the training motifs -------------------------
    fit = fit_collective_bank(
        normalized,
        adjacency,
        train_patterns,
        X,
        degree=args.degree,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        ridge=args.ridge,
        fit_seed=args.seed,
    )
    theta = fit["theta"]
    LOGGER.info(
        f"  lambda_min(Gamma) train: {fit['init_objective']:.6g} -> "
        f"{fit['objective']:.6g}"
    )
    train_cap = retained_energy(
        normalized, adjacency, train_patterns, X, theta, args.ridge
    )
    test_cap = retained_energy(
        normalized, adjacency, test_patterns, X, theta, args.ridge
    )
    LOGGER.info(
        f"  retained L-energy  train: min={train_cap['min_capture']:.3f} "
        f"mean={train_cap['mean_capture']:.3f}   "
        f"test: min={test_cap['min_capture']:.3f} "
        f"mean={test_cap['mean_capture']:.3f}"
    )

    # 5. embedding + target subspace R = span(Z) ------------------------------
    basis = build_bank_subspace(normalized, X, theta)

    # 6. Loukas RSA coarsening with the learned target ------------------------
    if args.epsilon is not None:
        # drive by the distortion budget: make the reduction cap non-binding
        # (n_target = 1) so epsilon is the sole stopping criterion.
        coarsen_budget = dict(
            reduction=args.reduction,
            # reduction=1.0 - 1.0 / adjacency.shape[0],
            epsilon=args.epsilon,
            epsilon_ramp_levels=args.epsilon_ramp_levels,
        )
    else:
        coarsen_budget = dict(reduction=args.reduction)
    coarsening = loukas_coarsen_pytorch(
        adjacency,
        basis,
        method=args.coarsening_method,
        laplacian=args.coarsening_laplacian,
        max_levels=args.max_levels,
        **coarsen_budget,
    )
    LOGGER.info(
        f"  coarsening: N={coarsening.n_original} -> n_coarse="
        f"{coarsening.n_coarse}  levels={len(coarsening.sigmas)}  "
        f"epsilon={coarsening.epsilon:.4g} (RSA exact; bound "
        f"{coarsening.epsilon_bound:.4g})"
    )

    # 7. recall / precision / detection rate ----------------------------------
    splits = {
        "train": train_patterns,
        "test": test_patterns,
        "all": patterns,
    }
    report = {}
    for name, split in splits.items():
        metrics = _alert_metrics(
            split, coarsening.node_to_supernode, graph.y, args.threshold
        )
        report[name] = {
            "detection_rate": metrics.get("detection_rate"),
            "mean_recall": metrics.get("mean_recall"),
            "mean_precision": metrics.get("mean_precision"),
            "detected": metrics.get("detected"),
            "total": metrics.get("total"),
        }

    header = f"  {'split':<6} {'recall':>8} {'precision':>10} {'detection':>10} {'det/tot':>9}"
    LOGGER.info(header)
    LOGGER.info("  " + "-" * (len(header) - 2))
    for name in ("train", "test", "all"):
        r = report[name]
        LOGGER.info(
            f"  {name:<6} {(r['mean_recall'] or 0):>8.3f} "
            f"{(r['mean_precision'] or 0):>10.3f} "
            f"{(r['detection_rate'] or 0):>10.1%} "
            f"{r['detected']:>4}/{r['total']:<4}"
        )

    # --- persist JSON + plot --------------------------------------------------
    out_dir = Path(args.output) if args.output else Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_out = out_dir / "collective_bank_detection.json"
    plot_out = out_dir / "collective_bank_detection.png"
    json_out.write_text(
        json.dumps(
            {
                "config": vars(args) | {"output": str(out_dir)},
                "learning": {
                    "lambda_min_gamma_init": fit["init_objective"],
                    "lambda_min_gamma_final": fit["objective"],
                    "theta": theta.detach().cpu().tolist(),
                    "train_capture": train_cap,
                    "test_capture": test_cap,
                },
                "coarsening": {
                    "n_original": coarsening.n_original,
                    "n_coarse": coarsening.n_coarse,
                    "reduction": coarsening.reduction,
                    "epsilon": coarsening.epsilon,
                    "n_levels": len(coarsening.sigmas),
                },
                "detection": report,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    _save_plot(report, args, fit, plot_out)

    LOGGER.info(f"\nJSON report:  {json_out}")
    LOGGER.info(f"Plot:         {plot_out}")


def _save_plot(report: dict, args, fit: dict, output: Path) -> None:
    """Two-panel figure: detection metrics per split and the training curve."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = ["train", "test", "all"]
    x = np.arange(len(names))
    w = 0.25
    metrics = ["mean_recall", "mean_precision", "detection_rate"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = ["recall", "precision", "detection rate"]

    ax = axes[0]
    for offset, metric, color, label in zip([-w, 0, w], metrics, colors, labels):
        vals = [report[n][metric] or 0.0 for n in names]
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
    ax.set_ylabel("score")
    ax.set_title(
        f"{args.num_motifs}x {args.motif_type} (size {args.motif_size})  "
        f"reduction {args.reduction:.0%}  {args.coarsening_laplacian}"
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.plot(fit["history"], color="tab:red")
    ax.axhline(fit["init_objective"], color="grey", ls="--", lw=1, label="init")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\lambda_{\min}(\Gamma)$")
    ax.set_title("Collective objective (train motifs)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
