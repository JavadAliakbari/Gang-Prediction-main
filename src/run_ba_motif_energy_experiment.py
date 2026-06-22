"""run_ba_motif_energy_experiment.py
====================================
Measure how planted motifs in a Barabási–Albert (BA) scale-free network
distribute their spectral energy across eigenvectors of the graph Laplacian.

Experiment design
-----------------
* Graph      : BA model, N >= 1000 nodes (default 1500), m=3 (scale-free)
* Motif types: clique, cycle, star
* Motif sizes: 10, 50, 100, 500
* Repetitions: 1, 3, 5 (capped by available non-overlapping nodes)
* Full spectral decomposition (all N eigenvectors via dense eigh)

For each planted motif instance the indicator vector 1_S / ‖1_S‖ is projected
onto the Laplacian eigenvectors.  energy[k] = (u_k^T v)^2 captures the
contribution of frequency component k to the motif.

Outputs (saved under results/<timestamp>/ba_motif_energy/)
-----------------------------------------------------------
* energy_dist_{motif_type}.png         – energy vs eigenvector index per size
* energy_compare_size{size}.png        – cross-motif comparison per size
* rep_effect_{motif_type}_s{size}.png  – effect of #repetitions on energy
* cumulative_energy_all.png            – cumulative energy comparison grid
* summary_k50_k90.png                  – bar chart of k50/k90
* summary_heatmap_k50.png / _k90.png   – (motif_type × size) heatmap

Usage (from repo root, with FedStruct conda env active)::

    python src/GangPrediction/run_ba_motif_energy_experiment.py
    python src/GangPrediction/run_ba_motif_energy_experiment.py --n_nodes 2000 --ba_m 5
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from src.coarsening_diagnostics import (
    compute_spectral_decomp,
    _ensure_graph_params,
)
from src.utils.utils import save_path, LOGGER

# ── constants ─────────────────────────────────────────────────────────────────
MOTIF_TYPES = ["clique", "cycle", "star"]
MOTIF_SIZES = [10, 50, 100, 250]
REPETITIONS = [
    1,
    3,
    5,
    10,
    20,
]

# colour palette per motif type
MOTIF_COLORS = {"clique": "#e41a1c", "cycle": "#377eb8", "star": "#4daf4a"}
SIZE_ALPHA = {10: 1.0, 50: 0.85, 100: 0.65, 250: 0.45}
REP_STYLES = {1: "-", 3: "--", 5: ":", 10: "-.", 20: (0, (3, 1, 1, 1))}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Graph construction
# ═══════════════════════════════════════════════════════════════════════════════


def build_ba_graph(n_nodes: int = 1500, m: int = 3, seed: int = 42) -> Data:
    """Return a PyG Data object for a Barabási–Albert random scale-free graph."""
    LOGGER.info(f"[BA graph] generating BA(n={n_nodes}, m={m}, seed={seed}) …")
    G_nx = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
    # Convert to undirected PyG graph
    edge_index = torch.tensor(list(G_nx.edges()), dtype=torch.long).t().contiguous()
    # Add reverse edges
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    # Remove duplicates
    edge_index = torch.unique(edge_index, dim=1)
    n = G_nx.number_of_nodes()
    data = Data(
        x=torch.eye(n, dtype=torch.float32),
        edge_index=edge_index,
        num_nodes=n,
    )
    data.edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float32)
    data = _ensure_graph_params(data)
    LOGGER.info(
        f"[BA graph] {data.num_nodes} nodes, {data.num_edges} edges  "
        f"(avg degree ≈ {data.num_edges / data.num_nodes:.1f})"
    )
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Motif edge generators
# ═══════════════════════════════════════════════════════════════════════════════


def clique_edges(nodes: np.ndarray) -> List[Tuple[int, int]]:
    """All C(k,2) pairs."""
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            edges.append((int(nodes[i]), int(nodes[j])))
    return edges


def cycle_edges(nodes: np.ndarray) -> List[Tuple[int, int]]:
    """Ring: 0→1→…→k-1→0."""
    k = len(nodes)
    return [(int(nodes[i]), int(nodes[(i + 1) % k])) for i in range(k)]


def star_edges(nodes: np.ndarray) -> List[Tuple[int, int]]:
    """Hub = nodes[0], spokes = nodes[1:]."""
    hub = int(nodes[0])
    return [(hub, int(n)) for n in nodes[1:]]


EDGE_GENERATORS = {
    "clique": clique_edges,
    "cycle": cycle_edges,
    "star": star_edges,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Motif planting
# ═══════════════════════════════════════════════════════════════════════════════


def plant_motifs(
    G: Data,
    motif_type: str,
    size: int,
    n_reps: int,
    seed: int = 0,
) -> Tuple[Data, List[np.ndarray]]:
    """Plant *n_reps* non-overlapping copies of *motif_type* of *size* nodes.

    Each copy:
      1. Picks a random seed from available nodes, then expands via BFS
         through the *original* graph edges until *size* unused neighbours
         are collected (retries up to 50 seeds if BFS yields too few nodes).
      2. Removes existing edges among them.
      3. Adds the motif-specific edges (both directions → undirected).

    Returns
    -------
    G_new        : modified Data object
    motif_nodes  : list of node-index arrays, one per planted copy
    """
    rng = np.random.default_rng(seed)
    N = G.num_nodes
    src_l, dst_l = G.edge_index[0].tolist(), G.edge_index[1].tolist()
    edge_set: set = set(zip(src_l, dst_l))

    # Build adjacency list once for BFS (uses original graph edges)
    adj: Dict[int, List[int]] = {}
    for s, d in zip(src_l, dst_l):
        adj.setdefault(s, []).append(d)

    gen = EDGE_GENERATORS[motif_type]
    used: set = set()
    planted: List[np.ndarray] = []

    for rep in range(n_reps):
        available_set = set(range(N)) - used
        available = np.array(sorted(available_set))
        if len(available) < size:
            LOGGER.warning(
                f"  [plant] only {len(available)} nodes left at rep {rep}; "
                f"stopping early (needed {size})"
            )
            break

        # BFS: try up to 50 random seeds to collect exactly *size* connected nodes
        nodes = None
        for _ in range(50):
            seed_node = int(rng.choice(available))
            visited: List[int] = [seed_node]
            visited_set: set = {seed_node}
            frontier: List[int] = [seed_node]
            while len(visited) < size and frontier:
                next_frontier: List[int] = []
                for node in frontier:
                    for nb in adj.get(node, []):
                        if nb not in visited_set and nb not in used:
                            visited_set.add(nb)
                            visited.append(nb)
                            next_frontier.append(nb)
                            if len(visited) >= size:
                                break
                    if len(visited) >= size:
                        break
                frontier = next_frontier
            if len(visited) >= size:
                nodes = np.array(visited[:size])
                break

        if nodes is None:
            LOGGER.warning(
                f"  [plant] BFS exhausted 50 seeds at rep {rep}; "
                f"stopping early (needed {size})"
            )
            break

        used.update(nodes.tolist())

        # Remove existing edges inside motif node set
        node_set = set(nodes.tolist())
        edge_set -= {(u, v) for u, v in edge_set if u in node_set and v in node_set}

        # Add motif edges (both directions)
        for u, v in gen(nodes):
            edge_set.add((u, v))
            edge_set.add((v, u))

        planted.append(nodes)

    # Rebuild PyG Data
    if edge_set:
        all_src, all_dst = zip(*sorted(edge_set))
    else:
        all_src, all_dst = [], []
    new_ei = torch.tensor([list(all_src), list(all_dst)], dtype=torch.long)
    G_new = Data(
        x=G.x.clone(),
        edge_index=new_ei,
        num_nodes=N,
    )
    G_new.edge_weight = torch.ones(G_new.edge_index.size(1), dtype=torch.float32)
    G_new = _ensure_graph_params(G_new)
    return G_new, planted


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Energy computation
# ═══════════════════════════════════════════════════════════════════════════════


def motif_energy(Uk: np.ndarray, node_sets: List[np.ndarray], N: int) -> np.ndarray:
    """Aggregate energy per eigenvector across all motif instances.

    For each instance with node set S, indicator v = 1_S / ‖1_S‖.
    energy_i[k] = (u_k^T v)^2.  Returns mean energy across instances.

    Parameters
    ----------
    Uk        : (N, K) eigenvector matrix
    node_sets : list of node-index arrays
    N         : number of graph nodes

    Returns
    -------
    mean_energy : (K,) array
    per_instance: (n_instances, K) array
    """
    K = Uk.shape[1]
    energies = []
    for nodes in node_sets:
        v = np.zeros(N, dtype=np.float64)
        v[nodes] = 1.0
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        proj = Uk.T @ v  # (K,)
        energies.append(proj**2)
    per_instance = np.array(energies)  # (n, K)
    mean_energy = per_instance.mean(axis=0)  # (K,)
    return mean_energy, per_instance


def random_baseline_energy(
    Uk: np.ndarray, node_sizes: List[int], N: int, n_trials: int = 50, seed: int = 0
) -> np.ndarray:
    """Expected energy for random node sets of the same sizes (uniform baseline)."""
    rng = np.random.default_rng(seed)
    K = Uk.shape[1]
    acc = np.zeros(K)
    count = 0
    for sz in node_sizes:
        for _ in range(n_trials):
            idx = rng.choice(N, size=max(1, sz), replace=False)
            v = np.zeros(N, dtype=np.float64)
            v[idx] = 1.0
            v /= np.linalg.norm(v)
            acc += (Uk.T @ v) ** 2
            count += 1
    return acc / max(1, count)


def cumulative_energy(energy: np.ndarray) -> np.ndarray:
    """Normalised cumulative sum of energy array."""
    total = energy.sum()
    if total == 0:
        return np.zeros_like(energy)
    return np.cumsum(energy) / total


def k_threshold(cum_energy: np.ndarray, threshold: float = 0.9) -> int:
    """First eigenvector index where cumulative energy exceeds *threshold*."""
    idx = np.searchsorted(cum_energy, threshold)
    return int(min(idx + 1, len(cum_energy)))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _smoothed(x: np.ndarray, window: int = 10) -> np.ndarray:
    """Running average with *window* samples; handles edge cases."""
    if len(x) <= window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def plot_energy_by_size(
    results: Dict,
    motif_type: str,
    n_reps: int,
    save_dir: str,
    smooth_window: int = 15,
) -> None:
    """
    4-panel figure: one subplot per motif size.
    Each subplot: smoothed mean energy vs eigenvector index k,
    compared against random baseline.
    """
    sizes = MOTIF_SIZES
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Motif energy across eigenvectors  –  {motif_type.upper()}  "
        f"({n_reps} rep{'s' if n_reps > 1 else ''})",
        fontsize=14,
    )

    for ax, size in zip(axes.flat, sizes):
        key = (motif_type, size, n_reps)
        if key not in results:
            ax.set_title(f"size={size}  [no data]")
            ax.axis("off")
            continue
        res = results[key]
        energy = res["mean_energy"]  # (K,)
        baseline = res["baseline"]  # (K,)
        K = len(energy)
        x = np.arange(K)
        e_sm = _smoothed(energy, smooth_window)
        b_sm = _smoothed(baseline, smooth_window)

        std_e = res.get("std_energy")
        std_b = res.get("std_baseline")
        if std_e is not None:
            s_sm = _smoothed(std_e, smooth_window)
            ax.fill_between(
                x,
                np.maximum(0, e_sm - s_sm),
                e_sm + s_sm,
                alpha=0.20,
                color=MOTIF_COLORS[motif_type],
            )
        else:
            ax.fill_between(x, e_sm, alpha=0.20, color=MOTIF_COLORS[motif_type])
        ax.plot(
            x,
            e_sm,
            color=MOTIF_COLORS[motif_type],
            lw=1.5,
            label=f"{motif_type} (size {size})",
        )
        if std_b is not None:
            sb_sm = _smoothed(std_b, smooth_window)
            ax.fill_between(
                x, np.maximum(0, b_sm - sb_sm), b_sm + sb_sm, alpha=0.12, color="grey"
            )
        ax.plot(x, b_sm, color="grey", lw=1.2, ls="--", label="random baseline")

        # Mark k50 / k90
        cum = cumulative_energy(energy)
        k50 = k_threshold(cum, 0.50)
        k90 = k_threshold(cum, 0.90)
        ax.axvline(k50, color="orange", ls=":", lw=1.2, label=f"k50={k50}")
        ax.axvline(k90, color="red", ls=":", lw=1.2, label=f"k90={k90}")

        ax.set_title(
            f"size={size}  |  k50={k50}, k90={k90}  " f"(planted: {res['n_planted']})",
            fontsize=9,
        )
        ax.set_xlabel("Eigenvector index  k  (sorted by eigenvalue)")
        ax.set_ylabel("Energy  (u_k^T v)²")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)

    plt.tight_layout()
    path = os.path.join(save_dir, f"energy_dist_{motif_type}_reps{n_reps}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"  Saved → {path}")


def plot_energy_compare_types(
    results: Dict,
    size: int,
    n_reps: int,
    save_dir: str,
    smooth_window: int = 15,
) -> None:
    """
    Single panel comparing energy distribution for all three motif types
    at the same size and repetition count.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title(
        f"Motif energy comparison  –  size={size}, reps={n_reps}",
        fontsize=13,
    )

    for mtype in MOTIF_TYPES:
        key = (mtype, size, n_reps)
        if key not in results:
            continue
        res = results[key]
        energy = res["mean_energy"]
        K = len(energy)
        e_sm = _smoothed(energy, smooth_window)
        cum = cumulative_energy(energy)
        k50 = k_threshold(cum, 0.50)
        k90 = k_threshold(cum, 0.90)
        lbl = f"{mtype}  (k50={k50}, k90={k90})"
        std_e = res.get("std_energy")
        if std_e is not None:
            s_sm = _smoothed(std_e, smooth_window)
            ax.fill_between(
                np.arange(K),
                np.maximum(0, e_sm - s_sm),
                e_sm + s_sm,
                alpha=0.15,
                color=MOTIF_COLORS[mtype],
            )
        ax.plot(e_sm, color=MOTIF_COLORS[mtype], lw=2, label=lbl)
        ax.axvline(k50, color=MOTIF_COLORS[mtype], ls=":", lw=1, alpha=0.7)
        ax.axvline(k90, color=MOTIF_COLORS[mtype], ls="--", lw=1, alpha=0.7)

    # Baseline (use last available)
    for mtype in MOTIF_TYPES:
        key = (mtype, size, n_reps)
        if key in results:
            b_sm = _smoothed(results[key]["baseline"], smooth_window)
            std_b = results[key].get("std_baseline")
            if std_b is not None:
                sb_sm = _smoothed(std_b, smooth_window)
                K = len(b_sm)
                ax.fill_between(
                    np.arange(K),
                    np.maximum(0, b_sm - sb_sm),
                    b_sm + sb_sm,
                    alpha=0.10,
                    color="grey",
                )
            ax.plot(b_sm, color="grey", lw=1.5, ls="--", label="random baseline")
            break

    ax.set_xlabel("Eigenvector index  k  (sorted by eigenvalue)")
    ax.set_ylabel("Energy  (u_k^T v)²")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    path = os.path.join(save_dir, f"energy_compare_size{size}_reps{n_reps}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"  Saved → {path}")


def plot_repetition_effect(
    results: Dict,
    motif_type: str,
    size: int,
    save_dir: str,
    smooth_window: int = 15,
) -> None:
    """
    Shows how planting more copies of the same motif changes the energy
    distribution: 3 curves (reps=1,3,5) on one panel.
    """
    fig, (ax_energy, ax_cum) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Repetition effect  –  {motif_type.upper()}, size={size}",
        fontsize=13,
    )

    for n_reps in REPETITIONS:
        key = (motif_type, size, n_reps)
        if key not in results:
            continue
        res = results[key]
        energy = res["mean_energy"]
        K = len(energy)
        e_sm = _smoothed(energy, smooth_window)
        cum = cumulative_energy(energy)
        k50 = k_threshold(cum, 0.50)
        k90 = k_threshold(cum, 0.90)
        ls = REP_STYLES[n_reps]
        lbl = f"reps={res['n_planted']}  (k50={k50}, k90={k90})"
        std_e = res.get("std_energy")
        if std_e is not None:
            s_sm = _smoothed(std_e, smooth_window)
            x_r = np.arange(K)
            ax_energy.fill_between(
                x_r,
                np.maximum(0, e_sm - s_sm),
                e_sm + s_sm,
                alpha=0.12,
                color=MOTIF_COLORS[motif_type],
            )
            # shading for cumulative: propagate std via cumsum
            cum_lo = cumulative_energy(np.maximum(0, energy - res["std_energy"]))
            cum_hi = cumulative_energy(energy + res["std_energy"])
            ax_cum.fill_between(
                x_r, cum_lo, cum_hi, alpha=0.12, color=MOTIF_COLORS[motif_type]
            )
        ax_energy.plot(
            e_sm, ls=ls, color=MOTIF_COLORS[motif_type], lw=1.8, label=lbl, alpha=0.9
        )
        ax_cum.plot(
            cum, ls=ls, color=MOTIF_COLORS[motif_type], lw=1.8, label=lbl, alpha=0.9
        )

    # Baseline
    for n_reps in REPETITIONS:
        key = (motif_type, size, n_reps)
        if key in results:
            b = results[key]["baseline"]
            ax_energy.plot(
                _smoothed(b, smooth_window),
                color="grey",
                ls="--",
                lw=1.2,
                label="random",
            )
            ax_cum.plot(
                cumulative_energy(b), color="grey", ls="--", lw=1.2, label="random"
            )
            break

    ax_energy.set_xlabel("Eigenvector index k")
    ax_energy.set_ylabel("Mean energy  (u_k^T v)²")
    ax_energy.set_title("Energy distribution")
    ax_energy.legend(fontsize=8)
    ax_energy.grid(alpha=0.25)

    ax_cum.axhline(0.5, color="orange", ls=":", lw=1, label="50 %")
    ax_cum.axhline(0.9, color="red", ls=":", lw=1, label="90 %")
    ax_cum.set_xlabel("Eigenvector index k")
    ax_cum.set_ylabel("Cumulative energy")
    ax_cum.set_title("Cumulative energy")
    ax_cum.legend(fontsize=8)
    ax_cum.grid(alpha=0.25)

    plt.tight_layout()
    path = os.path.join(save_dir, f"rep_effect_{motif_type}_s{size}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"  Saved → {path}")


def plot_cumulative_grid(
    results: Dict,
    n_reps: int,
    save_dir: str,
) -> None:
    """
    Grid of cumulative energy plots: rows = motif type, cols = size.
    All in one figure for easy comparison.
    """
    n_rows = len(MOTIF_TYPES)
    n_cols = len(MOTIF_SIZES)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharex=False, sharey=True
    )
    fig.suptitle(
        f"Cumulative spectral energy  –  {n_reps} repetition(s)",
        fontsize=15,
    )

    for r, mtype in enumerate(MOTIF_TYPES):
        for c, size in enumerate(MOTIF_SIZES):
            ax = axes[r, c]
            key = (mtype, size, n_reps)
            ax.axhline(0.5, color="orange", ls=":", lw=0.9, alpha=0.7)
            ax.axhline(0.9, color="red", ls=":", lw=0.9, alpha=0.7)
            if key not in results:
                ax.set_title(f"{mtype} / s={size}\n[no data]", fontsize=8)
                ax.set_ylim(0, 1.05)
                continue
            res = results[key]
            energy = res["mean_energy"]
            baseline = res["baseline"]
            cum = cumulative_energy(energy)
            cum_b = cumulative_energy(baseline)
            K = len(cum)
            x = np.arange(K) / (K - 1) * 100  # normalise to 0-100 %

            std_e = res.get("std_energy")
            if std_e is not None:
                cum_lo = cumulative_energy(np.maximum(0, energy - std_e))
                cum_hi = cumulative_energy(energy + std_e)
                ax.fill_between(
                    x, cum_lo, cum_hi, alpha=0.18, color=MOTIF_COLORS[mtype]
                )
            std_b = res.get("std_baseline")
            if std_b is not None:
                cb_lo = cumulative_energy(np.maximum(0, baseline - std_b))
                cb_hi = cumulative_energy(baseline + std_b)
                ax.fill_between(x, cb_lo, cb_hi, alpha=0.10, color="grey")
            ax.plot(x, cum, color=MOTIF_COLORS[mtype], lw=1.8)
            ax.plot(x, cum_b, color="grey", ls="--", lw=1.2)

            k50 = k_threshold(cum, 0.50)
            k90 = k_threshold(cum, 0.90)
            ax.set_title(
                f"{mtype}  size={size}\nk50={k50}, k90={k90}  n={res['n_planted']}",
                fontsize=8,
            )
            ax.set_ylim(0, 1.05)
            if c == 0:
                ax.set_ylabel("Cumulative energy")
            if r == n_rows - 1:
                ax.set_xlabel("% of eigenvectors used")
            ax.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(save_dir, f"cumulative_energy_reps{n_reps}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"  Saved → {path}")


def plot_summary_bar(results: Dict, save_dir: str) -> None:
    """Bar chart of k50 and k90 grouped by (motif_type × size × reps)."""
    labels, k50s, k90s = [], [], []
    for mtype in MOTIF_TYPES:
        for size in MOTIF_SIZES:
            for reps in REPETITIONS:
                key = (mtype, size, reps)
                if key not in results:
                    continue
                res = results[key]
                cum = cumulative_energy(res["mean_energy"])
                k50 = k_threshold(cum, 0.50)
                k90 = k_threshold(cum, 0.90)
                labels.append(f"{mtype[:2]}_s{size}_r{reps}")
                k50s.append(k50)
                k90s.append(k90)

    n = len(labels)
    if n == 0:
        return
    x = np.arange(n)
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(14, n * 0.55), 6))
    ax.bar(x - width / 2, k50s, width, label="k50 (50 % energy)", color="orange")
    ax.bar(x + width / 2, k90s, width, label="k90 (90 % energy)", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("Eigenvectors needed")
    ax.set_title("Spectral energy concentration across motif configurations (BA graph)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "summary_k50_k90.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"  Saved → {path}")


def plot_summary_heatmap(results: Dict, metric: str, save_dir: str) -> None:
    """
    Heatmap: rows = motif type, cols = size.
    One heatmap per repetition count, or aggregate the last rep.
    """
    for n_reps in REPETITIONS:
        mat = np.full((len(MOTIF_TYPES), len(MOTIF_SIZES)), np.nan)
        for r, mtype in enumerate(MOTIF_TYPES):
            for c, size in enumerate(MOTIF_SIZES):
                key = (mtype, size, n_reps)
                if key not in results:
                    continue
                cum = cumulative_energy(results[key]["mean_energy"])
                thresh = 0.5 if metric == "k50" else 0.9
                mat[r, c] = k_threshold(cum, thresh)

        if np.all(np.isnan(mat)):
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xticks(range(len(MOTIF_SIZES)))
        ax.set_xticklabels([f"size={s}" for s in MOTIF_SIZES], fontsize=9)
        ax.set_yticks(range(len(MOTIF_TYPES)))
        ax.set_yticklabels(MOTIF_TYPES, fontsize=10)
        ax.set_xlabel("Motif size")
        ax.set_ylabel("Motif type")
        ax.set_title(
            f"{metric} (eigenvectors for {int(float(metric[1:])/100*100)}% energy) "
            f"–  {n_reps} rep(s)  –  BA graph",
            fontsize=11,
        )
        for r in range(len(MOTIF_TYPES)):
            for c in range(len(MOTIF_SIZES)):
                val = mat[r, c]
                if not np.isnan(val):
                    ax.text(
                        c,
                        r,
                        f"{int(val)}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="white" if val > np.nanmax(mat) * 0.65 else "black",
                    )
        plt.colorbar(im, ax=ax, label=f"# eigenvectors ({metric})")
        plt.tight_layout()
        path = os.path.join(save_dir, f"summary_heatmap_{metric}_reps{n_reps}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        LOGGER.info(f"  Saved → {path}")


def plot_energy_heatmap_over_eigenvectors(
    results: Dict,
    n_reps: int,
    save_dir: str,
    n_bins: int = 100,
) -> None:
    """
    Full heatmap: rows = (motif_type, size) pairs, cols = binned eigenvector index.
    Shows the energy profile of each configuration as a row.
    """
    row_labels = []
    rows_data = []
    for mtype in MOTIF_TYPES:
        for size in MOTIF_SIZES:
            key = (mtype, size, n_reps)
            if key not in results:
                continue
            energy = results[key]["mean_energy"]
            K = len(energy)
            # Bin the energy into n_bins buckets
            bin_size = max(1, K // n_bins)
            n_full = K // bin_size
            energy_binned = (
                energy[: n_full * bin_size].reshape(n_full, bin_size).sum(axis=1)
            )
            row_labels.append(f"{mtype} s={size}")
            rows_data.append(energy_binned)

    if not rows_data:
        return

    # Pad to same length
    max_bins = max(len(r) for r in rows_data)
    mat = np.zeros((len(rows_data), max_bins))
    for i, row in enumerate(rows_data):
        mat[i, : len(row)] = row

    fig, ax = plt.subplots(
        figsize=(max(12, max_bins // 5), max(4, len(row_labels) * 0.5))
    )
    im = ax.imshow(mat, aspect="auto", cmap="hot_r", interpolation="bilinear")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel(f"Eigenvector bin  (each bin ≈ K/{n_bins} eigenvectors)")
    ax.set_title(
        f"Energy distribution heatmap across eigenvectors  –  reps={n_reps}  (BA graph)",
        fontsize=12,
    )
    plt.colorbar(im, ax=ax, label="Binned energy")
    plt.tight_layout()
    path = os.path.join(save_dir, f"energy_heatmap_eigenvectors_reps{n_reps}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Main experiment runner
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BA motif spectral energy experiment")
    p.add_argument(
        "--n_nodes",
        type=int,
        default=2000,
        help="Number of nodes in the BA graph (default: 2000)",
    )
    p.add_argument(
        "--ba_m",
        type=int,
        default=2,
        help="BA model m: edges to attach per new node (default: 2)",
    )
    p.add_argument("--seed", type=int, default=42, help="Global random seed")
    p.add_argument(
        "--k_max", type=int, default=0, help="Max eigenvectors (0 = full decomposition)"
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=20,
        help="Smoothing window for energy plots (default: 20)",
    )
    p.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Independent BA graph trials to average over (default: 10)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    SAVE_DIR = os.path.join(save_path, "ba_motif_energy")
    os.makedirs(SAVE_DIR, exist_ok=True)

    LOGGER.info("=" * 70)
    LOGGER.info("  BA Motif Spectral Energy Experiment")
    LOGGER.info(f"  BA graph    : n={args.n_nodes}, m={args.ba_m}, seed={args.seed}")
    LOGGER.info(f"  Motif types : {MOTIF_TYPES}")
    LOGGER.info(f"  Sizes       : {MOTIF_SIZES}")
    LOGGER.info(f"  Repetitions : {REPETITIONS}")
    LOGGER.info(f"  Trials      : {args.n_trials} (independent BA graphs, averaged)")
    LOGGER.info(f"  Output dir  : {SAVE_DIR}")
    LOGGER.info("=" * 70)

    configs = list(product(MOTIF_TYPES, MOTIF_SIZES, REPETITIONS))
    LOGGER.info(f"  Total configurations: {len(configs)}  × {args.n_trials} trials")

    # Accumulators: key → list of per-trial (K,) energy arrays
    trial_energies: Dict[tuple, List[np.ndarray]] = {}
    trial_baselines: Dict[tuple, List[np.ndarray]] = {}
    trial_n_planted: Dict[tuple, List[int]] = {}

    # ── Trial loop ───────────────────────────────────────────────────────────
    for trial in range(args.n_trials):
        trial_seed = args.seed + trial * 1000
        LOGGER.info(f"\n{'─'*70}")
        LOGGER.info(f"  Trial {trial + 1}/{args.n_trials}  (BA seed={trial_seed})")
        LOGGER.info(f"{'─'*70}")

        # Build a fresh BA graph for this trial
        G_base = build_ba_graph(n_nodes=args.n_nodes, m=args.ba_m, seed=trial_seed)
        N = G_base.num_nodes
        K_max = args.k_max if args.k_max > 0 else N
        dense_threshold = N + 10

        for idx, (mtype, size, n_reps) in enumerate(configs, 1):
            tag = f"{mtype}_s{size}_r{n_reps}"

            if size >= N:
                continue

            # Plant motifs (vary seed by config index too)
            G_planted, motif_node_sets = plant_motifs(
                G_base, mtype, size, n_reps, seed=trial_seed + idx
            )
            n_planted = len(motif_node_sets)
            if n_planted == 0:
                continue

            # Spectral decomposition
            lk, Uk = compute_spectral_decomp(
                G_planted, K_max=K_max, dense_threshold=dense_threshold
            )

            # Energy
            mean_e, _ = motif_energy(Uk, motif_node_sets, N)
            baseline = random_baseline_energy(
                Uk, [size] * n_planted, N, n_trials=20, seed=trial_seed
            )

            key = (mtype, size, n_reps)
            trial_energies.setdefault(key, []).append(mean_e)
            trial_baselines.setdefault(key, []).append(baseline)
            trial_n_planted.setdefault(key, []).append(n_planted)

        LOGGER.info(f"  Trial {trial + 1} done.")

    # ── Average across trials ─────────────────────────────────────────────────
    LOGGER.info("\n[avg] Averaging energy curves across trials …")
    results: Dict = {}
    for key in trial_energies:
        mtype, size, n_reps = key
        stack_e = np.array(trial_energies[key])  # (n_trials, K)
        stack_b = np.array(trial_baselines[key])  # (n_trials, K)
        mean_e = stack_e.mean(axis=0)
        std_e = stack_e.std(axis=0)
        mean_b = stack_b.mean(axis=0)
        std_b = stack_b.std(axis=0)
        n_p = int(round(np.mean(trial_n_planted[key])))
        cum = cumulative_energy(mean_e)
        k50 = k_threshold(cum, 0.50)
        k90 = k_threshold(cum, 0.90)
        LOGGER.info(
            f"  {mtype}_s{size}_r{n_reps}  "
            f"(trials={len(stack_e)})  k50={k50}, k90={k90}"
        )
        results[key] = {
            "mean_energy": mean_e,
            "std_energy": std_e,
            "baseline": mean_b,
            "std_baseline": std_b,
            "n_planted": n_p,
            "k50": k50,
            "k90": k90,
        }

    # Re-read N from last trial graph (all trials have same n_nodes)
    N = args.n_nodes

    # ── 3. Visualisation ─────────────────────────────────────────────────────
    LOGGER.info("\n[3] Generating visualisations …")
    sw = args.smooth

    # A. Energy distribution per motif type (one figure per type × rep-count)
    for mtype in MOTIF_TYPES:
        for n_reps in REPETITIONS:
            plot_energy_by_size(results, mtype, n_reps, SAVE_DIR, smooth_window=sw)

    # B. Cross-motif comparison per size
    for size in MOTIF_SIZES:
        for n_reps in REPETITIONS:
            plot_energy_compare_types(results, size, n_reps, SAVE_DIR, smooth_window=sw)

    # C. Repetition effect per (type, size)
    for mtype in MOTIF_TYPES:
        for size in MOTIF_SIZES:
            plot_repetition_effect(results, mtype, size, SAVE_DIR, smooth_window=sw)

    # D. Cumulative energy grid
    for n_reps in REPETITIONS:
        plot_cumulative_grid(results, n_reps, SAVE_DIR)

    # E. Big energy-vs-eigenvector heatmap
    for n_reps in REPETITIONS:
        plot_energy_heatmap_over_eigenvectors(results, n_reps, SAVE_DIR)

    # F. Summary bar chart
    plot_summary_bar(results, SAVE_DIR)

    # G. Summary heatmaps (k50, k90)
    plot_summary_heatmap(results, "k50", SAVE_DIR)
    plot_summary_heatmap(results, "k90", SAVE_DIR)

    # ── 4. Print summary table ───────────────────────────────────────────────
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info(f"{'Config':<35}  {'k50':>6}  {'k90':>6}  {'planted':>7}")
    LOGGER.info("-" * 70)
    for mtype in MOTIF_TYPES:
        for size in MOTIF_SIZES:
            for n_reps in REPETITIONS:
                key = (mtype, size, n_reps)
                if key not in results:
                    continue
                r = results[key]
                lbl = f"{mtype}_s{size}_r{n_reps}"
                LOGGER.info(
                    f"{lbl:<35}  {r['k50']:>6}  {r['k90']:>6}  {r['n_planted']:>7}"
                )
    LOGGER.info("=" * 70)

    n_plots = len(list(Path(SAVE_DIR).glob("*.png")))
    LOGGER.info(f"\nDone.  {n_plots} PNG files saved under {SAVE_DIR}")


if __name__ == "__main__":
    main()
