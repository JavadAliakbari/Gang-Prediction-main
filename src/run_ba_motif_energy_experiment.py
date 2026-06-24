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
import csv
from collections import defaultdict
import os
import sys
import warnings
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import scipy.sparse as sp

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
from src.pattern_models import Pattern, create_pattern

# ── constants ─────────────────────────────────────────────────────────────────
MOTIF_TYPES = ["clique", "cycle", "star"]
MOTIF_SIZES = [
    5,
    10,
    # 25,
    50,
    100,
]
REPETITIONS = [
    1,
    # 3,
    5,
    10,
    # 20,
]

# colour palette per motif type
MOTIF_COLORS = {"clique": "#e41a1c", "cycle": "#377eb8", "star": "#4daf4a"}
SIZE_ALPHA = {10: 1.0, 25: 0.9, 50: 0.85, 100: 0.65}
REP_STYLES = {1: "-", 3: "--", 5: ":", 10: "-.", 20: (0, (3, 1, 1, 1))}

# Bridge internal degree δ_br per motif (note Prop. 2): the number of internal
# motif edges incident to a node that carries a bridge to the host. This is the
# ONLY motif-specific quantity that enters the moments — and only the 3rd one.
DELTA_BR = {
    "clique": lambda s: s - 1,  # bridge node is adjacent to every other node
    "cycle": lambda s: 2,  # bridge node has two ring neighbours
    "star": lambda s: 1,  # bridge attaches at a leaf (one spoke)
}
DEFAULT_BRIDGES = 3  # b: bridge edges per planted copy (cut per copy)


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


def select_random_nodes(
    N: int, size: int, used: set, rng: np.random.Generator
) -> Optional[np.ndarray]:
    """Pick *size* nodes uniformly at random, avoiding *used* nodes."""
    available = np.array(sorted(set(range(N)) - used))
    if len(available) < size:
        return None
    chosen = rng.choice(available, size=size, replace=False)
    return chosen


def _adj_list_from_edge_index(edge_index: torch.Tensor, N: int) -> Dict[int, List[int]]:
    """Build an adjacency list dict from a PyG edge_index (cached per call site)."""
    adj: Dict[int, List[int]] = defaultdict(list)
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for s, d in zip(src, dst):
        adj[s].append(d)
    return adj


def select_bfs_nodes(
    edge_index: torch.Tensor,
    N: int,
    size: int,
    used: set,
    rng: np.random.Generator,
    adj: Optional[Dict[int, List[int]]] = None,
) -> Optional[np.ndarray]:
    """BFS from a random seed, collecting *size* neighbours.

    Avoids nodes in *used*.  Retries up to 50 seeds if BFS doesn't yield enough.
    """
    if adj is None:
        adj = _adj_list_from_edge_index(edge_index, N)
    available = list(set(range(N)) - used)
    if len(available) < size:
        return None

    for _ in range(50):
        seed = int(rng.choice(available))
        visited = [seed]
        visited_set = {seed}
        frontier = [seed]
        while len(visited) < size and frontier:
            next_frontier = []
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
            return np.array(visited[:size])
    return None  # exhausted retries


# ═══════════════════════════════════════════════════════════════════════════════
# Graph planting
# ═══════════════════════════════════════════════════════════════════════════════


def plant_patterns_fresh(
    G_original: Data,
    pattern_type: str,
    size: int,
    n_reps: int,
    bridges: int,
    seed: int = 42,
) -> Tuple[Data, List]:
    """Low-conductance planting on **fresh** vertices (the paper's planting model).

    For each of *n_reps* copies we append *size* brand-new vertices, wire them
    with the pattern's internal edges, and attach the copy to the host with
    exactly *bridges* (=b) edges, each from a distinct non-hub gang node (so the
    star attaches at a leaf ⇒ δ_br=1) to a globally distinct host node.  This
    realises the simple-boundary, low-conductance regime ϕ=b/s of the note, in
    which the closed forms m1=ϕ, σ²=2ϕ−ϕ² hold and only γ sees the motif type.

    Crucially the boundary (cut, boundary-degree profile) is *constructed* and
    therefore identical across motif types for a given (size, b) — so m1 and σ²
    come out type-independent regardless of which fresh nodes are used.
    """
    rng = np.random.default_rng(seed)
    N0 = G_original.num_nodes
    edge_gen = EDGE_GENERATORS[pattern_type]

    ei = G_original.edge_index
    edge_set = set(zip(ei[0].tolist(), ei[1].tolist()))

    b = min(bridges, size - 1)
    # global pool of distinct host endpoints (no two copies share a host node)
    host_endpoints = rng.choice(N0, size=n_reps * b, replace=False)

    patterns = []
    next_id = N0
    for rep_idx in range(n_reps):
        nodes = np.arange(next_id, next_id + size)
        next_id += size

        # internal motif edges (both directions for undirected)
        for u, v in edge_gen(nodes):
            edge_set.add((u, v))
            edge_set.add((v, u))

        # b bridges: distinct NON-hub gang nodes (index ≥ 1) → distinct host nodes
        gpos = rng.choice(np.arange(1, size), size=b, replace=False)
        for k in range(b):
            u, v = int(nodes[gpos[k]]), int(host_endpoints[rep_idx * b + k])
            edge_set.add((u, v))
            edge_set.add((v, u))

        patterns.append(
            create_pattern(
                pattern_id=f"{pattern_type}_{rep_idx}",
                nodes=nodes,
                pattern_type=pattern_type,
                label="alert",
            )
        )

    N_new = next_id
    all_src, all_dst = zip(*sorted(edge_set)) if edge_set else ([], [])
    new_edge_index = torch.tensor([list(all_src), list(all_dst)], dtype=torch.long)
    G_new = Data(
        x=torch.eye(N_new, dtype=torch.float32),
        edge_index=new_edge_index,
        num_nodes=N_new,
    )
    G_new.edge_weight = torch.ones(G_new.edge_index.size(1), dtype=torch.float32)
    G_new = _ensure_graph_params(G_new)
    return G_new, patterns


def plant_patterns_in_graph(
    G_original: Data,
    pattern_type: str,
    size: int,
    n_reps: int,
    strategy: str = "random",
    seed: int = 42,
    bridges: int = DEFAULT_BRIDGES,
) -> Tuple[Data, List]:
    """Plant *n_reps* copies of *pattern_type* into a copy of the graph.

    strategy="fresh"  → low-conductance planting on brand-new vertices (paper
                        model; required for the moment/theory comparison).
    strategy="random"/"bfs" → legacy high-conductance planting on existing nodes.

    Returns
    -------
    G_new : Data  — modified graph with patterns planted
    patterns : list[Pattern]  — Pattern objects for each planted instance
    """
    if strategy == "fresh":
        return plant_patterns_fresh(
            G_original, pattern_type, size, n_reps, bridges, seed=seed
        )

    rng = np.random.default_rng(seed)
    N = G_original.num_nodes

    # Work with edge sets for efficient add/remove
    ei = G_original.edge_index
    src_list, dst_list = ei[0].tolist(), ei[1].tolist()
    edge_set = set(zip(src_list, dst_list))

    # Precompute adjacency for BFS
    adj = _adj_list_from_edge_index(ei, N) if strategy == "bfs" else None

    edge_gen = EDGE_GENERATORS[pattern_type]
    used_nodes: set = set()
    patterns = []

    for rep_idx in range(n_reps):
        # ── select nodes ────────────────────────────────────────────────
        if strategy == "random":
            nodes = select_random_nodes(N, size, used_nodes, rng)
        else:
            nodes = select_bfs_nodes(ei, N, size, used_nodes, rng, adj=adj)
        if nodes is None:
            LOGGER.warning(
                f"  [plant] could not find {size} unused nodes at rep {rep_idx} "
                f"({strategy}); stopping at {rep_idx} reps"
            )
            break

        used_nodes.update(nodes.tolist())

        # ── remove existing edges among selected nodes ──────────────────
        node_set = set(nodes.tolist())
        to_remove = {(u, v) for u, v in edge_set if u in node_set and v in node_set}
        edge_set -= to_remove

        # ── add pattern edges (both directions for undirected) ──────────
        new_edges = edge_gen(nodes)
        for u, v in new_edges:
            edge_set.add((u, v))
            edge_set.add((v, u))

        # ── create Pattern object ───────────────────────────────────────
        p = create_pattern(
            pattern_id=f"{pattern_type}_{rep_idx}",
            nodes=nodes,
            pattern_type=pattern_type,
            label="alert",
        )
        patterns.append(p)

    # ── rebuild PyG Data ────────────────────────────────────────────────
    if edge_set:
        all_src, all_dst = zip(*sorted(edge_set))
    else:
        all_src, all_dst = [], []
    new_edge_index = torch.tensor([list(all_src), list(all_dst)], dtype=torch.long)

    G_new = Data(
        x=G_original.x.clone(),
        edge_index=new_edge_index,
        num_nodes=N,
    )
    if hasattr(G_original, "y") and G_original.y is not None:
        G_new.y = G_original.y.clone()
    G_new.edge_weight = torch.ones(G_new.edge_index.size(1), dtype=torch.float32)
    G_new = _ensure_graph_params(G_new)

    return G_new, patterns


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Energy computation
# ═══════════════════════════════════════════════════════════════════════════════


def motif_energy(Uk: np.ndarray, node_sets: List[Pattern], N: int) -> np.ndarray:
    """Aggregate energy per eigenvector across all motif instances.

    For each instance with node set S, indicator v = 1_S / ‖1_S‖.
    energy_i[k] = (u_k^T v)^2.  Returns mean energy across instances.

    Parameters
    ----------
    Uk        : (N, K) eigenvector matrix
    node_sets : list of Pattern objects
    N         : number of graph nodes

    Returns
    -------
    mean_energy : (K,) array
    per_instance: (n_instances, K) array
    """
    K = Uk.shape[1]
    energies = []
    for pattern in node_sets:
        v = np.zeros(N, dtype=np.float64)
        v[pattern.nodes] = 1.0
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        proj = Uk.T @ v  # (K,)
        energies.append(proj**2)
    per_instance = np.array(energies)  # (n, K)
    mean_energy = per_instance.mean(axis=0)  # (K,)
    return mean_energy, per_instance


# ── Energy moments: mean / variance / skewness (measured vs theory) ─────────────
def _laplacian_from_edge_index(edge_index: torch.Tensor, N: int) -> sp.csr_matrix:
    """Combinatorial Laplacian L = D − A as a scipy sparse matrix (simple graph)."""
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    A = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(N, N))
    A = A.minimum(1)  # collapse any duplicate directed entries to a simple edge
    deg = np.asarray(A.sum(axis=1)).ravel()
    return sp.diags(deg) - A


def measured_moments(
    edge_index: torch.Tensor, N: int, support: np.ndarray
) -> Dict[str, float]:
    """Exact moments of the energy measure ν of v = 1_S/√|S| (note Defs. 2–3).

    m_t = vᵀ Lᵗ v are computed directly from the Laplacian (no eigendecomposition,
    note eq. 5).  Returns mean m1, variance σ², (excess) skewness γ, plus the
    measured conductance ϕ = cut(S)/s.
    """
    L = _laplacian_from_edge_index(edge_index, N)
    s = len(support)
    v = np.zeros(N, dtype=np.float64)
    v[support] = 1.0 / np.sqrt(s)
    Lv = L @ v
    L2v = L @ Lv
    m1 = float(v @ Lv)  # = cut(S)/s = ϕ  (Prop. 1, an identity)
    m2 = float(Lv @ Lv)  # vᵀL²v = ‖Lv‖²  (L symmetric)
    m3 = float(Lv @ L2v)  # vᵀL³v = (Lv)ᵀ(L²v)
    sigma2 = m2 - m1 * m1
    mu3 = m3 - 3 * m1 * m2 + 2 * m1**3
    gamma = mu3 / sigma2**1.5 if sigma2 > 1e-15 else float("nan")
    return {"phi": m1, "m1": m1, "m2": m2, "sigma2": sigma2, "gamma": gamma}


def theoretical_moments(phi: float, motif_type: str, size: int) -> Dict[str, float]:
    """Closed-form predictions of the note for the simple-boundary regime:

        m1 = ϕ                         (Prop. 1 — boundary only, type/r independent)
        σ² = 2ϕ − ϕ²                   (Cor. 1)
        γ  = (ϕ δ_br + 4ϕ − 6ϕ² + 2ϕ³) / (2ϕ − ϕ²)^{3/2}   (Prop. 2, eq. 10)

    Only γ carries the motif type, through δ_br ∈ {s-1, 2, 1} for clique/cycle/star.
    """
    delta_br = DELTA_BR[motif_type](size)
    m1 = phi
    sigma2 = 2 * phi - phi**2
    num = phi * delta_br + 4 * phi - 6 * phi**2 + 2 * phi**3
    gamma = num / sigma2**1.5 if sigma2 > 1e-15 else float("nan")
    return {"m1": m1, "sigma2": sigma2, "gamma": gamma, "delta_br": delta_br}


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


def plot_energy_by_size_all_types(
    results: Dict,
    n_reps: int,
    save_dir: str,
    smooth_window: int = 15,
) -> None:
    """
    Same layout as ``plot_energy_by_size`` (one subplot per motif size) but with
    **all motif types overlaid** in every panel instead of a single type.  Each
    panel shows the smoothed mean energy ± std for clique/cycle/star against the
    shared random baseline, with each type's k50/k90 marked.
    """
    sizes = MOTIF_SIZES
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Motif energy across eigenvectors  –  ALL TYPES  "
        f"({n_reps} rep{'s' if n_reps > 1 else ''})",
        fontsize=14,
    )

    for ax, size in zip(axes.flat, sizes):
        present = [m for m in MOTIF_TYPES if (m, size, n_reps) in results]
        if not present:
            ax.set_title(f"size={size}  [no data]")
            ax.axis("off")
            continue

        title_bits = []
        for mtype in present:
            res = results[(mtype, size, n_reps)]
            energy = res["mean_energy"]
            K = len(energy)
            x = np.arange(K)
            e_sm = _smoothed(energy, smooth_window)
            std_e = res.get("std_energy")
            if std_e is not None:
                s_sm = _smoothed(std_e, smooth_window)
                ax.fill_between(
                    x, np.maximum(0, e_sm - s_sm), e_sm + s_sm,
                    alpha=0.15, color=MOTIF_COLORS[mtype],
                )
            cum = cumulative_energy(energy)
            k50 = k_threshold(cum, 0.50)
            k90 = k_threshold(cum, 0.90)
            ax.plot(
                x, e_sm, color=MOTIF_COLORS[mtype], lw=1.6,
                label=f"{mtype} (k50={k50}, k90={k90})",
            )
            ax.axvline(k50, color=MOTIF_COLORS[mtype], ls=":", lw=1.0, alpha=0.6)
            ax.axvline(k90, color=MOTIF_COLORS[mtype], ls="--", lw=1.0, alpha=0.6)
            title_bits.append(mtype)

        # shared random baseline (same node-set size → essentially type-independent)
        ref = results[(present[0], size, n_reps)]
        b_sm = _smoothed(ref["baseline"], smooth_window)
        ax.plot(np.arange(len(b_sm)), b_sm, color="grey", lw=1.2, ls="--",
                label="random baseline")

        ax.set_title(
            f"size={size}  (planted: {ref['n_planted']})", fontsize=10
        )
        ax.set_xlabel("Eigenvector index  k  (sorted by eigenvalue)")
        ax.set_ylabel("Energy  (u_k^T v)²")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)

    plt.tight_layout()
    path = os.path.join(save_dir, f"energy_dist_all_types_reps{n_reps}.png")
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

    moment_lines = []  # collected per-type mean/variance/skewness for the text box
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
        # Energy-distribution moments (mean m1, variance σ², skewness γ) of ν
        m1 = res.get("m1")
        sig2 = res.get("sigma2")
        gam = res.get("gamma")
        if m1 is not None:
            lbl = (
                f"{mtype}  (m1={m1:.3g}, σ²={sig2:.3g}, γ={gam:.1f}; "
                f"k50={k50}, k90={k90})"
            )
            moment_lines.append(
                f"{mtype:<6} m1={m1:.4f}  σ²={sig2:.4f}  γ={gam:.2f}"
            )
        else:
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

    # ── mean / variance / skewness box (theory: m1=ϕ, σ²=2ϕ−ϕ² are type-indep.) ──
    if moment_lines:
        any_key = next(((m, size, n_reps) for m in MOTIF_TYPES
                        if (m, size, n_reps) in results), None)
        ref = results[any_key]
        phi = ref.get("phi")
        head = "Energy moments (measured)"
        if phi is not None:
            head += (
                f"\ntheory: m1=ϕ={ref['m1_th']:.4f}, σ²=2ϕ−ϕ²={ref['sigma2_th']:.4f}"
                "  (type-independent)"
            )
        box = head + "\n" + "\n".join(moment_lines)
        ax.text(
            0.985,
            0.97,
            box,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85),
        )

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
        default=5000,
        help="Number of nodes in the BA graph (default: 5000)",
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
    p.add_argument(
        "--planting",
        choices=["fresh", "random", "bfs"],
        default="random",
        help="fresh = low-conductance planting on new nodes (paper model; "
        "needed for the moment/theory comparison). default: random",
    )
    p.add_argument(
        "--bridges",
        type=int,
        default=DEFAULT_BRIDGES,
        help=f"b: bridge edges per planted copy (cut/copy). default: {DEFAULT_BRIDGES}",
    )
    return p.parse_args()


def write_moment_csvs(moment_rows: List[Dict], save_dir: str) -> None:
    """Write per-trial and trial-averaged energy-moment CSVs (measured vs theory)
    and log a compact comparison table.
    """
    if not moment_rows:
        LOGGER.warning("  [moments] no rows collected; skipping CSV.")
        return

    # 1. Raw per-(trial, config) rows
    raw_path = os.path.join(save_dir, "energy_moments.csv")
    fields = list(moment_rows[0].keys())
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in moment_rows:
            w.writerow(r)
    LOGGER.info(f"  Saved → {raw_path}  ({len(moment_rows)} rows)")

    # 2. Trial-averaged summary per (motif_type, size, reps)
    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in moment_rows:
        groups[(r["motif_type"], r["size"], r["reps"])].append(r)

    summary_fields = [
        "motif_type",
        "size",
        "reps",
        "bridges",
        "delta_br",
        "n_trials",
        "phi_measured",
        "phi_theory",
        "m1_measured",
        "m1_theory",
        "m1_abs_err",
        "sigma2_measured",
        "sigma2_theory",
        "sigma2_abs_err",
        "gamma_measured",
        "gamma_std",
        "gamma_theory",
        "gamma_abs_err",
    ]
    summary_rows = []
    for (mtype, size, reps), rows in sorted(
        groups.items(), key=lambda kv: (MOTIF_TYPES.index(kv[0][0]), kv[0][1], kv[0][2])
    ):

        def mean(k):
            return float(np.mean([x[k] for x in rows]))

        srow = {
            "motif_type": mtype,
            "size": size,
            "reps": reps,
            "bridges": rows[0]["bridges"],
            "delta_br": rows[0]["delta_br"],
            "n_trials": len(rows),
            "phi_measured": mean("phi_measured"),
            "phi_theory": mean("phi_theory"),
            "m1_measured": mean("m1_measured"),
            "m1_theory": mean("m1_theory"),
            "m1_abs_err": abs(mean("m1_measured") - mean("m1_theory")),
            "sigma2_measured": mean("sigma2_measured"),
            "sigma2_theory": mean("sigma2_theory"),
            "sigma2_abs_err": abs(mean("sigma2_measured") - mean("sigma2_theory")),
            "gamma_measured": mean("gamma_measured"),
            "gamma_std": float(np.std([x["gamma_measured"] for x in rows])),
            "gamma_theory": mean("gamma_theory"),
            "gamma_abs_err": abs(mean("gamma_measured") - mean("gamma_theory")),
        }
        summary_rows.append(srow)

    sum_path = os.path.join(save_dir, "energy_moments_summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(
                {
                    k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k])
                    for k in summary_fields
                }
            )
    LOGGER.info(f"  Saved → {sum_path}  ({len(summary_rows)} configs)")

    # 3. Log a compact comparison table (measured | theory)
    LOGGER.info("\n" + "=" * 92)
    LOGGER.info(
        "  ENERGY MOMENTS: measured (vᵀLᵗv) vs theory  — mean/variance type-indep., skew type-stamped"
    )
    LOGGER.info("-" * 92)
    LOGGER.info(
        f"{'type':<7}{'s':>4}{'r':>4}{'ϕ':>8} | "
        f"{'m1 meas':>9}{'m1 th':>9} | {'σ² meas':>9}{'σ² th':>9} | "
        f"{'γ meas':>9}{'γ th':>9}{'δbr':>5}"
    )
    LOGGER.info("-" * 92)
    for r in summary_rows:
        LOGGER.info(
            f"{r['motif_type']:<7}{r['size']:>4}{r['reps']:>4}{r['phi_measured']:>8.3f} | "
            f"{r['m1_measured']:>9.4f}{r['m1_theory']:>9.4f} | "
            f"{r['sigma2_measured']:>9.4f}{r['sigma2_theory']:>9.4f} | "
            f"{r['gamma_measured']:>9.2f}{r['gamma_theory']:>9.2f}{r['delta_br']:>5}"
        )
    LOGGER.info("=" * 92)


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

    configs = list(product(MOTIF_SIZES, REPETITIONS, MOTIF_TYPES))
    LOGGER.info(f"  Total configurations: {len(configs)}  × {args.n_trials} trials")

    # Accumulators: key → list of per-trial (K,) energy arrays
    trial_energies: Dict[tuple, List[np.ndarray]] = {}
    trial_baselines: Dict[tuple, List[np.ndarray]] = {}
    trial_n_planted: Dict[tuple, List[int]] = {}
    trial_moments: Dict[tuple, List[Dict]] = {}  # key → list of measured-moment dicts

    # Per-(trial, config) energy-moment rows: measured vs theory (→ CSV)
    moment_rows: List[Dict] = []

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

        for idx, (size, n_reps, mtype) in enumerate(configs, 1):
            tag = f"{mtype}_s{size}_r{n_reps}"

            if size >= N:
                continue

            # Plant motifs (vary seed by config index too)
            G_planted, motif_node_sets = plant_patterns_in_graph(
                G_base,
                mtype,
                size,
                n_reps,
                strategy=args.planting,
                bridges=args.bridges,
                seed=trial_seed + idx,
            )
            n_planted = len(motif_node_sets)
            if n_planted == 0:
                continue

            Np = G_planted.num_nodes  # node count of the planted graph (grows if fresh)

            # ── Energy moments: measured (vᵀLᵗv) vs closed-form theory ──────
            support = np.concatenate([np.asarray(p.nodes) for p in motif_node_sets])
            meas = measured_moments(G_planted.edge_index, Np, support)
            theo = theoretical_moments(meas["phi"], mtype, size)
            moment_rows.append(
                {
                    "trial": trial,
                    "motif_type": mtype,
                    "size": size,
                    "reps": n_reps,
                    "bridges": args.bridges,
                    "delta_br": theo["delta_br"],
                    "phi_measured": meas["phi"],
                    "phi_theory": args.bridges / size,
                    "m1_measured": meas["m1"],
                    "m1_theory": theo["m1"],
                    "sigma2_measured": meas["sigma2"],
                    "sigma2_theory": theo["sigma2"],
                    "gamma_measured": meas["gamma"],
                    "gamma_theory": theo["gamma"],
                }
            )

            # Spectral decomposition
            lk, Uk = compute_spectral_decomp(
                G_planted, K_max=K_max, dense_threshold=dense_threshold
            )

            # Energy
            mean_e, _ = motif_energy(Uk, motif_node_sets, Np)
            baseline = random_baseline_energy(
                Uk, [size] * n_planted, Np, n_trials=20, seed=trial_seed
            )

            key = (mtype, size, n_reps)
            trial_energies.setdefault(key, []).append(mean_e)
            trial_baselines.setdefault(key, []).append(baseline)
            trial_n_planted.setdefault(key, []).append(n_planted)
            trial_moments.setdefault(key, []).append({**meas, **{f"{k}_th": v for k, v in theo.items()}})

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

        # Trial-averaged energy moments (measured + theory) for annotations
        mlist = trial_moments.get(key, [])
        if mlist:
            results[key].update(
                {
                    "m1": float(np.mean([m["m1"] for m in mlist])),
                    "sigma2": float(np.mean([m["sigma2"] for m in mlist])),
                    "gamma": float(np.mean([m["gamma"] for m in mlist])),
                    "m1_th": float(np.mean([m["m1_th"] for m in mlist])),
                    "sigma2_th": float(np.mean([m["sigma2_th"] for m in mlist])),
                    "gamma_th": float(np.mean([m["gamma_th"] for m in mlist])),
                    "phi": float(np.mean([m["phi"] for m in mlist])),
                }
            )

    # ── Energy moments → CSV (measured vs theory) ─────────────────────────────
    write_moment_csvs(moment_rows, SAVE_DIR)

    # Re-read N from last trial graph (all trials have same n_nodes)
    N = args.n_nodes

    # ── 3. Visualisation ─────────────────────────────────────────────────────
    LOGGER.info("\n[3] Generating visualisations …")
    sw = args.smooth

    # A. Energy distribution per motif type (one figure per type × rep-count)
    for mtype in MOTIF_TYPES:
        for n_reps in REPETITIONS:
            plot_energy_by_size(results, mtype, n_reps, SAVE_DIR, smooth_window=sw)

    # A2. Same layout but all motif types overlaid (one figure per rep-count)
    for n_reps in REPETITIONS:
        plot_energy_by_size_all_types(results, n_reps, SAVE_DIR, smooth_window=sw)

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
