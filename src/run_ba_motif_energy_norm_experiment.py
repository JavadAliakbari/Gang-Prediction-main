"""run_ba_motif_energy_norm_experiment.py
=========================================
Measure how planted motifs distribute their spectral energy across the
**symmetric normalized** Laplacian spectrum, and compare the measured energy
moments against the closed-form analysis of the paper
("Detecting Financial Crime Gangs via Spectral Conductance", Section 4).

Why a new file?
---------------
The sibling script ``run_ba_motif_energy_experiment.py`` uses the *combinatorial*
Laplacian L = D − A with the *uniform* indicator 1_S/√s, and its energy is the
plain projection (u_kᵀv)².  The paper's analysis has since moved to the
**symmetric normalized** Laplacian with self-loops and the **degree-weighted**
indicator, and to the λ-weighted L-energy measure.  The closed forms are
different, so this file re-implements the measurement in that metric.

The paper's setup (Def. 3.1–3.3, Thm. 4.2, Cor. 4.4)
----------------------------------------------------
* Self-loops:      W̃ = W + I,     D̃ = diag(W̃ 1),   d̃_i = d_i + 1
* Normalized adj:  Â = D̃^{-1/2} W̃ D̃^{-1/2}
* Normalized Lap:  L = I − Â,      eigenvalues 0 = λ_0 ≤ … < 2
* Gang indicator:  v_S = D̃^{1/2} 1_S / √vol(S),   vol(S) = Σ_{i∈S} d̃_i,  ‖v_S‖₂ = 1
* Conductance:     Φ = cut(S)/vol(S) = v_Sᵀ L v_S = m_1                     (identity)
* L-energy measure: q_k = λ_k (u_kᵀ v_S)² / Φ,   Σ_k q_k = 1     (λ-weighted!)
* Raw moments:     m_t = v_Sᵀ Lᵗ v_S
* Energy moments:  m̃_t = Σ_k λ_kᵗ q_k = m_{t+1}/m_1
      mean      m̃_1 = m_2/m_1
      variance  σ̃²  = m_3/m_1 − (m_2/m_1)²

Theory compared against (all EXACT or provable bounds):
  1. Boundary-edge identity (Thm. 4.2):   m̃_1 = ⟨ρ + β⟩_∂
       ρ_i = d_∂(i)/d̃_i  (S-side boundary fraction),  β_h = b_h/d̃_h  (host-side)
       — matches the measured m_2/m_1 to machine precision.
  2. Universal bounds (Thm. 4.2):          Φ ≤ m̃_1 ≤ 2,   σ̃² ≤ m̃_1(2 − m̃_1)
  3. Density upper bound (Cor. 4.4):       m̃_1 ≤ 1/(δ+2) + 1/(D+1)
                                           σ̃² ≤ 2/(δ+2) + 2/(D+1)
     with internal degree δ ∈ {s−1, 2, 1} for clique/cycle/star and host degree ≥ D.
  4. Gang-only asymptote (Cor. 4.4):       m̃_1^gang → 1/(δ+2):
       clique → 1/(s+1) → 0,   cycle → 1/4,   star → 1/3   (the star bottleneck).

Unlike the combinatorial metric (where the missed-energy floor 2/λ_K was
universal and demanded λ_K ≫ 2, impossible since λ ≤ 2), the normalized floor
m̃_1/λ_K is density-dependent and vanishes with the inverse density of the gang
(Cor. 4.6) — so a dense low-conductance gang concentrates at the bottom of the
normalized spectrum.

Outputs (under results/<timestamp>/ba_motif_energy_norm/)
---------------------------------------------------------
* energy_dist_norm_{motif}_reps{r}.png    – smooth energy density f(λ), λ∈[0,2], per size
* energy_compare_norm_size{s}_reps{r}.png – cross-motif energy density f(λ) with moment box
* cumulative_energy_norm_reps{r}.png       – cumulative energy vs frequency λ grid
* moments_validation_norm.png              – measured m̃_1/σ̃² vs theory & bounds
* energy_moments_norm.csv / _summary.csv   – measured vs theory tables

Averaging across graphs: each graph instance has different eigenvalues, so binning
gives jagged curves and index-averaging mixes frequencies.  Instead we treat each
eigenvalue λ_k as an energy-weighted sample (weight q_k) and use a weighted
Gaussian KDE (``--kde_bw``) on a shared fine grid over [0,2] (``--n_grid``); the
per-instance smooth densities are then averaged — smooth by construction, with
∫f dλ = 1 and no binning artefacts.

Usage (from repo root, with the FedStruct conda env active)::

    python src/run_ba_motif_energy_norm_experiment.py                 # full
    python src/run_ba_motif_energy_norm_experiment.py --moments_only  # fast, CSV only
    python src/run_ba_motif_energy_norm_experiment.py --planting fresh --bridges 3
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")

# ── path setup (mirror the sibling script so imports resolve when run directly) ─
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.utils.utils import save_path, LOGGER

# Reuse the *pure* (theory-free) machinery from the combinatorial script:
# graph construction, motif planting, motif metadata and generic plot utilities.
from run_ba_motif_energy_experiment import (  # noqa: E402
    build_ba_graph,
    plant_patterns_in_graph,
    DEFAULT_BRIDGES,
    MOTIF_TYPES,
    MOTIF_SIZES,
    REPETITIONS,
    MOTIF_COLORS,
)

# Internal (minimum) degree of a gang node per motif — this is the δ that enters
# the density bound m̃_1 ≤ 1/(δ+2)+1/(D+1) (Cor. 4.4).  For the star the bridge
# lands on a leaf (internal degree 1), which is why the star is the bottleneck.
DELTA_INTERNAL = {
    "clique": lambda s: s - 1,
    "cycle": lambda s: 2,
    "star": lambda s: 1,  # leaf-attached
}
# Gang-only asymptote m̃_1^gang → 1/(δ+2).
GANG_ASYMPTOTE = {
    "clique": lambda s: 1.0 / (s + 1),
    "cycle": lambda s: 1.0 / 4.0,
    "star": lambda s: 1.0 / 3.0,
}

# ── Smooth spectral energy density via weighted KDE ────────────────────────────
# Every graph instance has DIFFERENT eigenvalues, so we can neither average energy
# by eigenvector index (index k is a different λ per graph) nor get a smooth curve
# out of histogram binning.  Instead each eigenvalue λ_k is treated as an
# energy-WEIGHTED sample (weight q_k) and smeared by a Gaussian kernel of width h;
# summing these bumps over all eigenvalues — and averaging over graph instances —
# gives ONE smooth spectral energy density f(λ) on a shared fine grid over [0, 2].
# By construction ∫f dλ = Σ_k q_k = 1, and the average across graphs is exact
# (a mixture of every instance's kernels), with no binning artefacts.
N_LAMBDA_GRID = 400
LAMBDA_GRID = np.linspace(0.0, 2.0, N_LAMBDA_GRID)
KDE_BANDWIDTH = 0.03  # Gaussian σ in λ-units; overridable from the CLI


def set_lambda_grid(n_grid: int, bandwidth: float) -> None:
    """Configure the shared KDE evaluation grid and bandwidth (before threads)."""
    global N_LAMBDA_GRID, LAMBDA_GRID, KDE_BANDWIDTH
    N_LAMBDA_GRID = int(n_grid)
    LAMBDA_GRID = np.linspace(0.0, 2.0, N_LAMBDA_GRID)
    KDE_BANDWIDTH = float(bandwidth)


def kde_density(lk: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Smooth energy density f(λ) on ``LAMBDA_GRID`` from weighted samples (λ_k, q_k).

    f(λ) = Σ_k q_k · N(λ; λ_k, h²) — a Gaussian mixture, h = ``KDE_BANDWIDTH``.
    Samples are reflected across the boundaries λ=0 and λ=2 so the low-λ peak's
    mass is not lost past the edge, then f is renormalised so ∫f dλ = Σ_k q_k
    on [0, 2].  Averaging these densities across graphs stays smooth and exact.
    """
    h = KDE_BANDWIDTH
    centers = np.concatenate([lk, -lk, 4.0 - lk])  # reflect at 0 and 2
    weights = np.concatenate([q, q, q])
    d = (LAMBDA_GRID[:, None] - centers[None, :]) / h  # (G, 3M)
    f = (np.exp(-0.5 * d * d) @ weights) / (h * np.sqrt(2.0 * np.pi))
    area = float(np.trapz(f, LAMBDA_GRID))
    if area > 0:
        f *= q.sum() / area
    return f


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Symmetric normalized Laplacian operators
# ═══════════════════════════════════════════════════════════════════════════════


def _simple_adjacency(edge_index: torch.Tensor, N: int) -> sp.csr_matrix:
    """Symmetric 0/1 adjacency (no self-loops) from a PyG edge_index."""
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    A = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(N, N))
    A = A.minimum(1)  # collapse any duplicate directed entries
    A = A.maximum(A.T)  # force symmetry
    A.setdiag(0)  # no self loops in W
    A.eliminate_zeros()
    return A


def normalized_laplacian(
    edge_index: torch.Tensor, N: int
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Return (L, d̃) with L = I − D̃^{-1/2}(W+I)D̃^{-1/2}, d̃_i = deg_i + 1.

    L is the symmetric normalized Laplacian *with self-loops* (Def. 3.1); its
    eigenvalues lie in [0, 2).  d̃ is the self-looped degree used to build the
    degree-weighted gang indicator and the volume.
    """
    A = _simple_adjacency(edge_index, N)
    deg = np.asarray(A.sum(axis=1)).ravel()
    dt = deg + 1.0  # self-loop degree d̃
    dis = 1.0 / np.sqrt(dt)
    Dis = sp.diags(dis)
    What = A + sp.eye(N)  # W̃ = W + I
    Ahat = (Dis @ What @ Dis).tocsr()
    L = (sp.eye(N) - Ahat).tocsr()
    return L, dt


def degree_weighted_indicator(
    support: np.ndarray, dt: np.ndarray, N: int
) -> np.ndarray:
    """v_S = D̃^{1/2} 1_S / √vol(S),  vol(S) = Σ_{i∈S} d̃_i,  ‖v_S‖₂ = 1."""
    vol = float(dt[support].sum())
    v = np.zeros(N, dtype=np.float64)
    v[support] = np.sqrt(dt[support])
    v /= np.sqrt(vol)
    return v


def normalized_spectral_decomp(
    edge_index: torch.Tensor, N: int, k_max: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Full dense eigendecomposition of the symmetric normalized Laplacian.

    Returns (λ, U) sorted ascending.  If *k_max*>0 only the bottom-k columns are
    returned (the low-frequency band where low-conductance gangs concentrate).
    """
    L, _ = normalized_laplacian(edge_index, N)
    # eigh on the symmetric dense matrix — L is small enough (a few thousand).
    w, U = np.linalg.eigh(L.toarray())
    if k_max and k_max > 0:
        return w[:k_max], U[:, :k_max]
    return w, U


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Normalized L-energy distribution and moments
# ═══════════════════════════════════════════════════════════════════════════════


def energy_per_eigenvalue(
    lk: np.ndarray, Uk: np.ndarray, node_sets: List, dt: np.ndarray, N: int
) -> np.ndarray:
    """Per-eigenvalue normalized L-energy q_k, averaged over motif instances.

    For each instance q_k = λ_k(u_kᵀv_S)²/Φ (Σ_k q_k = 1, Def. 3.2); returns the
    mean over instances, aligned with *lk* (length = #eigenvalues).  Smoothing/
    averaging onto the shared λ-grid via ``kde_density`` is deferred to the
    aggregation step.
    """
    acc = np.zeros(len(lk))
    for pattern in node_sets:
        v = degree_weighted_indicator(np.asarray(pattern.nodes), dt, N)
        raw = lk * (Uk.T @ v) ** 2  # λ_k (u_kᵀ v)²
        phi = float(raw.sum())  # = Φ (full spectrum)
        acc += raw / phi if phi > 0 else raw
    return acc / len(node_sets)


def baseline_per_eigenvalue(
    lk: np.ndarray,
    Uk: np.ndarray,
    node_sizes: List[int],
    dt: np.ndarray,
    N: int,
    n_trials: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """Per-eigenvalue normalized L-energy for random node sets of the same sizes.

    Random sets have Φ ≈ 1 (high conductance) so their energy spreads to high λ
    (m̃_1 ≈ 1) — the frequency-domain contrast the theory predicts (Cor. C.21).
    """
    rng = np.random.default_rng(seed)
    acc = np.zeros(len(lk))
    count = 0
    for sz in node_sizes:
        for _ in range(n_trials):
            idx = rng.choice(N, size=max(1, sz), replace=False)
            v = degree_weighted_indicator(idx, dt, N)
            raw = lk * (Uk.T @ v) ** 2
            tot = raw.sum()
            acc += raw / tot if tot > 0 else raw
            count += 1
    return acc / max(1, count)


def lambda_threshold(density: np.ndarray, frac: float) -> float:
    """Frequency λ below which *frac* of the smooth energy density lies."""
    c = np.cumsum(density)
    if c[-1] <= 0:
        return float("nan")
    c = c / c[-1]
    i = min(int(np.searchsorted(c, frac)), len(LAMBDA_GRID) - 1)
    return float(LAMBDA_GRID[i])


def measured_moments_norm(
    edge_index: torch.Tensor, N: int, support: np.ndarray
) -> Dict[str, float]:
    """Exact moments of the normalized L-energy measure ν_L (no eigendecomposition).

    Uses only sparse mat-vecs against L (paper's O(K|E|) recipe):
        m_t = v_SᵀLᵗv_S,  computed via Lv, L²v.
        Φ   = m_1                                  (= cut/vol, the conductance)
        m̃_1 = m_2/m_1                              (energy mean)
        σ̃²  = m_3/m_1 − (m_2/m_1)²                 (energy variance)
        γ̃   = μ̃_3 / σ̃³                            (energy skewness)
    """
    L, dt = normalized_laplacian(edge_index, N)
    v = degree_weighted_indicator(np.asarray(support), dt, N)
    Lv = L @ v
    L2v = L @ Lv
    m1 = float(v @ Lv)  # Φ
    m2 = float(Lv @ Lv)  # vᵀL²v = ‖Lv‖²
    m3 = float(Lv @ L2v)  # vᵀL³v
    m4 = float(L2v @ L2v)  # vᵀL⁴v = ‖L²v‖²

    phi = m1
    tm1 = m2 / m1  # m̃_1
    tm2 = m3 / m1  # m̃_2
    tm3 = m4 / m1  # m̃_3
    tsig2 = tm2 - tm1 * tm1  # σ̃²
    tmu3 = tm3 - 3 * tm1 * tm2 + 2 * tm1**3
    tgamma = tmu3 / tsig2**1.5 if tsig2 > 1e-15 else float("nan")
    return {
        "phi": phi,
        "tm1": tm1,
        "tsigma2": tsig2,
        "tgamma": tgamma,
    }


def boundary_theory_norm(
    edge_index: torch.Tensor,
    N: int,
    support: np.ndarray,
    motif_type: str,
    size: int,
) -> Dict[str, float]:
    """Closed-form / provable predictions of the paper for the normalized energy.

    Returns
    -------
    tm1_boundary : exact ⟨ρ+β⟩_∂ of Thm. 4.2 (matches m_2/m_1 to machine eps)
    tm1_density_bound, tsigma2_density_bound : Cor. 4.4 upper bounds
    tm1_gang_asymptote : gang-only limit 1/(δ+2)
    phi_exact : cut(S)/vol(S)
    delta, host_min_deg : the δ and D that enter the bounds
    """
    A = _simple_adjacency(edge_index, N)
    indptr, indices = A.indptr, A.indices
    deg = np.asarray(A.sum(axis=1)).ravel()
    dt = deg + 1.0

    Sset = set(int(i) for i in support)
    vol = float(dt[list(Sset)].sum())

    def nbrs(i):
        return indices[indptr[i] : indptr[i + 1]]

    cut = 0
    tot = 0.0
    min_internal = np.inf
    host_min_deg = np.inf
    for i in Sset:
        nb = nbrs(i)
        internal_deg = int(sum(1 for j in nb if int(j) in Sset))
        d_partial = int(len(nb) - internal_deg)
        min_internal = min(min_internal, internal_deg)
        rho_i = d_partial / dt[i]
        for j in nb:
            h = int(j)
            if h in Sset:
                continue
            cut += 1
            b_h = int(sum(1 for k in nbrs(h) if int(k) in Sset))
            tot += rho_i + b_h / dt[h]
            host_min_deg = min(host_min_deg, deg[h])

    if cut == 0:
        # No boundary (disconnected planting): energy is degenerate.
        return {
            "tm1_boundary": float("nan"),
            "tm1_density_bound": float("nan"),
            "tsigma2_density_bound": float("nan"),
            "tm1_gang_asymptote": GANG_ASYMPTOTE[motif_type](size),
            "phi_exact": 0.0,
            "delta": DELTA_INTERNAL[motif_type](size),
            "host_min_deg": float("nan"),
        }

    tm1_bdy = tot / cut
    phi_exact = cut / vol

    # δ = min internal degree (equals the nominal {s−1,2,1} for fresh planting).
    delta = (
        int(min_internal)
        if np.isfinite(min_internal)
        else DELTA_INTERNAL[motif_type](size)
    )
    D = float(host_min_deg) if np.isfinite(host_min_deg) else 0.0
    tm1_bound = 1.0 / (delta + 2) + 1.0 / (D + 1)
    tsig2_bound = 2.0 / (delta + 2) + 2.0 / (D + 1)
    return {
        "tm1_boundary": tm1_bdy,
        "tm1_density_bound": tm1_bound,
        "tsigma2_density_bound": tsig2_bound,
        "tm1_gang_asymptote": GANG_ASYMPTOTE[motif_type](size),
        "phi_exact": phi_exact,
        "delta": delta,
        "host_min_deg": D,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Visualisation
# ═══════════════════════════════════════════════════════════════════════════════


def plot_energy_by_size_norm(
    results: Dict, motif_type: str, n_reps: int, save_dir: str, smooth_window: int = 0
) -> None:
    """One subplot per size: smooth normalized L-energy density vs frequency λ.

    The curve is the trial-averaged KDE density f(λ) of the energy over the
    spectrum (∫f dλ = 1).  A solid black line marks the energy mean m̃₁ (the
    distribution's centroid); λ50/λ90 mark the frequency cutoffs holding 50%/90%
    of the energy.  ``smooth_window`` is ignored (kept for call-site compat).
    """
    x = LAMBDA_GRID
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Normalized L-energy density vs graph frequency λ  –  {motif_type.upper()}  "
        f"({n_reps} rep{'s' if n_reps > 1 else ''})",
        fontsize=14,
    )
    for ax, size in zip(axes.flat, MOTIF_SIZES):
        key = (motif_type, size, n_reps)
        if key not in results:
            ax.set_title(f"size={size}  [no data]")
            ax.axis("off")
            continue
        res = results[key]
        energy = res["mean_energy"]
        baseline = res["baseline"]
        std_e = res.get("std_energy")
        if std_e is not None:
            ax.fill_between(
                x,
                np.maximum(0, energy - std_e),
                energy + std_e,
                alpha=0.20,
                color=MOTIF_COLORS[motif_type],
            )
        ax.plot(x, energy, color=MOTIF_COLORS[motif_type], lw=1.8,
                label=f"{motif_type} (size {size})")
        ax.plot(x, baseline, color="grey", lw=1.2, ls="--", label="random baseline")
        tm1, sig2 = res.get("tm1"), res.get("tsigma2")
        l50, l90 = lambda_threshold(energy, 0.50), lambda_threshold(energy, 0.90)
        if tm1 is not None:
            ax.axvline(
                tm1, color="black", lw=1.1, alpha=0.8, label=f"mean m̃₁={tm1:.3f}"
            )
        ax.axvline(l50, color="orange", ls=":", lw=1.2, label=f"λ50={l50:.2f}")
        ax.axvline(l90, color="red", ls=":", lw=1.2, label=f"λ90={l90:.2f}")
        ax.set_title(
            f"size={size}  |  m̃₁={tm1:.3f}, σ̃²={sig2:.3f}  |  λ50={l50:.2f}, λ90={l90:.2f}",
            fontsize=9,
        )
        ax.set_xlim(0, 2)
        ax.set_xlabel("Eigenvalue λ  (graph frequency, 0 … 2)")
        ax.set_ylabel("Energy density  f(λ)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)
    plt.tight_layout()
    path = os.path.join(save_dir, f"energy_dist_norm_{motif_type}_reps{n_reps}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"  Saved → {path}")


def plot_energy_compare_norm(
    results: Dict, size: int, n_reps: int, save_dir: str, smooth_window: int = 0
) -> None:
    """Cross-motif smooth energy density vs frequency λ at fixed (size, reps).

    Overlays the three motifs' KDE densities with a vertical line at each energy
    mean m̃₁ and a measured-vs-theory moment box.  ``smooth_window`` is ignored.
    """
    x = LAMBDA_GRID
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title(
        f"Normalized L-energy density vs graph frequency λ  –  size={size}, reps={n_reps}",
        fontsize=13,
    )
    moment_lines = []
    for mtype in MOTIF_TYPES:
        key = (mtype, size, n_reps)
        if key not in results:
            continue
        res = results[key]
        energy = res["mean_energy"]
        tm1, sig2, gam = res.get("tm1"), res.get("tsigma2"), res.get("tgamma")
        tm1_bdy = res.get("tm1_boundary")
        l50, l90 = lambda_threshold(energy, 0.50), lambda_threshold(energy, 0.90)
        lbl = f"{mtype}  (m̃₁={tm1:.3g}, σ̃²={sig2:.3g}; λ50={l50:.2f}, λ90={l90:.2f})"
        moment_lines.append(
            f"{mtype:<6} m̃₁={tm1:.4f} (bdy {tm1_bdy:.4f})  σ̃²={sig2:.4f}  γ̃={gam:.2f}"
        )
        std_e = res.get("std_energy")
        if std_e is not None:
            ax.fill_between(
                x, np.maximum(0, energy - std_e), energy + std_e,
                alpha=0.15, color=MOTIF_COLORS[mtype],
            )
        ax.plot(x, energy, color=MOTIF_COLORS[mtype], lw=2, label=lbl)
        if tm1 is not None:
            ax.axvline(tm1, color=MOTIF_COLORS[mtype], ls="-", lw=1.1, alpha=0.7)
    for mtype in MOTIF_TYPES:
        if (mtype, size, n_reps) in results:
            ax.plot(
                x, results[(mtype, size, n_reps)]["baseline"],
                color="grey", lw=1.5, ls="--", label="random baseline",
            )
            break
    ax.set_xlim(0, 2)
    ax.set_xlabel(
        "Eigenvalue λ  (graph frequency, 0 … 2)   —   vertical lines = energy mean m̃₁"
    )
    ax.set_ylabel("Energy density  f(λ)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    if moment_lines:
        box = (
            "Energy moments (measured vs Thm. 4.2 boundary form)\n"
            "theory: m̃₁=⟨ρ+β⟩_∂ (exact),  Φ≤m̃₁≤2,  σ̃²≤m̃₁(2−m̃₁)\n"
            + "\n".join(moment_lines)
        )
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
    path = os.path.join(save_dir, f"energy_compare_norm_size{size}_reps{n_reps}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"  Saved → {path}")


def _cdf(density: np.ndarray) -> np.ndarray:
    """Normalised cumulative energy from a density on ``LAMBDA_GRID``."""
    c = np.cumsum(density)
    return c / c[-1] if c[-1] > 0 else c


def plot_cumulative_grid_norm(results: Dict, n_reps: int, save_dir: str) -> None:
    """Grid of cumulative energy vs graph frequency λ: rows = type, cols = size."""
    x = LAMBDA_GRID
    n_rows, n_cols = len(MOTIF_TYPES), len(MOTIF_SIZES)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharey=True
    )
    fig.suptitle(
        f"Cumulative normalized L-energy vs frequency λ  –  {n_reps} repetition(s)",
        fontsize=15,
    )
    for r, mtype in enumerate(MOTIF_TYPES):
        for c, size in enumerate(MOTIF_SIZES):
            ax = axes[r, c]
            ax.axhline(0.5, color="orange", ls=":", lw=0.9, alpha=0.7)
            ax.axhline(0.9, color="red", ls=":", lw=0.9, alpha=0.7)
            key = (mtype, size, n_reps)
            if key not in results:
                ax.set_title(f"{mtype} / s={size}\n[no data]", fontsize=8)
                ax.set_ylim(0, 1.05)
                continue
            res = results[key]
            energy = res["mean_energy"]
            cum = _cdf(energy)
            cum_b = _cdf(res["baseline"])
            ax.plot(x, cum, color=MOTIF_COLORS[mtype], lw=1.8)
            ax.plot(x, cum_b, color="grey", ls="--", lw=1.2)
            l50, l90 = lambda_threshold(energy, 0.50), lambda_threshold(energy, 0.90)
            ax.axvline(l50, color="orange", ls=":", lw=0.9, alpha=0.7)
            ax.axvline(l90, color="red", ls=":", lw=0.9, alpha=0.7)
            ax.set_title(
                f"{mtype}  size={size}\nλ50={l50:.2f}, λ90={l90:.2f}", fontsize=8
            )
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 1.05)
            if c == 0:
                ax.set_ylabel("Cumulative energy")
            if r == n_rows - 1:
                ax.set_xlabel("Eigenvalue λ (frequency)")
            ax.grid(alpha=0.2)
    plt.tight_layout()
    path = os.path.join(save_dir, f"cumulative_energy_norm_reps{n_reps}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"  Saved → {path}")


def plot_moments_validation(summary_rows: List[Dict], save_dir: str) -> None:
    """The headline figure: measured energy moments vs the paper's analysis.

    Top row  (m̃_1): measured vs exact boundary form ⟨ρ+β⟩_∂, the gang asymptote
                     1/(δ+2) and the density upper bound 1/(δ+2)+1/(D+1).
    Bottom row (σ̃²): measured vs the universal bound m̃_1(2−m̃_1) and the density
                     bound 2/(δ+2)+2/(D+1).
    One column per motif type; x-axis = motif size.
    """
    # index by (type,size) averaged over reps for a clean size-sweep
    by_ts: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in summary_rows:
        by_ts[(r["motif_type"], r["size"])].append(r)

    fig, axes = plt.subplots(
        2, len(MOTIF_TYPES), figsize=(5 * len(MOTIF_TYPES), 8), sharex=True
    )
    if len(MOTIF_TYPES) == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(
        "Normalized L-energy moments: simulation vs. paper analysis (Thm. 4.2 / Cor. 4.4)",
        fontsize=14,
    )

    for c, mtype in enumerate(MOTIF_TYPES):
        sizes = sorted({s for (m, s) in by_ts if m == mtype})
        if not sizes:
            continue

        def agg(size, key):
            rows = by_ts[(mtype, size)]
            vals = [x[key] for x in rows if x[key] == x[key]]  # drop NaN
            return float(np.mean(vals)) if vals else float("nan")

        m_meas = [agg(s, "tm1_measured") for s in sizes]
        m_bdy = [agg(s, "tm1_boundary") for s in sizes]
        m_asym = [agg(s, "tm1_gang_asymptote") for s in sizes]
        m_bound = [agg(s, "tm1_density_bound") for s in sizes]
        s_meas = [agg(s, "tsigma2_measured") for s in sizes]
        s_univ = [t * (2 - t) for t in m_meas]  # m̃_1(2−m̃_1)
        s_bound = [agg(s, "tsigma2_density_bound") for s in sizes]

        col = MOTIF_COLORS[mtype]
        ax = axes[0, c]
        ax.plot(
            sizes, m_meas, "o-", color=col, lw=2, ms=7, label="m̃₁ measured (m₂/m₁)"
        )
        ax.plot(
            sizes,
            m_bdy,
            "x--",
            color="black",
            lw=1.2,
            ms=8,
            label="⟨ρ+β⟩_∂  (Thm. 4.2, exact)",
        )
        ax.plot(
            sizes,
            m_asym,
            ":",
            color="tab:green",
            lw=1.8,
            label="gang asymptote 1/(δ+2)",
        )
        ax.plot(
            sizes,
            m_bound,
            "-.",
            color="tab:red",
            lw=1.2,
            label="density bound 1/(δ+2)+1/(D+1)",
        )
        ax.axhline(2.0, color="grey", lw=0.8, alpha=0.5)
        ax.set_title(f"{mtype}  —  mean m̃₁", fontsize=11)
        ax.set_ylabel("m̃₁")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

        ax2 = axes[1, c]
        ax2.plot(sizes, s_meas, "o-", color=col, lw=2, ms=7, label="σ̃² measured")
        ax2.plot(sizes, s_univ, "--", color="black", lw=1.2, label="bound m̃₁(2−m̃₁)")
        ax2.plot(
            sizes,
            s_bound,
            "-.",
            color="tab:red",
            lw=1.2,
            label="density bound 2/(δ+2)+2/(D+1)",
        )
        ax2.set_title(f"{mtype}  —  variance σ̃²", fontsize=11)
        ax2.set_xlabel("Motif size s")
        ax2.set_ylabel("σ̃²")
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.25)

    plt.tight_layout()
    path = os.path.join(save_dir, "moments_validation_norm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CSV writers
# ═══════════════════════════════════════════════════════════════════════════════


def write_moment_csvs_norm(moment_rows: List[Dict], save_dir: str) -> List[Dict]:
    """Write per-(trial,config) and trial-averaged normalized-moment CSVs.

    Returns the trial-averaged summary rows (for the validation plot).
    """
    if not moment_rows:
        LOGGER.warning("  [moments] no rows collected; skipping CSV.")
        return []

    raw_path = os.path.join(save_dir, "energy_moments_norm.csv")
    fields = list(moment_rows[0].keys())
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in moment_rows:
            w.writerow(r)
    LOGGER.info(f"  Saved → {raw_path}  ({len(moment_rows)} rows)")

    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in moment_rows:
        groups[(r["motif_type"], r["size"], r["reps"])].append(r)

    summary_fields = [
        "motif_type",
        "size",
        "reps",
        "bridges",
        "delta",
        "host_min_deg",
        "n_trials",
        "phi_measured",
        "tm1_measured",
        "tm1_boundary",
        "tm1_abs_err",
        "tm1_gang_asymptote",
        "tm1_density_bound",
        "tm1_within_bound",
        "tsigma2_measured",
        "tsigma2_universal_bound",
        "tsigma2_density_bound",
        "tsigma2_within_bound",
        "tgamma_measured",
        "tgamma_std",
    ]
    summary_rows = []
    for (mtype, size, reps), rows in sorted(
        groups.items(), key=lambda kv: (MOTIF_TYPES.index(kv[0][0]), kv[0][1], kv[0][2])
    ):

        def mean(k):
            vals = [x[k] for x in rows if x[k] == x[k]]
            return float(np.mean(vals)) if vals else float("nan")

        tm1_m = mean("tm1_measured")
        tm1_b = mean("tm1_boundary")
        sig_m = mean("tsigma2_measured")
        sig_u = tm1_m * (2 - tm1_m)
        sig_d = mean("tsigma2_density_bound")
        srow = {
            "motif_type": mtype,
            "size": size,
            "reps": reps,
            "bridges": rows[0]["bridges"],
            "delta": rows[0]["delta"],
            "host_min_deg": mean("host_min_deg"),
            "n_trials": len(rows),
            "phi_measured": mean("phi_measured"),
            "tm1_measured": tm1_m,
            "tm1_boundary": tm1_b,
            "tm1_abs_err": abs(tm1_m - tm1_b),
            "tm1_gang_asymptote": mean("tm1_gang_asymptote"),
            "tm1_density_bound": mean("tm1_density_bound"),
            "tm1_within_bound": bool(tm1_m <= mean("tm1_density_bound") + 1e-9),
            "tsigma2_measured": sig_m,
            "tsigma2_universal_bound": sig_u,
            "tsigma2_density_bound": sig_d,
            "tsigma2_within_bound": bool(sig_m <= sig_u + 1e-9),
            "tgamma_measured": mean("tgamma_measured"),
            "tgamma_std": float(np.std([x["tgamma_measured"] for x in rows])),
        }
        summary_rows.append(srow)

    sum_path = os.path.join(save_dir, "energy_moments_norm_summary.csv")
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

    # Compact comparison table in the log
    LOGGER.info("\n" + "=" * 104)
    LOGGER.info(
        "  NORMALIZED L-ENERGY MOMENTS: measured vs paper analysis "
        "(Thm. 4.2 boundary form + Cor. 4.4 bounds)"
    )
    LOGGER.info("-" * 104)
    LOGGER.info(
        f"{'type':<7}{'s':>4}{'r':>4}{'δ':>4}{'Φ':>8} | "
        f"{'m̃1 meas':>9}{'m̃1 bdy':>9}{'|err|':>8}{'1/(δ+2)':>9}{'bound':>8}{'ok':>4} | "
        f"{'σ̃² meas':>9}{'m̃1(2-m̃1)':>10}{'ok':>4} | {'γ̃':>7}"
    )
    LOGGER.info("-" * 104)
    for r in summary_rows:
        LOGGER.info(
            f"{r['motif_type']:<7}{r['size']:>4}{r['reps']:>4}{r['delta']:>4}"
            f"{r['phi_measured']:>8.3f} | "
            f"{r['tm1_measured']:>9.4f}{r['tm1_boundary']:>9.4f}{r['tm1_abs_err']:>8.1e}"
            f"{r['tm1_gang_asymptote']:>9.4f}{r['tm1_density_bound']:>8.3f}"
            f"{('ok' if r['tm1_within_bound'] else 'OUT'):>4} | "
            f"{r['tsigma2_measured']:>9.4f}{r['tsigma2_universal_bound']:>10.4f}"
            f"{('ok' if r['tsigma2_within_bound'] else 'OUT'):>4} | "
            f"{r['tgamma_measured']:>7.2f}"
        )
    LOGGER.info("=" * 104)
    LOGGER.info(
        "  m̃1 meas ≡ m̃1 bdy confirms the exact boundary-edge identity (Thm. 4.2);"
    )
    LOGGER.info(
        "  clique m̃1→0, cycle→1/4, star→1/3 is the density/bottleneck ordering (Cor. 4.4)."
    )
    return summary_rows


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BA motif normalized-Laplacian energy experiment"
    )
    p.add_argument("--n_nodes", type=int, default=2000, help="BA graph node count")
    p.add_argument("--ba_m", type=int, default=1, help="BA attachment parameter m")
    p.add_argument("--seed", type=int, default=42, help="Global random seed")
    p.add_argument(
        "--k_max",
        type=int,
        default=0,
        help="Max eigenvectors for the energy plots (0 = full decomposition)",
    )
    p.add_argument(
        "--smooth", type=int, default=20, help="(unused) kept for CLI compatibility"
    )
    p.add_argument(
        "--n_grid",
        type=int,
        default=400,
        help="Resolution of the shared frequency grid over [0,2] on which the "
        "smooth KDE energy density is evaluated (default 400)",
    )
    p.add_argument(
        "--kde_bw",
        type=float,
        default=0.03,
        help="KDE Gaussian bandwidth h (σ in λ-units): larger = smoother, "
        "smaller = sharper low-λ peak (default 0.03)",
    )
    p.add_argument(
        "--n_trials", type=int, default=15, help="Independent BA graphs to average"
    )
    p.add_argument(
        "--planting",
        choices=["fresh", "random", "bfs"],
        default="random",
        help="fresh = low-conductance planting on new nodes (paper model; default)",
    )
    p.add_argument(
        "--bridges",
        type=int,
        default=DEFAULT_BRIDGES,
        help=f"b: bridge edges per planted copy (default {DEFAULT_BRIDGES})",
    )
    p.add_argument(
        "--n_jobs",
        type=int,
        default=min(1, os.cpu_count() or 1),
        help="Parallel worker threads for the per-config eigendecompositions",
    )
    p.add_argument(
        "--moments_only",
        action="store_true",
        default=False,
        help="Skip eigendecomposition/plots; compute moments + CSV only (near-instant)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_lambda_grid(args.n_grid, args.kde_bw)  # fix the KDE grid before threads start
    SAVE_DIR = os.path.join(save_path, "ba_motif_energy_norm")
    os.makedirs(SAVE_DIR, exist_ok=True)

    LOGGER.info("=" * 72)
    LOGGER.info("  BA Motif — NORMALIZED (symmetric) Laplacian energy experiment")
    LOGGER.info(f"  BA graph    : n={args.n_nodes}, m={args.ba_m}, seed={args.seed}")
    LOGGER.info(
        f"  Motif types : {MOTIF_TYPES}   sizes: {MOTIF_SIZES}   reps: {REPETITIONS}"
    )
    LOGGER.info(f"  Planting    : {args.planting}  (bridges b={args.bridges})")
    LOGGER.info(f"  Trials      : {args.n_trials}   Output: {SAVE_DIR}")
    LOGGER.info("=" * 72)

    configs = list(product(MOTIF_SIZES, REPETITIONS, MOTIF_TYPES))
    # Per config: list over trials of (lk, q_gang, q_base) raw spectra; the smooth
    # KDE density is built per trial and averaged at the end.
    trial_spectra: Dict[tuple, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    trial_moments: Dict[tuple, List[Dict]] = {}
    moment_rows: List[Dict] = []

    def _process_config(G_base, trial, trial_seed, idx, size, n_reps, mtype):
        N = G_base.num_nodes
        if size >= N:
            return None
        G_planted, node_sets = plant_patterns_in_graph(
            G_base,
            mtype,
            size,
            n_reps,
            strategy=args.planting,
            bridges=args.bridges,
            seed=trial_seed + idx,
        )
        if len(node_sets) == 0:
            return None
        Np = G_planted.num_nodes
        support = np.concatenate([np.asarray(p.nodes) for p in node_sets])

        meas = measured_moments_norm(G_planted.edge_index, Np, support)
        theo = boundary_theory_norm(G_planted.edge_index, Np, support, mtype, size)

        row = {
            "trial": trial,
            "motif_type": mtype,
            "size": size,
            "reps": n_reps,
            "bridges": args.bridges,
            "delta": theo["delta"],
            "host_min_deg": theo["host_min_deg"],
            "phi_measured": meas["phi"],
            "tm1_measured": meas["tm1"],
            "tm1_boundary": theo["tm1_boundary"],
            "tm1_gang_asymptote": theo["tm1_gang_asymptote"],
            "tm1_density_bound": theo["tm1_density_bound"],
            "tsigma2_measured": meas["tsigma2"],
            "tsigma2_density_bound": theo["tsigma2_density_bound"],
            "tgamma_measured": meas["tgamma"],
        }
        out = {
            "key": (mtype, size, n_reps),
            "row": row,
            "meas": meas,
            "theo": theo,
            "lk": None,
            "q_gang": None,
            "q_base": None,
        }

        if not args.moments_only:
            # Keep the raw per-eigenvalue energy; the smooth KDE density on the
            # shared λ-grid is built at the aggregation step.
            lk, Uk = normalized_spectral_decomp(
                G_planted.edge_index, Np, k_max=args.k_max
            )
            _, dt = normalized_laplacian(G_planted.edge_index, Np)
            out["lk"] = lk
            out["q_gang"] = energy_per_eigenvalue(lk, Uk, node_sets, dt, Np)
            out["q_base"] = baseline_per_eigenvalue(
                lk, Uk, [size] * len(node_sets), dt, Np, n_trials=20, seed=trial_seed
            )
        return out

    n_jobs = max(1, args.n_jobs)
    prev_threads = torch.get_num_threads()
    if n_jobs > 1:
        torch.set_num_threads(1)
    LOGGER.info(
        f"  Parallelism : n_jobs={n_jobs}"
        f"{'  |  moments_only' if args.moments_only else ''}"
    )

    for trial in range(args.n_trials):
        trial_seed = args.seed + trial * 1000
        LOGGER.info(
            f"\n{'─'*72}\n  Trial {trial + 1}/{args.n_trials}  (BA seed={trial_seed})\n{'─'*72}"
        )
        G_base = build_ba_graph(n_nodes=args.n_nodes, m=args.ba_m, seed=trial_seed)
        tasks = list(enumerate(configs, 1))

        def _run(item):
            idx, (size, n_reps, mtype) = item
            return _process_config(G_base, trial, trial_seed, idx, size, n_reps, mtype)

        if n_jobs > 1:
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                outs = list(ex.map(_run, tasks))
        else:
            outs = [_run(item) for item in tasks]

        for out in outs:
            if out is None:
                continue
            key = out["key"]
            moment_rows.append(out["row"])
            trial_moments.setdefault(key, []).append({**out["meas"], **out["theo"]})
            if out["lk"] is not None:
                trial_spectra.setdefault(key, []).append(
                    (out["lk"], out["q_gang"], out["q_base"])
                )
        LOGGER.info(f"  Trial {trial + 1} done.")

    if n_jobs > 1:
        torch.set_num_threads(prev_threads)

    # ── Smooth (KDE) each trial's spectrum, then average the densities ────────
    if trial_spectra:
        LOGGER.info(
            f"\n[grid] KDE on {N_LAMBDA_GRID}-pt grid over [0,2], "
            f"bandwidth h={KDE_BANDWIDTH:g}"
        )
    results: Dict = {}
    for key, spectra in trial_spectra.items():
        e_dens = np.array([kde_density(lk, qg) for lk, qg, _ in spectra])
        b_dens = np.array([kde_density(lk, qb) for lk, _, qb in spectra])
        results[key] = {
            "mean_energy": e_dens.mean(axis=0),
            "std_energy": e_dens.std(axis=0),
            "baseline": b_dens.mean(axis=0),
        }
        mlist = trial_moments.get(key, [])
        if mlist:
            results[key].update(
                {
                    "tm1": float(np.mean([m["tm1"] for m in mlist])),
                    "tsigma2": float(np.mean([m["tsigma2"] for m in mlist])),
                    "tgamma": float(np.nanmean([m["tgamma"] for m in mlist])),
                    "tm1_boundary": float(
                        np.nanmean([m["tm1_boundary"] for m in mlist])
                    ),
                    "phi": float(np.mean([m["phi"] for m in mlist])),
                }
            )

    summary_rows = write_moment_csvs_norm(moment_rows, SAVE_DIR)

    if args.moments_only:
        LOGGER.info(f"\n[moments_only] moment CSVs under {SAVE_DIR}")
        if summary_rows:
            plot_moments_validation(summary_rows, SAVE_DIR)
        return

    LOGGER.info("\n[plots] Generating normalized-energy visualisations …")
    sw = args.smooth
    for mtype in MOTIF_TYPES:
        for n_reps in REPETITIONS:
            plot_energy_by_size_norm(results, mtype, n_reps, SAVE_DIR, smooth_window=sw)
    for size in MOTIF_SIZES:
        for n_reps in REPETITIONS:
            plot_energy_compare_norm(results, size, n_reps, SAVE_DIR, smooth_window=sw)
    for n_reps in REPETITIONS:
        plot_cumulative_grid_norm(results, n_reps, SAVE_DIR)
    if summary_rows:
        plot_moments_validation(summary_rows, SAVE_DIR)

    n_plots = len(list(Path(SAVE_DIR).glob("*.png")))
    LOGGER.info(f"\nDone.  {n_plots} PNG files + 2 CSVs under {SAVE_DIR}")


if __name__ == "__main__":
    main()
