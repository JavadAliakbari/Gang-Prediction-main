"""run_shared_set_motif_demo.py
===============================
A small, fully controlled demonstration of the "Spectral Energy of Planted
Gangs" note on a single graph:

    1. build a random BA graph with N=100 nodes,
    2. choose ONE random set S of s=10 nodes,
    3. plant a clique, a cycle and a star **into that same set S**,
    4. compare the Laplacian eigenvectors of the three plantings visually,
    5. compare their spectral-energy distributions visually,
    6. compute the energy mean / variance / skewness by measurement and theory,
    7. compute k50 and k90 by measurement and theory.

Because all three motifs sit on the *same* vertices with the *same* boundary
(only the internal wiring differs), Propositions 1–2 predict that the mean m1=ϕ
and variance σ² are identical across the three types (boundary-only), and only
the skewness γ is type-stamped (clique ≫ cycle ≳ star).  The variance theory uses
the EXACT formula σ² = Var(d∂) + (1/s)·Σ_h b_h² (Prop. 1, un-simplified), which is
valid for any conductance ϕ — unlike the closed form 2ϕ−ϕ², which is only the
ϕ≪1 limit and goes negative for ϕ>2.

Host graph model
----------------
--graph_model {ba, er, ws, sbm}: Barabási–Albert (default), Erdős–Rényi,
Watts–Strogatz small-world, or stochastic block model.  Structured hosts (ws,
sbm) have a banded/gapped spectrum that makes the gang's energy stand out as
cleaner dominant peaks; er has a smooth bulk and washes peaks out.

Planting regime
---------------
* default (low-conductance, paper regime): the s chosen nodes are stripped of
  their previous edges, wired as the motif, and attached to the host by a small
  fixed number b of bridge edges (shared across the three motifs).  Then ϕ=b/s
  is small and the closed-form theory applies and matches.
* --keep_external: keep the chosen nodes' original external edges (only the
  internal edges among S are replaced).  The boundary is still shared across the
  three motifs (so m1, σ² stay type-independent) but ϕ is large (high
  conductance), so the *closed-form* σ²/γ fall out of regime (flagged).

Outputs (results/<timestamp>/shared_set_motif_demo/)
----------------------------------------------------
* graph_with_pattern.png     – the graph drawn with the planted motif highlighted
* eigenvectors_compare.png   – bottom-K eigenvectors, gang nodes highlighted
* energy_distribution.png    – energy vs eigenvector + cumulative, k50/k90 marked
* moments_k_table.csv        – measured vs theory: m1, σ², γ, k50, k90

Usage (FedStruct conda env, from repo root)::

    python src/run_shared_set_motif_demo.py
    python src/run_shared_set_motif_demo.py --graph_model sbm     # cleanest peaks
    python src/run_shared_set_motif_demo.py --graph_model er --er_p 0.04
    python src/run_shared_set_motif_demo.py --n_nodes 100 --size 10 --bridges 2
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.utils.utils import save_path, LOGGER
from src.run_ba_motif_energy_experiment import (
    theoretical_moments,
    cumulative_energy,
    k_threshold,
    DELTA_BR,
)

MOTIF_TYPES = ["clique", "cycle", "star"]
MOTIF_COLORS = {"clique": "#e41a1c", "cycle": "#377eb8", "star": "#4daf4a"}


# ═══════════════════════════════════════════════════════════════════════════════
# Motif internal edges  and  closed-form motif Laplacian spectra
# ═══════════════════════════════════════════════════════════════════════════════
def internal_edges(nodes: List[int], motif_type: str) -> List[Tuple[int, int]]:
    n = list(map(int, nodes))
    if motif_type == "clique":
        return [(n[i], n[j]) for i in range(len(n)) for j in range(i + 1, len(n))]
    if motif_type == "cycle":
        k = len(n)
        return [(n[i], n[(i + 1) % k]) for i in range(k)]
    if motif_type == "star":
        return [(n[0], n[i]) for i in range(1, len(n))]  # hub = nodes[0]
    raise ValueError(motif_type)


def motif_laplacian_spectrum(motif_type: str, s: int) -> np.ndarray:
    """Analytic Laplacian eigenvalues of a single size-s motif (note eqs. 17–19)."""
    if motif_type == "clique":  # K_s : {0, s (mult s-1)}
        return np.array([0.0] + [float(s)] * (s - 1))
    if motif_type == "cycle":  # C_s : 2 - 2cos(2πj/s)
        return np.array([2 - 2 * np.cos(2 * np.pi * j / s) for j in range(s)])
    if motif_type == "star":  # S_s : {0, 1 (mult s-2), s}
        return np.array([0.0] + [1.0] * (s - 2) + [float(s)])
    raise ValueError(motif_type)


def count_below(eigs: np.ndarray, lam: float) -> int:
    """n(λ) = #{eigenvalues ≤ λ}  (integrated density of states)."""
    return int(np.count_nonzero(eigs <= lam + 1e-9))


# ═══════════════════════════════════════════════════════════════════════════════
# Host graph models
# ═══════════════════════════════════════════════════════════════════════════════
def build_host(model: str, n: int, seed: int, args) -> nx.Graph:
    """Return a random host graph from the chosen model.

    ba   : Barabási–Albert scale-free (hubs → some localized modes).
    er   : Erdős–Rényi G(n,p) (smooth semicircle-like bulk → energy spreads).
    ws   : Watts–Strogatz small-world (banded, near-discrete spectrum → peaky).
    sbm  : stochastic block model (planted communities → spectral gap → the
           cleanest dominant low-frequency peaks; the gang IS a community).
    """
    if model == "ba":
        return nx.barabasi_albert_graph(n, args.ba_m, seed=seed)
    if model == "er":
        p = args.er_p if args.er_p > 0 else min(0.5, 2.0 * args.ba_m / (n - 1))
        return nx.gnp_random_graph(n, p, seed=seed)
    if model == "ws":
        return nx.watts_strogatz_graph(n, args.ws_k, args.ws_p, seed=seed)
    if model == "sbm":
        sizes = [n // args.sbm_blocks] * args.sbm_blocks
        sizes[-1] += n - sum(sizes)
        P = [
            [args.sbm_pin if i == j else args.sbm_pout for j in range(args.sbm_blocks)]
            for i in range(args.sbm_blocks)
        ]
        return nx.stochastic_block_model(sizes, P, seed=seed)
    raise ValueError(f"unknown graph model: {model}")


# ═══════════════════════════════════════════════════════════════════════════════
# Graph construction: plant one motif on the shared set S
# ═══════════════════════════════════════════════════════════════════════════════
def build_planted_graph(
    G_host: nx.Graph,
    S: List[int],
    motif_type: str,
    bridges: List[Tuple[int, int]],
    keep_external: bool,
) -> nx.Graph:
    """Return the host with *motif_type* planted on the shared node set S.

    keep_external=False : strip S's prior edges, wire the motif, add the shared
                          *bridges* (low conductance).
    keep_external=True  : keep S→host edges, replace only the internal edges
                          among S by the motif (high conductance).
    """
    N = G_host.number_of_nodes()
    Sset = set(S)
    G = nx.Graph()
    G.add_nodes_from(range(N))

    for u, v in G_host.edges():
        in_u, in_v = u in Sset, v in Sset
        if in_u and in_v:
            continue  # drop ALL original internal edges among S (replaced by motif)
        if (in_u or in_v) and not keep_external:
            continue  # drop S↔host edges in the low-conductance regime
        G.add_edge(u, v)

    G.add_edges_from(internal_edges(S, motif_type))  # internal motif edges
    if not keep_external:
        G.add_edges_from(bridges)  # shared, fixed boundary
    return G


def build_planted_laplacian(
    G_host: nx.Graph,
    S: List[int],
    motif_type: str,
    bridges: List[Tuple[int, int]],
    keep_external: bool,
) -> Tuple[np.ndarray, List[int]]:
    """Dense combinatorial Laplacian L=D−A of the planted graph (see build_planted_graph)."""
    G = build_planted_graph(G_host, S, motif_type, bridges, keep_external)
    A = nx.to_numpy_array(G, nodelist=range(G_host.number_of_nodes()))
    L = np.diag(A.sum(1)) - A
    return L, S


# ═══════════════════════════════════════════════════════════════════════════════
# Measured energy moments (exact, from the Laplacian)
# ═══════════════════════════════════════════════════════════════════════════════
def measured_moments_dense(L: np.ndarray, support: List[int]) -> Dict[str, float]:
    N = L.shape[0]
    s = len(support)
    v = np.zeros(N)
    v[support] = 1.0 / np.sqrt(s)
    Lv = L @ v
    L2v = L @ Lv
    m1 = float(v @ Lv)  # = cut(S)/s = ϕ
    m2 = float(Lv @ Lv)
    m3 = float(Lv @ L2v)
    sigma2 = m2 - m1 * m1
    mu3 = m3 - 3 * m1 * m2 + 2 * m1**3
    gamma = mu3 / sigma2**1.5 if sigma2 > 1e-12 else float("nan")
    return {"phi": m1, "m1": m1, "sigma2": sigma2, "gamma": gamma}


def exact_theory_moments(L: np.ndarray, support: List[int]) -> Dict[str, float]:
    """Exact moments from the boundary decomposition (note Prop. 1 / eq. 9) — the
    *un-simplified* formulas, valid for ANY conductance ϕ (unlike σ²=2ϕ−ϕ², which
    is only the ϕ≪1 limit and goes negative for ϕ>2).

        w = L·1_S          ⇒  w_i = d∂(i) on S,   w_h = −b_h on the host
        m1 = ϕ = cut/s
        σ² = Var(d∂) + (1/s)·Σ_h b_h²        (boundary only ⇒ type-independent)
        γ  = (m3 − 3 m1 m2 + 2 m1³)/σ³,  m3 = (1/s) wᵀ L w   (eq. 9; type enters here)

    Because m1 and σ² use only w on the boundary (internal motif edges cancel,
    Lemma 1), they are identical across clique/cycle/star; only m3 (hence γ) sees
    the internal wiring.  These exact values coincide with the measurement.
    """
    N = L.shape[0]
    s = len(support)
    mask = np.zeros(N, dtype=bool)
    mask[support] = True
    one_S = mask.astype(float)
    w = L @ one_S  # d∂ on S, −b_h on host
    cut = float(w[mask].sum())  # Σ d∂(i)
    m1 = cut / s  # = ϕ
    var_dbar = float((w[mask] ** 2).sum()) / s - m1 * m1  # Var(d∂)
    host_term = float((w[~mask] ** 2).sum()) / s  # (1/s) Σ_h b_h²
    sigma2 = var_dbar + host_term  # exact σ² (Prop. 1, any ϕ)
    m2 = sigma2 + m1 * m1
    m3 = float(w @ (L @ w)) / s  # (1/s) wᵀ L w  (eq. 9)
    mu3 = m3 - 3 * m1 * m2 + 2 * m1**3
    gamma = mu3 / sigma2**1.5 if sigma2 > 1e-12 else float("nan")
    return {
        "m1": m1,
        "sigma2": sigma2,
        "gamma": gamma,
        "var_dbar": var_dbar,
        "host_term": host_term,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════
def plot_eigenvectors(
    eigvecs: Dict[str, np.ndarray], S: List[int], k_show: int, save_dir: str
) -> None:
    """Heatmap of the bottom-k eigenvectors per type, gang nodes pulled to the top."""
    N = next(iter(eigvecs.values())).shape[0]
    perm = list(S) + [i for i in range(N) if i not in set(S)]  # gang rows first
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    for ax, mtype in zip(axes, MOTIF_TYPES):
        U = eigvecs[mtype][perm, :k_show]
        vmax = np.abs(U).max()
        im = ax.imshow(U, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.axhline(len(S) - 0.5, color="k", lw=1.5)  # gang | host divider
        ax.set_title(f"{mtype}  (bottom {k_show} eigenvectors)", fontsize=11)
        ax.set_xlabel("eigenvector index k (low → high freq)")
        ax.text(
            0.5,
            len(S) / 2,
            "gang S",
            color="k",
            fontsize=9,
            rotation=90,
            va="center",
            ha="center",
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("node (gang nodes on top, then host)")
    fig.suptitle(
        "Laplacian eigenvectors on the SAME node set — only internal wiring differs",
        fontsize=13,
    )
    plt.tight_layout()
    p = os.path.join(save_dir, "eigenvectors_compare.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"  Saved → {p}")


def plot_energy(
    energies: Dict[str, np.ndarray],
    eigvals: Dict[str, np.ndarray],
    kvals: Dict[str, Dict[str, int]],
    save_dir: str,
) -> None:
    fig, (ax, axc) = plt.subplots(1, 2, figsize=(14, 5))
    for mtype in MOTIF_TYPES:
        e = energies[mtype]
        cum = cumulative_energy(e)
        k50, k90 = kvals[mtype]["k50_meas"], kvals[mtype]["k90_meas"]
        ax.plot(
            e,
            color=MOTIF_COLORS[mtype],
            lw=1.6,
            marker=".",
            ms=4,
            label=f"{mtype} (k50={k50}, k90={k90})",
        )
        axc.plot(cum, color=MOTIF_COLORS[mtype], lw=1.8, label=mtype)
        axc.axvline(k50, color=MOTIF_COLORS[mtype], ls=":", lw=1, alpha=0.7)
        axc.axvline(k90, color=MOTIF_COLORS[mtype], ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("eigenvector index k (sorted by eigenvalue)")
    ax.set_ylabel("energy  (u_kᵀ v)²")
    ax.set_title("Spectral energy distribution")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    axc.axhline(0.5, color="orange", ls=":", lw=1, label="50%")
    axc.axhline(0.9, color="red", ls=":", lw=1, label="90%")
    axc.set_xlabel("eigenvector index k")
    axc.set_ylabel("cumulative energy")
    axc.set_title("Cumulative energy (k50 dotted, k90 dashed)")
    axc.legend(fontsize=9)
    axc.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(save_dir, "energy_distribution.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"  Saved → {p}")


def plot_graph_with_patterns(
    G_host: nx.Graph,
    S: List[int],
    bridges: List[Tuple[int, int]],
    keep_external: bool,
    save_dir: str,
    seed: int,
) -> None:
    """Draw the actual graph with the planted motif highlighted INSIDE it, one
    panel per type.  Host = light grey; gang nodes = colored; internal motif
    edges = thick colored; boundary (gang↔host) edges = dashed orange."""
    Sset = set(S)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, mtype in zip(axes, MOTIF_TYPES):
        G = build_planted_graph(G_host, S, mtype, bridges, keep_external)
        internal = [(u, v) for u, v in G.edges() if u in Sset and v in Sset]
        boundary = [(u, v) for u, v in G.edges() if (u in Sset) ^ (v in Sset)]
        host_e = [(u, v) for u, v in G.edges() if u not in Sset and v not in Sset]
        # layout: gang clustered (computed on the clique planting for a stable, tight blob)
        pos = nx.spring_layout(G, seed=seed, k=1.2 / np.sqrt(G.number_of_nodes()))
        host_nodes = [n for n in G.nodes() if n not in Sset]
        nx.draw_networkx_edges(
            G, pos, edgelist=host_e, edge_color="0.85", width=0.4, ax=ax
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=boundary,
            edge_color="orange",
            width=0.7,
            style="dashed",
            alpha=0.7,
            ax=ax,
        )
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=internal,
            edge_color=MOTIF_COLORS[mtype],
            width=1.3,
            alpha=0.8,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            G, pos, nodelist=host_nodes, node_size=10, node_color="0.7", ax=ax
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=S,
            node_size=55,
            node_color=MOTIF_COLORS[mtype],
            edgecolors="k",
            linewidths=0.4,
            ax=ax,
        )
        ax.set_title(
            f"{mtype}  — planted on the same {len(S)} nodes "
            f"({len(internal)} internal edges)",
            fontsize=11,
        )
        ax.axis("off")
    fig.suptitle(
        "The graph with the planted pattern highlighted (gang colored, "
        "boundary dashed)",
        fontsize=14,
    )
    plt.tight_layout()
    p = os.path.join(save_dir, "graph_with_pattern.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    LOGGER.info(f"  Saved → {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shared-set clique/cycle/star demo")
    p.add_argument("--n_nodes", type=int, default=250)
    p.add_argument("--size", type=int, default=50, help="gang size s")
    p.add_argument("--bridges", type=int, default=7, help="b bridges/gang (low-cond.)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--k_show", type=int, default=16, help="# eigenvectors in the heatmap"
    )
    p.add_argument(
        "--keep_external",
        action="store_true",
        default=True,
        help="keep S's external edges (high-conductance planting)",
    )
    # ── host graph model ────────────────────────────────────────────────────
    p.add_argument(
        "--graph_model",
        choices=["ba", "er", "ws", "sbm"],
        default="ba",
        help="random host model: ba=Barabási–Albert, er=Erdős–Rényi, "
        "ws=Watts–Strogatz, sbm=stochastic block model (best for dominant peaks)",
    )
    p.add_argument("--ba_m", type=int, default=2, help="BA: edges per new node")
    p.add_argument(
        "--er_p",
        type=float,
        default=0.0,
        help="ER: edge prob (0 = auto, matched to BA average degree)",
    )
    p.add_argument("--ws_k", type=int, default=5, help="WS: ring neighbours per node")
    p.add_argument("--ws_p", type=float, default=0.1, help="WS: rewiring probability")
    p.add_argument(
        "--sbm_blocks", type=int, default=4, help="SBM: number of communities"
    )
    p.add_argument("--sbm_pin", type=float, default=0.30, help="SBM: within-block prob")
    p.add_argument(
        "--sbm_pout", type=float, default=0.01, help="SBM: between-block prob"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    SAVE_DIR = os.path.join(save_path, "shared_set_motif_demo")
    os.makedirs(SAVE_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    s = args.size

    LOGGER.info("=" * 84)
    LOGGER.info("  Shared-set motif demo  (clique / cycle / star on the SAME nodes)")
    LOGGER.info(
        f"  host={args.graph_model.upper()}  n={args.n_nodes}  seed={args.seed}   "
        f"gang size s={s}"
    )
    LOGGER.info(
        f"  planting: {'KEEP external (high ϕ)' if args.keep_external else f'fresh bridges b={args.bridges} (low ϕ)'}"
    )
    LOGGER.info(f"  output → {SAVE_DIR}")
    LOGGER.info("=" * 84)

    # 1–2. host graph and ONE shared random node set S
    G_host = build_host(args.graph_model, args.n_nodes, args.seed, args)
    LOGGER.info(
        f"  host: {G_host.number_of_nodes()} nodes, {G_host.number_of_edges()} edges "
        f"(avg degree ≈ {2 * G_host.number_of_edges() / G_host.number_of_nodes():.1f})"
    )
    S = sorted(rng.choice(args.n_nodes, size=s, replace=False).tolist())
    LOGGER.info(f"  chosen gang nodes S = {S}")

    # Shared bridges (used by all three motifs in the low-conductance regime):
    # b distinct NON-hub gang nodes → b distinct host nodes.
    host_pool = [i for i in range(args.n_nodes) if i not in set(S)]
    b = min(args.bridges, s - 1)
    g_pos = rng.choice(np.arange(1, s), size=b, replace=False)  # avoid hub (index 0)
    h_end = rng.choice(host_pool, size=b, replace=False)
    bridges = [(int(S[g_pos[k]]), int(h_end[k])) for k in range(b)]

    # Host spectrum (induced subgraph on non-gang nodes) for the k_α counting theory
    L_host = nx.laplacian_matrix(G_host.subgraph(host_pool)).toarray().astype(float)
    host_eigs = np.linalg.eigvalsh(L_host)

    energies: Dict[str, np.ndarray] = {}
    eigvecs: Dict[str, np.ndarray] = {}
    eigvals: Dict[str, np.ndarray] = {}
    kvals: Dict[str, Dict[str, int]] = {}
    rows: List[Dict] = []

    for mtype in MOTIF_TYPES:
        L, support = build_planted_laplacian(
            G_host, S, mtype, bridges, args.keep_external
        )
        w, U = np.linalg.eigh(L)  # ascending eigenvalues
        v = np.zeros(L.shape[0])
        v[support] = 1.0 / np.sqrt(s)
        p = (U.T @ v) ** 2  # energy per eigenvector
        energies[mtype], eigvecs[mtype], eigvals[mtype] = p, U, w

        # 6. moments: measurement vs theory
        meas = measured_moments_dense(L, support)
        theo = exact_theory_moments(L, support)  # exact, valid for any ϕ
        lowphi = theoretical_moments(meas["phi"], mtype, s)  # ϕ≪1 closed form (ref)

        # 7. k50 / k90: measurement vs theory (counting decomposition, Prop. 3)
        cum = cumulative_energy(p)
        k50_m, k90_m = k_threshold(cum, 0.5), k_threshold(cum, 0.9)
        # λ at which the measured energy reaches 50 % / 90 %
        lam50 = w[min(k50_m - 1, len(w) - 1)]
        lam90 = w[min(k90_m - 1, len(w) - 1)]
        motif_eigs = motif_laplacian_spectrum(mtype, s)
        # k_α = N_host(λ_α) + n_motif(λ_α)
        k50_t = count_below(host_eigs, lam50) + count_below(motif_eigs, lam50)
        k90_t = count_below(host_eigs, lam90) + count_below(motif_eigs, lam90)
        kvals[mtype] = {
            "k50_meas": k50_m,
            "k90_meas": k90_m,
            "k50_theory": k50_t,
            "k90_theory": k90_t,
        }

        rows.append(
            {
                "motif_type": mtype,
                "size": s,
                "bridges": b,
                "delta_br": DELTA_BR[mtype](s),
                "phi": meas["phi"],
                "m1_meas": meas["m1"],
                "m1_theory": theo["m1"],
                # exact (Prop. 1 general) variance — valid for ANY ϕ
                "sigma2_meas": meas["sigma2"],
                "sigma2_theory_exact": theo["sigma2"],
                "var_dbar": theo["var_dbar"],  # Var(d∂)  (boundary-degree spread)
                "host_term": theo["host_term"],  # (1/s) Σ_h b_h²
                "sigma2_theory_lowphi": lowphi["sigma2"],  # 2ϕ−ϕ² (ϕ≪1 only)
                "gamma_meas": meas["gamma"],
                "gamma_theory_exact": theo["gamma"],
                "k50_meas": k50_m,
                "k50_theory": k50_t,
                "k90_meas": k90_m,
                "k90_theory": k90_t,
            }
        )

    # 4–5. visual comparisons
    plot_eigenvectors(eigvecs, S, args.k_show, SAVE_DIR)
    plot_energy(energies, eigvals, kvals, SAVE_DIR)
    plot_graph_with_patterns(
        G_host, S, bridges, args.keep_external, SAVE_DIR, args.seed
    )

    # CSV
    csv_path = os.path.join(SAVE_DIR, "moments_k_table.csv")
    with open(csv_path, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w_.writeheader()
        for r in rows:
            w_.writerow(
                {k: (f"{x:.5f}" if isinstance(x, float) else x) for k, x in r.items()}
            )
    LOGGER.info(f"  Saved → {csv_path}")

    # Log comparison table
    def f2(x):
        return "   —   " if (isinstance(x, float) and np.isnan(x)) else f"{x:7.4f}"

    LOGGER.info("\n" + "=" * 100)
    LOGGER.info(
        "  MEASUREMENT vs THEORY  (clique/cycle/star on the SAME node set; "
        "σ²/γ theory = exact Prop.1/eq.9, valid for any ϕ)"
    )
    LOGGER.info("-" * 100)
    LOGGER.info(
        f"{'type':<7}{'ϕ':>7} | {'m1 meas':>8}{'m1 th':>8} | "
        f"{'σ² meas':>9}{'σ² th':>9}{'(σ²=Var(d∂)+host)':>18} | "
        f"{'γ meas':>8}{'γ th':>8} | {'k50 m/th':>9}{'k90 m/th':>10}"
    )
    LOGGER.info("-" * 100)
    for r in rows:
        LOGGER.info(
            f"{r['motif_type']:<7}{r['phi']:>7.3f} | "
            f"{f2(r['m1_meas'])}{f2(r['m1_theory'])} | "
            f"{f2(r['sigma2_meas'])}{f2(r['sigma2_theory_exact'])}"
            f"   ({r['var_dbar']:.3f}+{r['host_term']:.3f}) | "
            f"{f2(r['gamma_meas'])}{f2(r['gamma_theory_exact'])} | "
            f"{r['k50_meas']:>4}/{r['k50_theory']:<4}{r['k90_meas']:>5}/{r['k90_theory']:<4}"
        )
    LOGGER.info("=" * 100)
    LOGGER.info(
        "  m1=ϕ and σ² are IDENTICAL across the three types (boundary-only, any ϕ); "
        "only γ is type-stamped (clique≫cycle≳star)."
    )
    LOGGER.info(
        f"  (low-ϕ closed form σ²=2ϕ−ϕ² would give "
        f"{rows[0]['sigma2_theory_lowphi']:.3f} at ϕ={rows[0]['phi']:.2f} — "
        "out of regime here; the exact formula above is used instead.)"
    )
    LOGGER.info(f"\nDone. Figures + CSV under {SAVE_DIR}")


if __name__ == "__main__":
    main()
