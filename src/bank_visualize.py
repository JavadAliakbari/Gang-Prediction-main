"""Rich visualisations for the collective-bank gang-detection experiment.

``save_rich_plots`` (called from ``run_for_tau``) writes five files per run:

* ``bank_graphs{suffix}.png``          -- original graph (gang-cluster layout) +
                                           coarsened graph (supernode view)
* ``bank_embedding{suffix}.png``       -- PCA of final filter-bank embedding Z,
                                           all N nodes + per-gang centroids
* ``bank_training{suffix}.png``        -- training curves: λ_min(Γ), mean
                                           retained energy, neg repulsion
* ``bank_separation_frames{suffix}.png`` -- static key-frames of gang-centroid
                                           separation in embedding space
* ``bank_separation{suffix}.gif``      -- animated version of the above
* ``bank_metrics{suffix}.png``         -- filter-hop weights, RSA sigma per
                                           level, per-gang retained energy, Γ
                                           matrix heatmap
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.collections as mc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402

try:
    from sklearn.decomposition import PCA as _PCA

    def _pca2d(X_np: np.ndarray, fit_on: np.ndarray | None = None) -> np.ndarray:
        pca = _PCA(n_components=2, random_state=0)
        pca.fit(fit_on if fit_on is not None else X_np)
        return pca.transform(X_np)

except ImportError:  # fall back to manual SVD

    def _pca2d(X_np: np.ndarray, fit_on: np.ndarray | None = None) -> np.ndarray:
        ref = fit_on if fit_on is not None else X_np
        mu = ref.mean(0)
        _, _, Vt = np.linalg.svd(ref - mu, full_matrices=False)
        return (X_np - mu) @ Vt[:2].T


try:
    from src.sgc_detection import propagation_stack as _prop_stack
except ImportError:
    _prop_stack = None

LOGGER = logging.getLogger(__name__)

# ── palette ─────────────────────────────────────────────────────────────────
_PALETTE = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)  # 40 colours


def _gang_colours(n: int) -> list:
    return [_PALETTE[i % len(_PALETTE)] for i in range(n)]


# ── local inline helpers (avoid circular import with run_collective_bank) ───


def _fb(propagated: list, theta: torch.Tensor) -> torch.Tensor:
    """Inline ``_filtered_bank``: Σ_k propagated[k] * theta[k]."""
    return sum(propagated[k] * theta[k].unsqueeze(0) for k in range(theta.shape[0]))


def _l_apply(a_hat: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    return S - torch.sparse.mm(a_hat, S)


def _dw_indicators(adjacency: torch.Tensor, patterns: list) -> torch.Tensor:
    """Degree-weighted gang indicators matching ``fit_collective_bank``.

    ``v_S = D_tilde^{1/2} 1_S / sqrt(vol_tilde(S))``,  ``D_tilde = D + I``.
    """
    n = adjacency.shape[0]
    dtype, device = adjacency.dtype, adjacency.device
    d = torch.sparse.sum(adjacency, dim=1).to_dense() + 1.0  # D_tilde
    V = torch.zeros(n, len(patterns), dtype=dtype, device=device)
    for j, p in enumerate(patterns):
        nodes = torch.as_tensor(_node_index_list(p), dtype=torch.long, device=device)
        V[nodes, j] = d[nodes].sqrt()
        vol = d[nodes].sum().clamp_min(torch.finfo(dtype).eps).sqrt()
        V[:, j] = V[:, j] / vol
    return V


def _mtau_m_vhat(
    adjacency: torch.Tensor,
    normalized: torch.Tensor,
    patterns: list,
    tau: float,
) -> torch.Tensor:
    """Precomputed ``M_tau V_hat``: ``(L + tau I) v_hat_j`` for all patterns.

    ``v_hat_j = v_j / ||v_j||_{M_tau}`` where ``v_j`` is the degree-weighted
    indicator.  This is snapshot-independent (depends only on the graph and
    the gang node sets, not on ``theta``).
    """
    V = _dw_indicators(adjacency, patterns)  # (N, m)
    l_v = _l_apply(normalized, V)  # L v
    phi = (V * l_v).sum(0)  # ||v_j||^2_L
    sq = (V * V).sum(0)  # ||v_j||^2_2
    denom = (phi + tau * sq).clamp_min(torch.finfo(V.dtype).eps)
    return (l_v + tau * V) / denom.sqrt().unsqueeze(0)  # (N, m)


def _whitened_coords(
    propagated: list,
    theta: torch.Tensor,
    m_vhat: torch.Tensor,
    normalized: torch.Tensor,
    tau: float,
    ridge: float,
) -> np.ndarray:
    """M_tau-whitened projected-indicator coordinates ``w_j`` in R^d.

    The M_tau-projection of gang indicator j onto span(Z) is
    ``pi_j = Z c_j``,  ``c_j = (Z^T M Z)^{-1} Z^T M v_hat_j``.
    Defining ``w_j = L_G^T c_j`` where ``G = Z^T M Z = L_G L_G^T``
    (Cholesky) turns the M_tau inner product into Euclidean:
    ``<w_i, w_j>_2 = Gamma_ij`` -- so Euclidean distance in w-space
    equals the M_tau-distance between the two projected indicators.

    Returns (m, d) array.
    """
    with torch.no_grad():
        Z = _fb(propagated, theta)  # (N, d)
        m_z = _l_apply(normalized, Z) + tau * Z  # M_tau Z  (N, d)
        G = Z.T @ m_z  # Z^T M Z  (d, d)
        G = 0.5 * (G + G.T)
        d = G.shape[0]
        G_reg = G + ridge * torch.eye(d, dtype=G.dtype, device=G.device)
        rhs = Z.T @ m_vhat  # Z^T M V_hat  (d, m)
        c = torch.linalg.solve(G_reg, rhs)  # (d, m)
        try:
            L = torch.linalg.cholesky(G_reg)  # G_reg = L L^T
            w = L.T @ c  # (d, m)
        except Exception:
            w = c
    return w.detach().numpy().T  # (m, d)


# ── helpers ──────────────────────────────────────────────────────────────────


def _node_index_list(p) -> list[int]:
    """Return pattern node indices as a plain Python list."""
    ni = p.node_indices
    if isinstance(ni, torch.Tensor):
        return ni.tolist()
    return list(ni)


def _gang_layout(patterns: list, n_nodes: int) -> np.ndarray:
    """Custom O(N) layout: gang nodes in labelled ring-clusters, bg random."""
    pos = np.zeros((n_nodes, 2), dtype=np.float32)
    n_gangs = len(patterns)
    ring_r = 5.0
    spoke_r = 0.38

    gang_set: set[int] = set()
    for j, p in enumerate(patterns):
        nodes = _node_index_list(p)
        gang_set.update(nodes)
        cx = ring_r * np.cos(2 * np.pi * j / n_gangs)
        cy = ring_r * np.sin(2 * np.pi * j / n_gangs)
        for k, node in enumerate(nodes):
            a = 2 * np.pi * k / max(len(nodes), 1)
            pos[node, 0] = cx + spoke_r * np.cos(a)
            pos[node, 1] = cy + spoke_r * np.sin(a)

    bg = [i for i in range(n_nodes) if i not in gang_set]
    rng = np.random.default_rng(42)
    r = rng.uniform(0.0, ring_r - 1.1, len(bg))
    a = rng.uniform(0, 2 * np.pi, len(bg))
    for i, node in enumerate(bg):
        pos[node] = [r[i] * np.cos(a[i]), r[i] * np.sin(a[i])]
    return pos


def _draw_graph_ax(
    ax,
    pos: np.ndarray,
    edge_index_np: np.ndarray,  # (2, E) int64
    node_rgba: np.ndarray,  # (N, 4) float
    node_size: np.ndarray,  # (N,)  float
    title: str,
    max_edges: int = 5000,
) -> None:
    """Draw nodes + sampled edges on *ax* using LineCollection for speed."""
    E = edge_index_np.shape[1]
    if E > max_edges:
        idx = np.random.default_rng(0).choice(E, max_edges, replace=False)
        edge_index_np = edge_index_np[:, idx]

    segs = list(zip(pos[edge_index_np[0]], pos[edge_index_np[1]]))
    lc = mc.LineCollection(segs, linewidths=0.25, colors="gray", alpha=0.12, zorder=1)
    ax.add_collection(lc)

    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=node_rgba,
        s=node_size,
        zorder=2,
        linewidths=0,
    )
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)


# ── Figure 1 : original + coarsened graph ────────────────────────────────────


def _plot_graphs(
    out_path: Path,
    graph,
    coarsening,
    patterns: list,
    train_patterns: list,
    test_patterns: list,
    args,
    tau: float,
) -> None:
    """Panel 1a original (gang-cluster layout)  +  panel 1b coarsened graph."""
    n_nodes = int(graph.num_nodes)
    n_gangs = len(patterns)
    colours = _gang_colours(n_gangs)
    pat_idx = {id(p): j for j, p in enumerate(patterns)}
    train_ids = {id(p) for p in train_patterns}
    test_ids = {id(p) for p in test_patterns}
    node_to_gang: dict[int, int] = {}
    for p in patterns:
        j = pat_idx[id(p)]
        for node in _node_index_list(p):
            node_to_gang[node] = j

    # ── original graph node colours / sizes ──────────────────────────────
    rgba = np.full((n_nodes, 4), [0.78, 0.78, 0.78, 0.30])
    sz = np.full(n_nodes, 3.5)
    for p in patterns:
        j = pat_idx[id(p)]
        c = colours[j]
        alpha = 0.90 if id(p) in train_ids else 0.55
        for node in _node_index_list(p):
            rgba[node] = (*c[:3], alpha)
            sz[node] = 20.0

    pos = _gang_layout(patterns, n_nodes)
    ei_np = graph.edge_index.numpy()  # (2, E)

    # ── coarsened graph ───────────────────────────────────────────────────
    n2s = coarsening.node_to_supernode.numpy()  # (N,)
    n_super = int(n2s.max()) + 1

    super_pos = np.zeros((n_super, 2), dtype=np.float32)
    super_count = np.zeros(n_super, dtype=np.int32)
    for i in range(n_nodes):
        super_pos[n2s[i]] += pos[i]
        super_count[n2s[i]] += 1
    super_pos /= np.clip(super_count[:, None], 1, None)

    # colour each supernode by its dominant gang
    super_votes: dict[int, list[int]] = {}
    for node, gang in node_to_gang.items():
        s = int(n2s[node])
        super_votes.setdefault(s, []).append(gang)

    super_rgba = np.full((n_super, 4), [0.78, 0.78, 0.78, 0.35])
    for s, votes in super_votes.items():
        dom = max(set(votes), key=votes.count)
        purity = votes.count(dom) / len(votes)
        c = colours[dom]
        super_rgba[s] = (*c[:3], 0.35 + 0.55 * purity)

    super_sz = np.clip(super_count * 1.8, 4, 200).astype(float)

    # coarsened edge_index (unique undirected pairs, no self-loops)
    s_src = n2s[ei_np[0]]
    s_dst = n2s[ei_np[1]]
    mask = s_src != s_dst
    raw_edges = np.unique(
        np.sort(np.stack([s_src[mask], s_dst[mask]], axis=1), axis=1), axis=0
    ).T  # (2, E')

    # ── figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    _draw_graph_ax(
        axes[0],
        pos,
        ei_np,
        rgba,
        sz,
        title=(
            f"Original graph  N={n_nodes}  |  "
            f"{n_gangs} gangs  ({len(train_patterns)} train / {len(test_patterns)} test)\n"
            rf"{args.motif_type}, size={args.motif_size}, "
            rf"$\phi$={args.motif_conductance:.2f},  $\tau$={tau:g}"
        ),
        max_edges=6000,
    )

    _draw_graph_ax(
        axes[1],
        super_pos,
        raw_edges,
        super_rgba,
        super_sz,
        title=(
            f"Coarsened  n={n_super}  "
            f"({coarsening.reduction:.0%} reduction,  "
            rf"$\varepsilon$={coarsening.epsilon:.3g},  "
            f"{len(coarsening.sigmas)} levels)\n"
            "node size ∝ supernode membership count"
        ),
        max_edges=6000,
    )

    # shared legend
    from matplotlib.patches import Patch

    axes[0].legend(
        handles=[
            Patch(fc="tab:blue", alpha=0.9, label="train gang"),
            Patch(fc="tab:blue", alpha=0.55, label="test gang"),
            Patch(fc="gray", alpha=0.3, label="background"),
        ],
        loc="upper right",
        fontsize=7,
    )
    axes[1].legend(
        handles=[
            Patch(fc="tab:blue", alpha=0.85, label="gang-dominant supernode"),
            Patch(fc="gray", alpha=0.35, label="background supernode"),
        ],
        loc="upper right",
        fontsize=7,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"  [viz] {out_path.name}")


# ── Figure 2 : embedding space ────────────────────────────────────────────────


def _plot_embedding(
    out_path: Path,
    Z_np: np.ndarray,
    graph,
    patterns: list,
    train_patterns: list,
    args,
    tau: float,
) -> None:
    """Panel 2a all-node PCA scatter  +  panel 2b gang-centroid PCA."""
    n_nodes = int(graph.num_nodes)
    n_gangs = len(patterns)
    colours = _gang_colours(n_gangs)
    train_ids = {id(p) for p in train_patterns}
    gang_node_set: set[int] = set()
    for p in patterns:
        gang_node_set.update(_node_index_list(p))

    emb2d = _pca2d(Z_np)  # (N, 2)

    # per-gang centroids
    gang_idx_arrays = [np.array(_node_index_list(p)) for p in patterns]
    centroids = np.stack([Z_np[idx].mean(0) for idx in gang_idx_arrays])  # (m, d)
    cent2d = _pca2d(centroids, fit_on=Z_np) if Z_np.shape[1] > 2 else centroids[:, :2]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── panel A: all nodes ────────────────────────────────────────────────
    ax = axes[0]
    bg_mask = np.array([i not in gang_node_set for i in range(n_nodes)])
    ax.scatter(
        emb2d[bg_mask, 0],
        emb2d[bg_mask, 1],
        c="lightgray",
        s=2.5,
        alpha=0.25,
        label="background",
        zorder=1,
    )
    for p in patterns:
        j = list(patterns).index(p)
        nodes = gang_idx_arrays[j]
        marker = "o" if id(p) in train_ids else "^"
        ax.scatter(
            emb2d[nodes, 0],
            emb2d[nodes, 1],
            c=[colours[j]],
            s=14,
            alpha=0.75,
            marker=marker,
            linewidths=0,
            zorder=2,
        )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(
        f"Node embeddings Z  (PCA, d={Z_np.shape[1]})\n"
        f"circle=train gang,  triangle=test gang"
    )
    ax.grid(alpha=0.18)

    # ── panel B: gang centroids ───────────────────────────────────────────
    ax = axes[1]
    for j, p in enumerate(patterns):
        marker = "o" if id(p) in train_ids else "^"
        ec = "k" if id(p) in train_ids else "gray"
        ax.scatter(
            cent2d[j, 0],
            cent2d[j, 1],
            c=[colours[j]],
            s=90,
            marker=marker,
            edgecolors=ec,
            linewidths=0.6,
            zorder=3,
        )
        ax.annotate(
            str(j),
            cent2d[j],
            fontsize=5,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
    ax.set_xlabel("PC 1  (gang centroid PCA)")
    ax.set_ylabel("PC 2")
    ax.set_title(
        "Gang centroid embeddings\n"
        "(circle=train, triangle=test;  each label = gang index)"
    )
    ax.grid(alpha=0.18)

    fig.suptitle(
        rf"{args.num_motifs}× {args.motif_type} (size {args.motif_size})  |  "
        rf"collective-bank encoder  |  $\tau$={tau:g}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"  [viz] {out_path.name}")


# ── Figure 3 : training dynamics ─────────────────────────────────────────────


def _plot_training(
    out_path: Path,
    fit: dict,
    train_cap: dict,
    test_cap: dict,
    args,
    tau: float,
) -> None:
    """Three training curves: λ_min, mean retained energy, neg repulsion."""
    history = fit.get("history", [])
    energy_hist = fit.get("energy_history", [])
    neg_hist = fit.get("neg_history", [])
    has_energy = len(energy_hist) > 0
    has_neg = any(v > 1e-9 for v in neg_hist)

    ncols = 1 + int(has_energy) + int(has_neg)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.2))
    if ncols == 1:
        axes = [axes]
    x = np.arange(len(history))

    # panel 0: λ_min
    ax = axes[0]
    ax.plot(x, history, color="tab:red", lw=1.4)
    ax.axhline(
        fit.get("init_objective", 0),
        color="gray",
        ls="--",
        lw=1.0,
        label=f"init={fit.get('init_objective', 0):.4f}",
    )
    ax.axhline(
        fit.get("objective", 0),
        color="tab:red",
        ls=":",
        lw=1.0,
        label=f"best={fit.get('objective', 0):.4f}",
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\lambda_{\min}(\Gamma)$")
    ax.set_title("Collective margin (train gangs)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # panel 1: retained energy
    if has_energy:
        ax = axes[1]
        ax.plot(x, energy_hist, color="tab:purple", lw=1.4, label="mean train")
        ax.axhline(
            train_cap.get("mean_capture", 0),
            color="tab:purple",
            ls=":",
            lw=1.0,
            label=f"final train mean={train_cap.get('mean_capture',0):.3f}",
        )
        ax.axhline(
            test_cap.get("mean_capture", 0),
            color="tab:orange",
            ls=":",
            lw=1.0,
            label=f"final test mean={test_cap.get('mean_capture',0):.3f}",
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"mean retained $M_\tau$-energy $\bar C_j$")
        ax.set_title("Retained energy (train gang indicators)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

    # panel 2: negative objective
    if has_neg:
        ax = axes[-1]
        ax.plot(x, neg_hist, color="tab:cyan", lw=1.0, alpha=0.75)
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"soft $\lambda_{\max}(\Gamma^-)$")
        ax.set_title("Negative repulsion (want ↓)")
        ax.grid(alpha=0.25)

    fig.suptitle(
        rf"{args.num_motifs}× {args.motif_type}  $\tau$={tau:g}  "
        f"degree={args.degree}  epochs={args.epochs}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"  [viz] {out_path.name}")


# ── Figure 4 : LDA-type separation animation ─────────────────────────────────


def _plot_separation(
    anim_path: Path,
    frames_path: Path,
    fit: dict,
    propagated: list,
    normalized: torch.Tensor,
    m_vhat_all: torch.Tensor,
    ridge: float,
    patterns: list,
    train_patterns: list,
    test_patterns: list,
    args,
    tau: float,
) -> None:
    """Two-view animation: raw Z-centroid PCA  +  M_tau-whitened projected indicators.

    **Raw view** (left / top row): each gang is represented by the Euclidean
    mean of its node embeddings ``Z_i`` and projected to 2D via PCA.  This
    tracks absolute motion in the filter output space but the objective is
    *invariant* to invertible linear reparametrisations of Z, so it need not
    show any separation even when lambda_min rises.

    **M_tau-whitened view** (right / bottom row): each gang is represented by
    ``w_j = L_G^T c_j`` where ``G = Z^T M_tau Z = L_G L_G^T`` and
    ``c_j = G^{-1} Z^T M_tau v_hat_j``.  In this coordinate system
    ``<w_i, w_j>_2 = Gamma_ij``, so Euclidean distance equals the
    M_tau-distance between the projected indicators.  When ``lambda_min(Gamma)``n
    is high the w_j vectors are nearly M_tau-orthogonal and appear spread
    out; when lambda_min ~ 0 they cluster.  This is the space the objective
    actually optimises.
    """
    snapshots = fit.get("snapshots", [])
    if len(snapshots) < 2:
        LOGGER.info("  [viz] fewer than 2 snapshots - skipping separation animation")
        return

    n_gangs = len(patterns)
    colours = _gang_colours(n_gangs)
    train_ids = {id(p) for p in train_patterns}
    gang_idx = [np.array(_node_index_list(p)) for p in patterns]

    lam_vals = [s["lam_min"] for s in snapshots]
    epoch_vals = [s["epoch"] for s in snapshots]
    gdiag_vals = [np.array(s["gamma_diag"]) for s in snapshots]

    # ── compute both representations at every snapshot ────────────────────
    all_raw: list[np.ndarray] = []  # raw Z-centroid  (m, d)
    all_white: list[np.ndarray] = []  # M_tau-whitened  (m, d)
    for snap in snapshots:
        Z_t = _fb(propagated, snap["theta"]).detach().numpy()
        all_raw.append(np.stack([Z_t[idx].mean(0) for idx in gang_idx]))
        all_white.append(
            _whitened_coords(
                propagated, snap["theta"], m_vhat_all, normalized, tau, ridge
            )
        )

    # PCA projections: fit on final snapshot
    def _project_all(all_c):
        final = all_c[-1]
        if final.shape[1] > 2:
            return [_pca2d(c, fit_on=final) for c in all_c]
        return [c[:, :2] for c in all_c]

    proj_raw = _project_all(all_raw)
    proj_white = _project_all(all_white)

    def _limits(projs, pad_frac=0.18):
        xy = np.concatenate(projs, axis=0)
        span = max((xy.max(0) - xy.min(0)).max() * pad_frac, 1e-6)
        return (xy[:, 0].min() - span, xy[:, 0].max() + span), (
            xy[:, 1].min() - span,
            xy[:, 1].max() + span,
        )

    xlim_r, ylim_r = _limits(proj_raw)
    xlim_w, ylim_w = _limits(proj_white)

    # ── static key-frames (2 rows: raw top, whitened bottom) ─────────────
    n_frames = len(snapshots)
    ki_list = sorted(
        set(np.linspace(0, n_frames - 1, min(6, n_frames), dtype=int).tolist())
    )
    n_ki = len(ki_list)
    fig_kf, kaxes = plt.subplots(2, n_ki, figsize=(3.2 * n_ki, 7.0))
    if n_ki == 1:
        kaxes = kaxes[:, None]  # keep 2-D indexing

    for col, ki in enumerate(ki_list):
        ep = epoch_vals[ki]
        lam = lam_vals[ki]
        gd = gdiag_vals[ki]
        ep_title = (
            f"ep {ep}\n" + rf"$\lambda_{{min}}$={lam:.4f}" + f"\nmean C={gd.mean():.3f}"
        )

        for row, (xy, xlim, ylim, ylabel) in enumerate(
            [
                (proj_raw[ki], xlim_r, ylim_r, "raw PCA"),
                (proj_white[ki], xlim_w, ylim_w, r"$M_\tau$-whitened"),
            ]
        ):
            ax = kaxes[row, col]
            for j, p in enumerate(patterns):
                marker = "o" if id(p) in train_ids else "^"
                ec = "k" if id(p) in train_ids else "dimgray"
                alpha = float(
                    np.clip(0.4 + 0.55 * (gd[j] if j < len(gd) else 0.5), 0.3, 1.0)
                )
                ax.scatter(
                    xy[j, 0],
                    xy[j, 1],
                    c=[colours[j]],
                    s=60,
                    marker=marker,
                    edgecolors=ec,
                    linewidths=0.5,
                    alpha=alpha,
                    zorder=3,
                )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(alpha=0.18)
            if row == 0:
                ax.set_title(ep_title, fontsize=7)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)

    fig_kf.suptitle(
        rf"Gang separation  ($\tau$={tau:g}, {args.motif_type})  "
        "circle=train  triangle=test  opacity\u221dC_j\n"
        "top: raw Z-centroid PCA  |  bottom: $M_\\tau$-whitened projected indicators",
        fontsize=8,
    )
    fig_kf.tight_layout()
    fig_kf.savefig(frames_path, dpi=150, bbox_inches="tight")
    plt.close(fig_kf)
    LOGGER.info(f"  [viz] {frames_path.name}")

    # ── animated figure (left=raw, right=whitened, bottom=curve) ─────────
    fig_a = plt.figure(figsize=(10.5, 7.0))
    gs = fig_a.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.35, wspace=0.3)
    ax_raw = fig_a.add_subplot(gs[0, 0])
    ax_white = fig_a.add_subplot(gs[0, 1])
    ax_curve = fig_a.add_subplot(gs[1, :])

    def _make_scatters(ax):
        scs = []
        for j, p in enumerate(patterns):
            marker = "o" if id(p) in train_ids else "^"
            ec = "k" if id(p) in train_ids else "dimgray"
            sc = ax.scatter(
                [],
                [],
                c=[colours[j]],
                s=55,
                marker=marker,
                edgecolors=ec,
                linewidths=0.5,
                zorder=3,
            )
            scs.append(sc)
        return scs

    scs_raw = _make_scatters(ax_raw)
    scs_white = _make_scatters(ax_white)

    for ax, xlim, ylim, title in [
        (ax_raw, xlim_r, ylim_r, "Raw Z-centroid (PCA)"),
        (
            ax_white,
            xlim_w,
            ylim_w,
            r"$M_\tau$-whitened  ($\langle w_i,w_j\rangle_2=\Gamma_{ij}$)",
        ),
    ]:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.grid(alpha=0.18)
        ax.set_title(title, fontsize=8)

    ttl = fig_a.suptitle("", fontsize=9, y=0.98)

    ax_curve.plot(epoch_vals, lam_vals, "-", color="tab:red", lw=1.0, alpha=0.4)
    (dot,) = ax_curve.plot([], [], "o", color="tab:red", ms=7)
    ax_curve.set_xlabel("epoch")
    ax_curve.set_ylabel(r"$\lambda_{\min}(\Gamma)$")
    ax_curve.grid(alpha=0.25)

    def _init():
        for sc in scs_raw + scs_white:
            sc.set_offsets(np.empty((0, 2)))
        dot.set_data([], [])
        ttl.set_text("")
        return scs_raw + scs_white + [dot, ttl]

    def _update(fi: int):
        gd = gdiag_vals[fi]
        for j, (sr, sw) in enumerate(zip(scs_raw, scs_white)):
            alpha = float(
                np.clip(0.4 + 0.55 * (gd[j] if j < len(gd) else 0.5), 0.3, 1.0)
            )
            for sc, xy in [(sr, proj_raw[fi]), (sw, proj_white[fi])]:
                sc.set_offsets([[xy[j, 0], xy[j, 1]]])
                sc.set_alpha(alpha)
        dot.set_data([epoch_vals[fi]], [lam_vals[fi]])
        ttl.set_text(
            rf"Epoch {epoch_vals[fi]}   $\lambda_{{min}}$={lam_vals[fi]:.4f}"
            f"   mean C={gd.mean():.3f}"
        )
        return scs_raw + scs_white + [dot, ttl]

    anim = FuncAnimation(
        fig_a,
        _update,
        frames=n_frames,
        init_func=_init,
        interval=160,
        blit=True,
    )

    for writer, ext in [("pillow", ".gif"), ("ffmpeg", ".mp4")]:
        out = anim_path.with_suffix(ext)
        try:
            anim.save(str(out), writer=writer, fps=6)
            LOGGER.info(f"  [viz] {out.name}")
            break
        except Exception as e:
            LOGGER.debug(f"  [viz] {writer} writer failed: {e}")

    plt.close(fig_a)


# ── Figure 5 : extra metrics ──────────────────────────────────────────────────


def _plot_metrics(
    out_path: Path,
    theta_np: np.ndarray,  # (K+1, d)
    coarsening,
    fit: dict,
    train_cap: dict,
    test_cap: dict,
    report: dict,  # {"train": {...}, "test": {...}, "all": {...}}
    args,
    tau: float,
    gamma_np: np.ndarray | None = None,  # (m, m) Γ matrix at final epoch
) -> None:
    """4-panel metrics: filter hops, RSA sigmas, per-gang energy, detection bar."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # ── 5A: filter-bank hop weights ───────────────────────────────────────
    ax = axes[0, 0]
    hop_mean = np.abs(theta_np).mean(axis=1)  # mean |theta_k| per hop
    ax.bar(np.arange(len(hop_mean)), hop_mean, color="tab:blue", alpha=0.82)
    ax.set_xlabel("hop k")
    ax.set_ylabel(r"mean $|\theta_k|$ across channels")
    ax.set_title("Learned filter: hop importance")
    ax.set_xticks(np.arange(len(hop_mean)))
    ax.grid(axis="y", alpha=0.25)

    # ── 5B: RSA sigma per level ───────────────────────────────────────────
    ax = axes[0, 1]
    sigmas = coarsening.sigmas
    sizes = coarsening.sizes
    if sigmas:
        ax.bar(np.arange(len(sigmas)), sigmas, color="tab:orange", alpha=0.82)
        ax.set_xlabel("coarsening level ℓ")
        ax.set_ylabel(r"$\sigma_\ell$ (per-level RSA distortion)")
        ax.set_title(
            f"RSA distortion per level  "
            rf"(cumulative $\varepsilon$={coarsening.epsilon:.3g})"
        )
        ax.grid(axis="y", alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(
            np.arange(len(sizes[1:])),
            sizes[1:],
            "s--",
            color="dimgray",
            ms=2.5,
            alpha=0.7,
            lw=0.8,
        )
        ax2.set_ylabel("graph size after level", color="dimgray", fontsize=8)
    else:
        ax.text(
            0.5, 0.5, "no sigma data", ha="center", va="center", transform=ax.transAxes
        )

    # ── 5C: per-gang retained energy ──────────────────────────────────────
    ax = axes[1, 0]
    train_per = np.array(train_cap.get("per_gang_capture", []))
    test_per = np.array(test_cap.get("per_gang_capture", []))
    if len(train_per):
        ax.bar(
            np.arange(len(train_per)),
            train_per,
            color="tab:blue",
            alpha=0.75,
            label=f"train (mean={train_per.mean():.3f})",
        )
    if len(test_per):
        offset = len(train_per)
        ax.bar(
            np.arange(offset, offset + len(test_per)),
            test_per,
            color="tab:orange",
            alpha=0.75,
            label=f"test (mean={test_per.mean():.3f})",
        )
    if len(train_per):
        ax.axhline(train_per.mean(), color="tab:blue", ls="--", lw=1.0, alpha=0.8)
    if len(test_per):
        ax.axhline(test_per.mean(), color="tab:orange", ls="--", lw=1.0, alpha=0.8)
    ax.set_xlabel("gang index  (train | test)")
    ax.set_ylabel(r"retained $M_\tau$-energy $C_j$")
    ax.set_title("Per-gang retained energy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    # ── 5D: detection metrics per split ───────────────────────────────────
    ax = axes[1, 1]
    splits = ["train", "test", "all"]
    x = np.arange(len(splits))
    w = 0.25
    metrics = ["mean_recall", "mean_precision", "detection_rate"]
    mlabs = ["recall", "precision", "det. rate"]
    mcols = ["tab:blue", "tab:orange", "tab:green"]
    for offset, metric, color, label in zip([-w, 0, w], metrics, mcols, mlabs):
        vals = [(report[s][metric] or 0.0) for s in splits]
        bars = ax.bar(x + offset, vals, w * 0.95, color=color, alpha=0.82, label=label)
        for bar, v in zip(bars, vals):
            if v > 0.015:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.01,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.set_title("Detection metrics per split")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        rf"{args.num_motifs}× {args.motif_type} (size {args.motif_size})  "
        rf"$\tau$={tau:g}  reduction {args.reduction:.0%}  {args.coarsening_laplacian}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"  [viz] {out_path.name}")


# ── Entry point ───────────────────────────────────────────────────────────────


def save_rich_plots(
    *,
    normalized: torch.Tensor,
    adjacency: torch.Tensor,
    X: torch.Tensor,
    graph,
    patterns: list,
    train_patterns: list,
    test_patterns: list,
    theta: torch.Tensor,
    fit: dict,
    coarsening,
    basis: torch.Tensor,
    report: dict,
    train_cap: dict,
    test_cap: dict,
    args,
    out_dir: Path,
    tau: float = 0.0,
    suffix: str = "",
) -> None:
    """Produce all visualisation figures for one ``tau`` run.

    All errors are caught individually so a failed figure never aborts the
    main experiment.
    """
    if _prop_stack is None:
        LOGGER.warning("  [viz] propagation_stack unavailable; skipping rich plots")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = suffix  # e.g. "_tau3"

    # ── shared: recompute embedding once ─────────────────────────────────
    degree = int(theta.shape[0]) - 1
    with torch.no_grad():
        propagated = _prop_stack(normalized, X, degree)
        Z = _fb(propagated, theta)
    Z_np = Z.detach().numpy()

    # ── Figure 1 ─────────────────────────────────────────────────────────
    try:
        _plot_graphs(
            out_dir / f"bank_graphs{tag}.png",
            graph,
            coarsening,
            patterns,
            train_patterns,
            test_patterns,
            args,
            tau,
        )
    except Exception as e:
        LOGGER.warning(f"  [viz] bank_graphs: {e}")

    # ── Figure 2 ─────────────────────────────────────────────────────────
    try:
        _plot_embedding(
            out_dir / f"bank_embedding{tag}.png",
            Z_np,
            graph,
            patterns,
            train_patterns,
            args,
            tau,
        )
    except Exception as e:
        LOGGER.warning(f"  [viz] bank_embedding: {e}")

    # ── Figure 3 ─────────────────────────────────────────────────────────
    try:
        _plot_training(
            out_dir / f"bank_training{tag}.png",
            fit,
            train_cap,
            test_cap,
            args,
            tau,
        )
    except Exception as e:
        LOGGER.warning(f"  [viz] bank_training: {e}")

    # ── Figure 4 (frames + animation) ────────────────────────────────────
    try:
        # precompute M_tau V_hat for ALL patterns (snapshot-independent)
        with torch.no_grad():
            m_vhat_all = _mtau_m_vhat(adjacency, normalized, patterns, tau)
        _plot_separation(
            out_dir / f"bank_separation{tag}.gif",
            out_dir / f"bank_separation_frames{tag}.png",
            fit,
            propagated,
            normalized,
            m_vhat_all,
            float(args.ridge),
            patterns,
            train_patterns,
            test_patterns,
            args,
            tau,
        )
    except Exception as e:
        LOGGER.warning(f"  [viz] bank_separation: {e}")

    # ── Figure 5 ─────────────────────────────────────────────────────────
    try:
        _plot_metrics(
            out_dir / f"bank_metrics{tag}.png",
            theta.detach().numpy(),
            coarsening,
            fit,
            train_cap,
            test_cap,
            report,
            args,
            tau,
        )
    except Exception as e:
        LOGGER.warning(f"  [viz] bank_metrics: {e}")
