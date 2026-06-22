"""Coarsening diagnostics: five self-contained experiments to understand
*why* spectral eigenvectors matter for gang detection and *why* fast epsilon
schedules outperform gradual ones.

All experiments are pure graph-structure analyses — no GNN training needed.

Experiments
-----------
1. spectral_fingerprint         – how much energy do pattern indicators carry at each eigenvector
2. merge_recall_precision_vs_K  – does using more eigenvectors (larger K) improve merge recall/precision?
3. plot_epsilon_schedules       – visualise every schedule power; when does each "unlock"?
4. epsilon_schedule_ablation    – run multilevel coarsening under each schedule; track merge recall/precision
5. supernode_entropy_analysis   – track label entropy inside super-nodes across levels & schedules
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from torch_geometric.data import Data

from src.utils.utils import *

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_graph_params(G: Data) -> Data:
    """Attach W, L, dw to a PyG graph if not already present."""
    if not hasattr(G, "L") or G.L is None:
        if not hasattr(G, "edge_weight") or G.edge_weight is None:
            G.edge_weight = torch.ones(G.edge_index.size(1), dtype=torch.float32)
        G.W, G.L, G.dw = graph_params(G)
    return G


def compute_spectral_decomp(
    G: Data,
    K_max: int = 300,
    dense_threshold: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (eigenvalues, eigenvectors) of the graph Laplacian.

    For graphs with N ≤ *dense_threshold* the full dense decomposition is used.
    For larger graphs lobpcg is used (K_max smallest eigenvectors).

    Returns
    -------
    lk : ndarray, shape (K,)
    Uk : ndarray, shape (N, K)
    """
    G = _ensure_graph_params(G)
    N = G.num_nodes
    K = min(K_max, N)
    L = G.L

    if N <= dense_threshold:
        d, V = torch.linalg.eigh(L.to_dense())
        lk = _to_numpy(d[:K])
        Uk = _to_numpy(V[:, :K])
    else:
        # Use shift-invert via lobpcg to get smallest K eigenvectors.
        try:

            offset = float(2 * G.dw.max())
            T = offset * sparse_eye(N).to(L.device) - L
            X_init = torch.randn(N, K, device=L.device)
            lk_t, Uk_t = torch.lobpcg(T, k=K, X=X_init, largest=True, tol=1e-4)
            lk_t = torch.flip(offset - lk_t, [0])
            Uk_t = torch.flip(Uk_t, [1])
            lk = _to_numpy(lk_t)
            Uk = _to_numpy(Uk_t)
        except Exception as e:
            LOGGER.info(
                f"[spectral_decomp] lobpcg failed ({e}); falling back to dense for first {K} vectors"
            )
            d, V = torch.linalg.eigh(L.to_dense())
            lk = _to_numpy(d[:K])
            Uk = _to_numpy(V[:, :K])
    return lk, Uk
