r"""Smooth Dual Ward -- adjacency-constrained agglomerative graph coarsening.

Classical (Euclidean) Ward on an ``L``-orthonormal embedding scores a merge by
the *unnormalized* increase of the Frobenius RSA residual.  That increment grows
with the mass of the merged clusters, so it silently prefers merging small/low
degree clusters regardless of whether the target subspace is actually smooth
across them.  **Smooth Dual Ward** repairs this by scoring a merge in the *dual*
(screened) metric and dividing by the structural cost the merge would incur:

    Delta_DW(A, B) = (v_A v_B)/(v_A + v_B) ||mu_A - mu_B||^2   = ||U_t^T M_t g||^2
    m_t(A, B)      = ell(A, B) + tau                           = g^T M_t g
    sigma_DW(A, B) = Delta_DW / m_t                            in [0, 1]
    s_alpha(A, B)  = m_t^alpha * sigma_DW = Delta_DW / m_t^(1 - alpha)

with the ``M_tau``-normalized contrast vector

    g_{A,B} = sqrt(v_A v_B/(v_A + v_B)) * D_t^{1/2} (1_A/v_A - 1_B/v_B),

``M_tau = L_sym + tau I``, ``L_sym = I - D_t^{-1/2}(W + I)D_t^{-1/2}``, and
``U_tau`` an ``M_tau``-orthonormal basis of the learned target ``span(Z)``.

``sigma_DW`` is the fraction of the merge direction's *screened energy* that the
target subspace can see -- a scale-free measure -- while ``Delta_DW`` is the raw
visible energy.  Dividing by ``m_tau`` removes the bias toward absolute merge
energy and ranks contrasts by relative target visibility; note the direction: for
a fixed ``Delta_DW`` a *larger* ``m_tau`` makes a merge *cheaper*, so the
normalization does not protect high-cut directions, it standardizes them away.
``alpha`` smoothly interpolates:

* ``alpha = 0`` -> ``s_0 = sigma_DW`` (pure normalized / conductance-like);
* ``alpha = 1`` -> ``s_1 = Delta_DW`` (pure dual Ward increment; the sum over
  merges telescopes to ``F_Y(P) = ||(I - Pi_P) Y_tau||_F^2`` with the *same*
  Euclidean degree-weighted block projector ``Pi_P`` the coarsening applies --
  it is NOT the residual of an ``M_tau``-orthogonal projector.  The mismatch
  with the reported constant is that the screened RSA trace is
  ``T_RSA(P) = ||(I - Pi_P) U_tau||^2_{M_tau,F}``, and ``F_Y(P) != T_RSA(P)``
  unless ``Y_tau = U_tau``, i.e. unless ``embedding="primal"``);
* ``0 < alpha < 1`` -> a mixture that keeps the normalized score's indifference
  to cluster mass while still penalizing expensive (high-cut) merges.

Only graph-adjacent clusters are ever merged, so every produced cluster induces a
connected subgraph and the output is a legal Laplacian-consistent coarsening.
All graph algebra is sparse; no dense ``N x N`` matrix is ever formed.

Run ``python -m src.smooth_dual_ward`` to execute the validation suite of
Section 12 (target normalization, numerator/denominator identities, score
bounds, endpoint identities, incremental-statistic consistency, connectivity,
and the ``alpha = 1`` cumulative identity).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import scipy.sparse as sp

__all__ = [
    "SmoothDualWardResult",
    "smooth_dual_ward",
    "screened_operators",
    "m_orthonormal_basis",
    "exact_rsa_epsilon",
    "run_validation_suite",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SmoothDualWardResult:
    """Full agglomeration output (Section 9).

    * ``labels_`` -- ``(N,)`` contiguous cluster label per original node.
    * ``children_`` -- ``(m, 2)`` merged cluster ids per step; step ``t`` creates
      cluster id ``N + t`` (sklearn's ``children_`` convention).
    * ``merge_records_`` -- per-merge diagnostics (see :func:`smooth_dual_ward`).
    * ``n_effective_target_dims_`` -- retained rank of ``Z^T M_tau Z``.
    * ``target_basis_`` -- ``U_tau`` (``M_tau``-orthonormal basis of ``span(Z)``).
    * ``dual_embedding_`` -- ``B_dual``, row ``i`` = ``(M_tau U_tau)_i/sqrt(d~_i)``.
    * ``rsa_curve_`` -- ``[(n_clusters, epsilon_exact), ...]`` when requested.
    """

    labels_: np.ndarray
    children_: np.ndarray
    merge_records_: List[dict]
    n_effective_target_dims_: int
    target_basis_: np.ndarray
    dual_embedding_: np.ndarray
    rsa_curve_: List[tuple] = field(default_factory=list)
    n_leaves_: int = 0
    n_clusters_: int = 0

    # -- hierarchy replay ---------------------------------------------------
    def labels_at(self, n_clusters: int) -> np.ndarray:
        """Re-cut the stored hierarchy to (at most) ``n_clusters`` clusters."""

        n_merges = max(0, self.n_leaves_ - int(n_clusters))
        return self._cut(min(n_merges, int(self.children_.shape[0])))

    def _cut(self, n_merges_to_apply: int) -> np.ndarray:
        n = self.n_leaves_
        m = int(self.children_.shape[0])
        apply = max(0, min(int(n_merges_to_apply), m))
        parent = list(range(n + m))

        def find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:  # path compression
                parent[x], x = root, parent[x]
            return root

        pairs = self.children_.tolist()
        for t in range(apply):
            a, b = pairs[t]
            parent[find(int(a))] = n + t
            parent[find(int(b))] = n + t

        seen: Dict[int, int] = {}
        labels = np.empty(n, dtype=np.int64)
        for leaf in range(n):
            root = find(leaf)
            labels[leaf] = seen.setdefault(root, len(seen))
        return labels


# ---------------------------------------------------------------------------
# Section 2 / 3 / 4 -- operators, target basis, dual embedding
# ---------------------------------------------------------------------------


def _sanitize_adjacency(W) -> sp.csr_matrix:
    """Symmetric, nonnegative, loop-free CSR copy of ``W``."""

    A = sp.csr_matrix(W, dtype=np.float64)
    A = 0.5 * (A + A.T)  # enforce exact symmetry
    A = sp.csr_matrix(A)
    A.setdiag(0.0)
    A.eliminate_zeros()
    if A.nnz and A.data.min() < 0.0:
        raise ValueError("adjacency must be nonnegative")
    return A


def screened_operators(W, tau: float):
    r"""Return ``(W_offdiag, d_tilde, L_sym, M_tau)`` for ``M_tau = L_sym + tau I``.

    ``W_offdiag`` keeps the *original* off-diagonal weights (used for cluster cuts
    -- self-loops never contribute to a cut), ``d_tilde = (W + I) 1`` are the
    augmented degrees, and ``L_sym = I - D_t^{-1/2}(W + I) D_t^{-1/2}``.
    """

    if tau <= 0.0:
        raise ValueError("tau must be strictly positive (M_tau must be PD)")
    A = _sanitize_adjacency(W)
    n = A.shape[0]
    identity = sp.identity(n, format="csr", dtype=np.float64)
    W_tilde = (A + identity).tocsr()
    d_tilde = np.asarray(W_tilde.sum(axis=1)).ravel()
    inv_sqrt = 1.0 / np.sqrt(np.maximum(d_tilde, np.finfo(np.float64).tiny))
    scale = sp.diags(inv_sqrt)
    a_hat = (scale @ W_tilde @ scale).tocsr()
    L = (identity - a_hat).tocsr()
    M = (L + tau * identity).tocsr()
    return A, d_tilde, L, M


def m_orthonormal_basis(Z: np.ndarray, M: sp.spmatrix, rank_tol: float = 1e-10):
    r"""``U_tau = Z V_r Lambda_r^{-1/2}`` with ``U_tau^T M_tau U_tau = I`` (Section 3).

    Handles a rank-deficient / non-orthogonal ``Z``: eigenvalues of the small
    Gram ``G_tau = Z^T M_tau Z`` below ``rank_tol * lambda_max`` are dropped, so
    the returned basis spans the numerically well-conditioned part of
    ``span(Z)``.  Never forms an ``N x N`` matrix.
    """

    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    G = Z.T @ (M @ Z)
    G = 0.5 * (G + G.T)
    evals, evecs = np.linalg.eigh(G)
    lam_max = float(evals[-1]) if evals.size else 0.0
    if lam_max <= 0.0:
        raise ValueError("Z^T M_tau Z is not positive definite; empty target")
    keep = evals > rank_tol * lam_max
    V_r = evecs[:, keep]
    lam_r = evals[keep]
    U = Z @ (V_r / np.sqrt(lam_r))
    return U, int(lam_r.size)


def dual_embedding(
    U: np.ndarray,
    M: sp.spmatrix,
    d_tilde: np.ndarray,
    embedding: str = "dual",
):
    r"""Scoring embedding ``(Y, B)`` with ``B_i = Y_i / sqrt(d~_i)``.

    ``embedding="dual"`` (the spec) uses ``Y = M_tau U_tau``.  Then
    ``Delta_DW = ||U^T M g||^2`` and, at ``alpha = 1``, the merge scores
    telescope to ``||(I - Pi_P) M_tau U_tau||_F^2``.

    ``embedding="primal"`` uses ``Y = U_tau``, giving ``Delta = ||U^T g||^2``
    and the telescoped objective ``||(I - Pi_P) U_tau||_F^2``.  It is also the
    *raw* damage of the merge normalized once: the newly removed component is
    ``g g^T U_tau``, whose isolated ``M_tau`` energy is
    ``(g^T M_tau g) ||U_tau^T g||^2 = m_tau ||U_tau^T g||^2``, so dividing that
    by the structural energy ``m_tau`` leaves exactly ``||U_tau^T g||^2``.
    Dividing it by ``m_tau`` a *second* time (``alpha < 1``) is therefore not
    obviously principled for this embedding -- test ``alpha = 1`` first.

    The primal objective brackets the screened RSA trace,

        tau ||(I - Pi_P)U||_F^2 <= ||(I - Pi_P)U||^2_{M_tau,F}
                               <= (lambda_max + tau) ||(I - Pi_P)U||_F^2,

    which the dual objective does not.

    Why it matters: ``M_tau`` scales each spectral direction by ``lambda + tau``,
    so ``M_tau U_tau`` *attenuates* precisely the low-frequency band the target
    subspace is meant to retain -- the dual embedding scores merges in high-passed
    coordinates.  The primal embedding removes that distortion while leaving the
    contrast vector ``g``, the volumes and the ``m_tau`` denominator untouched, so
    every other identity of the construction still holds.  ``primal`` is also the
    ``tau -> inf`` limit of ``dual`` (``M_tau -> tau I``), i.e. ``tau`` is itself a
    continuous low-pass knob.
    """

    if embedding == "dual":
        Y = M @ U
    elif embedding == "primal":
        Y = np.asarray(U)
    else:
        raise ValueError("embedding must be 'dual' or 'primal'")
    B = Y / np.sqrt(d_tilde)[:, None]
    return Y, B


def exact_rsa_epsilon(
    U: np.ndarray, M: sp.spmatrix, d_tilde: np.ndarray, labels: np.ndarray
) -> float:
    r"""Exact RSA constant ``sqrt(lambda_max(E^T M_tau E))`` (Section 10).

    ``E = (I - Pi_P) U_tau`` with the degree-weighted Euclidean block-averaging
    projector ``(Pi_P U)_i = sqrt(d~_i) r_{C(i)}``,
    ``r_C = (1/v_C) sum_{j in C} sqrt(d~_j) U_j``.  Applied without materializing
    ``Pi_P``.  Note ``Pi_P`` is Euclidean, not ``M_tau``-orthogonal, so this is
    *not* guaranteed to be monotone in the coarsening level.
    """

    labels = np.asarray(labels, dtype=np.int64)
    k = int(labels.max()) + 1
    root = np.sqrt(d_tilde)
    vol = np.bincount(labels, weights=d_tilde, minlength=k)
    numer = np.zeros((k, U.shape[1]), dtype=np.float64)
    np.add.at(numer, labels, root[:, None] * U)
    r = numer / np.maximum(vol, np.finfo(np.float64).tiny)[:, None]
    E = U - root[:, None] * r[labels]
    H = E.T @ (M @ E)
    H = 0.5 * (H + H.T)
    top = float(np.linalg.eigvalsh(H)[-1])
    return math.sqrt(max(top, 0.0))


# ---------------------------------------------------------------------------
# Section 5 -- candidate merge quantities
# ---------------------------------------------------------------------------


def _pair_scores(
    delta: float,
    ell: float,
    tau: float,
    alpha: float,
    bound_tol: float,
    enforce_bound: bool = True,
):
    """``(s_alpha, sigma_DW, Delta_DW, m_tau, ell)`` from the raw ingredients.

    ``enforce_bound`` asserts ``sigma in [0, 1]``.  That bound is specific to the
    *dual* embedding, where ``Delta = ||P_R g||_M^2 <= ||g||_M^2 = m_tau`` because
    ``P_R`` is the ``M_tau``-orthogonal projector onto the target.  With the
    primal embedding the numerator is ``||U^T g||^2``, bounded by
    ``g^T M_tau^{-1} g`` rather than ``m_tau``, so the ratio may legitimately
    exceed 1 and must not be clipped (clipping would flatten the ranking).
    """

    delta = max(delta, 0.0)  # clip roundoff only
    ell = max(ell, 0.0)
    m = ell + tau  # strictly positive because tau > 0
    sigma = delta / m
    if enforce_bound:
        if sigma > 1.0 + 1e-3 or sigma < -1e-3:
            raise ValueError(
                f"sigma_DW = {sigma!r} outside [0, 1] (Delta={delta!r}, m_tau={m!r}): "
                "inconsistent normalization or cut bookkeeping"
            )
        sigma = min(max(sigma, 0.0), 1.0)  # absorb the remaining float roundoff
    else:
        sigma = max(sigma, 0.0)
    if alpha >= 1.0:
        score = delta
    elif alpha <= 0.0:
        score = sigma
    elif delta == 0.0:
        score = 0.0
    else:  # log-space for stability: Delta * exp((alpha - 1) log m)
        score = math.exp(math.log(delta) + (alpha - 1.0) * math.log(m))
    return score, sigma, delta, m, ell


# ---------------------------------------------------------------------------
# Sections 6-9 -- the agglomeration
# ---------------------------------------------------------------------------


def smooth_dual_ward(
    W,
    Z: np.ndarray,
    tau: float,
    alpha: float,
    *,
    n_clusters: int | None = None,
    build_full_tree: bool = True,
    rank_tol: float = 1e-10,
    score_tol: float = 1e-12,
    max_cluster_size: int = 0,
    embedding: str = "dual",
    evaluate_rsa: bool = False,
    rsa_levels: Sequence[int] | None = None,
) -> SmoothDualWardResult:
    r"""Adjacency-constrained agglomeration minimizing ``s_alpha`` (Section 13 API).

    Parameters
    ----------
    W:
        Symmetric nonnegative (sparse) adjacency.  Self-loops are ignored for
        cuts; the algorithm adds its own ``+I`` renormalization internally.
    Z:
        ``(N, d0)`` target basis.  Columns need not be orthogonal or independent.
    tau:
        Screening level, strictly positive (guarantees ``m_tau > 0``).
    alpha:
        Smoothing exponent in ``[0, 1]``; ``0`` = normalized score, ``1`` = raw
        dual-Ward increment.
    n_clusters:
        Requested final cluster count.  Agglomeration always stops when no
        adjacent pair remains, so the achieved count can exceed ``n_clusters``
        on a disconnected graph.
    build_full_tree:
        Build the entire hierarchy (then cut to ``n_clusters``) instead of
        stopping early.  ``False`` is much cheaper when only one cut is needed.
    max_cluster_size:
        Optional cap on the *cardinality* of a super-node (``0`` = uncapped, the
        normal setting).  This is a **cost bound, not a correctness fix**: the
        harmonic mass factor ``v_A v_B/(v_A + v_B)`` saturates at
        ``min(v_A, v_B)``, so neither this score nor classical Ward penalises an
        already-large cluster for absorbing one more node -- measured on
        Elliptic++ days 24-26 at the best cut, Ward's largest super-node is
        3,333 nodes and uncapped dual-ward's (alpha=0) is 2,602.  Cap only when
        the pure-Python agglomeration's worst case (alpha=1 reached 15.8k) is
        too slow to run many variants in one pass.
    embedding:
        ``"dual"`` (the spec) scores merges against ``M_tau U_tau``; ``"primal"``
        scores against ``U_tau``, removing the ``M_tau`` high-pass so that the
        telescoped objective matches the evaluated RSA constant.  See
        :func:`dual_embedding`.
    evaluate_rsa / rsa_levels:
        Evaluate the exact RSA constant (Section 10) at the listed cluster counts
        (default: a log-spaced sweep) and return it as ``rsa_curve_``.

    Every merge record contains ``children``, ``new_id``, ``n_clusters``,
    ``s_alpha``, ``sigma_dw``, ``delta_dw``, ``m_tau``, ``ell`` (``g^T L g``),
    ``volume`` and ``cut`` -- enough to replay the hierarchy without storing any
    intermediate partition.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")

    A, d_tilde, L, M = screened_operators(W, tau)
    n = A.shape[0]
    U, n_dims = m_orthonormal_basis(Z, M, rank_tol=rank_tol)
    Y, B = dual_embedding(U, M, d_tilde, embedding)
    # the sigma <= 1 projection bound holds only for the dual embedding
    bounded = embedding == "dual"

    # ---- cluster state, indexed by immutable cluster id -------------------
    max_ids = 2 * n
    dim = B.shape[1]
    mu = np.zeros((max_ids, dim), dtype=np.float64)
    mu[:n] = B
    vol = np.zeros(max_ids, dtype=np.float64)
    vol[:n] = d_tilde
    ocut = np.zeros(max_ids, dtype=np.float64)
    ocut[:n] = np.asarray(A.sum(axis=1)).ravel()  # cut(i, V \ {i})
    version = np.zeros(max_ids, dtype=np.int64)
    active = np.zeros(max_ids, dtype=bool)
    active[:n] = True
    size = np.zeros(max_ids, dtype=np.int64)
    size[:n] = 1

    coo = sp.triu(A, k=1).tocoo()
    nbr: List[Dict[int, float]] = [dict() for _ in range(max_ids)]
    for i, j, w in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
        if w <= 0.0:
            continue
        nbr[i][j] = w
        nbr[j][i] = w

    def stats(a: int, b: int):
        va, vb = vol[a], vol[b]
        s = va + vb
        diff = mu[a] - mu[b]
        delta = (va * vb / s) * float(diff @ diff)
        wab = nbr[a].get(b, 0.0)
        ell = (
            s * wab / (va * vb)
            + vb * (ocut[a] - wab) / (va * s)
            + va * (ocut[b] - wab) / (vb * s)
        )
        return _pair_scores(delta, ell, tau, alpha, score_tol, bounded) + (wab,)

    def stats_many(c: int, ks: np.ndarray, ws: np.ndarray):
        """Vectorized ``(s_alpha, sigma, Delta)`` for the pairs ``(c, ks)``.

        Same algebra as :func:`stats`, batched over all neighbours of ``c`` --
        the merge loop touches every neighbour of the freshly merged cluster, so
        this is the inner loop's hot path.
        """

        va = vol[c]
        vb = vol[ks]
        s = va + vb
        diff = mu[ks] - mu[c]
        delta = (va * vb / s) * np.einsum("ij,ij->i", diff, diff)
        np.maximum(delta, 0.0, out=delta)
        ell = (
            s * ws / (va * vb)
            + vb * (ocut[c] - ws) / (va * s)
            + va * (ocut[ks] - ws) / (vb * s)
        )
        m = np.maximum(ell, 0.0) + tau
        sigma = delta / m
        if bounded:
            np.clip(sigma, 0.0, 1.0, out=sigma)
        else:
            np.maximum(sigma, 0.0, out=sigma)
        if alpha >= 1.0:
            scores = delta
        elif alpha <= 0.0:
            scores = sigma
        else:
            scores = delta * np.power(m, alpha - 1.0)
        return scores, sigma, delta

    heap: List[tuple] = []
    for i, j, w in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
        if w <= 0.0:
            continue
        score, sigma, delta, _m, _ell, _w = stats(i, j)
        heap.append((score, sigma, delta, i, j, 0, 0))
    heapq.heapify(heap)
    n_edges = len(heap)

    n_active = n
    children: List[tuple] = []
    records: List[dict] = []
    rsa_curve: List[tuple] = []
    next_id = n
    cap = int(max_cluster_size) if max_cluster_size else 0

    target = int(n_clusters) if n_clusters is not None else 1
    rsa_targets = set()
    if evaluate_rsa:
        if rsa_levels is not None:
            rsa_targets = {int(x) for x in rsa_levels}
        else:
            lo = max(1, target)
            rsa_targets = {
                int(round(x))
                for x in np.geomspace(max(lo, 1), n, num=min(12, n))
                if 1 <= round(x) <= n
            }

    def stop_now() -> bool:
        if build_full_tree:
            return False
        return n_clusters is not None and n_active <= target

    # Heap growth bound.  For alpha < 1 the score is scale free in cluster mass,
    # so a cluster can keep absorbing neighbours (chaining) and its candidate
    # list -- hence the number of pushes per merge -- grows without bound.  The
    # compaction below drops entries that lazy invalidation would discard
    # anyway, keeping memory linear in the live candidate count.
    compact_at = max(4 * (n_edges + n), 1 << 20)

    while heap and n_active > 1 and not stop_now():
        score, sigma, delta, a, b, va_ver, vb_ver = heapq.heappop(heap)
        # ---- lazy invalidation (Section 8) --------------------------------
        if not active[a] or not active[b]:
            continue
        if va_ver != version[a] or vb_ver != version[b]:
            continue
        wab = nbr[a].get(b, 0.0)
        if wab <= 0.0:
            continue
        if cap and size[a] + size[b] > cap:
            continue  # refused: would exceed the super-node size cap
        # the entry is current, so re-deriving is exact and yields m_tau / ell
        score, sigma, delta, m_tau, ell, wab = stats(a, b)

        # ---- Section 7: cluster update ------------------------------------
        new = next_id
        next_id += 1
        va, vb = vol[a], vol[b]
        vol[new] = va + vb
        mu[new] = (va * mu[a] + vb * mu[b]) / (va + vb)
        ocut[new] = ocut[a] + ocut[b] - 2.0 * wab
        size[new] = size[a] + size[b]
        active[a] = active[b] = False
        active[new] = True
        version[new] = 0

        merged: Dict[int, float] = {}
        for k, w in nbr[a].items():
            if k != b and active[k]:
                merged[k] = merged.get(k, 0.0) + w
        for k, w in nbr[b].items():
            if k != a and active[k]:
                merged[k] = merged.get(k, 0.0) + w
        for k in nbr[a]:
            nbr[k].pop(a, None)
        for k in nbr[b]:
            nbr[k].pop(b, None)
        nbr[a] = {}
        nbr[b] = {}
        for k, w in merged.items():
            if w <= 0.0:
                continue
            nbr[new][k] = w
            nbr[k][new] = w
        # No version bump is needed for the surviving neighbours ``k``: merging
        # ``a, b`` leaves ``v_k``, ``mu_k`` and ``o_k`` untouched, and the pair
        # ``(k, new)`` carries a *fresh* id, so every heap entry that survives
        # the active/adjacency checks is still numerically current.  Bumping
        # ``version[k]`` here would silently discard valid ``(k, j)`` candidates.

        n_active -= 1
        children.append((a, b))
        records.append(
            {
                "children": (int(a), int(b)),
                "new_id": int(new),
                "n_clusters": int(n_active),
                "s_alpha": float(score),
                "sigma_dw": float(sigma),
                "delta_dw": float(delta),
                "m_tau": float(m_tau),
                "ell": float(ell),
                "volume": float(vol[new]),
                "cut": float(wab),
            }
        )

        # ---- recompute every affected candidate (no Lance-Williams) -------
        if nbr[new]:
            ks = np.fromiter(nbr[new].keys(), dtype=np.int64, count=len(nbr[new]))
            ws = np.fromiter(nbr[new].values(), dtype=np.float64, count=len(nbr[new]))
            if cap:
                admissible = size[ks] + size[new] <= cap
                ks, ws = ks[admissible], ws[admissible]
            if ks.size:
                s_k, sig_k, del_k = stats_many(new, ks, ws)
                ver_new = int(version[new])
                for t in range(ks.size):
                    k = int(ks[t])
                    lo, hi = (k, new) if k < new else (new, k)
                    heapq.heappush(
                        heap,
                        (
                            float(s_k[t]),
                            float(sig_k[t]),
                            float(del_k[t]),
                            lo,
                            hi,
                            int(version[k]) if lo == k else ver_new,
                            ver_new if lo == k else int(version[k]),
                        ),
                    )

        if len(heap) > compact_at:  # drop entries lazy invalidation would reject
            heap = [
                e
                for e in heap
                if active[e[3]]
                and active[e[4]]
                and e[5] == version[e[3]]
                and e[6] == version[e[4]]
            ]
            heapq.heapify(heap)
            compact_at = max(compact_at, 2 * len(heap))

        if rsa_targets and n_active in rsa_targets:
            labels_now = _current_labels(n, children)
            rsa_curve.append(
                (int(n_active), exact_rsa_epsilon(U, M, d_tilde, labels_now))
            )

    children_arr = (
        np.asarray(children, dtype=np.int64)
        if children
        else np.empty((0, 2), dtype=np.int64)
    )
    result = SmoothDualWardResult(
        labels_=np.zeros(n, dtype=np.int64),
        children_=children_arr,
        merge_records_=records,
        n_effective_target_dims_=n_dims,
        target_basis_=U,
        dual_embedding_=B,
        rsa_curve_=rsa_curve,
        n_leaves_=n,
    )
    if n_clusters is not None and build_full_tree:
        result.labels_ = result.labels_at(int(n_clusters))
    else:
        result.labels_ = result._cut(len(children))
    result.n_clusters_ = int(result.labels_.max()) + 1 if n else 0
    return result


def _current_labels(n: int, children: Sequence[tuple]) -> np.ndarray:
    """Contiguous leaf labels after applying every merge in ``children``."""

    m = len(children)
    parent = list(range(n + m))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for t, (a, b) in enumerate(children):
        parent[find(int(a))] = n + t
        parent[find(int(b))] = n + t
    seen: Dict[int, int] = {}
    labels = np.empty(n, dtype=np.int64)
    for leaf in range(n):
        labels[leaf] = seen.setdefault(find(leaf), len(seen))
    return labels


# ---------------------------------------------------------------------------
# Section 12 -- validation suite
# ---------------------------------------------------------------------------


def _brute_pair(U, M, d_tilde, A, labels, ca, cb, tau, alpha, embedding="dual"):
    """Explicitly built ``g_{A,B}`` reference quantities for a candidate pair."""

    idx_a = np.where(labels == ca)[0]
    idx_b = np.where(labels == cb)[0]
    va = float(d_tilde[idx_a].sum())
    vb = float(d_tilde[idx_b].sum())
    s = va + vb
    g = np.zeros(U.shape[0])
    g[idx_a] = math.sqrt(vb / (va * s)) * np.sqrt(d_tilde[idx_a])
    g[idx_b] = -math.sqrt(va / (vb * s)) * np.sqrt(d_tilde[idx_b])
    proj = U.T @ (M @ g) if embedding == "dual" else U.T @ g
    num = float(proj @ proj)
    den = float(g @ (M @ g))
    return num, den, g


def run_validation_suite(seed: int = 0, verbose: bool = True) -> None:
    """Small-graph checks from Section 12; raises ``AssertionError`` on failure."""

    rng = np.random.default_rng(seed)
    n, tau = 40, 0.35
    # connected random graph: a spanning path plus random chords (so the
    # agglomeration can always reach the requested cluster count)
    Wd = np.zeros((n, n))
    for i in range(n - 1):
        Wd[i, i + 1] = rng.uniform(0.5, 2.0)
    extra = rng.random((n, n)) < 0.12
    Wd = np.where(np.triu(extra, 1), rng.uniform(0.5, 2.0, size=(n, n)), Wd)
    Wd = np.triu(Wd, 1)
    for i in range(n - 1):  # keep the path after the overwrite above
        Wd[i, i + 1] = max(Wd[i, i + 1], 0.5)
    Wd = Wd + Wd.T
    W = sp.csr_matrix(Wd)
    Z = rng.normal(size=(n, 5))
    Z = np.hstack([Z, Z[:, :1]])  # deliberately rank deficient

    A, d_tilde, L, M = screened_operators(W, tau)
    U, n_dims = m_orthonormal_basis(Z, M)
    assert n_dims == 5, f"expected effective rank 5, got {n_dims}"

    # (1) target normalization
    err = np.linalg.norm(U.T @ (M @ U) - np.eye(n_dims), 2)
    assert err < 1e-8, f"U^T M U != I (err={err:.3e})"

    # (2)-(8) identities, bounds, endpoints and the telescoping objective --
    # run for BOTH scoring embeddings (see :func:`dual_embedding`).
    summary = {}
    for embedding in ("dual", "primal"):
        res = smooth_dual_ward(
            W, Z, tau, 0.5, n_clusters=max(2, n // 3), embedding=embedding
        )
        labels = res.labels_
        Y, B = dual_embedding(U, M, d_tilde, embedding)

        k = int(labels.max()) + 1
        vol = np.bincount(labels, weights=d_tilde, minlength=k)
        cent = np.zeros((k, B.shape[1]))
        np.add.at(cent, labels, d_tilde[:, None] * B)
        cent /= vol[:, None]

        Acoo = sp.triu(A, k=1).tocoo()
        pair_w: Dict[tuple, float] = {}
        for i, j, w in zip(Acoo.row, Acoo.col, Acoo.data):
            ca, cb = int(labels[i]), int(labels[j])
            if ca == cb:
                continue
            key = (min(ca, cb), max(ca, cb))
            pair_w[key] = pair_w.get(key, 0.0) + float(w)
        deg = np.asarray(A.sum(axis=1)).ravel()
        ocut = np.bincount(labels, weights=deg, minlength=k)
        intra = np.zeros(k)
        for i, j, w in zip(Acoo.row, Acoo.col, Acoo.data):
            if labels[i] == labels[j]:
                intra[labels[i]] += 2.0 * float(w)
        ocut = ocut - intra

        checked = 0
        for (ca, cb), wab in list(pair_w.items())[:25]:
            num, den, _ = _brute_pair(
                U, M, d_tilde, A, labels, ca, cb, tau, 0.5, embedding
            )
            va, vb = vol[ca], vol[cb]
            s = va + vb
            diff = cent[ca] - cent[cb]
            delta = (va * vb / s) * float(diff @ diff)
            ell = (
                s * wab / (va * vb)
                + vb * (ocut[ca] - wab) / (va * s)
                + va * (ocut[cb] - wab) / (vb * s)
            )
            assert abs(delta - num) <= 1e-8 * max(
                1.0, abs(num)
            ), f"[{embedding}] numerator identity failed: {delta} vs {num}"
            assert abs((ell + tau) - den) <= 1e-8 * max(
                1.0, abs(den)
            ), f"[{embedding}] denominator identity failed: {ell + tau} vs {den}"
            s0 = _pair_scores(delta, ell, tau, 0.0, 1e-12, embedding == "dual")
            s1 = _pair_scores(delta, ell, tau, 1.0, 1e-12, embedding == "dual")
            if embedding == "dual":
                # sigma = ||P g||_M^2 / ||g||_M^2 is a projection ratio only for
                # the dual embedding; the primal numerator is not bounded by m_tau.
                assert -1e-9 <= s0[1] <= 1.0 + 1e-9, f"sigma out of bounds: {s0[1]}"
            assert abs(s0[0] - s0[1]) < 1e-12, "s_0 != sigma_DW"
            assert abs(s1[0] - s1[2]) < 1e-12, "s_1 != Delta_DW"
            checked += 1
        assert checked > 0, "no cross-cluster pair was checked"

        # (6) incremental cluster statistics vs recomputation from memberships
        ids = _leaf_members(res)
        for cid, members in list(ids.items())[:20]:
            v_ref = float(d_tilde[members].sum())
            mu_ref = (d_tilde[members, None] * B[members]).sum(0) / v_ref
            sub = A[members][:, members]
            o_ref = float(deg[members].sum() - sub.sum())
            lab = int(labels[members[0]])
            assert abs(v_ref - vol[lab]) < 1e-9 * max(1.0, v_ref), "volume mismatch"
            assert np.allclose(mu_ref, cent[lab], atol=1e-9), "centroid mismatch"
            assert abs(o_ref - ocut[lab]) < 1e-8 * max(1.0, abs(o_ref)), "cut mismatch"

        # (7) connectivity of every produced cluster
        for cid in range(k):
            members = np.where(labels == cid)[0]
            sub = A[members][:, members]
            ncomp = sp.csgraph.connected_components(sub, directed=False)[0]
            assert ncomp == 1, f"cluster {cid} is disconnected ({ncomp} components)"

        # (8) alpha = 1 cumulative identity: sum Delta == ||(I - Pi_P) Y||_F^2,
        # with Y = M_tau U_tau (dual) or Y = U_tau (primal).  For 'primal' the
        # right-hand side is the Frobenius relaxation of the constant
        # exact_rsa_epsilon actually evaluates.
        res1 = smooth_dual_ward(
            W,
            Z,
            tau,
            1.0,
            n_clusters=max(2, n // 3),
            build_full_tree=False,
            embedding=embedding,
        )
        total = sum(r["delta_dw"] for r in res1.merge_records_)
        lab1 = res1.labels_
        k1 = int(lab1.max()) + 1
        vol1 = np.bincount(lab1, weights=d_tilde, minlength=k1)
        cent1 = np.zeros((k1, B.shape[1]))
        np.add.at(cent1, lab1, d_tilde[:, None] * B)
        cent1 /= vol1[:, None]
        resid = Y - np.sqrt(d_tilde)[:, None] * cent1[lab1]
        frob = float((resid**2).sum())
        assert abs(total - frob) <= 1e-6 * max(
            1.0, frob
        ), f"[{embedding}] alpha=1 cumulative identity failed: {total} vs {frob}"
        rsa = exact_rsa_epsilon(U, M, d_tilde, lab1)
        summary[embedding] = (checked, total, frob, rsa)

    if verbose:
        print("smooth_dual_ward validation suite: all checks passed")
        print(f"  effective target dims       : {n_dims}")
        print(f"  ||U^T M U - I||_2           : {err:.3e}")
        for embedding, (checked, total, frob, rsa) in summary.items():
            print(f"  [{embedding}] pairs checked      : {checked}")
            print(f"  [{embedding}] sum Delta (a=1)    : {total:.10f}")
            print(f"  [{embedding}] ||(I-Pi)Y||_F^2    : {frob:.10f}")
            print(f"  [{embedding}] exact RSA epsilon  : {rsa:.6f}")


def _leaf_members(result: SmoothDualWardResult) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for cid in np.unique(result.labels_):
        out[int(cid)] = np.where(result.labels_ == cid)[0]
    return out


if __name__ == "__main__":  # pragma: no cover
    run_validation_suite()
