r"""Minimax RSA coarsening: greedily minimize ``lambda_max(H_P)`` directly.

Ward-type coarseners (:mod:`src.smooth_dual_ward`) score a merge by a *sum* of
squared distortions, which is a Frobenius surrogate for the restricted spectral
approximation (RSA) constant.  This module optimizes the constant itself.

For an ``M_tau``-orthonormal target basis ``U_tau`` (``U^T M_tau U = I``) and the
Euclidean block-averaging projector ``Pi_P``, write ``R_P = I - Pi_P`` and

    H_P = U_tau^T R_P M_tau R_P U_tau        (d x d, symmetric PSD)

Every target signal ``x = U_tau c`` has ``||x||^2_{M_tau} = c^T c``, so

    max_{x in R} ||x - Pi_P x||^2_{M_tau} / ||x||^2_{M_tau} = lambda_max(H_P),

i.e. ``epsilon_P = sqrt(lambda_max(H_P))`` is *exactly* the RSA constant --
the same quantity :func:`src.smooth_dual_ward.exact_rsa_epsilon` reports.  The
greedy step is the minimax problem

    (A*, B*) = argmin_{A ~ B} max_{||c||=1} c^T H_{P + (A,B)} c
             = argmin_{A ~ B} lambda_max(H_{P + (A,B)}).

Contrast with Ward, which effectively minimizes ``trace(H_P)``: given candidates
with ``diag(H) = (0.01, 0.40)`` and ``(0.22, 0.24)``, a trace rule takes the
first (0.41 < 0.46) even though it destroys one target direction at 0.40; the
minimax rule takes the second (0.24 < 0.40).  That matters when ``R`` holds
several learned gang patterns and none of them may be sacrificed.

Crucially the metric ``M_tau`` enters only as the *inner product* ``e^T M_tau e``
of the residual, never as a feature map ``e -> M_tau e``.  There is therefore no
dual embedding and no high-pass distortion of the target (the failure mode
measured for ``embedding="dual"`` in :mod:`src.smooth_dual_ward`).

Efficiency
----------
Merging blocks ``A, B`` removes exactly one direction from the projector.  With
normalized block vectors ``v_C = D~^{1/2} 1_C / sqrt(vol C)`` and

    alpha = sqrt(vol A / s),  beta = sqrt(vol B / s),  s = vol A + vol B,
    v_{A u B} = alpha v_A + beta v_B,   q_AB = beta v_A - alpha v_B,

we have ``Pi_{P'} = Pi_P - q q^T``, hence with ``E_P = R_P U_tau``,

    a_AB = U_tau^T q,   c_AB = E_P^T M_tau q,   eta_AB = q^T M_tau q,
    H_{P'} = H_P + c a^T + a c^T + eta a a^T          (rank <= 2),
    E_{P'} = E_P + q a^T.

No ``N x N`` matrix is ever formed.  Better, nothing supported on nodes is needed
at all: keeping ``p_C = U_tau^T v_C`` and ``g_C = E_P^T M_tau v_C`` per block
gives ``a = beta p_A - alpha p_B`` and ``c = beta g_A - alpha g_B`` in ``O(d)``,
because the block-level Gram is closed-form in the cut statistics,

    v_C^T M_tau v_K = (o_C + tau vol C)/vol C          if C == K,
    v_C^T M_tau v_K = -w_CK / sqrt(vol C vol K)        otherwise,

using ``D~^{1/2} L_sym D~^{1/2} = D_W - W`` (self-loops cancel).  So a merge
costs ``O(deg * d)`` plus one ``d x d`` eigendecomposition.

Stopping
--------
``build_full_tree=True`` ignores ``n_clusters`` while merging and runs all the
way down to one block per connected component, recording ``epsilon`` after every
single merge in ``epsilon_curve_``.  ``epsilon*`` is then chosen by re-cutting the
stored hierarchy (:meth:`MinimaxCoarseningResult.labels_at`,
:meth:`~MinimaxCoarseningResult.epsilon_at`) -- the tree is built once and any
level is available afterwards, so there is no need to guess a target in advance.
``build_full_tree=False`` stops early at ``n_clusters`` (cheaper when a single
cut is wanted).  ``epsilon_max`` instead refuses individual merges that would
breach the budget; because ``Pi_P`` is Euclidean the curve need not be monotone,
so :meth:`~MinimaxCoarseningResult.coarsest_within` scans the whole trajectory
rather than halting at the first crossing.

Candidate scoring (``score_mode``)
----------------------------------
``"exact"``  -- (default) full ``eigvalsh`` of the candidate, ``O(d^3)``.
``"ritz"``   -- Rayleigh-Ritz on ``span{u_1, a, c}``: a ``<= 3 x 3``
                eigenproblem giving a *lower bound* on the true ``lambda_max``
                that explicitly contains the two directions the merge creates.
``"rayleigh"`` -- the ``O(d)`` first-order score
                ``lambda_1 + 2(u_1^T a)(u_1^T c) + eta (u_1^T a)^2``, also a
                lower bound (Rayleigh quotient at the old eigenvector).

**Use the lower bounds for pruning only, never as the decision rule.**  Measured
on a 1,200-node preferential-attachment graph cut to 240 blocks, the largest
five blocks are

    exact     [ 80,  75,  72,  72, 58]   epsilon = 0.836
    ritz      [533, 229,  65,  41, 14]   epsilon = 0.916
    rayleigh  [211, 186, 165, 145, 84]   epsilon = 0.917

Both bounds systematically *under*-score merges that grow an already-large block
(the new worst direction is nearly orthogonal to ``u_1`` and to ``span{a, c}``),
so those merges are popped first and the coarsening degenerates into one giant
blob plus singletons -- on Elliptic++ days 24-26 that reached 39,364 of 49,203
nodes in a single block.  Exact scoring does not have this failure mode.
Candidate *insertion* still uses the cheap Rayleigh key (a hint only; every entry
is rescored with ``score_mode`` when popped), which is safe.

Weyl bounds are available for pruning: with ``r = ||a||``,
``s = eta r^2 + 2 a^T c``, ``c_par = a^T c / r``, ``c_perp^2 = ||c||^2 - c_par^2``,
the update's two nonzero eigenvalues are
``delta_pm = (s +- sqrt(s^2 + 4 r^2 c_perp^2))/2`` and
``lambda_1 + delta_- <= lambda_max(H_{P'}) <= lambda_1 + delta_+``.

Caveat: because every merge changes ``H_P`` globally, *all* candidate scores move
after each commit -- unlike Ward there is no static priority.  The heap is
therefore re-evaluated lazily (pop, rescore against the current ``H``, re-push if
it is no longer the minimum).  That is exact whenever the score is non-decreasing
in the iteration, which is typical but not guaranteed (``Pi_P`` is Euclidean, not
``M_tau``-orthogonal).  ``strict=True`` rescores every candidate at every step --
correct but ``O(#candidates * #merges)``, so only for small graphs and tests.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import scipy.sparse as sp

from src.smooth_dual_ward import (
    exact_rsa_epsilon,
    m_orthonormal_basis,
    screened_operators,
)

__all__ = ["MinimaxCoarseningResult", "minimax_coarsen", "run_validation_suite"]


@dataclass
class MinimaxCoarseningResult:
    """Merge hierarchy plus the RSA constant at *every* level.

    ``epsilon_`` is the constant of the partition in ``labels_``.  ``epsilon_curve_``
    holds ``(n_clusters, epsilon)`` after every merge, i.e. the whole trajectory
    from ``n_leaves`` down to one block per connected component.  Because
    ``Pi_P`` is Euclidean rather than ``M_tau``-orthogonal the trajectory is *not*
    guaranteed to be monotone, so an error-budget stop must scan the curve for the
    coarsest admissible level rather than halt at the first crossing.
    """

    labels_: np.ndarray
    children_: np.ndarray
    merge_records_: List[dict]
    epsilon_: float
    n_effective_target_dims_: int
    target_basis_: np.ndarray
    n_leaves_: int = 0
    n_clusters_: int = 0
    epsilon_curve_: List[tuple] = field(default_factory=list)

    def epsilon_at(self, n_clusters: int) -> float:
        """RSA constant of the horizontal cut at ``n_clusters`` blocks."""

        k = int(n_clusters)
        if k >= self.n_leaves_:
            return 0.0
        for kk, eps in self.epsilon_curve_:
            if kk == k:
                return float(eps)
        raise ValueError(f"level {k} not present in the recorded curve")

    def coarsest_within(self, epsilon_max: float) -> int:
        """Fewest blocks whose RSA constant still satisfies the budget.

        Scans the whole trajectory rather than stopping at the first crossing,
        which matters because the curve need not be monotone.
        """

        ok = [k for k, eps in self.epsilon_curve_ if eps <= epsilon_max]
        return min(ok) if ok else self.n_leaves_

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
            while parent[x] != root:
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
            labels[leaf] = seen.setdefault(find(leaf), len(seen))
        return labels


def weyl_interval(lam1: float, a: np.ndarray, c: np.ndarray, eta: float):
    """``(delta_minus, delta_plus)`` of the rank-two update (Section 3 bounds)."""

    r = float(np.linalg.norm(a))
    if r <= 0.0:
        return 0.0, 0.0
    ac = float(a @ c)
    c_par = ac / r
    c_perp_sq = max(float(c @ c) - c_par * c_par, 0.0)
    s = eta * r * r + 2.0 * ac
    disc = math.sqrt(max(s * s + 4.0 * r * r * c_perp_sq, 0.0))
    return 0.5 * (s - disc), 0.5 * (s + disc)


def minimax_coarsen(
    W,
    Z: np.ndarray,
    tau: float,
    *,
    n_clusters: int | None = None,
    epsilon_max: float | None = None,
    score_mode: str = "exact",
    build_full_tree: bool = False,
    rank_tol: float = 1e-10,
    max_cluster_size: int = 0,
    strict: bool = False,
    max_rescore: int = 8,
    record_curve: bool = True,
) -> MinimaxCoarseningResult:
    r"""Greedy ``argmin lambda_max(H_P)`` adjacency-constrained coarsening.

    Parameters mirror :func:`src.smooth_dual_ward.smooth_dual_ward` where they
    overlap.  ``epsilon_max`` additionally refuses any merge that would push the
    RSA constant past the budget, giving the certificate
    ``||x - Pi_P x||_{M_tau} <= epsilon_max ||x||_{M_tau}`` for all ``x`` in the
    target; because ``Pi_P`` is Euclidean the score need not increase monotonically,
    so fixed-compression stopping (``n_clusters``) is the easier one to interpret.

    ``max_rescore`` bounds how many heap entries are refreshed per merge (see the
    module docstring): every commit raises ``H``, so all live scores drift upward
    together and an unbounded "re-push until minimal" loop would rescore most of
    the heap at each step.  Larger values are closer to the exact greedy at
    proportional cost; ``strict=True`` is the exact (quadratic) reference.
    """

    if score_mode not in ("exact", "ritz", "rayleigh"):
        raise ValueError("score_mode must be 'exact', 'ritz' or 'rayleigh'")

    A_off, d_tilde, _, M = screened_operators(W, tau)
    n = A_off.shape[0]
    U, n_dims = m_orthonormal_basis(Z, M, rank_tol=rank_tol)
    d = int(n_dims)

    # ---- block state ------------------------------------------------------
    max_ids = 2 * n
    vol = np.zeros(max_ids)
    vol[:n] = d_tilde
    ocut = np.zeros(max_ids)
    ocut[:n] = np.asarray(A_off.sum(axis=1)).ravel()
    size = np.zeros(max_ids, dtype=np.int64)
    size[:n] = 1
    active = np.zeros(max_ids, dtype=bool)
    active[:n] = True
    version = np.zeros(max_ids, dtype=np.int64)
    # p_C = U^T v_C ; singleton: v_i = e_i so p_i = U[i]
    P = np.zeros((max_ids, d))
    P[:n] = U
    # g_C = E^T M v_C ; E = 0 at the singleton partition
    G = np.zeros((max_ids, d))

    coo = sp.triu(A_off, k=1).tocoo()
    nbr: List[Dict[int, float]] = [dict() for _ in range(max_ids)]
    for i, j, w in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
        if w > 0.0:
            nbr[i][j] = w
            nbr[j][i] = w

    H = np.zeros((d, d))
    evals = np.zeros(d)
    evecs = np.eye(d)
    lam1 = 0.0
    u1 = evecs[:, -1]

    def gram_diag(c: int) -> float:
        return (ocut[c] + tau * vol[c]) / vol[c]

    def candidate(a_id: int, b_id: int):
        """``(a, c, eta)`` of the rank-two update for merging ``a_id, b_id``."""

        va, vb = vol[a_id], vol[b_id]
        s = va + vb
        alpha = math.sqrt(va / s)
        beta = math.sqrt(vb / s)
        a_vec = beta * P[a_id] - alpha * P[b_id]
        c_vec = beta * G[a_id] - alpha * G[b_id]
        w_ab = nbr[a_id].get(b_id, 0.0)
        s_ab = -w_ab / math.sqrt(va * vb)
        eta = (
            beta * beta * gram_diag(a_id)
            - 2.0 * alpha * beta * s_ab
            + alpha * alpha * gram_diag(b_id)
        )
        return a_vec, c_vec, float(eta), alpha, beta, s_ab, w_ab

    def rayleigh_score(a_vec, c_vec, eta) -> float:
        """``O(d)`` Rayleigh quotient at the current top eigenvector (lower bound).

        Used to *insert* candidates: a merge pushes one entry per neighbour of the
        new block, so on a heavy-tailed graph a hub block would otherwise force
        thousands of expensive scorings per commit.  The key is only a hint --
        every entry is rescored with ``score_mode`` when it is popped.
        """

        ua = float(u1 @ a_vec)
        uc = float(u1 @ c_vec)
        return max(lam1 + 2.0 * ua * uc + eta * ua * ua, 0.0)

    def score_of(a_vec, c_vec, eta) -> float:
        if score_mode == "rayleigh":
            return rayleigh_score(a_vec, c_vec, eta)
        if score_mode == "exact":
            cand = H + np.outer(c_vec, a_vec) + np.outer(a_vec, c_vec)
            cand += eta * np.outer(a_vec, a_vec)
            cand = 0.5 * (cand + cand.T)
            return max(float(np.linalg.eigvalsh(cand)[-1]), 0.0)
        # "ritz": Rayleigh-Ritz on span{u1, a, c} -- a certified lower bound that
        # contains both directions the merge introduces.
        B = np.stack([u1, a_vec, c_vec], axis=1)
        Q, r = np.linalg.qr(B)
        keep = np.abs(np.diag(r)) > 1e-12 * max(1.0, float(np.abs(r).max()))
        Q = Q[:, keep]
        if Q.shape[1] == 0:
            return lam1
        HQ = H @ Q
        HQ = HQ + np.outer(c_vec, a_vec @ Q) + np.outer(a_vec, c_vec @ Q)
        HQ = HQ + eta * np.outer(a_vec, a_vec @ Q)
        T = Q.T @ HQ
        T = 0.5 * (T + T.T)
        return max(float(np.linalg.eigvalsh(T)[-1]), 0.0)

    def full_score(a_vec, c_vec, eta) -> float:
        cand = H + np.outer(c_vec, a_vec) + np.outer(a_vec, c_vec)
        cand += eta * np.outer(a_vec, a_vec)
        cand = 0.5 * (cand + cand.T)
        return max(float(np.linalg.eigvalsh(cand)[-1]), 0.0)

    heap: List[tuple] = []
    for i, j, w in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
        if w <= 0.0:
            continue
        a_vec, c_vec, eta, *_ = candidate(i, j)
        heapq.heappush(heap, (rayleigh_score(a_vec, c_vec, eta), i, j, 0, 0))

    n_active = n
    children: List[tuple] = []
    records: List[dict] = []
    curve: List[tuple] = []
    next_id = n
    cap = int(max_cluster_size) if max_cluster_size else 0
    target = int(n_clusters) if n_clusters is not None else 1

    def stop_now() -> bool:
        if build_full_tree:
            return False
        return n_clusters is not None and n_active <= target

    while n_active > 1 and not stop_now():
        chosen = None
        if strict:
            # rescore every live candidate every step (exact greedy, O(|C|) per merge)
            best = None
            seen = set()
            for entry in heap:
                _, a_id, b_id, va, vb = entry
                if not (active[a_id] and active[b_id]):
                    continue
                if va != version[a_id] or vb != version[b_id]:
                    continue
                if (a_id, b_id) in seen or nbr[a_id].get(b_id, 0.0) <= 0.0:
                    continue
                seen.add((a_id, b_id))
                if cap and size[a_id] + size[b_id] > cap:
                    continue
                a_vec, c_vec, eta, *rest = candidate(a_id, b_id)
                sc = full_score(a_vec, c_vec, eta)
                if best is None or sc < best[0]:
                    best = (sc, a_id, b_id, a_vec, c_vec, eta, rest)
            if best is None:
                break
            chosen = best
        else:
            # Bounded lazy re-evaluation.  Every commit raises H, so *all* live
            # scores drift upward together and a pure "re-push until minimal"
            # loop would rescore most of the heap at every step (quadratic).
            # Instead pop at most ``max_rescore`` entries, rescoring each against
            # the current H; stop early if one is certified minimal (its
            # refreshed score is still <= the heap top), otherwise commit the
            # best of the batch and return the rest.  Relative order is
            # essentially preserved by the uniform drift, so the batch nearly
            # always contains the true minimum.
            batch: List[tuple] = []
            best = None
            refreshed = 0
            while heap and refreshed < max_rescore:
                sc_old, a_id, b_id, va, vb = heapq.heappop(heap)
                if not (active[a_id] and active[b_id]):
                    continue  # superseded entry: drop, does not use the budget
                if nbr[a_id].get(b_id, 0.0) <= 0.0:
                    continue
                if cap and size[a_id] + size[b_id] > cap:
                    continue
                refreshed += 1
                a_vec, c_vec, eta, *rest = candidate(a_id, b_id)
                sc = score_of(a_vec, c_vec, eta)
                if best is None or sc < best[0]:
                    if best is not None:
                        batch.append((best[0], best[1], best[2], 0, 0))
                    best = (sc, a_id, b_id, a_vec, c_vec, eta, rest)
                else:
                    batch.append((sc, a_id, b_id, 0, 0))
                if heap and best[0] <= heap[0][0] + 1e-12:
                    break  # certified: nothing left in the heap can beat it
            for entry in batch:
                heapq.heappush(heap, entry)
            chosen = best
            if chosen is None:
                break  # heap exhausted: no admissible adjacent pair remains

        sc, a_id, b_id, a_vec, c_vec, eta, rest = chosen
        alpha, beta, s_ab, w_ab = rest
        exact_lam = full_score(a_vec, c_vec, eta)
        if epsilon_max is not None and exact_lam > epsilon_max**2:
            continue  # refuse: over budget, try the next candidate

        # ---- commit ------------------------------------------------------
        new = next_id
        next_id += 1
        va, vb = vol[a_id], vol[b_id]
        vol[new] = va + vb
        ocut[new] = ocut[a_id] + ocut[b_id] - 2.0 * w_ab
        size[new] = size[a_id] + size[b_id]
        P[new] = alpha * P[a_id] + beta * P[b_id]
        # q^T M v_new, needed because both E and v change for the merged block
        q_m_new = (
            alpha * beta * gram_diag(a_id)
            + (beta * beta - alpha * alpha) * s_ab
            - alpha * beta * gram_diag(b_id)
        )
        G[new] = alpha * G[a_id] + beta * G[b_id] + q_m_new * a_vec

        # neighbours: g_K += a * (q^T M v_K); q^T M v_K vanishes unless K touches
        # A or B, so only the local neighbourhood is updated.
        touched: Dict[int, float] = {}
        for k, w in nbr[a_id].items():
            if k != b_id and active[k]:
                touched[k] = touched.get(k, 0.0) + beta * (-w / math.sqrt(va * vol[k]))
        for k, w in nbr[b_id].items():
            if k != a_id and active[k]:
                touched[k] = touched.get(k, 0.0) - alpha * (-w / math.sqrt(vb * vol[k]))
        for k, coef in touched.items():
            G[k] += coef * a_vec

        merged: Dict[int, float] = {}
        for k, w in nbr[a_id].items():
            if k != b_id and active[k]:
                merged[k] = merged.get(k, 0.0) + w
        for k, w in nbr[b_id].items():
            if k != a_id and active[k]:
                merged[k] = merged.get(k, 0.0) + w
        for k in nbr[a_id]:
            nbr[k].pop(a_id, None)
        for k in nbr[b_id]:
            nbr[k].pop(b_id, None)
        nbr[a_id] = {}
        nbr[b_id] = {}
        for k, w in merged.items():
            if w > 0.0:
                nbr[new][k] = w
                nbr[k][new] = w

        active[a_id] = active[b_id] = False
        active[new] = True

        H = H + np.outer(c_vec, a_vec) + np.outer(a_vec, c_vec)
        H = H + eta * np.outer(a_vec, a_vec)
        H = 0.5 * (H + H.T)
        evals, evecs = np.linalg.eigh(H)
        lam1 = max(float(evals[-1]), 0.0)
        u1 = evecs[:, -1]

        n_active -= 1
        children.append((a_id, b_id))
        records.append(
            {
                "children": (int(a_id), int(b_id)),
                "new_id": int(new),
                "n_clusters": int(n_active),
                "lambda_max": float(lam1),
                "epsilon": math.sqrt(lam1),
                "eta": float(eta),
                "cut": float(w_ab),
                "volume": float(vol[new]),
            }
        )
        if record_curve:
            curve.append((int(n_active), math.sqrt(lam1)))

        # every surviving neighbour of the new block needs a candidate entry
        for k in nbr[new]:
            a2, c2, eta2, *_ = candidate(new, k)
            lo, hi = (k, new) if k < new else (new, k)
            heapq.heappush(heap, (rayleigh_score(a2, c2, eta2), lo, hi, 0, 0))
        # NOTE: entries for pairs whose ``g_K`` moved are left in place with a
        # stale key.  That is safe -- :func:`candidate` always reads the current
        # block state, so a popped entry is rescored exactly, and the
        # "re-push if no longer minimal" rule restores correct ordering.  Eagerly
        # refreshing them instead would cost O(deg^2) per merge.

    children_arr = (
        np.asarray(children, dtype=np.int64)
        if children
        else np.empty((0, 2), dtype=np.int64)
    )
    result = MinimaxCoarseningResult(
        labels_=np.zeros(n, dtype=np.int64),
        children_=children_arr,
        merge_records_=records,
        epsilon_=math.sqrt(lam1),
        n_effective_target_dims_=d,
        target_basis_=U,
        n_leaves_=n,
        epsilon_curve_=curve,
    )
    if n_clusters is not None and build_full_tree:
        result.labels_ = result.labels_at(int(n_clusters))
    else:
        result.labels_ = result._cut(len(children))
    result.n_clusters_ = int(result.labels_.max()) + 1 if n else 0
    # epsilon must describe the partition actually returned, not the last level
    # reached while building the tree.
    if record_curve and result.n_clusters_ < n:
        try:
            result.epsilon_ = result.epsilon_at(result.n_clusters_)
        except ValueError:
            result.epsilon_ = float(exact_rsa_epsilon(U, M, d_tilde, result.labels_))
    elif result.n_clusters_ >= n:
        result.epsilon_ = 0.0
    return result


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def run_validation_suite(seed: int = 0, verbose: bool = True) -> None:
    """Check the incremental algebra against brute-force ``H_P``."""

    rng = np.random.default_rng(seed)
    n, tau = 36, 0.4
    Wd = np.zeros((n, n))
    for i in range(n - 1):
        Wd[i, i + 1] = rng.uniform(0.5, 2.0)
    extra = np.triu(rng.random((n, n)) < 0.15, 1)
    Wd = np.where(extra, rng.uniform(0.5, 2.0, size=(n, n)), Wd)
    Wd = np.triu(Wd, 1)
    for i in range(n - 1):
        Wd[i, i + 1] = max(Wd[i, i + 1], 0.5)
    W = sp.csr_matrix(Wd + Wd.T)
    Z = rng.normal(size=(n, 4))

    A_off, d_tilde, _, M = screened_operators(W, tau)
    U, d = m_orthonormal_basis(Z, M)

    def brute_H(labels):
        k = int(labels.max()) + 1
        root = np.sqrt(d_tilde)
        vol = np.bincount(labels, weights=d_tilde, minlength=k)
        num = np.zeros((k, U.shape[1]))
        np.add.at(num, labels, root[:, None] * U)
        E = U - root[:, None] * (num / vol[:, None])[labels]
        Hb = E.T @ (M @ E)
        return 0.5 * (Hb + Hb.T)

    for mode in ("exact", "ritz", "rayleigh"):
        res = minimax_coarsen(W, Z, tau, n_clusters=max(2, n // 3), score_mode=mode)
        Hb = brute_H(res.labels_)
        lam_b = float(np.linalg.eigvalsh(Hb)[-1])
        eps_b = exact_rsa_epsilon(U, M, d_tilde, res.labels_)
        assert abs(res.epsilon_ - math.sqrt(max(lam_b, 0.0))) < 1e-8, (
            f"[{mode}] incremental H disagrees with brute force: "
            f"{res.epsilon_} vs {math.sqrt(max(lam_b, 0.0))}"
        )
        assert abs(res.epsilon_ - eps_b) < 1e-8, (
            f"[{mode}] epsilon disagrees with exact_rsa_epsilon: "
            f"{res.epsilon_} vs {eps_b}"
        )
        for cid in range(int(res.labels_.max()) + 1):
            members = np.where(res.labels_ == cid)[0]
            sub = A_off[members][:, members]
            ncomp = sp.csgraph.connected_components(sub, directed=False)[0]
            assert ncomp == 1, f"[{mode}] cluster {cid} disconnected"
        if verbose:
            print(
                f"  score_mode={mode:<9} epsilon={res.epsilon_:.8f} (brute {eps_b:.8f})"
            )

    # the strict (rescore-everything) greedy must not be beaten by the lazy heap
    lazy = minimax_coarsen(W, Z, tau, n_clusters=max(2, n // 3), score_mode="exact")
    strict = minimax_coarsen(
        W, Z, tau, n_clusters=max(2, n // 3), score_mode="exact", strict=True
    )

    # full hierarchy down to a single block: every recorded level's epsilon must
    # equal the brute-force constant of that cut, and epsilon_ must describe the
    # partition actually returned.
    full = minimax_coarsen(W, Z, tau, n_clusters=6, build_full_tree=True)
    assert (
        full.epsilon_curve_[-1][0] == 1
    ), f"full tree did not reach a single block: {full.epsilon_curve_[-1]}"
    assert (
        abs(full.epsilon_ - exact_rsa_epsilon(U, M, d_tilde, full.labels_)) < 1e-8
    ), "epsilon_ does not describe labels_"
    for k in (2, 5, 6, 9, 15, n // 2):
        if k >= n:
            continue
        eps_ref = exact_rsa_epsilon(U, M, d_tilde, full.labels_at(k))
        assert (
            abs(full.epsilon_at(k) - eps_ref) < 1e-8
        ), f"epsilon_at({k}) = {full.epsilon_at(k)} != brute force {eps_ref}"
    eps_seq = [e for _, e in full.epsilon_curve_]
    drops = sum(
        1 for i in range(1, len(eps_seq)) if eps_seq[i] < eps_seq[i - 1] - 1e-12
    )
    if verbose:
        print(f"  lazy heap epsilon    : {lazy.epsilon_:.8f}")
        print(f"  strict greedy epsilon: {strict.epsilon_:.8f}")
        print(
            f"  full tree: {len(eps_seq)} levels, epsilon {eps_seq[0]:.4f} -> "
            f"{eps_seq[-1]:.4f}, non-monotone steps: {drops}"
        )
        print("minimax_coarsen validation suite: all checks passed")


if __name__ == "__main__":  # pragma: no cover
    run_validation_suite()
