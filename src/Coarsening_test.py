"""Empirical confirmation of Corollary 4.7 (planted Erdos-Renyi motif).

Corollary 4.7 says: plant a size-``s`` subset ``S`` whose internal pairs are
``Bern(p)`` inside an ``N``-node graph whose every other pair (host-host and
S-host) is ``Bern(q)``, with ``p > q``.  With
``delta = p(s-1)``, ``b = q(N-s)``, ``dbar = delta + b + 1``, ``Dbar = q(N-s)+1``,
the M_0-energy measure ``nu`` of the degree-weighted indicator ``v_S`` has, up to
``(1+o(1))``:

    Phi   = b / dbar                                            (conductance = mean)
    m1t   = Phi + (1-Phi)/dbar + s/N + 1/Dbar                   (location m_tilde_1)
    sig2 <= (1-Phi)/dbar + 2/Dbar                               (variance)

and, for the eigenspace target ``R = span(U_K)`` the retained energy
``C_K = sum_{k<K} q_k`` obeys the tail bounds (tau = 0)

    1 - C_K <= m1t / lambda_K                        (Markov)
    1 - C_K <= sig2 / (sig2 + (lambda_K - m1t)^2)    (Cantelli, lambda_K > m1t)

with ``lambda_K`` the ``K``-th smallest Laplacian eigenvalue (the lowest frequency
left in the complement of ``span(U_K)``).

This script builds the graph, computes ``C_K`` exactly from the eigendecomposition
of ``L = I - Ahat`` (self-loop renormalized, matching the paper), and checks
(i) the empirical moments match the closed forms and (ii) the empirical
``1 - C_K`` stays under both bounds, for every valid ``K`` and every random seed.

Run::

    python -m src.run_corollary_4_7 --N 1500 --s 60 --p 0.7 --q 0.012 --seeds 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# torch and scikit-learn each bundle their own libomp; on macOS the duplicate
# OpenMP runtime aborts with a "trace trap" when both are loaded (e.g. the Ward
# coarsening method).  Allow the duplicate load (set before torch/sklearn import).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

from arrow import now
import numpy as np
from src.utils.utils import *

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAVE_PLT = True
except Exception:  # pragma: no cover - plotting is optional
    _HAVE_PLT = True


# --------------------------------------------------------------------------- #
# 1-2.  random host graph G(N, q) with a planted G(s, p) motif
# --------------------------------------------------------------------------- #
def build_planted_er(N: int, s: int, p: float, q: float, rng: np.random.Generator):
    """Symmetric 0/1 adjacency: internal S-pairs Bern(p), all others Bern(q).

    ``S`` is the first ``s`` nodes.  No self-loops in the raw adjacency (they are
    added later by the renormalization ``W_tilde = W + I``).
    """

    prob = np.full((N, N), float(q))
    prob[:s, :s] = float(p)  # planted dense block
    upper = (rng.random((N, N)) < prob).astype(np.float64)
    upper = np.triu(upper, 1)  # keep strict upper triangle
    W = upper + upper.T  # symmetric, zero diagonal
    return W


# --------------------------------------------------------------------------- #
# 3-4.  span(U_K) target and the retained energy C_K = sum_{k<K} q_k^tau
# --------------------------------------------------------------------------- #
def operators(W: np.ndarray):
    """Self-loop renormalized ``Ahat = D_t^{-1/2}(W+I)D_t^{-1/2}`` and ``L = I-Ahat``."""

    N = W.shape[0]
    Wt = W + np.eye(N)
    dt = Wt.sum(1)  # d_tilde = degree + 1 (self-loop augmented)
    dinv = 1.0 / np.sqrt(dt)
    Ahat = (dinv[:, None] * Wt) * dinv[None, :]
    L_sym = np.eye(N) - Ahat
    L_sym = 0.5 * (L_sym + L_sym.T)  # symmetrize against round-off
    # LL = np.diag(dt) - Wt  # unnormalized Laplacian (for comparison)
    return L_sym, dt


def degree_indicator(dt: np.ndarray, s: int) -> np.ndarray:
    """Degree-weighted gang indicator ``v_S = D_t^{1/2} 1_S / sqrt(vol(S))``."""

    v = np.zeros_like(dt)
    v[:s] = np.sqrt(dt[:s])
    v /= np.sqrt(dt[:s].sum())  # vol(S) = sum_{i in S} d_tilde_i ; ||v_S|| = 1
    return v


def spectral_capture(L: np.ndarray, v: np.ndarray, tau: float):
    """Eigendecompose ``L`` and return the M_tau energy measure and ``C_K`` curve.

    Returns ``lam`` (ascending eigenvalues), ``Phi`` (= m_1 = ||v||_L^2), the
    location ``m1t`` and variance ``sig2`` of the measure ``nu^tau``, the per-mode
    weights ``q`` (a probability vector), and the cumulative capture ``C`` with
    ``C[K-1] = C_K = sum_{k<K} q_k``.
    """

    lam, U = np.linalg.eigh(L)  # ascending; L symmetric PSD
    lam = np.clip(lam, 0.0, None)
    pk = (U.T @ v) ** 2  # p_k = (u_k^T v_S)^2 , sum_k p_k = ||v||^2 = 1
    Phi = float((lam * pk).sum())  # m_1 = v^T L v = conductance
    # M_tau spectral energy weights q_k = (lam_k + tau) p_k / (Phi + tau)
    qk = (lam + tau) * pk / (Phi + tau)
    qk = qk / qk.sum()  # normalize against round-off (sum is 1 in exact arithmetic)
    m1t = float((lam * qk).sum())  # m_tilde_1 (mean of nu^tau)
    m2t = float((lam**2 * qk).sum())
    sig2 = float(m2t - m1t**2)  # variance of nu^tau
    C = np.cumsum(qk)  # C[K-1] = C_K
    return lam, Phi, m1t, sig2, qk, C


def theoretical_moments(N: int, s: int, p: float, q: float) -> dict:
    """Closed forms of Corollary 4.7, eq. (20) (the tau = 0 statistics)."""

    delta = p * (s - 1)  # expected internal degree
    b = q * (N - s)  # expected boundary degree
    dbar = delta + b + 1.0  # self-loop augmented gang degree
    Dbar = q * (N - s) + 1.0  # self-loop augmented host degree
    Phi = b / dbar
    m1t = Phi + (1.0 - Phi) / dbar + s / N + 1.0 / Dbar
    sig2 = (1.0 - Phi) / dbar + 2.0 / Dbar
    return {
        "delta": delta,
        "b": b,
        "dbar": dbar,
        "Dbar": Dbar,
        "Phi": Phi,
        "m1t": m1t,
        "sig2": sig2,
        "min_deg": min(delta, b),
        "logN": np.log(N),
    }


# --------------------------------------------------------------------------- #
# 6.  coarsening sweep: span(U_K) target -> precision-recall AUC over levels
# --------------------------------------------------------------------------- #
def _incremental_pr_curve(
    adjacency,
    U_K,
    pattern,
    y_labels,
    tau: float,
    max_levels: int,
    eval_threshold: float,
    stop_precision: float,
    helpers,
):
    """Coarsen one Loukas level at a time, recording (recall, precision) per level.

    A single edge-matching per level (``n_target=1``) contracts as many disjoint
    edges as possible, so each level roughly halves the node count -- exactly one
    level of Loukas Algorithm 1.  The intermediate coarsened graph / basis are
    *reused* between levels, so tracing the whole precision-recall trajectory
    costs one full coarsening run (not one run per stopping level).  The sweep
    stops as soon as precision drops below ``stop_precision`` (or the graph can no
    longer be contracted).  This avoids any reduction/epsilon stopping rule: the
    trajectory itself is the object of interest, summarised later by its AUC.
    """
    import math

    (
        torch,
        _normalized_laplacian,
        _screened_metric,
        _edge_partition,
        _reduce_adjacency,
        _reduce_basis,
        evaluate_loukas_patterns,
    ) = helpers

    def laplacian_fn(adj):
        return _screened_metric(_normalized_laplacian(adj), tau)

    n_original = adjacency.shape[0]
    current_adjacency, basis = adjacency, U_K
    original_to_current = torch.arange(n_original)

    recalls, precisions, n_coarses = [], [], []
    for _level in range(max_levels):
        n_current = current_adjacency.shape[0]
        if n_current <= 2:
            break
        # n_target=1 => contract as many disjoint edges as possible this level
        groups, _sigma = _edge_partition(
            current_adjacency,
            basis,
            1,
            math.inf,
            laplacian_fn=laplacian_fn,
            tau=tau,
        )
        n_new = int(groups.max().item()) + 1
        if n_new >= n_current:
            break  # no further contraction possible

        original_to_current = groups[original_to_current]
        current_adjacency = _reduce_adjacency(current_adjacency, groups)
        basis = _reduce_basis(basis, groups)

        _, dense_ids = torch.unique(
            original_to_current, sorted=True, return_inverse=True
        )
        _, by_label = evaluate_loukas_patterns(
            [pattern], dense_ids, y_labels, threshold=eval_threshold
        )
        g = by_label.get("alert", {})
        p = g.get("mean_precision", 0.0) or 0.0
        r = g.get("mean_recall", 0.0) or 0.0
        recalls.append(r)
        precisions.append(p)
        n_coarses.append(n_new)
        if p < stop_precision:
            break

    return np.array(recalls), np.array(precisions), np.array(n_coarses)


def _sequential_edge_variation_pr_curve(
    adjacency,
    U_K,
    pattern,
    y_labels,
    tau: float,
    max_contractions: int,
    eval_threshold: float,
    stop_precision: float,
    helpers,
    epsilon_budget: float = float("inf"),
    return_history: bool = False,
    refresh_every: int = 1,
    self_loop_aware: bool = False,
    exact_epsilon_every: int = 1,
    chained: bool = True,
    max_cluster_size: int = 0,
):
    r"""Sequential Loukas edge-variation coarsening, ``refresh_every`` edges/level.

    Each level contracts up to ``refresh_every`` edges, then refreshes the target
    embedding ``A = B (B^T L B)^{+1/2}`` from the *updated* graph, recomputes the
    exact RSA epsilon, and continues.  Two ways to pick the edges of a level:

    * ``chained=True`` (default) -- :func:`_chained_edge_partition`: cheapest-first
      union-find with **no matching constraint**, so a vertex may be contracted
      several times in one level (agglomerative chaining, like Ward).  Exactly
      ``n_current - n_target`` merges happen, so the graph reduces by exactly
      ``refresh_every`` nodes per level regardless of how the cheap edges overlap
      -- a just-merged node is free to keep absorbing partners instead of being
      locked out until the next level.  This is what the caller asked for.
    * ``chained=False`` -- :func:`_edge_partition`: the original maximal
      cheapest-first *matching* (each vertex contracted at most once per level),
      so a level reduces by at most ``n_current / 2`` and possibly fewer than
      ``refresh_every`` when the cheap edges share endpoints.

    ``refresh_every`` is the speed / faithfulness knob (the "middle ground"):

    * ``1`` is the exact one-edge-per-refresh method: every contraction is scored
      against a freshly rebuilt embedding.  Most faithful, ``O(n)`` refreshes,
      slowest.  (With ``chained`` this is identical to the matching version --
      one edge cannot chain.)
    * ``b > 1`` refreshes once per ``b`` contractions, amortising the Laplacian
      rebuild + embedding recompute (the real cost -- see below) over ``b`` edges.
      ``b = inf`` collapses the whole graph in a single scored level.

    ``max_cluster_size > 0`` caps how many original nodes a supernode may hold
    (only meaningful with ``chained``; ``0`` = unlimited), a guard against
    single-linkage chaining swallowing the graph within one level.

    Note on cost: the per-step bottleneck is rebuilding the sparse Laplacian and
    recomputing every edge cost on the whole graph, *not* the eigendecomposition
    -- that is a ``K x K`` Gram ``eigh`` (K = target-subspace width), which is
    negligible.  ``refresh_every`` attacks the real cost.

    Stopping is driven by the **exact** restricted-spectral-approximation (RSA)
    constant when ``self_loop_aware`` or a finite ``epsilon_budget`` is given:
    every ``exact_epsilon_every`` contractions the cumulative distortion

        epsilon_exact = max_{x in R} ||x - Pi x||_{M} / ||x||_{M}
                      = _exact_rsa_epsilon(a0, L0, original_to_current)

    is measured against the *original* graph (``a0``, ``L0`` are formed once),
    which is the tight quantity Loukas' Theorem bounds -- not the looser product
    ``prod_l (1 + sigma_l) - 1`` (kept as ``epsilon_bound`` in the history for
    reference only).

    ``self_loop_aware`` switches the reduction to the volume-preserving
    ``W_c = S^T W S`` form: each supernode stores its internal weight on the
    diagonal (``d_s = d_u + d_v``), and the metric becomes the self-loop-aware
    normalized Laplacian ``I - D^{-1/2} W_c D^{-1/2}`` (no renormalization
    trick).  ``False`` keeps the legacy renorm-normalized metric byte-for-byte.

    ``max_contractions`` is the number of single-edge contractions, not the
    number of matching levels.
    """
    import math

    (
        torch,
        _normalized_laplacian,
        _screened_metric,
        _edge_partition,
        _reduce_adjacency,
        _reduce_basis,
        _l_orthonormalize,
        evaluate_loukas_patterns,
        _weighted_normalized_laplacian,
        _exact_rsa_epsilon,
        _chained_edge_partition,
    ) = helpers

    base_laplacian = (
        _weighted_normalized_laplacian if self_loop_aware else _normalized_laplacian
    )

    def laplacian_fn(adj):
        return _screened_metric(base_laplacian(adj), tau)

    def as_float(value) -> float:
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    n_original = int(adjacency.shape[0])
    current_adjacency = adjacency.coalesce()
    basis = U_K
    original_to_current = torch.arange(
        n_original, device=current_adjacency.device, dtype=torch.long
    )

    # Exact-RSA reference: the L-orthonormal target basis and metric of the
    # *original* graph.  Only the partition ``original_to_current`` changes as we
    # coarsen, so these are formed once and reused for every epsilon check.
    track_exact = self_loop_aware or math.isfinite(epsilon_budget)
    a0 = None
    original_laplacian = None
    if track_exact:
        original_laplacian = laplacian_fn(current_adjacency)
        try:
            a0 = _l_orthonormalize(basis, original_laplacian)
        except ValueError:
            a0 = None
        if a0 is None or a0.shape[1] == 0:
            track_exact = False

    batch = max(1, int(refresh_every))
    check_every = max(1, int(exact_epsilon_every))

    recalls: list[float] = []
    precisions: list[float] = []
    n_coarses: list[int] = []
    history: list[dict] = []

    epsilon_bound = 0.0  # loose product estimate prod (1 + sigma_l) - 1
    epsilon_exact = 0.0
    n_done = 0

    while n_done < max_contractions:
        n_current = int(current_adjacency.shape[0])
        if n_current <= 2:
            break

        # Stopping is governed by the EXACT RSA constant (checked after the
        # contraction), not by the per-edge product bound -- that estimate is
        # hopelessly loose (it can reach 1e50+ while the exact epsilon is ~1), so
        # using it to pre-prune the greedy would truncate the sweep far too
        # early.  The greedy therefore always contracts the cheapest ``want``
        # edges (sigma_limit = inf) and the exact gate decides when to stop.
        want = min(batch, max_contractions - n_done)
        target_n = n_current - want
        try:
            if chained:
                # Union-find chaining: reduces by exactly ``want`` nodes/level,
                # a just-merged node may keep absorbing partners (no matching
                # lock-out).  ``merges`` is this level's dendrogram.
                groups, sigma_step_raw, merges = _chained_edge_partition(
                    current_adjacency,
                    basis,
                    target_n,
                    math.inf,
                    laplacian_fn=laplacian_fn,
                    tau=tau,
                    max_cluster_size=max_cluster_size,
                )
            else:
                groups, sigma_step_raw = _edge_partition(
                    current_adjacency,
                    basis,
                    target_n,
                    math.inf,
                    laplacian_fn=laplacian_fn,
                    tau=tau,
                )
                merges = None
        except ValueError:
            break  # degenerate target subspace: no positive-energy direction

        groups = groups.to(dtype=torch.long)
        n_new = int(groups.max().item()) + 1
        if n_new >= n_current:
            # No admissible edge, usually because of the sigma budget.
            break
        n_contracted = n_current - n_new

        sigma_step = as_float(sigma_step_raw)
        epsilon_bound = (1.0 + epsilon_bound) * (1.0 + sigma_step) - 1.0

        # Compose the original -> coarse map and apply the contraction.  The
        # reduced basis feeds the next refresh; with ``self_loop_aware`` the
        # diagonal of ``W_c`` accumulates each supernode's internal weight.
        original_to_current = groups[original_to_current]
        current_adjacency = _reduce_adjacency(
            current_adjacency, groups, keep_self_loops=self_loop_aware
        ).coalesce()
        basis = _reduce_basis(basis, groups)
        n_done += n_contracted

        # Dense labels are used both for pattern evaluation and the exact RSA.
        _, dense_ids = torch.unique(
            original_to_current, sorted=True, return_inverse=True
        )
        _, by_label = evaluate_loukas_patterns(
            [pattern], dense_ids, y_labels, threshold=eval_threshold
        )
        gang_stats = by_label.get("alert", {})
        precision = float(gang_stats.get("mean_precision", 0.0) or 0.0)
        recall = float(gang_stats.get("mean_recall", 0.0) or 0.0)

        # Exact cumulative RSA distortion of the coarsening so far.
        if track_exact and (
            (n_done % check_every == 0) or math.isfinite(epsilon_budget)
        ):
            epsilon_exact = _exact_rsa_epsilon(a0, original_laplacian, dense_ids)

        recalls.append(recall)
        precisions.append(precision)
        n_coarses.append(n_new)

        history.append(
            {
                "step": n_done,
                "n_before": n_current,
                "n_after": n_new,
                "n_contracted": n_contracted,
                "sigma_step": sigma_step,
                "epsilon_bound": epsilon_bound,  # loose product estimate
                "epsilon_exact": epsilon_exact,  # tight RSA constant
                "precision": precision,
                "recall": recall,
                # This level's dendrogram (chained mode): ordered (a, b)
                # current-node endpoints contracted, i.e. which nodes combined.
                "merges": merges,
            }
        )

        if precision < stop_precision:
            break
        if math.isfinite(epsilon_budget) and epsilon_exact >= epsilon_budget:
            break

    result = (
        np.asarray(recalls, dtype=float),
        np.asarray(precisions, dtype=float),
        np.asarray(n_coarses, dtype=int),
    )
    if return_history:
        return (*result, history)
    return result


def _labels_from_tree(children, n_samples: int, k: int):
    """Cut an agglomerative merge tree to ``k`` clusters -> contiguous labels.

    ``children`` is the sklearn ``children_`` array: row ``i`` records the two
    nodes merged to form node ``n_samples + i`` (leaves are ``0..n_samples-1``).
    Cutting to ``k`` clusters applies the first ``n_samples - k`` merges; a
    path-compressed union-find then labels every leaf by its surviving root.
    """
    n_merges = max(0, n_samples - k)
    parent = np.arange(n_samples + len(children))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    for i in range(n_merges):
        a, b = int(children[i, 0]), int(children[i, 1])
        parent[find(a)] = n_samples + i
        parent[find(b)] = n_samples + i

    roots: dict[int, int] = {}
    labels = np.empty(n_samples, dtype=np.int64)
    for leaf in range(n_samples):
        r = find(leaf)
        labels[leaf] = roots.setdefault(r, len(roots))
    return labels


def _ward_pr_curve(
    adjacency,
    U_K,
    pattern,
    y_labels,
    tau: float,
    max_levels: int,
    eval_threshold: float,
    stop_precision: float,
    helpers,
):
    r"""Contiguity-constrained Ward agglomeration on the rows of ``A = U_K Lam_K^{+1/2}``.

    Loukas' edge greedy fixes each merge cost from *pairwise* differences on the
    coarse graph and can therefore compound boundary errors.  Ward instead keeps
    a partition into connected clusters and repeatedly merges the *adjacent* pair
    minimising the Ward increment

        Delta(C1, C2) = |C1||C2| / (|C1|+|C2|) * || abar_C1 - abar_C2 ||^2,

    which is exactly the increase of ``||A - Pi_P A||_F^2`` caused by the merge.
    The connectivity constraint (original graph adjacency) keeps every
    contraction set connected, so the result is a legal Laplacian-consistent
    coarsening.  At the singleton stage the increment reduces to Loukas' edge
    cost -- the greedy *is* Ward's first level -- but thereafter Ward compares
    centroids over *all* absorbed rows, so a growing gang fragment's boundary
    statistic averages ``m`` rows and its noise shrinks by ``1/sqrt(m)``
    (noise annealing), the opposite of the greedy's compounding contamination.

    The full merge tree is built once with ``A`` computed on the *original*
    graph; the precision-recall trajectory is read off by cutting the tree at a
    geometric schedule of cluster counts (fine -> coarse) until precision drops
    below ``stop_precision``.
    """
    (
        torch,
        _normalized_laplacian,
        _screened_metric,
        _l_orthonormalize,
        evaluate_loukas_patterns,
    ) = helpers

    try:
        from sklearn.cluster import AgglomerativeClustering
        from scipy.sparse import csr_matrix
    except ImportError:
        return np.array([]), np.array([]), np.array([])

    def laplacian_fn(adj):
        return _screened_metric(_normalized_laplacian(adj), tau)

    n = adjacency.shape[0]
    # L-orthonormal embedding rows A (drops the zero-energy/constant direction);
    # rows of this A are exactly the objects Ward's Frobenius objective acts on.
    try:
        A = _l_orthonormalize(U_K, laplacian_fn(adjacency)).cpu().numpy()
    except ValueError:
        return np.array([]), np.array([]), np.array([])
    if A.shape[1] == 0:
        return np.array([]), np.array([]), np.array([])

    # connectivity = original adjacency sparsity -> merges stay connected
    idx = adjacency.indices().cpu().numpy()
    conn = csr_matrix(
        (np.ones(idx.shape[1], dtype=np.float64), (idx[0], idx[1])), shape=(n, n)
    )

    try:
        model = AgglomerativeClustering(
            n_clusters=2,
            linkage="ward",
            connectivity=conn,
            compute_full_tree=True,
        ).fit(A)
    except Exception:  # pragma: no cover - sklearn/version edge cases
        return np.array([]), np.array([]), np.array([])
    children = np.asarray(model.children_)

    # cluster-count schedule (fine -> coarse), matched in count to the greedy's
    # levels so the two methods trace comparably-sampled PR trajectories
    n_pts = max(2 * max_levels, 20)
    ns = np.unique(np.round(np.geomspace(2, n - 1, n_pts)).astype(int))
    ns = ns[(ns >= 2) & (ns < n)][::-1]  # descending: many clusters -> few

    recalls, precisions, n_coarses = [], [], []
    for k in ns.tolist():
        labels = _labels_from_tree(children, n, int(k))
        node_to_super = torch.from_numpy(labels)
        _, by_label = evaluate_loukas_patterns(
            [pattern], node_to_super, y_labels, threshold=eval_threshold
        )
        g = by_label.get("alert", {})
        p = g.get("mean_precision", 0.0) or 0.0
        r = g.get("mean_recall", 0.0) or 0.0
        recalls.append(r)
        precisions.append(p)
        n_coarses.append(int(k))
        if p < stop_precision:
            break

    return np.array(recalls), np.array(precisions), np.array(n_coarses)


def _pr_auc(recalls, precisions):
    """Area under the precision-recall trajectory + best-F1 operating point.

    Points are ordered by recall and integrated trapezoidally from a
    ``recall = 0`` anchor.  Returns ``(auc, best_precision, best_recall,
    best_f1)``; the best-F1 point supplies the representative precision / recall
    scalars plotted alongside the AUC.  NaN when the curve is empty.
    """
    if len(recalls) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    r = np.asarray(recalls, dtype=float)
    p = np.asarray(precisions, dtype=float)
    order = np.argsort(r)
    r_s, p_s = r[order], p[order]
    r_int = np.concatenate([[0.0], r_s])  # anchor the integral at recall = 0
    p_int = np.concatenate([[p_s[0]], p_s])
    trapz = getattr(np, "trapezoid", np.trapz)
    auc = float(trapz(p_int, r_int))
    denom = p + r
    f1 = np.where(denom > 0, 2.0 * p * r / np.where(denom > 0, denom, 1.0), 0.0)
    bi = int(np.argmax(f1))
    return auc, float(p[bi]), float(r[bi]), float(f1[bi])


def run_coarsening_sweep(args) -> None:
    """Sweep K over `coarsen_n_seeds` independent planted-ER instances.

    For each (seed, K) pair the graph is coarsened one Loukas level at a time;
    precision and recall are recorded after every level (see
    :func:`_incremental_pr_curve`) until precision collapses.  The trajectory is
    summarised by its precision-recall AUC (and the best-F1 operating point).
    Averaging over seeds, the sweep reports AUC / precision / recall against
    ``K``, ``C_K`` and ``lambda_K / Phi`` -- confirming that more retained energy
    (up to the sharp threshold) yields better motif contraction.
    """
    try:
        import torch
        from src.loukas_sgc_detection import (
            _normalized_laplacian,
            _screened_metric,
            _edge_partition,
            _reduce_adjacency,
            _reduce_basis,
            _l_orthonormalize,
            evaluate_loukas_patterns,
            _weighted_normalized_laplacian,
            _exact_rsa_epsilon,
            _chained_edge_partition,
        )
        from src.pattern_models import create_pattern
    except ImportError as exc:
        LOGGER.info(f"  run_coarsening_sweep requires torch + src package: {exc}")
        return

    use_ward = args.coarsen_method == "ward"
    use_sequential_edges = args.coarsen_method == "sequential_edges"
    helpers = (
        torch,
        _normalized_laplacian,
        _screened_metric,
        _edge_partition,
        _reduce_adjacency,
        _reduce_basis,
        evaluate_loukas_patterns,
    )
    ward_helpers = (
        torch,
        _normalized_laplacian,
        _screened_metric,
        _l_orthonormalize,
        evaluate_loukas_patterns,
    )
    sequential_helpers = (
        torch,
        _normalized_laplacian,
        _screened_metric,
        _edge_partition,
        _reduce_adjacency,
        _reduce_basis,
        _l_orthonormalize,
        evaluate_loukas_patterns,
        _weighted_normalized_laplacian,
        _exact_rsa_epsilon,
        _chained_edge_partition,
    )
    if use_ward:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            LOGGER.info("  --coarsen-method ward requires scikit-learn; aborting sweep")
            return

    th = theoretical_moments(args.N, args.s, args.p, args.q)
    algo = (
        "Ward agglomeration"
        if use_ward
        else (
            "sequential one-edge Loukas variation"
            if use_sequential_edges
            else "edge-greedy matching"
        )
    )
    LOGGER.info("=" * 78)
    LOGGER.info(
        f"Coarsening sweep: level-by-level precision-recall AUC vs span(U_K) "
        f"[{algo}]"
    )
    LOGGER.info(
        f"  N={args.N}  s={args.s}  p={args.p}  q={args.q}  tau={args.tau}\n"
        f"  theory: Phi={th['Phi']:.4f}  m1t={th['m1t']:.4f}\n"
        f"  stop_precision={args.coarsen_stop_precision:.0%}  "
        f"n_seeds={args.coarsen_n_seeds}  method={args.coarsen_method}  "
        f"max_levels={args.coarsen_max_levels}"
    )
    LOGGER.info("=" * 78)

    # fixed gang pattern and node labels (same structure across seeds)
    pattern = create_pattern("gang_0", list(range(args.s)), "er_motif", label="alert")
    y_labels = torch.zeros(args.N, dtype=torch.long)
    y_labels[: args.s] = 1

    # K grid based on theoretical Phi (fixed across seeds for comparability)
    # Use the first seed to determine K_thresh and C_K
    rng0 = np.random.default_rng(args.seed0)
    W0 = build_planted_er(args.N, args.s, args.p, args.q, rng0)
    L0, dt0 = operators(W0)
    v0 = degree_indicator(dt0, args.s)
    lam0, U0 = np.linalg.eigh(L0)
    lam0 = np.clip(lam0, 0.0, None)
    pk0 = (U0.T @ v0) ** 2
    Phi0 = float((lam0 * pk0).sum())
    qk0 = (lam0 * pk0) / max(Phi0, 1e-12)
    qk0 /= qk0.sum()
    C_all0 = np.cumsum(qk0)
    K_thresh0 = max(2, min(int(np.searchsorted(lam0, Phi0)), args.N - 2))

    K_max = min(args.N - 2, args.coarsen_k_max)
    Ks_geo = (
        np.unique(np.round(np.geomspace(2, K_max, args.coarsen_k_points)).astype(int))
        if K_max >= 2
        else np.array([2], dtype=int)
    )
    step = max(1, 60 // 20)
    Ks_dense = np.arange(max(2, K_thresh0 - 30), min(K_max, K_thresh0 + 30) + 1, step)
    Ks = np.unique(np.concatenate([[2], Ks_geo, Ks_dense]))
    Ks = Ks[(Ks >= 2) & (Ks <= K_max)]

    LOGGER.info(
        f"  Phi_emp(seed0)={Phi0:.4f}  K_thresh(seed0)={K_thresh0}  "
        f"({len(Ks)} K values)]"
    )

    # ---- main loop: seeds -------------------------------------------------
    # per_K_*[i] = list of scalar summaries over seeds for Ks[i]
    per_K_auc: list[list] = [[] for _ in Ks]
    per_K_prec: list[list] = [[] for _ in Ks]
    per_K_recall: list[list] = [[] for _ in Ks]
    per_K_lam: list[list] = [[] for _ in Ks]
    per_K_CK: list[list] = [[] for _ in Ks]

    for si in range(args.coarsen_n_seeds):
        rng = np.random.default_rng(args.seed0 + si)
        W = build_planted_er(args.N, args.s, args.p, args.q, rng)
        L_np, dt = operators(W)
        v = degree_indicator(dt, args.s)
        lam, U = np.linalg.eigh(L_np)
        lam = np.clip(lam, 0.0, None)

        # per-seed C_K and lam_K
        pk = (U.T @ v) ** 2
        Phi_i = float((lam * pk).sum())
        qk_i = (lam * pk) / max(Phi_i, 1e-12)
        qk_i /= qk_i.sum()
        C_i = np.cumsum(qk_i)

        rows_w, cols_w = np.nonzero(W)
        vals_w = W[rows_w, cols_w]
        adjacency = torch.sparse_coo_tensor(
            torch.tensor(np.vstack([rows_w, cols_w]), dtype=torch.long),
            torch.tensor(vals_w, dtype=torch.float64),
            (args.N, args.N),
        ).coalesce()

        LOGGER.info(
            f"  seed {si + 1}/{args.coarsen_n_seeds}  "
            f"Phi_emp={Phi_i:.4f}  K_thresh="
            + str(max(2, min(int(np.searchsorted(lam, Phi_i)), args.N - 2)))
        )

        for ki, K in enumerate(Ks):
            K = int(K)
            U_K = torch.from_numpy(U[:, :K].copy().astype(np.float64))
            lam_K = float(lam[K]) if K < len(lam) else 2.0
            C_K = float(C_i[K - 1]) if K <= len(C_i) else 1.0

            if use_ward:
                recalls, precisions, _n_coarses = _ward_pr_curve(
                    adjacency,
                    U_K,
                    pattern,
                    y_labels,
                    args.tau,
                    args.coarsen_max_levels,
                    args.threshold,
                    args.coarsen_stop_precision,
                    ward_helpers,
                )
            elif use_sequential_edges:
                max_contraction = args.coarsen_reduction * args.N
                recalls, precisions, _n_coarses = _sequential_edge_variation_pr_curve(
                    adjacency,
                    U_K,
                    pattern,
                    y_labels,
                    args.tau,
                    max_contraction,
                    args.threshold,
                    args.coarsen_stop_precision,
                    sequential_helpers,
                    epsilon_budget=args.coarsen_epsilon,
                    refresh_every=args.coarsen_refresh_every,
                    self_loop_aware=args.coarsen_self_loops,
                    exact_epsilon_every=args.coarsen_exact_eps_every,
                    chained=args.coarsen_chained,
                    max_cluster_size=args.coarsen_max_cluster_size,
                )
            else:
                recalls, precisions, _n_coarses = _incremental_pr_curve(
                    adjacency,
                    U_K,
                    pattern,
                    y_labels,
                    args.tau,
                    args.coarsen_max_levels,
                    args.threshold,
                    args.coarsen_stop_precision,
                    helpers,
                )
            auc, best_p, best_r, _best_f1 = _pr_auc(recalls, precisions)
            per_K_auc[ki].append(auc)
            per_K_prec[ki].append(best_p)
            per_K_recall[ki].append(best_r)
            per_K_lam[ki].append(lam_K)
            per_K_CK[ki].append(C_K)

    # ---- aggregate --------------------------------------------------------
    def _mean_std(vals):
        good = [v for v in vals if not np.isnan(v)]
        mu = float(np.mean(good)) if good else float("nan")
        sd = float(np.std(good)) if len(good) > 1 else 0.0
        return mu, sd, len(good)

    coarsen_rows = []
    LOGGER.info(
        f"  {'K':>6} {'lam_K':>8} {'C_K':>8} {'AUC':>8} "
        f"{'prec*':>8} {'recall*':>8}   (n)"
    )
    LOGGER.info("  " + "-" * 60)
    for ki, K in enumerate(Ks):
        auc_mu, auc_sd, n_valid = _mean_std(per_K_auc[ki])
        prec_mu, prec_sd, _ = _mean_std(per_K_prec[ki])
        rec_mu, rec_sd, _ = _mean_std(per_K_recall[ki])
        lam_K = float(np.nanmean(per_K_lam[ki]))
        C_K = float(np.nanmean(per_K_CK[ki]))
        coarsen_rows.append(
            {
                "K": K,
                "lam_K": lam_K,
                "C_K": C_K,
                "auc_mean": auc_mu,
                "auc_std": auc_sd,
                "prec_mean": prec_mu,
                "prec_std": prec_sd,
                "recall_mean": rec_mu,
                "recall_std": rec_sd,
                "n_valid": n_valid,
            }
        )
        LOGGER.info(
            f"  {K:>6} {lam_K:>8.4f} {C_K:>8.4f} {auc_mu:>8.3f} "
            f"{prec_mu:>8.3f} {rec_mu:>8.3f}   (n={n_valid})"
        )

    if _HAVE_PLT and args.plot:
        _plot_coarsening_sweep(coarsen_rows, th, Phi0, K_thresh0, args)


def _plot_coarsening_sweep(
    rows: list, th: dict, Phi_emp: float, K_thresh: int, args
) -> None:
    """Three panels (vs K, C_K, lambda_K/Phi), each showing precision, recall and
    precision-recall AUC (mean +/- std over seeds) of the level-by-level sweep."""

    Ks = np.array([r["K"] for r in rows])
    lam_Ks = np.array([r["lam_K"] for r in rows])
    C_Ks = np.array([r["C_K"] for r in rows])

    metrics = {
        "recall": (
            np.array([r["recall_mean"] for r in rows]),
            np.array([r["recall_std"] for r in rows]),
            "tab:blue",
        ),
        "precision": (
            np.array([r["prec_mean"] for r in rows]),
            np.array([r["prec_std"] for r in rows]),
            "tab:orange",
        ),
        "PR-AUC": (
            np.array([r["auc_mean"] for r in rows]),
            np.array([r["auc_std"] for r in rows]),
            "tab:purple",
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        rf"Level-by-level coarsening [{args.coarsen_method}] "
        rf"(stop precision $<${args.coarsen_stop_precision:.0%}) "
        rf"-- precision / recall / PR-AUC vs span$(U_K)$  "
        f"(N={args.N}, s={args.s}, p={args.p}, q={args.q}, "
        rf"$\tau$={args.tau}, {args.coarsen_n_seeds} seeds)",
        fontsize=10,
    )

    def _panel(ax, x, xlog=False):
        """Plot all three metrics (mean line + +/-1 sigma band) against x."""
        order = np.argsort(x)
        xs = x[order]
        for name, (mu, sd, color) in metrics.items():
            m, s = mu[order], sd[order]
            mask = ~np.isnan(m)
            xv, mv, sv = xs[mask], m[mask], s[mask]
            plot = ax.semilogx if xlog else ax.plot
            plot(xv, mv, "o-", color=color, ms=4, lw=1.5, label=name)
            ax.fill_between(
                xv, (mv - sv).clip(0), (mv + sv).clip(0, 1), color=color, alpha=0.15
            )
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3, which="both" if xlog else "major")

    # ---- Panel A: vs K (log scale) ----------------------------------------
    ax = axes[0]
    _panel(ax, Ks, xlog=True)
    ax.axvline(
        K_thresh,
        ls="--",
        lw=1.2,
        color="grey",
        label=rf"$K^*$ ($\lambda_K\approx\Phi$)",
    )
    ax.set_xlabel(r"$K$ (eigenvectors in span$(U_K)$)")
    ax.set_ylabel("score")
    ax.set_title("(A) vs K")
    ax.legend(fontsize=8)

    # ---- Panel B: vs C_K --------------------------------------------------
    ax = axes[1]
    _panel(ax, C_Ks)
    idx_thresh = int(np.argmin(np.abs(Ks - K_thresh)))
    ax.axvline(
        C_Ks[idx_thresh],
        ls="--",
        lw=1.2,
        color="grey",
        label=rf"$C_K$ at $K^*\approx{K_thresh}$",
    )
    ax.set_xlabel(r"$C_K$ (fraction of $L$-energy in span$(U_K)$)")
    ax.set_ylabel("score")
    ax.set_title(r"(B) vs retained energy $C_K$")
    ax.legend(fontsize=8)

    # ---- Panel C: vs lambda_K / Phi ---------------------------------------
    ax = axes[2]
    lk_norm = lam_Ks / max(Phi_emp, 1e-12)
    _panel(ax, lk_norm)
    ax.axvline(
        1.0, ls="--", lw=1.5, color="grey", label=r"$\lambda_K = \Phi$ (threshold)"
    )
    ax.axvline(
        th["m1t"] / max(Phi_emp, 1e-12),
        ls=":",
        lw=1.2,
        color="black",
        label=r"$\lambda_K = \tilde m_1$",
    )
    ax.set_xlabel(r"$\lambda_K \,/\, \Phi$")
    ax.set_ylabel("score")
    ax.set_title(r"(C) Sharp threshold at $\lambda_K = \Phi$")
    ax.legend(fontsize=8)

    fig.tight_layout()
    coarsen_out = args.out.replace(".png", f"_coarsening_{args.coarsen_method}.png")
    fig.savefig(coarsen_out, dpi=150, bbox_inches="tight")
    LOGGER.info(f"\ncoarsening sweep plot -> {coarsen_out}")
    plt.close(fig)


def main() -> None:
    result_path = "results/Coarsening_test/"
    path = f"{result_path}{now}/"
    os.makedirs(path, exist_ok=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=1000, help="total nodes")
    ap.add_argument("--s", type=int, default=40, help="planted motif size |S|")
    ap.add_argument("--p", type=float, default=0.2, help="internal edge prob (Bern p)")
    ap.add_argument(
        "--q", type=float, default=0.012, help="background edge prob (Bern q)"
    )
    ap.add_argument("--tau", type=float, default=1.0, help="screening (0 = Cor 4.7)")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument(
        "--log-const",
        type=float,
        default=1.0,
        help="C in the min{delta,b} >= C log N regularity assumption",
    )
    ap.add_argument("--plot", action="store_true", default=True)
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    ap.add_argument("--out", type=str, default=f"{path}corollary_4_7.png")
    ap.add_argument(
        "--mean-host-deg",
        type=float,
        default=10.0,
        help="expected host degree held fixed while N grows",
    )
    # coarsening sweep
    ap.add_argument(
        "--coarsening",
        action="store_true",
        help="also run the K-sweep coarsening experiment (span(U_K) target vs "
        "recall/precision vs C_K); requires torch + src package",
    )
    ap.add_argument(
        "--coarsen-method",
        choices=[
            "edges",
            "sequential_edges",
            "neighborhood",
            "capped",
            "linkage",
            "ward",
        ],
        default="ward",
    )
    ap.add_argument(
        "--coarsen-reduction",
        type=float,
        default=0.90,
        help="node-count reduction target (stop when n_coarse <= (1-r)*N)",
    )
    ap.add_argument(
        "--coarsen-epsilon",
        type=float,
        default=float("inf"),
        help="RSA distortion budget (inf = reduction-only stop). For "
        "--coarsen-method sequential_edges this gates on the EXACT cumulative "
        "RSA constant (_exact_rsa_epsilon), not the loose product bound",
    )
    ap.add_argument(
        "--coarsen-refresh-every",
        type=int,
        default=10,
        help="sequential_edges only: contract this many disjoint edges per "
        "basis/embedding refresh (1 = exact one-at-a-time; larger = faster, "
        "closer to a full matching). The speed/faithfulness middle ground",
    )
    ap.add_argument(
        "--coarsen-exact-eps-every",
        type=int,
        default=10,
        help="sequential_edges only: recompute the exact RSA epsilon every this "
        "many contractions (>1 amortises the check when epsilon is inf)",
    )
    ap.add_argument(
        "--coarsen-self-loops",
        default=True,
        action="store_true",
        help="sequential_edges only: use the volume-preserving W_c = S^T W S "
        "reduction (supernode internal weight on the diagonal) with the "
        "self-loop-aware normalized Laplacian I - D^-1/2 W_c D^-1/2",
    )
    ap.add_argument(
        "--coarsen-chained",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="sequential_edges only: contract each level with cheapest-first "
        "union-find CHAINING (a just-merged node may keep absorbing partners, "
        "reducing by exactly --coarsen-refresh-every nodes/level) instead of a "
        "disjoint matching. Use --no-coarsen-chained for the matching behaviour",
    )
    ap.add_argument(
        "--coarsen-max-cluster-size",
        type=int,
        default=0,
        help="sequential_edges + chaining only: cap the number of original nodes "
        "a supernode may hold within one level (0 = unlimited); guards against "
        "single-linkage chaining swallowing the graph",
    )
    ap.add_argument(
        "--coarsen-max-levels",
        type=int,
        default=500,
        help="maximum coarsening levels",
    )
    ap.add_argument(
        "--coarsen-k-max",
        type=int,
        default=360,
        help="maximum K value in the sweep",
    )
    ap.add_argument(
        "--coarsen-k-points",
        type=int,
        default=15,
        help="number of logarithmically spaced K points (dense region near K* is added)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.51,
        help="fraction of pattern nodes that must land in a single supernode to "
        "count as detected (passed to evaluate_loukas_patterns)",
    )
    ap.add_argument(
        "--coarsen-n-seeds",
        type=int,
        default=10,
        help="number of independent planted-ER instances to average over in the "
        "coarsening sweep",
    )
    ap.add_argument(
        "--coarsen-stop-precision",
        type=float,
        default=0.10,
        help="stop the level-by-level coarsening once gang precision drops below "
        "this (the trajectory up to here defines the precision-recall AUC)",
    )
    args = ap.parse_args()
    # run(args)
    # if args.scaling:
    #     run_scaling(args)
    # if args.coarsening:
    run_coarsening_sweep(args)
    # if args.density:
    #     run_density_sweep(args)
    # if args.size:
    #     run_size_sweep(args)


if __name__ == "__main__":
    main()
