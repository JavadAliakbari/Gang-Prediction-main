"""PyTorch SGC target construction, Loukas RSA coarsening, and pattern recall.

This is the end-to-end path requested by the Graph_Coarsening note and the
Loukas paper:

1. fit ``theta`` with Eq. (48) on training patterns;
2. form ``Z = g_theta(A_hat) X`` and ``R = span(Z)``;
3. run edge-based local-variation RSA coarsening with ``R`` as its target;
4. declare a pattern detected only when its Pattern-model recall and precision
   are both strictly greater than the supplied threshold.

All graph algebra and the Loukas Algorithm 1/2 implementation below are
PyTorch based.  No NumPy/SciPy coarsening path is used, with a single
exception: ``method="ward"`` (:func:`_ward_partition`) lazily imports
scikit-learn and SciPy for connectivity-constrained Ward agglomeration --
those packages are optional and only required when that method is selected.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import json
import math

import torch
import torch.nn.functional as F

from src.sgc_detection import (
    SGCTrainingResult,
    apply_feature_channel,
    apply_graph_filter,
    filter_signals,
    normalized_adjacency,
    propagation_stack,
)


@dataclass
class LoukasCoarseningResult:
    """Original-node mapping and RSA diagnostics from Algorithm 1."""

    node_to_supernode: torch.Tensor
    n_original: int
    n_coarse: int
    epsilon: float
    epsilon_bound: float
    sigmas: List[float]
    sizes: List[int]

    @property
    def reduction(self) -> float:
        return 1.0 - self.n_coarse / self.n_original


@dataclass
class LoukasPatternDetection:
    """Pattern-model recall/precision after RSA coarsening."""

    pattern_id: str
    pattern_type: str
    label: str
    recall: float
    precision: float
    f1: float
    detected: bool


def _symmetric_adjacency_without_loops(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor | None,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Build a sparse symmetric adjacency for the combinatorial Laplacian."""

    rows, cols = edge_index[0], edge_index[1]
    non_self = rows != cols
    rows, cols = rows[non_self], cols[non_self]
    if edge_weight is None:
        values = torch.ones(rows.numel(), dtype=dtype, device=edge_index.device)
    else:
        values = edge_weight.to(device=edge_index.device, dtype=dtype)[non_self]

    indices = torch.cat((torch.stack((rows, cols)), torch.stack((cols, rows))), dim=1)
    values = torch.cat((values, values)) * 0.5
    return torch.sparse_coo_tensor(
        indices,
        values,
        (num_nodes, num_nodes),
        dtype=dtype,
        device=edge_index.device,
    ).coalesce()


def _degrees(adjacency: torch.Tensor) -> torch.Tensor:
    degree = torch.zeros(
        adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device
    )
    degree.scatter_add_(0, adjacency.indices()[0], adjacency.values())
    return degree


def _laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    """Return combinatorial ``L = D - W`` without forming a dense matrix."""

    n = adjacency.shape[0]
    diagonal = torch.arange(n, device=adjacency.device)
    indices = torch.cat((torch.stack((diagonal, diagonal)), adjacency.indices()), dim=1)
    values = torch.cat((_degrees(adjacency), -adjacency.values()))
    return torch.sparse_coo_tensor(
        indices,
        values,
        (n, n),
        dtype=adjacency.dtype,
        device=adjacency.device,
    ).coalesce()


def _normalized_laplacian(
    adjacency: torch.Tensor, *, add_self_loops: bool = True
) -> torch.Tensor:
    """Return the symmetric normalized Laplacian ``L_sym = I - A_hat``.

    This is the metric the collective learnable-filter objective is derived in
    (``L = I - A_hat`` of the analysis), as opposed to the combinatorial
    ``L = D - W`` of :func:`_laplacian`.  With ``add_self_loops`` (the
    renormalization trick, matching :func:`src.sgc_detection.normalized_adjacency`)
    the operator is ``I - D_tilde^{-1/2}(W + I) D_tilde^{-1/2}`` with
    ``D_tilde = D + I``; otherwise it is the classical
    ``I - D^{-1/2} W D^{-1/2}``.  Off-diagonal entries are ``-A_hat_ij`` and the
    diagonal is ``1 - A_hat_ii``, so every eigenvalue lies in ``[0, 2]``.
    """

    n = adjacency.shape[0]
    device, dtype = adjacency.device, adjacency.dtype
    indices = adjacency.indices()
    values = adjacency.values()
    off_diagonal = indices[0] != indices[1]
    indices, values = indices[:, off_diagonal], values[off_diagonal]

    if add_self_loops:
        loop = torch.arange(n, device=device)
        indices = torch.cat((indices, torch.stack((loop, loop))), dim=1)
        values = torch.cat((values, torch.ones(n, dtype=dtype, device=device)))

    degree = torch.zeros(n, dtype=dtype, device=device)
    degree.scatter_add_(0, indices[0], values)
    inv_sqrt = degree.clamp_min(torch.finfo(dtype).eps).rsqrt()
    a_hat_values = values * inv_sqrt[indices[0]] * inv_sqrt[indices[1]]

    # L = I - A_hat: negate the (self-loop-augmented) A_hat entries and add the
    # identity on the diagonal.  Coalescing sums the +1 with any -A_hat_ii entry.
    loop = torch.arange(n, device=device)
    lap_indices = torch.cat((indices, torch.stack((loop, loop))), dim=1)
    lap_values = torch.cat((-a_hat_values, torch.ones(n, dtype=dtype, device=device)))
    return torch.sparse_coo_tensor(
        lap_indices, lap_values, (n, n), dtype=dtype, device=device
    ).coalesce()


def _screened_metric(operator: torch.Tensor, tau: float) -> torch.Tensor:
    """Return the screened metric ``M_tau = operator + tau I`` (Eq. 8/9).

    ``operator`` is the base Laplacian (combinatorial ``L = D - W`` or symmetric
    normalized ``L = I - A_hat``); adding ``tau`` on the diagonal turns the
    ``L``-seminorm into the positive-definite screened norm
    ``||x||^2_{M_tau} = ||x||^2_L + tau ||x||^2_2``.  ``tau = 0`` returns the
    operator unchanged, so every RSA quantity reduces to the plain-``L`` metric.
    """

    if not tau:
        return operator
    n = operator.shape[0]
    device, dtype = operator.device, operator.dtype
    loop = torch.arange(n, device=device)
    indices = torch.cat((operator.indices(), torch.stack((loop, loop))), dim=1)
    values = torch.cat(
        (
            operator.values(),
            torch.full((n,), float(tau), dtype=dtype, device=device),
        )
    )
    return torch.sparse_coo_tensor(
        indices, values, operator.shape, dtype=dtype, device=device
    ).coalesce()


def build_sgc_subspace(
    normalized_adjacency_: torch.Tensor,
    theta: torch.Tensor,
    features: torch.Tensor | None = None,
    *,
    width: int | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """Build an orthonormal basis for ``R = span(g_theta(A_hat) X)``.

    ``X`` is chosen by the ``features`` argument, giving two options:

    * ``features=graph.x`` -- the embedding is the learned SGC filter applied to
      the real node features, ``Z = g_theta(A_hat) X``.  When ``width`` is given
      and smaller than the feature dimension, ``X`` is first compressed with a
      seeded Gaussian sketch ``X @ Omega`` (a randomized range finder that
      preserves ``span(g_theta(A_hat) X)``).
    * ``features=None`` -- ``X`` is a seeded Gaussian matrix of ``width``
      columns (isotropic features), recovering the structural target subspace.

    The QR step only changes the basis, not the subspace Loukas preserves.
    """

    n = normalized_adjacency_.shape[0]
    device = normalized_adjacency_.device
    dtype = normalized_adjacency_.dtype

    if features is None:
        if width is None or width <= 0:
            raise ValueError("gaussian features require a positive width")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        X = torch.randn(
            n, min(width, n), dtype=dtype, device=device, generator=generator
        )
    else:
        X = features.to(device=device, dtype=dtype)
        if X.dim() == 1:
            X = X.unsqueeze(1)
        if X.shape[0] != n:
            raise ValueError("features must have one row per graph node")
        if width is not None:
            if width <= 0:
                raise ValueError("subspace width must be positive")
            if width < X.shape[1]:
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
                sketch = torch.randn(
                    X.shape[1], width, dtype=dtype, device=device, generator=generator
                )
                X = X @ sketch

    theta = theta.to(device=device, dtype=dtype)
    propagated = propagation_stack(normalized_adjacency_, X, theta.numel() - 1)
    Z = filter_signals(propagated, theta)
    return _orthonormal_range(Z)


def _orthonormal_range(Z: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis for ``span(Z)`` via rank-revealing QR.

    The QR step only changes the basis, not the subspace Loukas preserves.
    """

    Q, R = torch.linalg.qr(Z, mode="reduced")
    diagonal = torch.abs(torch.diagonal(R))
    tolerance = torch.finfo(Z.dtype).eps * max(Z.shape) * diagonal.max().clamp_min(1.0)
    rank = int((diagonal > tolerance).sum().item())
    if rank == 0:
        raise ValueError("subspace generator has zero numerical rank")
    return Q[:, :rank]


def build_joint_subspace(
    normalized_adjacency_: torch.Tensor,
    theta: torch.Tensor,
    features: torch.Tensor,
    feature_map: torch.Tensor,
    *,
    structural_width: int,
    seed: int = 0,
    per_hop: bool = False,
) -> torch.Tensor:
    """Basis for ``span([g_theta(A_hat) Omega | feature channel])``.

    Concatenates a random structural range-finder channel ``g_theta(A_hat) Omega``
    (so community structure is retained at least as well as the structural
    target) with the learned feature channel (so the useful feature directions
    are kept on top).  This *augments* the structural directions rather than
    replacing them -- the cure for "features worse than random", since the
    filtered-feature span alone cannot reach the structural eigenspace when the
    features are low-dimensional or non-structural.

    The feature channel matches the encoder: ``g_theta(A_hat) X W`` for a shared
    map, or ``sum_k A_hat^k X W_k`` when ``per_hop`` (``feature_map`` shape
    ``(K+1, f, d)``).
    """

    device = normalized_adjacency_.device
    dtype = normalized_adjacency_.dtype
    n = normalized_adjacency_.shape[0]
    theta = theta.to(device=device, dtype=dtype)

    blocks: List[torch.Tensor] = []
    if structural_width > 0:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        omega = torch.randn(
            n, structural_width, dtype=dtype, device=device, generator=generator
        )
        blocks.append(apply_graph_filter(normalized_adjacency_, omega, theta))

    X = features.to(device=device, dtype=dtype)
    if X.dim() == 1:
        X = X.unsqueeze(1)
    if X.shape[0] != n:
        raise ValueError("features must have one row per graph node")
    blocks.append(
        apply_feature_channel(
            normalized_adjacency_, X, feature_map, theta, per_hop=per_hop
        )
    )

    return _orthonormal_range(torch.cat(blocks, dim=1))


def build_laplacian_subspace(
    adjacency: torch.Tensor,
    *,
    width: int,
) -> torch.Tensor:
    """Orthonormal basis for ``R = span(U_K)``: the bottom-``K`` eigenvectors of
    the combinatorial Laplacian ``L = D - W``.

    This is the classical, learning-free spectral target subspace from Loukas
    (2019).  It is the baseline against which ``span(g_theta(A_hat) X)`` is
    compared: the SGC subspace is tuned to the planted patterns through the
    Eq. (48) ``theta``, whereas ``U_K`` only captures the globally smoothest
    directions of the graph regardless of where the gangs sit.

    The Laplacian null space (eigenvalue ``~0``: vectors that are constant on
    each connected component) is skipped.  Those directions carry no RSA
    constraint -- a contraction never merges nodes across components, so they
    are preserved exactly -- and including them makes ``B^T L B`` singular,
    breaking the ``L``-orthonormalization.  ``U_K`` is therefore the ``K``
    lowest *positive*-frequency eigenvectors.
    """

    if width <= 0:
        raise ValueError("subspace width must be positive")
    n = adjacency.shape[0]
    laplacian = _laplacian(adjacency).to_dense()
    laplacian = 0.5 * (laplacian + laplacian.T)
    # ``eigh`` returns ascending eigenvalues; drop the (near-)zero null space and
    # keep the ``K`` smallest strictly-positive-frequency eigenvectors.
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    tolerance = (
        eigenvalues.abs().max().clamp_min(1.0) * torch.finfo(eigenvalues.dtype).eps * n
    )
    positive = eigenvalues > tolerance
    eigenvectors = eigenvectors[:, positive]
    k = min(width, eigenvectors.shape[1])
    if k == 0:
        raise ValueError("Laplacian has no positive-frequency eigenvector")
    return eigenvectors[:, :k].contiguous()


def _l_orthonormalize(B: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
    """Compute the paper's ``A=B(B^T L B)^(-1/2)`` on its non-null range."""

    gram = B.T @ torch.sparse.mm(laplacian, B)
    gram = 0.5 * (gram + gram.T)
    values, vectors = torch.linalg.eigh(gram)
    maximum = values.abs().max().clamp_min(1.0)
    keep = values > torch.finfo(B.dtype).eps * max(B.shape) * maximum
    if not torch.any(keep):
        raise ValueError("target subspace has no positive Laplacian-energy direction")
    return B @ (vectors[:, keep] * values[keep].rsqrt())


def _edge_partition(
    adjacency: torch.Tensor,
    target_basis: torch.Tensor,
    n_target: int,
    sigma_max: float,
    *,
    laplacian_fn: Callable[[torch.Tensor], torch.Tensor] = _laplacian,
    tau: float = 0.0,
) -> tuple[torch.Tensor, float]:
    """Algorithm 2: greedy, edge-based local-variation contractions."""

    n = adjacency.shape[0]
    indices = adjacency.indices()
    upper = indices[0] < indices[1]
    edge_i, edge_j = indices[0, upper], indices[1, upper]
    if edge_i.numel() == 0:
        return torch.arange(n, device=adjacency.device), 0.0

    A = _l_orthonormalize(target_basis, laplacian_fn(adjacency))
    diff_sq = (A[edge_i] - A[edge_j]).square().sum(dim=1)
    degree = _degrees(adjacency)
    costs = 0.25 * degree[edge_i].add(degree[edge_j]).square() * diff_sq.square()
    if tau:
        d_sum = degree[edge_i].add(degree[edge_j])
        frob = (degree[edge_i].square() + degree[edge_j].square()) / d_sum.square()
        costs = costs + tau * frob * diff_sq
    order = torch.argsort(costs)

    marked = torch.zeros(n, dtype=torch.bool, device=adjacency.device)
    groups = torch.full((n,), -1, dtype=torch.long, device=adjacency.device)
    n_current, n_groups, sigma_sq = n, 0, 0.0
    sigma_limit_sq = math.inf if math.isinf(sigma_max) else sigma_max * sigma_max

    # The greedy matching is inherently sequential (Algorithm 2).  Tensor
    # operations still compute every cost and every reduced graph.
    for edge in order.tolist():
        if n_current <= n_target:
            break
        i, j = int(edge_i[edge]), int(edge_j[edge])
        cost = float(costs[edge])
        if sigma_sq + cost > sigma_limit_sq:
            break
        if marked[i] or marked[j]:
            continue
        marked[i] = marked[j] = True
        groups[i] = groups[j] = n_groups
        n_groups += 1
        n_current -= 1
        sigma_sq += cost

    for vertex in torch.nonzero(~marked, as_tuple=False).flatten().tolist():
        groups[vertex] = n_groups
        n_groups += 1
    return groups, math.sqrt(sigma_sq)


def _chained_edge_partition(
    adjacency: torch.Tensor,
    target_basis: torch.Tensor,
    n_target: int,
    sigma_max: float,
    *,
    laplacian_fn: Callable[[torch.Tensor], torch.Tensor] = _laplacian,
    tau: float = 0.0,
    max_cluster_size: int = 0,
) -> tuple[torch.Tensor, float, List[tuple[int, int]]]:
    r"""Cheapest-first *chained* edge contraction (no matching constraint).

    This is :func:`_edge_partition`'s twin, but it drops the ``marked`` rule that
    lets each vertex be contracted at most once per level.  Edge costs are the
    same static Loukas edge local-variation costs

        c_ij = (1/4) (d_i + d_j)^2 ||A_i - A_j||^4   ( + tau screening term ),

    computed **once** on the level's ``L``-orthonormal embedding
    ``A = B(B^T L B)^{+1/2}``.  Edges are then contracted cheapest-first with a
    union-find, and a vertex may take part in *many* contractions in the same
    level: contracting ``(a, b)`` when ``a`` already sits in a growing supernode
    simply unions ``b`` into it (agglomerative chaining, like Ward, but scored
    against the fixed level embedding rather than recomputed centroids).

    Exactly ``n - n_target`` merges are performed -- one per edge that joins two
    *distinct* current groups -- so the graph is reduced from ``n`` to
    ``n_target`` nodes in a single level (as long as that many inter-group edges
    exist), instead of the ``<= n/2`` a matching can reach.  The outer loop then
    recomputes the exact RSA epsilon, rebuilds ``L`` and ``B`` on the coarsened
    graph, and calls this again for the next level.

    Returns ``(groups, sigma, merges)`` where ``groups`` maps each of the ``n``
    current nodes to a contiguous supernode id ``0..n_target-1``, ``sigma`` is the
    root of the summed accepted edge costs (the per-level product-bound
    contribution; the tight distortion is the exact epsilon measured outside),
    and ``merges`` is the dendrogram for this level: the ordered list of
    ``(a, b)`` current-node endpoints actually contracted.  ``sigma_max`` caps the
    cumulative per-level cost and ``max_cluster_size > 0`` caps a supernode's
    membership (``0`` = unlimited).
    """

    n = adjacency.shape[0]
    indices = adjacency.indices()
    upper = indices[0] < indices[1]
    edge_i, edge_j = indices[0, upper], indices[1, upper]
    if edge_i.numel() == 0:
        return torch.arange(n, device=adjacency.device), 0.0, []

    A = _l_orthonormalize(target_basis, laplacian_fn(adjacency))
    diff_sq = (A[edge_i] - A[edge_j]).square().sum(dim=1)
    degree = _degrees(adjacency)
    costs = 0.25 * degree[edge_i].add(degree[edge_j]).square() * diff_sq.square()
    if tau:
        d_sum = degree[edge_i].add(degree[edge_j])
        frob = (degree[edge_i].square() + degree[edge_j].square()) / d_sum.square()
        costs = costs + tau * frob * diff_sq
    order = torch.argsort(costs).tolist()
    ei, ej, cst = edge_i.tolist(), edge_j.tolist(), costs.tolist()

    parent = list(range(n))
    size = [1] * n
    cap = max_cluster_size if (max_cluster_size and max_cluster_size > 0) else n

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    # Union-find, cheapest edge first, chaining allowed (no ``marked`` guard).
    merges: List[tuple[int, int]] = []
    n_comp, sigma_sq = n, 0.0
    sigma_limit_sq = math.inf if math.isinf(sigma_max) else sigma_max * sigma_max
    for e in order:
        if n_comp <= n_target:
            break
        a, b = ei[e], ej[e]
        ra, rb = find(a), find(b)
        if ra == rb:  # already in the same supernode
            continue
        if size[ra] + size[rb] > cap:
            continue
        c = cst[e]
        if sigma_sq + c > sigma_limit_sq:
            break
        parent[ra] = rb
        size[rb] += size[ra]
        n_comp -= 1
        sigma_sq += c
        merges.append((a, b))

    roots = torch.tensor([find(i) for i in range(n)], device=adjacency.device)
    _, groups = torch.unique(roots, sorted=True, return_inverse=True)
    return groups, math.sqrt(sigma_sq), merges


def _neighborhood_partition(
    adjacency: torch.Tensor,
    target_basis: torch.Tensor,
    n_target: int,
    sigma_max: float,
    *,
    max_set_size: int = 32,
    laplacian_fn: Callable[[torch.Tensor], torch.Tensor] = _laplacian,
    tau: float = 0.0,
) -> tuple[torch.Tensor, float]:
    """Algorithm 2 with the *neighborhood* local-variation candidate family.

    Each candidate contraction set is a vertex with its neighbors,
    ``C_i = {i} u N(i)``, so one greedy pick can merge a whole neighborhood
    rather than a single edge.  The cost of contracting ``C`` is the Loukas
    local-variation cost

        c(C) = trace(R.T L_C R) / (|C| - 1),   R = (I - 1 p.T) A_C,

    where ``A`` is the ``L``-orthonormal target basis, ``L_C`` is the induced
    subgraph Laplacian on ``C``, and ``p = d_C / sum(d_C)`` are the
    degree-weighted contraction coefficients (the supernode is the
    degree-weighted average, so ``R`` is the residual the contraction discards).
    With single edges this reduces to ``w_ij ||A[i] - A[j]||^2``.
    """

    n = adjacency.shape[0]
    indices = adjacency.indices()
    if indices.numel() == 0:
        return torch.arange(n, device=adjacency.device), 0.0

    A = _l_orthonormalize(target_basis, laplacian_fn(adjacency))
    degree = _degrees(adjacency)
    eps = torch.finfo(A.dtype).eps

    rows = indices[0].tolist()
    cols = indices[1].tolist()
    vals = adjacency.values().tolist()
    neighbors: List[List[int]] = [[] for _ in range(n)]
    weight: Dict[tuple[int, int], float] = {}
    for r, c, w in zip(rows, cols, vals):
        if r == c:
            continue
        neighbors[r].append(c)
        key = (r, c) if r < c else (c, r)
        weight[key] = w

    candidate_sets: List[List[int]] = []
    candidate_costs: List[float] = []
    for i in range(n):
        members = list(dict.fromkeys([i, *neighbors[i]]))
        size = len(members)
        if size < 2 or size > max_set_size:
            continue
        position = {member: p for p, member in enumerate(members)}
        idx = torch.tensor(members, device=adjacency.device)
        A_C = A[idx]
        d_C = degree[idx]

        local_laplacian = torch.zeros((size, size), dtype=A.dtype, device=A.device)
        for member in members:
            a = position[member]
            for other in neighbors[member]:
                b = position.get(other)
                if b is None or member >= other:
                    continue
                w = weight[(member, other)]
                local_laplacian[a, a] += w
                local_laplacian[b, b] += w
                local_laplacian[a, b] -= w
                local_laplacian[b, a] -= w

        p_vec = (d_C / d_C.sum().clamp_min(eps)).unsqueeze(1)
        residual = A_C - (p_vec * A_C).sum(dim=0, keepdim=True)
        energy = torch.trace(residual.T @ (local_laplacian @ residual))
        if tau:
            energy = energy + tau * residual.square().sum()
        cost = energy / (size - 1)
        candidate_sets.append(members)
        candidate_costs.append(float(cost.clamp_min(0.0)))

    if not candidate_sets:
        return torch.arange(n, device=adjacency.device), 0.0

    order = sorted(range(len(candidate_costs)), key=candidate_costs.__getitem__)
    marked = bytearray(n)
    groups = torch.full((n,), -1, dtype=torch.long, device=adjacency.device)
    n_current, n_groups, sigma_sq = n, 0, 0.0
    sigma_limit_sq = math.inf if math.isinf(sigma_max) else sigma_max * sigma_max

    for candidate in order:
        if n_current <= n_target:
            break
        members = candidate_sets[candidate]
        cost = candidate_costs[candidate]
        if sigma_sq + cost > sigma_limit_sq:
            break
        if any(marked[member] for member in members):
            continue
        for member in members:
            marked[member] = 1
            groups[member] = n_groups
        n_groups += 1
        n_current -= len(members) - 1
        sigma_sq += cost

    for vertex in range(n):
        if not marked[vertex]:
            groups[vertex] = n_groups
            n_groups += 1
    return groups, math.sqrt(sigma_sq)


def _adjacency_lists(
    adjacency: torch.Tensor,
) -> tuple[List[List[int]], Dict[tuple[int, int], float]]:
    """Build per-node neighbor lists and a canonical-keyed edge-weight map."""

    indices = adjacency.indices()
    rows, cols = indices[0].tolist(), indices[1].tolist()
    vals = adjacency.values().tolist()
    neighbors: List[List[int]] = [[] for _ in range(adjacency.shape[0])]
    weight: Dict[tuple[int, int], float] = {}
    for r, c, w in zip(rows, cols, vals):
        if r == c:
            continue
        neighbors[r].append(c)
        weight[(r, c) if r < c else (c, r)] = w
    return neighbors, weight


def _local_variation_cost(
    members: Sequence[int],
    A: torch.Tensor,
    degree: torch.Tensor,
    neighbors: List[List[int]],
    weight: Dict[tuple[int, int], float],
    eps: float,
    tau: float = 0.0,
) -> float:
    """Loukas local-variation cost ``c(C) = trace(R.T L_C R) / (|C| - 1)``.

    ``R = (I - 1 p.T) A_C`` with degree-weighted coefficients ``p = d_C / sum d_C``
    and ``L_C`` the induced-subgraph Laplacian.  This is the same quantity the
    neighborhood family uses; it is factored out so the capped and star families
    can score arbitrary contraction sets identically.
    """

    size = len(members)
    position = {member: p for p, member in enumerate(members)}
    idx = torch.tensor(list(members), device=A.device)
    A_C = A[idx]
    d_C = degree[idx]
    local_laplacian = torch.zeros((size, size), dtype=A.dtype, device=A.device)
    for member in members:
        a = position[member]
        for other in neighbors[member]:
            b = position.get(other)
            if b is None or member >= other:
                continue
            w = weight[(member, other)]
            local_laplacian[a, a] += w
            local_laplacian[b, b] += w
            local_laplacian[a, b] -= w
            local_laplacian[b, a] -= w
    p_vec = (d_C / d_C.sum().clamp_min(eps)).unsqueeze(1)
    residual = A_C - (p_vec * A_C).sum(dim=0, keepdim=True)
    # Remark C.30: screened local-variation cost is the L-Dirichlet energy of the
    # discarded residual plus ``tau`` times its squared Frobenius (l2) norm.
    energy = torch.trace(residual.T @ (local_laplacian @ residual))
    if tau:
        energy = energy + tau * residual.square().sum()
    cost = energy / max(size - 1, 1)
    return float(cost.clamp_min(0.0))


def _capped_partition(
    adjacency: torch.Tensor,
    target_basis: torch.Tensor,
    n_target: int,
    sigma_max: float,
    *,
    max_contraction_size: int = 4,
    laplacian_fn: Callable[[torch.Tensor], torch.Tensor] = _laplacian,
    tau: float = 0.0,
) -> tuple[torch.Tensor, float]:
    """In-between candidate family: bounded contraction sets of size ``<= cap``.

    Edge matching contracts one pair at a time (conservative); the neighborhood
    family contracts a whole ``{i} u N(i)`` (aggressive).  This interpolates: each
    candidate is ``{i}`` plus the up-to ``cap - 1`` neighbors closest to ``i`` in
    the ``L``-orthonormal target embedding (smallest ``||A[i] - A[j]||^2``).
    ``cap = 2`` reproduces the edge family and a large ``cap`` approaches the
    neighborhood family, so ``max_contraction_size`` is the conservative<->
    aggressive knob.  Sets are scored by :func:`_local_variation_cost` and
    contracted greedily, lowest cost first, on disjoint sets.
    """

    n = adjacency.shape[0]
    if adjacency.indices().numel() == 0:
        return torch.arange(n, device=adjacency.device), 0.0

    A = _l_orthonormalize(target_basis, laplacian_fn(adjacency))
    degree = _degrees(adjacency)
    eps = torch.finfo(A.dtype).eps
    neighbors, weight = _adjacency_lists(adjacency)
    cap = max(2, int(max_contraction_size))

    candidate_sets: List[List[int]] = []
    candidate_costs: List[float] = []
    for i in range(n):
        nb = neighbors[i]
        if not nb:
            continue
        if len(nb) > cap - 1:
            nb_idx = torch.tensor(nb, device=A.device)
            dist_sq = (A[i].unsqueeze(0) - A[nb_idx]).square().sum(dim=1)
            keep = torch.topk(dist_sq, cap - 1, largest=False).indices.tolist()
            chosen = [nb[k] for k in keep]
        else:
            chosen = nb
        members = list(dict.fromkeys([i, *chosen]))
        if len(members) < 2:
            continue
        candidate_sets.append(members)
        candidate_costs.append(
            _local_variation_cost(members, A, degree, neighbors, weight, eps, tau)
        )

    if not candidate_sets:
        return torch.arange(n, device=adjacency.device), 0.0

    order = sorted(range(len(candidate_costs)), key=candidate_costs.__getitem__)
    marked = bytearray(n)
    groups = torch.full((n,), -1, dtype=torch.long, device=adjacency.device)
    n_current, n_groups, sigma_sq = n, 0, 0.0
    sigma_limit_sq = math.inf if math.isinf(sigma_max) else sigma_max * sigma_max

    for candidate in order:
        if n_current <= n_target:
            break
        members = candidate_sets[candidate]
        cost = candidate_costs[candidate]
        if sigma_sq + cost > sigma_limit_sq:
            break
        if any(marked[member] for member in members):
            continue
        for member in members:
            marked[member] = 1
            groups[member] = n_groups
        n_groups += 1
        n_current -= len(members) - 1
        sigma_sq += cost

    for vertex in range(n):
        if not marked[vertex]:
            groups[vertex] = n_groups
            n_groups += 1
    return groups, math.sqrt(sigma_sq)


def _star_partition(
    adjacency: torch.Tensor,
    target_basis: torch.Tensor,
    n_target: int,
    sigma_max: float,
    *,
    leaf_degree: int = 1,
    max_star_size: int = 64,
    min_spokes: int = 4,
    laplacian_fn: Callable[[torch.Tensor], torch.Tensor] = _laplacian,
    tau: float = 0.0,
) -> tuple[torch.Tensor, float]:
    """Star-aware candidate family: a genuine-fan pre-pass, then edge matching.

    Hub-and-spoke patterns (fan_in / fan_out) fragment under edge matching: the
    spokes connect only through the hub, so a near-matching contracts at most one
    spoke per hub per level.  A *cost-blind* star pre-pass fixes that -- each hub,
    in descending degree order so it claims its spokes before they are matched
    away, contracts ``{hub} u {spokes}`` -- but firing on every hub shreds
    regular/dense motifs (a cycle node with one higher-degree neighbor looks like
    a tiny star).  Two gates restrict the pre-pass to *genuine* fans:

    * a spoke must have combinatorial degree ``<= leaf_degree`` **and strictly
      below the hub's** (so a uniform-degree cycle yields no stars), and
    * a hub must have at least ``min_spokes`` such spokes (so isolated
      low-degree pairs are left to edge matching, which handles them better).

    Sets are capped at ``max_star_size``.  After the pre-pass, ordinary edge
    matching runs on the still-unmarked nodes, so non-fan regions coarsen exactly
    as in ``"edges"`` and per-level progress is guaranteed.
    """

    n = adjacency.shape[0]
    indices = adjacency.indices()
    if indices.numel() == 0:
        return torch.arange(n, device=adjacency.device), 0.0

    A = _l_orthonormalize(target_basis, laplacian_fn(adjacency))
    degree = _degrees(adjacency)
    eps = torch.finfo(A.dtype).eps
    neighbors, weight = _adjacency_lists(adjacency)
    comb_degree = [len(neighbors[i]) for i in range(n)]

    marked = bytearray(n)
    groups = torch.full((n,), -1, dtype=torch.long, device=adjacency.device)
    n_current, n_groups, sigma_sq = n, 0, 0.0
    sigma_limit_sq = math.inf if math.isinf(sigma_max) else sigma_max * sigma_max

    # --- genuine-fan star pre-pass: big hubs first, claim their spokes ---
    for hub in sorted(range(n), key=lambda h: comb_degree[h], reverse=True):
        if n_current <= n_target:
            break
        if marked[hub]:
            continue
        hub_degree = comb_degree[hub]
        spokes = [
            j
            for j in neighbors[hub]
            if not marked[j]
            and comb_degree[j] <= leaf_degree
            and comb_degree[j] < hub_degree
        ]
        if len(spokes) < max(2, min_spokes):
            continue
        members = [hub, *dict.fromkeys(spokes)][:max_star_size]
        cost = _local_variation_cost(members, A, degree, neighbors, weight, eps, tau)
        if sigma_sq + cost > sigma_limit_sq:
            continue
        for member in members:
            marked[member] = 1
            groups[member] = n_groups
        n_groups += 1
        n_current -= len(members) - 1
        sigma_sq += cost

    # --- edge matching on the remaining unmarked nodes ---
    upper = indices[0] < indices[1]
    edge_i, edge_j = indices[0, upper], indices[1, upper]
    if edge_i.numel() > 0:
        diff_sq = (A[edge_i] - A[edge_j]).square().sum(dim=1)
        costs = 0.25 * degree[edge_i].add(degree[edge_j]).square() * diff_sq.square()
        if tau:
            d_sum = degree[edge_i].add(degree[edge_j])
            frob = (degree[edge_i].square() + degree[edge_j].square()) / d_sum.square()
            costs = costs + tau * frob * diff_sq
        for edge in torch.argsort(costs).tolist():
            if n_current <= n_target:
                break
            i, j = int(edge_i[edge]), int(edge_j[edge])
            if marked[i] or marked[j]:
                continue
            cost = float(costs[edge])
            if sigma_sq + cost > sigma_limit_sq:
                break
            marked[i] = marked[j] = 1
            groups[i] = groups[j] = n_groups
            n_groups += 1
            n_current -= 1
            sigma_sq += cost

    for vertex in range(n):
        if not marked[vertex]:
            groups[vertex] = n_groups
            n_groups += 1
    return groups, math.sqrt(sigma_sq)


def _kmeans_labels(
    embedding: torch.Tensor, k: int, *, iters: int, generator: torch.Generator
) -> torch.Tensor:
    """k-means++ seeding + Lloyd iterations on the rows of ``embedding`` (n, K)."""

    n = embedding.shape[0]
    if k >= n:
        return torch.arange(n, device=embedding.device)

    centers = torch.empty(
        k, embedding.shape[1], dtype=embedding.dtype, device=embedding.device
    )
    start = int(torch.randint(0, n, (1,), generator=generator, device=embedding.device))
    centers[0] = embedding[start]
    dist_sq = (embedding - centers[0]).square().sum(dim=1)
    for c in range(1, k):
        if float(dist_sq.sum()) <= 0.0:
            pick = int(
                torch.randint(0, n, (1,), generator=generator, device=embedding.device)
            )
        else:
            pick = int(
                torch.multinomial(dist_sq.clamp_min(0.0), 1, generator=generator)
            )
        centers[c] = embedding[pick]
        dist_sq = torch.minimum(dist_sq, (embedding - centers[c]).square().sum(dim=1))

    labels = torch.cdist(embedding, centers).argmin(dim=1)
    for _ in range(iters):
        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, embedding)
        counts = torch.bincount(labels, minlength=k)
        nonempty = counts > 0
        new_centers = centers.clone()
        new_centers[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1).to(
            embedding.dtype
        )
        new_labels = torch.cdist(embedding, new_centers).argmin(dim=1)
        centers = new_centers
        if torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
    return labels


def _connected_subclusters(
    adjacency: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Split each cluster into its graph-connected components (union-find).

    k-means clusters by embedding similarity and can be graph-disconnected; this
    keeps two nodes together only if they share an edge *and* a cluster label, so
    every returned supernode is a connected contraction set -- a valid Loukas
    reduction.  Nodes isolated within their cluster become singletons.
    """

    n = labels.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    indices = adjacency.indices()
    rows, cols = indices[0].tolist(), indices[1].tolist()
    lab = labels.tolist()
    for a, b in zip(rows, cols):
        if a < b and lab[a] == lab[b]:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    roots = torch.tensor([find(i) for i in range(n)], device=adjacency.device)
    _, groups = torch.unique(roots, sorted=True, return_inverse=True)
    return groups


def _kmeans_partition(
    adjacency: torch.Tensor,
    target_basis: torch.Tensor,
    n_target: int,
    sigma_max: float,
    *,
    kmeans_iters: int = 10,
    kmeans_seed: int = 0,
    laplacian_fn: Callable[[torch.Tensor], torch.Tensor] = _laplacian,
    tau: float = 0.0,
) -> tuple[torch.Tensor, float]:
    """Global, subspace-driven coarsening: k-means on the embedding + connectivity.

    Rather than greedy local contractions, this targets the partition directly:
    the RSA distortion of a partition equals (up to degree weighting) the
    within-cluster variance of the rows of the ``L``-orthonormal embedding
    ``A = _l_orthonormalize(B, L)``, so the best subspace-preserving coarsening is
    (weighted) k-means on ``A``.  k-means++ seeding gives the standard
    ``O(log k)`` approximation to that NP-hard objective -- a global, bounded
    sub-optimal alternative to the local-variation families.  Each cluster is
    then split into its graph-connected components (:func:`_connected_subclusters`)
    so every supernode is a valid connected contraction set.

    Caveats vs the greedy families: this optimizes the *global* subspace fit but
    does not enforce the per-level ``sigma_max`` RSA budget (it is one-shot); the
    reported ``sigma`` is the realized Loukas cost summed over the resulting
    supernodes, so the cumulative ``epsilon`` stays comparable across methods.
    Connectivity splitting can overshoot ``n_target`` when clusters are graph-
    fragmented; the outer level loop then re-clusters to refine toward the target.
    """

    n = adjacency.shape[0]
    if n <= n_target or adjacency.indices().numel() == 0:
        return torch.arange(n, device=adjacency.device), 0.0

    A = _l_orthonormalize(target_basis, laplacian_fn(adjacency))
    k = max(1, min(int(n_target), n))
    generator = torch.Generator(device=adjacency.device)
    generator.manual_seed(int(kmeans_seed))
    labels = _kmeans_labels(A, k, iters=kmeans_iters, generator=generator)
    groups = _connected_subclusters(adjacency, labels)

    # Realized RSA cost: sum the Loukas local-variation cost over each supernode.
    neighbors, weight = _adjacency_lists(adjacency)
    degree = _degrees(adjacency)
    eps = torch.finfo(A.dtype).eps
    members_by_group: Dict[int, List[int]] = defaultdict(list)
    for node, group in enumerate(groups.tolist()):
        members_by_group[group].append(node)
    sigma_sq = 0.0
    for members in members_by_group.values():
        if len(members) >= 2:
            sigma_sq += _local_variation_cost(
                members, A, degree, neighbors, weight, eps, tau
            )
    return groups, math.sqrt(sigma_sq)


def _ward_partition(
    adjacency: torch.Tensor,
    target_basis: torch.Tensor,
    n_target: int,
    sigma_max: float,
    *,
    laplacian_fn: Callable[[torch.Tensor], torch.Tensor] = _laplacian,
    tau: float = 0.0,
) -> tuple[torch.Tensor, float]:
    r"""Contiguity-constrained Ward agglomeration on the rows of the target embedding.

    The greedy families above fix each merge's cost from *pairwise* differences,
    which is a property of pairwise decisions, not of coarsening per se: once an
    early merge is wrong, every later cost derived from it is contaminated, and
    the contamination compounds.  This method replaces the pairwise cost with
    the *global* Ward objective on ``A = _l_orthonormalize(target_basis, L)``: it
    maintains a partition into connected clusters (initially singletons, only
    ``adjacency``-adjacent clusters are ever merged) and repeatedly merges the
    pair minimizing the Ward increment

        Delta(C1, C2) = |C1||C2| / (|C1|+|C2|) * || abar_C1 - abar_C2 ||^2,
        abar_C = (1/|C|) sum_{i in C} A[i, :],

    which is *exactly* the increase of ``||A - Pi_P A||_F^2`` (the Frobenius
    relaxation of the RSA objective) caused by the merge.  Three properties make
    this the right replacement for pairwise greedy costs:

    (i) *Validity*: restricting merges to graph-adjacent clusters keeps every
        contraction set connected, so the output is a legal Laplacian-consistent
        coarsening (unlike unconstrained k-means on the embedding, whose
        clusters need not be connected -- see :func:`_kmeans_partition`, which
        needs a separate connectivity-splitting pass to repair this).
    (ii) *Strictly generalizes the edge greedy*: at the singleton stage
        ``Delta(a,b) = 0.5 ||A[a] - A[b]||^2`` is exactly the edge family's
        local-variation cost (:func:`_edge_partition`) -- Loukas' greedy *is*
        Ward's first level -- but thereafter Ward compares *centroids over all
        absorbed original rows*, whereas the edge greedy re-derives pairwise
        costs from scratch on the coarse graph each level.
    (iii) *Noise annealing*: once a cluster has absorbed ``m`` rows, its
        boundary decision statistic aggregates ``m`` rows and its noise shrinks
        by ``1/sqrt(m)``; correct early merges make later boundary decisions
        progressively sharper -- the opposite of the greedy's compounding
        contamination.

    This is a *global*, one-shot solve (the whole merge tree is built once by
    ``scipy``/``scikit-learn`` and cut directly to ``n_target`` clusters), so
    -- like :func:`_kmeans_partition` -- it does not consume the per-level
    ``sigma_max`` RSA budget incrementally; ``sigma_max`` is accepted for
    signature compatibility with the other partition families but ignored, and
    the returned ``sigma`` is the *realized* Loukas local-variation cost summed
    over the resulting supernodes (comparable to the other methods'). Requires
    ``scikit-learn`` and ``scipy`` (only imported when this method is used).
    """

    n = adjacency.shape[0]
    if n <= n_target or adjacency.indices().numel() == 0:
        return torch.arange(n, device=adjacency.device), 0.0

    try:
        import numpy as np
        from scipy.sparse import csr_matrix
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "method='ward' requires scikit-learn and scipy "
            "(pip install scikit-learn scipy)"
        ) from exc

    A = _l_orthonormalize(target_basis, laplacian_fn(adjacency))
    A_np = A.detach().cpu().numpy()

    indices = adjacency.indices().cpu().numpy()
    # connectivity = original adjacency sparsity -> every merge stays connected
    conn = csr_matrix(
        (np.ones(indices.shape[1], dtype=np.float64), (indices[0], indices[1])),
        shape=(n, n),
    )

    k = max(1, min(int(n_target), n))
    # model = custom_ward_coarsen(
    #     target_basis=A, n_target=k, adjacency=adjacency, volume_weighted=False, tau=tau, epsilon_budget=sigma_max,
    # )
    # model = custom_ward(
    #     A, n_clusters=k, connectivity=adjacency, distance_threshold=None
    # )
    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward",
        connectivity=conn,
        distance_threshold=None,
    ).fit(A_np)

    groups = torch.as_tensor(model.labels_, dtype=torch.long, device=adjacency.device)
    # groups = model[0]
    _, groups = torch.unique(groups, sorted=True, return_inverse=True)

    # Realized RSA cost: sum the Loukas local-variation cost over each supernode
    # (same accounting as _kmeans_partition, for comparability across methods).
    neighbors, weight = _adjacency_lists(adjacency)
    degree = _degrees(adjacency)
    eps = torch.finfo(A.dtype).eps
    members_by_group: Dict[int, List[int]] = defaultdict(list)
    for node, group in enumerate(groups.tolist()):
        members_by_group[group].append(node)
    sigma_sq = 0.0
    for members in members_by_group.values():
        if len(members) >= 2:
            sigma_sq += _local_variation_cost(
                members, A, degree, neighbors, weight, eps, tau
            )
    return groups, math.sqrt(sigma_sq)


@dataclass
class CustomWardResult:
    """Result of :func:`custom_ward` -- a full agglomeration tree, cuttable anywhere.

    Attributes mirror scikit-learn's ``AgglomerativeClustering`` so the tree can be
    used interchangeably:

    * ``children`` -- ``(m, 2)`` long tensor; row ``t`` holds the two cluster ids
      merged at step ``t`` to form the new cluster ``n_leaves + t`` (leaves are
      ``0..n_leaves-1``).  Identical id convention to sklearn's ``children_``.
    * ``distances`` -- ``(m,)`` tensor; ``distances[t]`` is the Ward increment of
      merge ``t`` (the increase in within-cluster sum of squares
      ``||A - Pi_P A||_F^2``, i.e. the merged pair's Loukas Frobenius cost).
    * ``sizes`` -- ``(n_leaves + m,)`` tensor with the size of every cluster id.
    * ``labels_`` -- the fitted partition (contiguous ids), cut at ``n_clusters``
      / ``distance_threshold`` if given, else the full collapse (one cluster per
      connected component).
    * ``n_leaves`` / ``n_clusters_`` -- number of original nodes / fitted clusters.

    Because the whole tree is retained, :meth:`labels_at` and
    :meth:`labels_at_threshold` recut it at any granularity without re-fitting.
    """

    children: torch.Tensor
    distances: torch.Tensor
    sizes: torch.Tensor
    n_leaves: int
    labels_: torch.Tensor
    n_clusters_: int

    def _cut(self, n_merges_to_apply: int) -> torch.Tensor:
        """Apply the first ``n_merges_to_apply`` merges -> contiguous leaf labels."""
        n = self.n_leaves
        m = int(self.children.shape[0])
        apply = max(0, min(int(n_merges_to_apply), m))
        parent = list(range(n + m))

        def find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:  # path compression
                parent[x], x = root, parent[x]
            return root

        pairs = self.children.tolist()
        for t in range(apply):
            a, b = pairs[t]
            parent[find(int(a))] = n + t
            parent[find(int(b))] = n + t

        roots: dict[int, int] = {}
        labels = [0] * n
        for leaf in range(n):
            r = find(leaf)
            labels[leaf] = roots.setdefault(r, len(roots))
        return torch.tensor(labels, dtype=torch.long, device=self.children.device)

    def labels_at(self, n_clusters: int) -> torch.Tensor:
        """Contiguous labels for the horizontal cut yielding ``n_clusters`` clusters."""
        return self._cut(self.n_leaves - int(n_clusters))

    def labels_at_threshold(self, threshold: float) -> torch.Tensor:
        """Labels after every leading merge with increment ``< threshold``.

        Stops at the first merge whose Ward increment reaches ``threshold`` -- the
        same prefix semantics as sklearn's ``distance_threshold`` (a distortion
        budget rather than a fixed cluster count).
        """
        d = self.distances.tolist()
        apply = 0
        for x in d:
            if x < threshold:
                apply += 1
            else:
                break
        return self._cut(apply)


def _custom_ward_edges(connectivity, n: int):
    """Normalize a connectivity argument to deduped undirected edges ``(ei < ej)``.

    Returns two ``int64`` numpy arrays ``(ei, ej)`` with ``ei < ej`` and no
    duplicates.  Accepts a sparse ``torch`` adjacency, a ``(2, E)`` edge-index
    tensor/array, a SciPy sparse matrix, or ``None`` (fully connected --
    unconstrained Ward).  The dedup is vectorized (``i*n + j`` key) rather than a
    per-edge Python set.
    """
    import numpy as np

    if connectivity is None:
        ii, jj = np.triu_indices(n, k=1)
        return ii.astype(np.int64), jj.astype(np.int64)

    if torch.is_tensor(connectivity):
        if connectivity.is_sparse:
            idx = connectivity.coalesce().indices()
            rows, cols = idx[0].cpu().numpy(), idx[1].cpu().numpy()
        elif connectivity.dim() == 2 and connectivity.shape[0] == 2:
            rows, cols = connectivity[0].cpu().numpy(), connectivity[1].cpu().numpy()
        else:  # dense adjacency
            idx = connectivity.nonzero(as_tuple=False)
            rows, cols = idx[:, 0].cpu().numpy(), idx[:, 1].cpu().numpy()
    elif hasattr(connectivity, "tocoo"):  # SciPy sparse
        coo = connectivity.tocoo()
        rows, cols = np.asarray(coo.row), np.asarray(coo.col)
    else:  # array-like (2, E)
        rows, cols = np.asarray(connectivity[0]), np.asarray(connectivity[1])

    rows = rows.astype(np.int64, copy=False)
    cols = cols.astype(np.int64, copy=False)
    lo = np.minimum(rows, cols)
    hi = np.maximum(rows, cols)
    keep = lo != hi
    lo, hi = lo[keep], hi[keep]
    keys = np.unique(lo * n + hi)  # dedup undirected edges in one pass
    return keys // n, keys % n


def custom_ward(
    X: torch.Tensor,
    connectivity=None,
    *,
    n_clusters: int | None = None,
    distance_threshold: float | None = None,
    full_tree: bool = True,
    merge_cost: Callable[[torch.Tensor, int, torch.Tensor, int], float] | None = None,
    node_weights: torch.Tensor | None = None,
) -> CustomWardResult:
    r"""Self-contained connectivity-constrained Ward agglomeration (no scikit-learn).

    A from-scratch reimplementation of ``AgglomerativeClustering(linkage="ward")``
    that exposes every piece of the algorithm so it can be inspected and extended
    (custom merge cost, distortion-threshold stopping, arbitrary re-cuts) -- the
    things sklearn's one-shot ``fit`` hides.

    The rows of ``X`` are the objects clustered; in the coarsening pipeline these
    are the ``L``-orthonormal embedding rows ``A = B(B^T L B)^{-1/2}``, so squared
    Euclidean distance on ``X`` **is** the ``L``-metric (RSA) distortion -- feeding
    that ``A`` here is what makes Euclidean Ward an ``L_sym`` clustering (no metric
    swap needed; the metric lives in the whitening of ``X``).

    Algorithm (exact Ward via centroids, lazy-deletion heap):

    * every node starts as a singleton cluster (centroid = its row, size 1);
    * only ``connectivity``-adjacent clusters are merge candidates, so every
      supernode stays a connected subgraph (a legal Laplacian-consistent
      coarsening) -- pass the graph adjacency as ``connectivity``;
    * the globally cheapest admissible merge is taken, where the cost is the Ward
      increment ``Delta(a,b) = (n_a n_b)/(n_a + n_b) ||mu_a - mu_b||^2``; centroid
      and neighbour sets update in place and the merged cluster's new candidate
      costs are pushed to the heap.  Cluster ids are immutable, so a heap entry is
      valid exactly while both its endpoints are still active -- no stale-key
      bookkeeping is needed beyond an ``active`` check.

    Parameters mirror sklearn: ``n_clusters`` and/or ``distance_threshold`` set the
    fitted cut in ``labels_``; ``full_tree`` (default ``True``, like
    ``compute_full_tree``) builds the entire dendrogram so
    :meth:`CustomWardResult.labels_at` can recut at any ``k`` afterwards, whereas
    ``full_tree=False`` stops the agglomeration as soon as the cut is reached.
    ``merge_cost`` overrides the Ward increment with any
    ``f(centroid_a, size_a, centroid_b, size_b) -> float`` (e.g. a different metric
    on centroids); the default is exact Ward.  With ``connectivity=None`` every
    pair is a candidate (unconstrained Ward, ``O(n^2)`` edges).

    ``node_weights`` (default ``None`` = all ones = standard cardinality Ward)
    turns this into **weighted** Ward: cluster weight ``W_C = sum_{i in C} w_i``,
    the centroid becomes the ``w``-weighted mean, and the increment becomes
    ``(W_a W_b)/(W_a + W_b) ||mu_a - mu_b||^2``.  Passing the node degrees makes it
    *volume*-weighted -- the natural weighting for graph coarsening, matching
    Loukas' degree-weighted contraction.  Caveat: that only stays consistent with
    the RSA distortion if the epsilon is also degree-weighted;
    :func:`_exact_rsa_epsilon` uses *uniform* block-averaging, which the default
    (uniform) Ward already matches, so leave ``node_weights=None`` unless you have
    switched the RSA metric to match.
    """
    import heapq

    import numpy as np

    Xt = (
        X.detach().to(torch.float64)
        if torch.is_tensor(X)
        else torch.as_tensor(X, dtype=torch.float64)
    )
    n = int(Xt.shape[0])
    device = Xt.device
    if n == 0:
        empty = torch.empty((0, 2), dtype=torch.long, device=device)
        return CustomWardResult(
            empty,
            torch.empty(0, device=device),
            torch.empty(0, device=device),
            0,
            torch.empty(0, dtype=torch.long, device=device),
            0,
        )

    ei, ej = _custom_ward_edges(connectivity, n)

    # Cluster state in preallocated arrays indexed by (immutable) cluster id.
    # A merge only allocates a new id, so at most ``2n - 1`` ids ever exist.
    max_ids = 2 * n
    d_dim = int(Xt.shape[1]) if Xt.dim() == 2 else 1
    C = np.zeros((max_ids, d_dim), dtype=np.float64)  # centroids
    C[:n] = Xt.detach().cpu().numpy().reshape(n, d_dim)
    w = np.ones(max_ids, dtype=np.float64)  # cluster weights
    size_arr = np.ones(max_ids, dtype=np.int64)  # cluster cardinalities
    if node_weights is not None:
        wv = np.asarray([float(x) for x in node_weights], dtype=np.float64)
        if wv.shape[0] != n:
            raise ValueError("node_weights must have one entry per row of X")
        w[:n] = wv

    neighbors: Dict[int, set] = defaultdict(set)
    use_custom = merge_cost is not None

    def cost_scalar(a: int, b: int) -> float:  # custom-cost fallback only
        return float(
            merge_cost(
                torch.from_numpy(C[a]),
                int(size_arr[a]),
                torch.from_numpy(C[b]),
                int(size_arr[b]),
            )
        )

    # ---- initial candidate costs, vectorized over all connectivity edges ----
    n_edges = int(ei.shape[0])
    if n_edges:
        ei_l, ej_l = ei.tolist(), ej.tolist()
        for i, j in zip(ei_l, ej_l):
            neighbors[i].add(j)
            neighbors[j].add(i)
        if use_custom:
            init = [cost_scalar(i, j) for i, j in zip(ei_l, ej_l)]
        else:
            diff = C[ei] - C[ej]
            d2 = np.einsum("ij,ij->i", diff, diff)
            init = ((w[ei] * w[ej] / (w[ei] + w[ej])) * d2).tolist()
        heap: list[tuple[float, int, int]] = [
            (init[t], ei_l[t], ej_l[t]) for t in range(n_edges)
        ]
        heapq.heapify(heap)
    else:
        heap = []

    active = set(range(n))
    children: List[tuple[int, int]] = []
    distances: List[float] = []
    sizes: List[int] = [1] * n  # indexed by cluster id (leaves + internals)
    next_id = n

    def stop_now() -> bool:
        if full_tree:
            return False
        return n_clusters is not None and len(active) <= int(n_clusters)

    while heap and len(active) > 1 and not stop_now():
        d, a, b = heapq.heappop(heap)
        if a not in active or b not in active:
            continue  # a stale candidate: one endpoint already merged away
        if not full_tree and distance_threshold is not None and d >= distance_threshold:
            break

        new = next_id  # always the largest id so far, so new > every neighbour k
        next_id += 1
        wa, wb = w[a], w[b]
        C[new] = (wa * C[a] + wb * C[b]) / (wa + wb)
        w[new] = wa + wb
        size_arr[new] = size_arr[a] + size_arr[b]
        sizes.append(int(size_arr[new]))
        children.append((a, b))
        distances.append(d)

        merged_neighbors = (neighbors[a] | neighbors[b]) - {a, b}
        active.discard(a)
        active.discard(b)
        active.add(new)
        nbr_list = [k for k in merged_neighbors if k in active]
        neighbors[new] = set(nbr_list)
        for k in nbr_list:
            nk = neighbors[k]
            nk.discard(a)
            nk.discard(b)
            nk.add(new)
        # ---- costs from the merged cluster to all its neighbours, batched ----
        if nbr_list:
            if use_custom:
                for k in nbr_list:
                    heapq.heappush(heap, (cost_scalar(new, k), k, new))
            else:
                karr = np.fromiter(nbr_list, dtype=np.int64, count=len(nbr_list))
                diff = C[karr] - C[new]
                d2 = np.einsum("ij,ij->i", diff, diff)
                costs = (w[karr] * w[new] / (w[karr] + w[new])) * d2
                cl = costs.tolist()
                for t, k in enumerate(nbr_list):
                    heapq.heappush(heap, (cl[t], k, new))

    children_t = (
        torch.tensor(children, dtype=torch.long, device=device)
        if children
        else torch.empty((0, 2), dtype=torch.long, device=device)
    )
    distances_t = torch.tensor(distances, dtype=torch.float64, device=device)
    sizes_t = torch.tensor(sizes, dtype=torch.long, device=device)

    result = CustomWardResult(
        children=children_t,
        distances=distances_t,
        sizes=sizes_t,
        n_leaves=n,
        labels_=torch.empty(0, dtype=torch.long, device=device),
        n_clusters_=0,
    )
    if n_clusters is not None:
        labels = result.labels_at(int(n_clusters))
    elif distance_threshold is not None:
        labels = result.labels_at_threshold(float(distance_threshold))
    else:  # full collapse -> one cluster per connected component
        labels = result._cut(len(children))
    result.labels_ = labels
    result.n_clusters_ = int(labels.max().item()) + 1
    return result


def custom_ward_coarsen(
    adjacency: torch.Tensor,
    target_basis: torch.Tensor,
    *,
    n_target: int = 1,
    epsilon_budget: float = float("inf"),
    refresh_every: int | None = None,
    laplacian_fn: Callable[[torch.Tensor], torch.Tensor] = _normalized_laplacian,
    tau: float = 0.0,
    self_loop_aware: bool = False,
    volume_weighted: bool = False,
    return_history: bool = False,
):
    r"""Ward coarsening with a *refreshed* embedding and exact-RSA stopping.

    Plain :func:`custom_ward` (like scikit-learn) builds one tree from the
    embedding ``A = B(B^T L B)^{-1/2}`` computed *once* on the original graph, so
    deep merges are scored against a stale metric.  This driver instead coarsens
    in levels: it runs one Ward pass that reduces the node count by
    ``refresh_every``, rebuilds ``L``, ``B`` and ``A`` on the *coarsened* graph,
    and repeats -- so every level's merges are scored against an up-to-date
    embedding (the same fix the sequential edge method uses, applied to Ward).

    * ``refresh_every=None`` -- one global Ward pass cut to ``n_target`` (the
      static behaviour, equivalent to sklearn); no refresh.
    * ``refresh_every=b`` -- rebuild the embedding every ``b`` contractions; small
      ``b`` = fresher metric, more cost.  ``b=1`` refreshes after every merge.

    Stopping uses the **exact** RSA constant (:func:`_exact_rsa_epsilon` against
    the original ``a0``/``L0``, formed once): coarsening halts at ``n_target`` or
    as soon as the exact epsilon would exceed ``epsilon_budget``.

    ``self_loop_aware`` uses the volume-preserving ``W_c = S^T W S`` reduction and
    the self-loop-aware normalized Laplacian; ``volume_weighted`` runs *degree*-
    weighted Ward each level (see ``node_weights`` in :func:`custom_ward` -- pair
    it with a degree-weighted RSA to stay consistent).

    Returns ``(groups, coarse_adjacency, epsilon_exact)`` -- ``groups`` maps each
    original node to a contiguous supernode id -- plus a per-level ``history`` list
    when ``return_history`` is set.
    """

    base_laplacian = _weighted_normalized_laplacian if self_loop_aware else laplacian_fn

    def metric(adj: torch.Tensor) -> torch.Tensor:
        return _screened_metric(base_laplacian(adj), tau)

    n_original = int(adjacency.shape[0])
    current_adjacency = adjacency.coalesce()
    basis = target_basis
    original_to_current = torch.arange(
        n_original, device=current_adjacency.device, dtype=torch.long
    )

    # Exact-RSA reference on the original graph (partition is all that changes).
    original_laplacian = metric(current_adjacency)
    try:
        a0 = _l_orthonormalize(basis, original_laplacian)
    except ValueError:
        a0 = None

    history: List[dict] = []
    epsilon_exact = 0.0

    while int(current_adjacency.shape[0]) > max(1, n_target):
        n_current = int(current_adjacency.shape[0])
        want = (
            (n_current - n_target)
            if refresh_every is None
            else min(int(refresh_every), n_current - n_target)
        )
        if want <= 0:
            break

        try:
            A = _l_orthonormalize(basis, metric(current_adjacency))
        except ValueError:
            break
        if A.shape[1] == 0:
            break

        weights = _degrees(current_adjacency) if volume_weighted else None
        res = custom_ward(
            A,
            current_adjacency,
            n_clusters=n_current - want,
            full_tree=False,
            node_weights=weights,
        )
        groups = res.labels_.to(dtype=torch.long)
        n_new = int(groups.max().item()) + 1
        if n_new >= n_current:
            break  # no admissible merge (e.g. disconnected below n_target)

        original_to_current = groups[original_to_current]
        current_adjacency = _reduce_adjacency(
            current_adjacency, groups, keep_self_loops=self_loop_aware
        ).coalesce()
        basis = _reduce_basis(basis, groups)

        _, dense_ids = torch.unique(
            original_to_current, sorted=True, return_inverse=True
        )
        if a0 is not None:
            epsilon_exact = _exact_rsa_epsilon(a0, original_laplacian, dense_ids)

        if return_history:
            history.append(
                {
                    "n_after": n_new,
                    "epsilon_exact": epsilon_exact,
                    "ward_increment_max": (
                        float(res.distances.max()) if res.distances.numel() else 0.0
                    ),
                }
            )
        if math.isfinite(epsilon_budget) and epsilon_exact >= epsilon_budget:
            break

    _, groups_final = torch.unique(
        original_to_current, sorted=True, return_inverse=True
    )
    if return_history:
        return groups_final, current_adjacency, epsilon_exact, history
    return groups_final, current_adjacency, epsilon_exact


def _linkage_partition(
    adjacency: torch.Tensor,
    target_basis: torch.Tensor,
    n_target: int,
    sigma_max: float,
    *,
    max_cluster_size: int = 0,
    laplacian_fn: Callable[[torch.Tensor], torch.Tensor] = _laplacian,
    tau: float = 0.0,
) -> tuple[torch.Tensor, float]:
    """Cost-cheapest-first agglomerative clustering with a per-round ``A`` refresh.

    Edge cost is the Loukas single-edge local-variation cost
    ``c_ij = w_ij ||A_i - A_j||^2`` on the ``L``-orthonormal embedding
    ``A = B(B^T L B)^(-1/2)``.  Each *call* performs **one agglomeration round**:
    a maximal cheapest-first *matching* (every node is contracted at most once)
    in ascending cost order, subject to the per-level ``sigma_max`` RSA budget and
    the optional ``max_cluster_size`` cap.  Only graph edges are ever unioned, so
    every supernode is connected **by construction**.

    The round boundary is the whole point: the outer level loop rebuilds
    ``A = B_l (B_l^T L_l B_l)^(-1/2)`` (with ``B_l = P_l B_{l-1}``) on the freshly
    coarsened graph before the next call, so merges are always scored against an
    up-to-date embedding instead of one static ``A`` computed once for the entire
    collapse.  Refreshing ``A`` between rounds is what curbs single-linkage
    *chaining* -- a hub accretes at most one spoke per round rather than its whole
    star against a stale embedding -- which keeps supernodes pure and lifts
    detection precision.  ``max_cluster_size > 0`` further caps the merged size.
    """

    n = adjacency.shape[0]
    indices = adjacency.indices()
    upper = indices[0] < indices[1]
    edge_i, edge_j = indices[0, upper], indices[1, upper]
    if edge_i.numel() == 0:
        return torch.arange(n, device=adjacency.device), 0.0

    A = _l_orthonormalize(target_basis, laplacian_fn(adjacency))
    weights = adjacency.values()[upper]
    diff_sq = (A[edge_i] - A[edge_j]).square().sum(dim=1)
    cost = (weights * diff_sq).clamp_min(0.0)
    if tau:
        degree = _degrees(adjacency)
        d_sum = degree[edge_i].add(degree[edge_j])
        frob = (degree[edge_i].square() + degree[edge_j].square()) / d_sum.square()
        cost = (cost + tau * frob * diff_sq).clamp_min(0.0)
    order = torch.argsort(cost).tolist()
    ei, ej, cst = edge_i.tolist(), edge_j.tolist(), cost.tolist()

    parent = list(range(n))
    size = [1] * n
    matched = [False] * n  # each node is contracted at most once per round
    cap = max_cluster_size if (max_cluster_size and max_cluster_size > 0) else n

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    # One agglomeration generation: merge disjoint cheapest edges (a maximal
    # cheapest-first matching), then return so the outer level loop rebuilds ``A``
    # on the freshly coarsened graph before the next round.
    n_comp, sigma_sq = n, 0.0
    sigma_limit_sq = math.inf if math.isinf(sigma_max) else sigma_max * sigma_max
    for e in order:
        if n_comp <= n_target:
            break
        a, b = ei[e], ej[e]
        if matched[a] or matched[b]:
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if size[ra] + size[rb] > cap:
            continue
        c = cst[e]
        if sigma_sq + c > sigma_limit_sq:
            break
        parent[ra] = rb
        size[rb] += size[ra]
        matched[a] = matched[b] = True
        n_comp -= 1
        sigma_sq += c

    roots = torch.tensor([find(i) for i in range(n)], device=adjacency.device)
    _, groups = torch.unique(roots, sorted=True, return_inverse=True)
    return groups, math.sqrt(sigma_sq)


def _reduce_adjacency(
    adjacency: torch.Tensor,
    groups: torch.Tensor,
    *,
    keep_self_loops: bool = False,
) -> torch.Tensor:
    """Apply the Laplacian-consistent Loukas reduction to a sparse adjacency.

    This is exactly ``W_c = S^T W S`` with ``S`` the 0/1 assignment matrix
    (``S[i, r] = 1`` iff original node ``i`` is in supernode ``r``): the
    contraction sums every original edge weight into its supernode pair.

    ``keep_self_loops`` controls the diagonal of ``W_c``:

    * ``False`` (default) drops it, so ``_degrees(W_c)`` is the *cut* degree and
      the combinatorial ``L = D - W`` is unchanged (self-loops cancel there).
      This preserves the historical behaviour of every caller.
    * ``True`` keeps ``(W_c)_{rr} = sum_{i, j in C_r} W_{ij}``, i.e. the total
      internal weight of the supernode (each internal undirected edge counted
      twice).  A single merge ``u, v -> s`` then satisfies the volume-preserving
      rules ``(W_c)_{ss} = W_{uu} + W_{vv} + 2 W_{uv}`` and ``d_s = d_u + d_v``
      automatically, because the two off-diagonal ``(u, v)`` / ``(v, u)`` entries
      collapse onto ``(s, s)`` and add.  Use this with
      :func:`_weighted_normalized_laplacian`, which reads the stored diagonal.
    """

    n_new = int(groups.max().item()) + 1
    old_indices = adjacency.indices()
    new_indices = groups[old_indices]
    if keep_self_loops:
        keep = slice(None)
    else:
        keep = new_indices[0] != new_indices[1]
    return torch.sparse_coo_tensor(
        new_indices[:, keep],
        adjacency.values()[keep],
        (n_new, n_new),
        dtype=adjacency.dtype,
        device=adjacency.device,
    ).coalesce()


def _weighted_normalized_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
    r"""Self-loop-aware symmetric normalized Laplacian ``L_sym = I - D^{-1/2} W D^{-1/2}``.

    Unlike :func:`_normalized_laplacian` (which discards any stored diagonal and
    applies the renormalization trick ``W + I``, ``D_tilde = D + I``), this reads
    the diagonal ``W_{ii}`` that :func:`_reduce_adjacency` (``keep_self_loops=True``)
    stores as a supernode's internal weight.  With ``D = diag(W \mathbf 1)`` the
    full row sum (diagonal included),

        (L_sym)_{ii} = 1 - W_{ii} / d_i,
        (L_sym)_{ij} = - W_{ij} / sqrt(d_i d_j)   (i != j),

    so a dense supernode with large internal weight has a *smaller* normalized
    diagonal, reflecting that more of its volume stays inside it.  On a graph
    with no self-loops (``W_{ii} = 0``) this is the classical
    ``I - D^{-1/2} W D^{-1/2}`` and eigenvalues still lie in ``[0, 2]``.
    """

    n = adjacency.shape[0]
    device, dtype = adjacency.device, adjacency.dtype
    indices = adjacency.indices()
    values = adjacency.values()

    degree = torch.zeros(n, dtype=dtype, device=device)
    degree.scatter_add_(0, indices[0], values)
    inv_sqrt = degree.clamp_min(torch.finfo(dtype).eps).rsqrt()
    a_hat_values = values * inv_sqrt[indices[0]] * inv_sqrt[indices[1]]

    # L = I - A_hat: negate every A_hat entry (diagonal included) and add the
    # identity.  Coalescing sums the +1 with the -W_ii / d_i diagonal term.
    loop = torch.arange(n, device=device)
    lap_indices = torch.cat((indices, torch.stack((loop, loop))), dim=1)
    lap_values = torch.cat((-a_hat_values, torch.ones(n, dtype=dtype, device=device)))
    return torch.sparse_coo_tensor(
        lap_indices, lap_values, (n, n), dtype=dtype, device=device
    ).coalesce()


def _reduce_basis(basis: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Apply ``B_l=P_l B_{l-1}``, with P averaging each contraction set."""

    n_new = int(groups.max().item()) + 1
    reduced = torch.zeros(n_new, basis.shape[1], dtype=basis.dtype, device=basis.device)
    reduced.index_add_(0, groups, basis)
    counts = torch.bincount(groups, minlength=n_new).to(dtype=basis.dtype).unsqueeze(1)
    return reduced / counts


def _exact_rsa_epsilon(
    a0: torch.Tensor,
    laplacian: torch.Tensor,
    original_to_supernode: torch.Tensor,
) -> float:
    """Exact restricted-spectral-approximation constant of a coarsening.

    The RSA definition (Loukas 2019, Def. 2) is the smallest ``epsilon`` with
    ``||x - Pi x||_L <= epsilon ||x||_L`` for every ``x`` in the target subspace
    ``R``, where ``Pi = P^+ P`` is the block-averaging projection onto vectors
    that are constant on each supernode.  This is the *exact* worst-case
    distortion of the cumulative coarsening -- not the looser per-level product
    bound ``prod_l (1 + sigma_l) - 1``.

    With ``a0`` an ``L``-orthonormal basis of ``R`` (``a0^T L a0 = I``), any
    ``x = a0 c`` has ``||x||_L = ||c||``, so

        epsilon^2 = max_c (c^T Y^T L Y c) / (c^T c) = lambda_max(Y^T L Y),

    with ``Y = (I - Pi) a0`` -- each row of ``a0`` minus its supernode mean.
    ``laplacian`` and ``a0`` are those of the *original* graph; only the
    partition ``original_to_supernode`` changes across levels.
    """

    n_super = int(original_to_supernode.max().item()) + 1
    counts = (
        torch.bincount(original_to_supernode, minlength=n_super)
        .to(dtype=a0.dtype)
        .clamp_min(1.0)
        .unsqueeze(1)
    )
    sums = torch.zeros(n_super, a0.shape[1], dtype=a0.dtype, device=a0.device)
    sums.index_add_(0, original_to_supernode, a0)
    residual = a0 - (sums / counts)[original_to_supernode]  # (I - Pi) a0
    gram = residual.T @ torch.sparse.mm(laplacian, residual)
    gram = 0.5 * (gram + gram.T)
    top = torch.linalg.eigvalsh(gram)[-1].clamp_min(0.0)
    return float(top.sqrt())


def loukas_coarsen_pytorch(
    adjacency: torch.Tensor,
    target_basis: torch.Tensor,
    *,
    reduction: float = 0.7,
    epsilon: float = math.inf,
    max_levels: int = 30,
    method: str = "edges",
    max_contraction_size: int = 4,
    leaf_degree: int = 1,
    max_star_size: int = 64,
    min_spokes: int = 4,
    kmeans_iters: int = 10,
    kmeans_seed: int = 0,
    max_cluster_size: int = 8,
    epsilon_ramp_levels: int | None = None,
    laplacian: str = "combinatorial",
    tau: float = 0.0,
) -> LoukasCoarseningResult:
    """Loukas Algorithm 1 using the supplied ``R=span(target_basis)``.

    ``method`` selects the local-variation candidate family:

    * ``"edges"`` -- contract single edges (Algorithm 2, edge family), the most
      conservative (one pair per match);
    * ``"neighborhood"`` -- contract a vertex with all its neighbors
      ``C_i = {i} u N(i)``, the most aggressive (a whole neighborhood at once);
    * ``"capped"`` -- the *in-between* family: bounded sets of size
      ``<= max_contraction_size`` ({i} plus its closest neighbors).  ``cap = 2``
      reproduces ``"edges"`` and a large cap approaches ``"neighborhood"``;
    * ``"star"`` -- a hub-priority star pre-pass (``{hub} u {spokes}`` with spoke
      combinatorial degree ``<= leaf_degree``, capped at ``max_star_size``)
      followed by edge matching, to hold hub-and-spoke (fan) patterns together;
    * ``"kmeans"`` -- global, *non-greedy* subspace clustering: k-means++ on the
      ``L``-orthonormal embedding (a bounded sub-optimal solver for the partition
      that best preserves ``R``), then connectivity splitting so supernodes stay
      connected.  See :func:`_kmeans_partition`.
    * ``"linkage"`` -- cost-cheapest-first agglomerative matching on the
      cost-weighted graph: one round per level, with ``A`` refreshed between
      rounds so merges never chain against a stale embedding (connected by
      construction).  See :func:`_linkage_partition`.
    * ``"ward"`` -- global, non-greedy contiguity-constrained Ward agglomeration
      on the ``L``-orthonormal embedding: repeatedly merges the *adjacent*
      cluster pair minimizing the Ward increment (the Frobenius-exact cost of
      the merge), which strictly generalizes the edge greedy (its first level)
      while comparing centroids over all absorbed rows thereafter, so early
      correct merges anneal later boundary decisions instead of compounding
      their errors.  One-shot (like ``"kmeans"``) and connected by construction
      (unlike ``"kmeans"``, no post-hoc connectivity split is needed).  Requires
      scikit-learn and scipy.  See :func:`_ward_partition`.

    ``laplacian`` chooses the metric the RSA distortion is measured in:
    ``"combinatorial"`` (default) uses ``L = D - W`` (:func:`_laplacian`), while
    ``"symmetric"`` (aliases ``"normalized"``, ``"sym"``) uses the symmetric
    normalized ``L = I - A_hat`` (:func:`_normalized_laplacian`) -- the metric the
    collective learnable-filter objective is derived in.

    ``tau`` screens that metric into ``M_tau = L + tau I`` (Eq. 8/9): the target
    is ``M_tau``-orthonormalized, each greedy contraction pays the Remark C.30
    screened cost ``||(I - Pi_C) A||^2_{L,loc} + tau ||(I - Pi_C) A||^2_F``, and
    the exact epsilon is the worst-case ``M_tau`` distortion.  ``tau = 0``
    recovers the plain-``L`` RSA exactly, so existing callers are unaffected.

    The cumulative distortion is the exact restricted-spectral-approximation
    constant ``epsilon = max_{x in R} ||x - Pi x||_{M_tau} / ||x||_{M_tau}`` of the
    final coarsening (:func:`_exact_rsa_epsilon`); the looser product estimate
    ``prod_l (1 + sigma_l) - 1`` is kept as ``epsilon_bound`` for reference.
    """

    if not 0.0 <= reduction < 1.0:
        raise ValueError("reduction must be in [0, 1)")
    if tau < 0.0:
        raise ValueError("tau must be non-negative")
    if method not in (
        "edges",
        "neighborhood",
        "capped",
        "star",
        "kmeans",
        "linkage",
        "ward",
    ):
        raise ValueError(
            "method must be 'edges', 'neighborhood', 'capped', 'star', 'kmeans', "
            "'linkage', or 'ward'"
        )
    if laplacian in ("combinatorial", "comb"):
        base_fn: Callable[[torch.Tensor], torch.Tensor] = _laplacian
    elif laplacian in ("symmetric", "normalized", "sym", "norm"):
        base_fn = _normalized_laplacian
    else:
        raise ValueError("laplacian must be 'combinatorial' or 'symmetric'")

    # Screen the RSA metric with ``tau``: every partition orthonormalizes and
    # scores its target against ``M_tau = base_fn(A) + tau I`` (Eq. 8/9), and the
    # exact epsilon below is measured in the same screened norm.  ``tau = 0``
    # leaves ``laplacian_fn`` equal to the base Laplacian, so all RSA quantities
    # collapse to the plain-``L`` metric and legacy callers are unaffected.
    def laplacian_fn(
        adj: torch.Tensor, _base: Callable[[torch.Tensor], torch.Tensor] = base_fn
    ) -> torch.Tensor:
        return _screened_metric(_base(adj), tau)

    if method == "edges":
        partition = partial(_edge_partition, laplacian_fn=laplacian_fn, tau=tau)
    elif method == "neighborhood":
        partition = partial(_neighborhood_partition, laplacian_fn=laplacian_fn, tau=tau)
    elif method == "linkage":
        partition = partial(
            _linkage_partition,
            max_cluster_size=max_cluster_size,
            laplacian_fn=laplacian_fn,
            tau=tau,
        )
    elif method == "capped":
        partition = partial(
            _capped_partition,
            max_contraction_size=max_contraction_size,
            laplacian_fn=laplacian_fn,
            tau=tau,
        )
    elif method == "star":
        partition = partial(
            _star_partition,
            leaf_degree=leaf_degree,
            max_star_size=max_star_size,
            min_spokes=min_spokes,
            laplacian_fn=laplacian_fn,
            tau=tau,
        )
    elif method == "ward":
        partition = partial(
            _ward_partition,
            laplacian_fn=laplacian_fn,
            tau=tau,
        )
    else:
        partition = partial(
            _kmeans_partition,
            kmeans_iters=kmeans_iters,
            kmeans_seed=kmeans_seed,
            laplacian_fn=laplacian_fn,
            tau=tau,
        )
    n_original = adjacency.shape[0]
    n_target = max(1, int(round((1.0 - reduction) * n_original)))
    current_adjacency, basis = adjacency, target_basis
    original_to_current = torch.arange(n_original, device=adjacency.device)
    # Exact RSA is measured against the *original* graph: ``a0`` is an
    # ``L0``-orthonormal basis of ``R`` and ``l0`` its Laplacian, both fixed
    # across levels (see :func:`_exact_rsa_epsilon`).
    l0 = laplacian_fn(adjacency)
    a0 = _l_orthonormalize(target_basis, l0)
    epsilon_current = 0.0  # exact RSA(R) of the cumulative coarsening
    epsilon_bound = 0.0  # legacy product estimate prod_l (1 + sigma_l) - 1
    sigmas: List[float] = []
    sizes = [n_original]

    for level in range(max_levels):
        n_current = current_adjacency.shape[0]
        if n_current <= n_target or epsilon_current >= epsilon:
            break
        if math.isinf(epsilon):
            sigma_max = math.inf
        else:
            # Gradual budget: instead of offering the whole remaining epsilon to
            # every level (which lets level 0 spend all of it), ration it as a
            # linear ramp -- the cumulative budget at level l is capped at
            # epsilon * (l+1)/ramp_levels, so each level may add at most one
            # chunk.  ramp_levels=None keeps the original all-at-once behaviour.
            if epsilon_ramp_levels:
                budget = epsilon * min(1.0, (level + 1) / epsilon_ramp_levels)
            else:
                budget = epsilon
            sigma_max = max(0.0, (1.0 + budget) / (1.0 + epsilon_current) - 1.0)
        # k-means is a one-shot global partition; connectivity splitting leaves it
        # well above n_target, so the first level clusters and later levels switch
        # to cheap edge matching to refine the leftover fragments down to target
        # (re-clustering instead would inflate the RSA epsilon).  Ward is also
        # one-shot but connectivity-constrained by construction, so it lands
        # exactly on n_target after its single call -- no refinement fallback
        # is needed.
        level_partition = (
            partial(_edge_partition, laplacian_fn=laplacian_fn, tau=tau)
            if (method == "kmeans" and level > 0)
            else partition
        )
        groups, sigma = level_partition(current_adjacency, basis, n_target, sigma_max)
        n_new = int(groups.max().item()) + 1
        if n_new >= n_current:
            break

        original_to_current = groups[original_to_current]
        current_adjacency = _reduce_adjacency(current_adjacency, groups)
        basis = _reduce_basis(basis, groups)
        epsilon_current = _exact_rsa_epsilon(a0, l0, original_to_current)
        epsilon_bound = (1.0 + epsilon_bound) * (1.0 + sigma) - 1.0
        sigmas.append(sigma)
        sizes.append(n_new)

    _, dense_ids = torch.unique(original_to_current, sorted=True, return_inverse=True)
    return LoukasCoarseningResult(
        node_to_supernode=dense_ids,
        n_original=n_original,
        n_coarse=int(dense_ids.max().item()) + 1,
        epsilon=epsilon_current,
        epsilon_bound=epsilon_bound,
        sigmas=sigmas,
        sizes=sizes,
    )


def evaluate_loukas_patterns(
    patterns: Sequence[Any],
    node_to_supernode: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float = 0.51,
) -> tuple[List[LoukasPatternDetection], Dict[str, Dict[str, Any]]]:
    """Evaluate Pattern.compute_detection_metrics at the final coarsening level."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if labels.numel() != node_to_supernode.numel():
        raise ValueError(
            "labels and node_to_supernode must refer to original graph nodes"
        )

    classes = max(2, int(labels.max().item()) + 1)
    pseudo_labels = F.one_hot(labels.to(torch.long), num_classes=classes).to(
        torch.float32
    )
    results: List[LoukasPatternDetection] = []
    grouped: Dict[str, Dict[str, List[LoukasPatternDetection]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for pattern in patterns:
        # Test patterns are fresh loader objects, but clear this explicitly when
        # callers evaluate the same objects more than once.
        pattern.level_data.clear()
        metrics = pattern.capture_level(
            node_to_supernode=node_to_supernode,
            pseudo_labels=pseudo_labels,
        )
        recall, precision = float(metrics["recall"]), float(metrics["precision"])
        result = LoukasPatternDetection(
            pattern_id=str(pattern.id),
            pattern_type=str(pattern.pattern_type),
            label=str(pattern.label),
            recall=recall,
            precision=precision,
            f1=float(metrics["f1"]),
            detected=recall > threshold and precision > threshold,
        )
        results.append(result)
        grouped[result.label][result.pattern_type].append(result)

    by_label: Dict[str, Dict[str, Any]] = {}
    for label, by_type in sorted(grouped.items()):
        type_metrics: Dict[str, Dict[str, float]] = {}
        all_entries: List[LoukasPatternDetection] = []
        for pattern_type, entries in sorted(by_type.items()):
            detected = sum(entry.detected for entry in entries)
            type_metrics[pattern_type] = {
                "detected": detected,
                "total": len(entries),
                "detection_rate": detected / len(entries),
                "mean_recall": sum(entry.recall for entry in entries) / len(entries),
                "mean_precision": sum(entry.precision for entry in entries)
                / len(entries),
            }
            all_entries.extend(entries)
        detected = sum(entry.detected for entry in all_entries)
        by_label[label] = {
            "detected": detected,
            "total": len(all_entries),
            "detection_rate": detected / len(all_entries),
            "mean_recall": sum(entry.recall for entry in all_entries)
            / len(all_entries),
            "mean_precision": sum(entry.precision for entry in all_entries)
            / len(all_entries),
            "by_pattern_type": type_metrics,
        }
    return results, by_label


def save_loukas_report(
    path: str | Path,
    fit: SGCTrainingResult,
    coarsening: LoukasCoarseningResult,
    detections: Iterable[LoukasPatternDetection],
    by_label: Mapping[str, Mapping[str, Any]],
    *,
    subspace_width: int,
    threshold: float,
) -> None:
    """Save reproducible theta, RSA diagnostics, and Pattern-model detections."""

    payload = {
        "theta": fit.theta.detach().cpu().tolist(),
        "theta_training_pattern_count": len(fit.train_labels),
        "theta_training_pattern_labels": sorted(set(fit.train_labels)),
        "objective_lambda_min_G": fit.objective,
        "subspace_width": subspace_width,
        "detection_threshold": threshold,
        "detection_rule": "recall > threshold and precision > threshold",
        "coarsening": {
            "n_original": coarsening.n_original,
            "n_coarse": coarsening.n_coarse,
            "reduction": coarsening.reduction,
            "epsilon": coarsening.epsilon,
            "sigmas": coarsening.sigmas,
            "sizes": coarsening.sizes,
        },
        "detection_rate_by_label": by_label,
        "patterns": [asdict(detection) for detection in detections],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def graph_operators(graph: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the SGC and Loukas sparse graph operators from a loader graph."""

    normalized = normalized_adjacency(
        graph.edge_index, int(graph.num_nodes), getattr(graph, "edge_weight", None)
    )
    adjacency = _symmetric_adjacency_without_loops(
        graph.edge_index, int(graph.num_nodes), getattr(graph, "edge_weight", None)
    )
    return normalized, adjacency
