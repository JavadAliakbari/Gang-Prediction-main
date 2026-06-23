"""PyTorch SGC target construction, Loukas RSA coarsening, and pattern recall.

This is the end-to-end path requested by the Graph_Coarsening note and the
Loukas paper:

1. fit ``theta`` with Eq. (48) on training patterns;
2. form ``Z = g_theta(A_hat) X`` and ``R = span(Z)``;
3. run edge-based local-variation RSA coarsening with ``R`` as its target;
4. declare a pattern detected only when its Pattern-model recall and precision
   are both strictly greater than the supplied threshold.

All graph algebra and the Loukas Algorithm 1/2 implementation below are
PyTorch based.  No NumPy/SciPy coarsening path is used.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

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
) -> tuple[torch.Tensor, float]:
    """Algorithm 2: greedy, edge-based local-variation contractions."""

    n = adjacency.shape[0]
    indices = adjacency.indices()
    upper = indices[0] < indices[1]
    edge_i, edge_j = indices[0, upper], indices[1, upper]
    if edge_i.numel() == 0:
        return torch.arange(n, device=adjacency.device), 0.0

    A = _l_orthonormalize(target_basis, _laplacian(adjacency))
    diff_sq = (A[edge_i] - A[edge_j]).square().sum(dim=1)
    degree = _degrees(adjacency)
    costs = 0.25 * degree[edge_i].add(degree[edge_j]).square() * diff_sq.square()
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


def _neighborhood_partition(
    adjacency: torch.Tensor,
    target_basis: torch.Tensor,
    n_target: int,
    sigma_max: float,
    *,
    max_set_size: int = 32,
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

    A = _l_orthonormalize(target_basis, _laplacian(adjacency))
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
        cost = torch.trace(residual.T @ (local_laplacian @ residual)) / (size - 1)
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
    cost = torch.trace(residual.T @ (local_laplacian @ residual)) / max(size - 1, 1)
    return float(cost.clamp_min(0.0))


def _capped_partition(
    adjacency: torch.Tensor,
    target_basis: torch.Tensor,
    n_target: int,
    sigma_max: float,
    *,
    max_contraction_size: int = 4,
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

    A = _l_orthonormalize(target_basis, _laplacian(adjacency))
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
            _local_variation_cost(members, A, degree, neighbors, weight, eps)
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

    A = _l_orthonormalize(target_basis, _laplacian(adjacency))
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
        cost = _local_variation_cost(members, A, degree, neighbors, weight, eps)
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
            pick = int(torch.multinomial(dist_sq.clamp_min(0.0), 1, generator=generator))
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

    A = _l_orthonormalize(target_basis, _laplacian(adjacency))
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
            sigma_sq += _local_variation_cost(members, A, degree, neighbors, weight, eps)
    return groups, math.sqrt(sigma_sq)


def _linkage_partition(
    adjacency: torch.Tensor,
    target_basis: torch.Tensor,
    n_target: int,
    sigma_max: float,
    *,
    max_cluster_size: int = 0,
) -> tuple[torch.Tensor, float]:
    """Single-linkage agglomerative clustering on the local-variation cost graph.

    This is the "weight the adjacency by edge cost, then cluster on the graph"
    idea: edge cost is the Loukas single-edge local-variation cost
    ``c_ij = w_ij ||A_i - A_j||^2`` on the ``L``-orthonormal embedding ``A``, and
    union-find merges edge endpoints in *ascending cost* order until the component
    count reaches ``n_target``.  Because only graph edges are ever unioned, every
    cluster is connected **by construction** -- the connectivity guarantee that
    plain embedding k-means lacked.

    Unlike the edge *matching* family (each node contracted at most once per
    level, hence many levels and at most one spoke per hub), a cluster here grows
    by accreting all of its cheap edges in a single global pass, so a hub can
    absorb its whole star at once.  Caveat: single linkage can *chain* -- a path
    of cheap edges links distant nodes into one large cluster -- which is its
    classic failure mode and yields unbalanced, impure supernodes (it tanks
    detection precision).  ``max_cluster_size > 0`` caps the merged size to curb
    the chaining (size-constrained single linkage), keeping supernodes small and
    pure.
    """

    n = adjacency.shape[0]
    indices = adjacency.indices()
    upper = indices[0] < indices[1]
    edge_i, edge_j = indices[0, upper], indices[1, upper]
    if edge_i.numel() == 0:
        return torch.arange(n, device=adjacency.device), 0.0

    A = _l_orthonormalize(target_basis, _laplacian(adjacency))
    weights = adjacency.values()[upper]
    cost = (weights * (A[edge_i] - A[edge_j]).square().sum(dim=1)).clamp_min(0.0)
    order = torch.argsort(cost).tolist()
    ei, ej, cst = edge_i.tolist(), edge_j.tolist(), cost.tolist()

    parent = list(range(n))
    size = [1] * n
    cap = max_cluster_size if (max_cluster_size and max_cluster_size > 0) else n

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    n_comp, sigma_sq = n, 0.0
    sigma_limit_sq = math.inf if math.isinf(sigma_max) else sigma_max * sigma_max
    for e in order:
        if n_comp <= n_target:
            break
        ra, rb = find(ei[e]), find(ej[e])
        if ra == rb:
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

    roots = torch.tensor([find(i) for i in range(n)], device=adjacency.device)
    _, groups = torch.unique(roots, sorted=True, return_inverse=True)
    return groups, math.sqrt(sigma_sq)


def _reduce_adjacency(adjacency: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Apply the Laplacian-consistent Loukas reduction to a sparse adjacency."""

    n_new = int(groups.max().item()) + 1
    old_indices = adjacency.indices()
    new_indices = groups[old_indices]
    keep = new_indices[0] != new_indices[1]
    return torch.sparse_coo_tensor(
        new_indices[:, keep],
        adjacency.values()[keep],
        (n_new, n_new),
        dtype=adjacency.dtype,
        device=adjacency.device,
    ).coalesce()


def _reduce_basis(basis: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Apply ``B_l=P_l B_{l-1}``, with P averaging each contraction set."""

    n_new = int(groups.max().item()) + 1
    reduced = torch.zeros(n_new, basis.shape[1], dtype=basis.dtype, device=basis.device)
    reduced.index_add_(0, groups, basis)
    counts = torch.bincount(groups, minlength=n_new).to(dtype=basis.dtype).unsqueeze(1)
    return reduced / counts


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
    * ``"linkage"`` -- single-linkage union-find on the cost-weighted graph: merge
      edge endpoints cheapest-first until ``n_target`` components, connected by
      construction (no fragmentation, hits the target in one pass).  See
      :func:`_linkage_partition`.

    The cumulative RSA bound is ``prod_l (1 + sigma_l) - 1``.
    """

    if not 0.0 <= reduction < 1.0:
        raise ValueError("reduction must be in [0, 1)")
    if method not in ("edges", "neighborhood", "capped", "star", "kmeans", "linkage"):
        raise ValueError(
            "method must be 'edges', 'neighborhood', 'capped', 'star', 'kmeans', "
            "or 'linkage'"
        )
    if method == "edges":
        partition = _edge_partition
    elif method == "neighborhood":
        partition = _neighborhood_partition
    elif method == "linkage":
        partition = partial(_linkage_partition, max_cluster_size=max_cluster_size)
    elif method == "capped":
        partition = partial(
            _capped_partition, max_contraction_size=max_contraction_size
        )
    elif method == "star":
        partition = partial(
            _star_partition,
            leaf_degree=leaf_degree,
            max_star_size=max_star_size,
            min_spokes=min_spokes,
        )
    else:
        partition = partial(
            _kmeans_partition, kmeans_iters=kmeans_iters, kmeans_seed=kmeans_seed
        )
    n_original = adjacency.shape[0]
    n_target = max(1, int(round((1.0 - reduction) * n_original)))
    current_adjacency, basis = adjacency, target_basis
    original_to_current = torch.arange(n_original, device=adjacency.device)
    epsilon_current = 0.0
    sigmas: List[float] = []
    sizes = [n_original]

    for level in range(max_levels):
        n_current = current_adjacency.shape[0]
        if n_current <= n_target or epsilon_current >= epsilon:
            break
        sigma_max = (
            math.inf
            if math.isinf(epsilon)
            else (1.0 + epsilon) / (1.0 + epsilon_current) - 1.0
        )
        # k-means is a one-shot global partition; connectivity splitting leaves it
        # well above n_target, so the first level clusters and later levels switch
        # to cheap edge matching to refine the leftover fragments down to target
        # (re-clustering instead would inflate the RSA epsilon).
        level_partition = (
            _edge_partition if (method == "kmeans" and level > 0) else partition
        )
        groups, sigma = level_partition(current_adjacency, basis, n_target, sigma_max)
        n_new = int(groups.max().item()) + 1
        if n_new >= n_current:
            break

        original_to_current = groups[original_to_current]
        current_adjacency = _reduce_adjacency(current_adjacency, groups)
        basis = _reduce_basis(basis, groups)
        epsilon_current = (1.0 + epsilon_current) * (1.0 + sigma) - 1.0
        sigmas.append(sigma)
        sizes.append(n_new)

    _, dense_ids = torch.unique(original_to_current, sorted=True, return_inverse=True)
    return LoukasCoarseningResult(
        node_to_supernode=dense_ids,
        n_original=n_original,
        n_coarse=int(dense_ids.max().item()) + 1,
        epsilon=epsilon_current,
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
