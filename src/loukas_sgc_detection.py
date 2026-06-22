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
) -> LoukasCoarseningResult:
    """Loukas Algorithm 1 using the supplied ``R=span(target_basis)``.

    ``method`` selects the local-variation candidate family:

    * ``"edges"`` -- contract single edges (Algorithm 2, edge family);
    * ``"neighborhood"`` -- contract a vertex together with its neighbors
      ``C_i = {i} u N(i)`` (Algorithm 2, neighborhood family), which merges
      whole neighborhoods per level and typically coarsens in fewer levels.

    The cumulative RSA bound is ``prod_l (1 + sigma_l) - 1``.
    """

    if not 0.0 <= reduction < 1.0:
        raise ValueError("reduction must be in [0, 1)")
    if method not in ("edges", "neighborhood"):
        raise ValueError("method must be 'edges' or 'neighborhood'")
    partition = _edge_partition if method == "edges" else _neighborhood_partition
    n_original = adjacency.shape[0]
    n_target = max(1, int(round((1.0 - reduction) * n_original)))
    current_adjacency, basis = adjacency, target_basis
    original_to_current = torch.arange(n_original, device=adjacency.device)
    epsilon_current = 0.0
    sigmas: List[float] = []
    sizes = [n_original]

    for _ in range(max_levels):
        n_current = current_adjacency.shape[0]
        if n_current <= n_target or epsilon_current >= epsilon:
            break
        sigma_max = (
            math.inf
            if math.isinf(epsilon)
            else (1.0 + epsilon) / (1.0 + epsilon_current) - 1.0
        )
        groups, sigma = partition(current_adjacency, basis, n_target, sigma_max)
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
