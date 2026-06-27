"""PyTorch implementation of collective trainable SGC motif detection.

This module implements the objective in Eq. (48) of ``Graph_Coarsening``:

    max_{||theta||_2 = 1} lambda_min(G(theta)),
    G(theta) = V.T g_theta(A_hat).T g_theta(A_hat) V.

``V`` contains one normalized indicator vector per *training* pattern and
``g_theta(A_hat) = sum_k theta_k A_hat**k``.  The graph operator is sparse and
all propagation, optimization, Gram construction, and evaluation use PyTorch.
No test pattern is used when fitting theta or calibrating the decision threshold.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import json
import math

import torch


@dataclass
class SGCTrainingResult:
    """Parameters and diagnostics returned after optimizing Eq. (48)."""

    theta: torch.Tensor
    objective: float
    vanilla_sgc_objective: float
    train_threshold: float
    degree: int
    epochs: int
    history: List[float]
    train_scores: List[float]
    train_labels: List[str]
    mode: str = "lambda_min"
    feature_aware: bool = False
    separation_ratio: float | None = None
    alert_scores: List[float] = field(default_factory=list)
    normal_scores: List[float] = field(default_factory=list)
    auc: float | None = None
    retention_side: str = "pattern"


@dataclass
class PatternDetection:
    """Detection result for one held-out pattern."""

    pattern_id: str
    pattern_type: str
    label: str
    retained_energy: float
    separation_energy: float
    score: float
    detected: bool


@dataclass
class FeatureDiscriminantResult:
    """A feature-space (ridge-)LDA alert classifier, separate from ``theta``.

    Unlike the spectral energy ``theta.T M_j theta`` -- which traces over the
    feature covariance ``X X^T`` and therefore cannot select a discriminative
    feature *direction* -- this head keeps the feature axis and learns a weight
    ``w`` that scores each pattern by ``s_j = w.T z_j`` on its mean-pooled
    feature signature ``z_j`` (so it reads the label, while ``theta`` is used
    only to build the coarsening target).
    """

    weight: torch.Tensor
    center: torch.Tensor
    head: str
    degree: int
    ridge: float
    train_auc: float
    train_scores: List[float]
    train_labels: List[str]


@dataclass
class JointEncoderResult:
    """A jointly-learned linear encoder ``Z = [g_theta(A_hat) Omega | g_theta(A_hat) X W]``.

    Both the spectral filter ``theta`` (length ``K+1``) and the feature map
    ``W in R^{f x d}`` are trained by gradient on ``lambda_min(G(theta, W))`` with
    ``G(theta, W) = P.T g_theta(A_hat) X W W.T X.T g_theta(A_hat) P`` -- the
    learnable-feature-metric (linear GNN/SGC) upgrade of the fixed
    ``Sigma_X = X X.T``.  Because the pattern signature ``v_j.T g_theta(A_hat) X W``
    keeps ``d`` directions, the embedding no longer collapses to a scalar energy:
    ``W`` selects the ``d`` feature combinations that resolve every pattern, which
    a length-``(K+1)`` ``theta`` alone could never do.  ``theta``/``W`` feed the
    coarsening target; a separate discriminant reads the label.
    """

    theta: torch.Tensor
    feature_map: torch.Tensor
    degree: int
    embed_dim: int
    structural_width: int
    ridge: float
    seed: int
    objective: float
    vanilla_objective: float
    history: List[float]
    train_labels: List[str]
    label_weight: float = 0.0
    combined_objective: float | None = None
    label_separation: float | None = None
    per_hop_features: bool = False
    retention_side: str = "pattern"
    contrastive_weight: float = 0.0
    contrastive_ratio: float | None = None


def normalized_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor | None = None,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return sparse symmetric ``A_hat = D^-1/2 (A + I) D^-1/2``.

    The AMLGenTex loader may return a directed transaction graph.  Eq. (48)
    assumes the symmetric normalized adjacency, so directed edges are mirrored.
    Existing reciprocal edges retain their original total weight; one-way edges
    receive the same weight in both directions.
    """

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")

    device = edge_index.device
    rows, cols = edge_index[0], edge_index[1]
    non_self = rows != cols
    rows, cols = rows[non_self], cols[non_self]
    if edge_weight is None:
        values = torch.ones(rows.numel(), dtype=dtype, device=device)
    else:
        values = edge_weight.to(device=device, dtype=dtype)[non_self]

    # A + A.T, divided by two: reciprocal entries retain their weight while a
    # one-way transaction becomes an undirected edge of half weight per side.
    symmetric_indices = torch.cat(
        [torch.stack((rows, cols)), torch.stack((cols, rows))], dim=1
    )
    symmetric_values = torch.cat((values, values)) * 0.5
    adjacency = torch.sparse_coo_tensor(
        symmetric_indices,
        symmetric_values,
        (num_nodes, num_nodes),
        device=device,
        dtype=dtype,
    ).coalesce()

    loop_nodes = torch.arange(num_nodes, device=device)
    with_loops = torch.sparse_coo_tensor(
        torch.cat((adjacency.indices(), torch.stack((loop_nodes, loop_nodes))), dim=1),
        torch.cat(
            (adjacency.values(), torch.ones(num_nodes, dtype=dtype, device=device))
        ),
        (num_nodes, num_nodes),
        device=device,
        dtype=dtype,
    ).coalesce()

    degree = torch.zeros(num_nodes, dtype=dtype, device=device)
    degree.scatter_add_(0, with_loops.indices()[0], with_loops.values())
    inv_sqrt_degree = degree.clamp_min(torch.finfo(dtype).eps).rsqrt()
    indices = with_loops.indices()
    normalized_values = (
        with_loops.values() * inv_sqrt_degree[indices[0]] * inv_sqrt_degree[indices[1]]
    )
    return torch.sparse_coo_tensor(
        indices,
        normalized_values,
        (num_nodes, num_nodes),
        device=device,
        dtype=dtype,
    ).coalesce()


def pattern_indicator_matrix(
    patterns: Sequence[Any], num_nodes: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Build ``V`` with normalized indicator columns for non-empty patterns."""

    if not patterns:
        raise ValueError("at least one pattern is required")

    columns = []
    for pattern in patterns:
        nodes = torch.as_tensor(pattern.node_indices, dtype=torch.long, device=device)
        nodes = torch.unique(nodes)
        nodes = nodes[(nodes >= 0) & (nodes < num_nodes)]
        if nodes.numel() == 0:
            raise ValueError(f"pattern {pattern.id!r} has no nodes in the graph")
        column = torch.zeros(num_nodes, dtype=dtype, device=device)
        column[nodes] = 1.0 / math.sqrt(nodes.numel())
        columns.append(column)
    return torch.stack(columns, dim=1)


def propagation_stack(
    adjacency: torch.Tensor, signals: torch.Tensor, degree: int
) -> List[torch.Tensor]:
    """Compute ``[V, A_hat V, ..., A_hat**degree V]`` by sparse matvecs."""

    propagated = [signals]
    for _ in range(degree):
        propagated.append(torch.sparse.mm(adjacency, propagated[-1]))
    return propagated


def filter_signals(
    propagated: Sequence[torch.Tensor], theta: torch.Tensor
) -> torch.Tensor:
    """Apply ``g_theta`` to precomputed propagated pattern indicators."""

    if len(propagated) != theta.numel():
        raise ValueError("theta degree does not match the propagated signal stack")
    return sum(coefficient * signal for coefficient, signal in zip(theta, propagated))


def apply_graph_filter(
    adjacency: torch.Tensor, signals: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """Return ``g_theta(A_hat) @ signals = sum_k theta_k A_hat^k signals`` (no QR).

    A convenience wrapper around :func:`propagation_stack` + :func:`filter_signals`
    for building node-level encoder channels such as ``g_theta(A_hat) Omega`` or
    ``g_theta(A_hat) X W`` without re-orthonormalizing.
    """

    theta = theta.to(device=signals.device, dtype=signals.dtype)
    propagated = propagation_stack(adjacency, signals, theta.numel() - 1)
    return filter_signals(propagated, theta)


def apply_feature_channel(
    adjacency: torch.Tensor,
    features: torch.Tensor,
    feature_map: torch.Tensor,
    theta: torch.Tensor,
    *,
    per_hop: bool = False,
) -> torch.Tensor:
    """Node embedding of the learned feature channel ``Z_feat in R^{N x d}``.

    * ``per_hop=False`` -- ``g_theta(A_hat) (X W)`` with a single shared map
      ``W in R^{f x d}`` (``feature_map`` shape ``(f, d)``).  The scalar filter
      ``theta_k`` is the only per-hop freedom, so every smoothing depth uses the
      *same* feature combination ``X W``.
    * ``per_hop=True`` -- ``sum_k A_hat^k (X W_k)`` with one map ``W_k`` per
      propagation depth (``feature_map`` shape ``(K+1, f, d)``).  Each hop is free
      to read a different feature combination, so the channel can keep raw
      features at ``k=0`` and a distinct mix of over-smoothed features at deep
      ``k`` -- the linear "filterbank" upgrade of the shared-``W`` channel.
      ``theta`` is absorbed into the per-hop maps and is not used here.
    """

    X = features.to(device=adjacency.device, dtype=adjacency.dtype)
    if X.dim() == 1:
        X = X.unsqueeze(1)
    W = feature_map.to(device=X.device, dtype=X.dtype)
    if not per_hop:
        if W.dim() != 2 or W.shape[0] != X.shape[1]:
            raise ValueError("feature_map must be (f, d) when per_hop is False")
        return apply_graph_filter(adjacency, X @ W, theta)
    if W.dim() != 3 or W.shape[1] != X.shape[1]:
        raise ValueError("feature_map must be (K+1, f, d) when per_hop is True")
    propagated = propagation_stack(adjacency, X, W.shape[0] - 1)
    return sum(signal @ W[k] for k, signal in enumerate(propagated))


def gram_matrix(
    propagated: Sequence[torch.Tensor],
    theta: torch.Tensor,
    features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the (feature-aware) filtered pattern Gram matrix ``G(theta)``.

    With ``features=None`` this is the Eq. (46) structural Gram
    ``G = F.T F`` (equivalently ``Sigma_X = I``).  With node features
    ``X in R^{N x f}`` it becomes the feature-aware sandwich

        G(theta) = P.T g_theta(A_hat) X X.T g_theta(A_hat) P
                 = Y(theta).T Y(theta),  Y(theta) = X.T g_theta(A_hat) P,

    so each pattern is represented by its ``f``-dim feature signature
    ``y_j = X.T g_theta(A_hat) v_j`` instead of its ``N``-dim filtered indicator.
    """

    filtered = filter_signals(propagated, theta)
    if features is not None:
        filtered = features.T @ filtered
    return filtered.T @ filtered


def _retention_side(r: int, m: int, mode: str) -> str:
    """Which Gram :func:`_retention_lambda_min` uses: ``'pattern'`` or ``'channel'``."""

    if mode == "channel":
        return "channel"
    if mode == "auto" and m > r:
        return "channel"
    return "pattern"


def _reduce_eigs(
    evals: torch.Tensor,
    *,
    reduce: str = "min",
    temp: float = 0.1,
    positive_tol: float = 1e-10,
) -> torch.Tensor:
    """Reduce ascending PSD eigenvalues to a scalar retention objective.

    Only the *resolvable* eigenvalues (above ``positive_tol * lambda_max``) take
    part, so rank-deficiency zeros never dominate the reduction.

    * ``"min"``     -- smallest resolvable eigenvalue (the strict ``lambda_min``);
    * ``"mean"``    -- mean of the resolvable eigenvalues (the trace/energy limit,
      provided mostly to confirm it under-performs);
    * ``"softmin"`` -- soft minimum ``-tau * log mean_i exp(-lambda_i / tau)`` over
      the ``k`` resolvable eigenvalues: a smooth, Schur-concave *eigenvalue-
      weighted trace* whose gradient weights are ``softmax(-lambda / tau)`` (peaked
      on the smallest eigenvalues).  The ``log mean`` (i.e. the ``- log k``
      normalization) keeps it bounded in ``[min, mean]``: ``tau = temp *
      lambda_max`` is scale free (``temp`` relative to the spectrum), ``temp -> 0``
      recovers ``"min"`` and ``temp -> inf`` approaches ``"mean"``.

    ``"min"`` reproduces :func:`_retention_lambda_min`'s original behaviour
    exactly.
    """

    lam_max = evals[-1].clamp_min(0.0)
    floor = positive_tol * lam_max
    mask = evals > floor
    resolvable = evals[mask] if bool(mask.any()) else evals[-1:]
    if reduce == "min":
        return resolvable[0]
    if reduce == "mean":
        return resolvable.mean()
    if reduce == "softmin":
        eps = torch.finfo(evals.dtype).eps
        tau = (temp * lam_max).detach().clamp_min(eps)
        # log-mean-exp: the - log(k) keeps the soft-min bounded in [min, mean].
        log_k = math.log(resolvable.shape[0]) if resolvable.shape[0] > 1 else 0.0
        return -tau * (torch.logsumexp(-resolvable / tau, dim=0) - log_k)
    raise ValueError("retention_reduce must be 'min', 'mean', or 'softmin'")


def _retention_lambda_min(
    signatures: torch.Tensor,
    *,
    mode: str = "auto",
    ridge: float = 0.0,
    positive_tol: float = 1e-10,
    reduce: str = "min",
    temp: float = 0.1,
) -> torch.Tensor:
    """Smallest *resolvable* retention eigenvalue, robust to the over-complete regime.

    ``signatures`` is the channel signature matrix ``Y`` of shape ``(r, m)``
    (``r`` = encoder output dimension, ``m`` = number of patterns).  The pattern
    Gram ``Y.T Y`` (``m x m``) and the channel Gram ``Y Y.T`` (``r x r``) share
    the same *non-zero* spectrum, so this evaluates ``lambda_min`` on whichever
    side is cheaper and not rank starved:

    * ``mode="pattern"`` -- always ``Y.T Y`` (the strict Eq. (48) Gram);
    * ``mode="channel"`` -- always ``Y Y.T``;
    * ``mode="auto"``    -- the smaller Gram: pattern side when ``m <= r`` else
      channel side.

    Two distinct things force a zero eigenvalue: (i) ``m > r`` (more patterns
    than channel directions) and (ii) the channel itself being rank deficient
    (``rank(Y) < min(r, m)``, e.g. collinear node features or over-smoothing).
    The side switch only cures (i).  To also survive (ii) we return the smallest
    eigenvalue *above a relative floor* ``positive_tol * lambda_max`` -- the
    worst-resolved direction the encoder can actually lift off zero -- instead of
    the absolute minimum.  When ``Y`` is full rank (the well-posed regime, incl.
    every ``m <= r`` default run) every eigenvalue clears the floor and this is
    byte-for-byte the original ``lambda_min``.  ``channel`` is meant for the
    feature/joint encoders where ``r`` is small; forcing it on the structural
    encoder builds an ``N x N`` Gram and should be avoided.

    ``reduce`` selects how the (resolvable) spectrum is collapsed to a scalar --
    ``"min"`` (default, strict ``lambda_min``), ``"softmin"`` (smooth
    eigenvalue-weighted trace, temperature ``temp``), or ``"mean"`` -- via
    :func:`_reduce_eigs`.
    """

    r, m = signatures.shape
    use_channel = mode == "channel" or (mode == "auto" and m > r)
    gram = signatures @ signatures.T if use_channel else signatures.T @ signatures
    gram = 0.5 * (gram + gram.T)
    if ridge:
        eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        gram = gram + ridge * eye
    evals = torch.linalg.eigvalsh(gram)  # ascending; PSD so >= 0
    return _reduce_eigs(evals, reduce=reduce, temp=temp, positive_tol=positive_tol)


def _soft_lambda_max(
    signatures: torch.Tensor,
    *,
    mode: str = "auto",
    ridge: float = 0.0,
    reduce: str = "softmax",
    temp: float = 0.1,
) -> torch.Tensor:
    """Largest retained eigenvalue of the (smaller-side) pattern/channel Gram.

    The dual of :func:`_retention_lambda_min` used by the ``contrastive_ratio``
    objective: ``lambda_max(G_-)`` is the *best-retained* negative direction (the
    worst look-alike).  ``lambda_max`` is shared by the pattern and channel Grams,
    so we evaluate it on whichever is smaller.

    * ``"max"``     -- strict ``lambda_max``;
    * ``"softmax"`` -- smooth maximum ``tau * (logsumexp(lambda / tau) - log k)``,
      ``tau = temp * lambda_max`` (scale free, symmetric to the soft-min), which
      smooths the eigenvalue-crossing kink so the gradient is well behaved.
    """

    r, m = signatures.shape
    use_channel = mode == "channel" or (mode == "auto" and m > r)
    gram = signatures @ signatures.T if use_channel else signatures.T @ signatures
    gram = 0.5 * (gram + gram.T)
    if ridge:
        eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        gram = gram + ridge * eye
    evals = torch.linalg.eigvalsh(gram)  # ascending; PSD so >= 0
    lam_max = evals[-1].clamp_min(0.0)
    if reduce == "max":
        return lam_max
    if reduce == "softmax":
        eps = torch.finfo(evals.dtype).eps
        tau = (temp * lam_max).detach().clamp_min(eps)
        log_k = math.log(evals.shape[0]) if evals.shape[0] > 1 else 0.0
        return tau * (torch.logsumexp(evals / tau, dim=0) - log_k)
    raise ValueError("reduce must be 'max' or 'softmax'")


def feature_moment_matrices(
    propagated: Sequence[torch.Tensor],
    features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-pattern feature-moment matrices ``M_j in R^{(K+1) x (K+1)}``.

    ``(M_j)_kl = (X.T A_hat^k v_j).T (X.T A_hat^l v_j)`` so that the feature-aware
    retained energy of pattern ``j`` is the quadratic form ``theta.T M_j theta``.
    With ``features=None`` (``Sigma_X = I``) this reduces to the structural
    moment ``(A_hat^k v_j).T (A_hat^l v_j)``.  Returns shape ``(m, K+1, K+1)``.
    """

    if features is not None:
        signatures = [features.T @ signal for signal in propagated]
    else:
        signatures = list(propagated)
    stacked = torch.stack(signatures, dim=0)  # (K+1, d, m)
    return torch.einsum("kdj,ldj->jkl", stacked, stacked)


def quadratic_scores(moments: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Per-pattern retained energy ``theta.T M_j theta`` (shape ``(m,)``)."""

    theta = theta.to(device=moments.device, dtype=moments.dtype)
    return torch.einsum("jkl,k,l->j", moments, theta, theta)


def discriminative_score_ratio(
    scores: torch.Tensor, is_alert: torch.Tensor, *, eps: float = 1e-8
) -> torch.Tensor:
    """Soft Fisher ratio on the per-pattern energies ``s_j = theta.T M_j theta``.

        J(theta) = (mean_alert(s) - mean_normal(s))**2
                   / (var_alert(s) + var_normal(s) + eps)

    The numerator is squared so the objective is invariant to which class has
    the larger energy, and it is fully differentiable in ``theta``.  Unlike the
    closed-form :func:`fisher_theta` (which separates the *class-mean* moment
    matrices), this maximizes the separation of the actual energy distribution
    that is later thresholded, so it can be optimized by gradient descent.
    """

    pos = scores[is_alert]
    neg = scores[~is_alert]
    mean_gap = pos.mean() - neg.mean()
    spread = pos.var(unbiased=False) + neg.var(unbiased=False) + eps
    return (mean_gap * mean_gap) / spread


def fisher_theta(
    moments: torch.Tensor,
    labels: Sequence[str],
    *,
    ridge: float = 1e-6,
) -> tuple[torch.Tensor, float]:
    """Top generalized eigenvector of the class-mean feature-moment pair.

    Maximizes the alert/normal separation ratio in filter-coefficient space,

        theta* = argmax_theta (theta.T S_+ theta) / (theta.T (S_- + eps I) theta),

    where ``S_+`` and ``S_-`` are the mean feature-moment matrices ``M_j`` over
    the alert and normal training patterns.  This is Fisher/LDA on the
    ``(K+1)``-dim filter coefficients, picking the spectral band *and* feature
    directions where alerts differ most from normals.
    """

    is_alert = torch.tensor(
        [str(label) == "alert" for label in labels], device=moments.device
    )
    if not bool(is_alert.any()) or not bool((~is_alert).any()):
        raise ValueError(
            "Fisher objective needs both alert and normal training patterns"
        )

    s_plus = moments[is_alert].mean(dim=0)
    s_minus = moments[~is_alert].mean(dim=0)
    s_plus = 0.5 * (s_plus + s_plus.T)
    s_minus = 0.5 * (s_minus + s_minus.T)
    eye = torch.eye(s_plus.shape[0], dtype=moments.dtype, device=moments.device)
    chol = torch.linalg.cholesky(s_minus + ridge * eye)
    # whiten: C = L^-1 S_+ L^-T, then theta = L^-T (top eigenvector of C).
    whitened = torch.linalg.solve_triangular(chol, s_plus, upper=False)
    whitened = torch.linalg.solve_triangular(chol, whitened.T, upper=False).T
    whitened = 0.5 * (whitened + whitened.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(whitened)
    top = eigenvectors[:, -1]
    theta = torch.linalg.solve_triangular(chol.T, top.unsqueeze(1), upper=True).squeeze(
        1
    )
    return _unit(theta), float(eigenvalues[-1])


def _roc_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Rank-based ROC AUC (Mann-Whitney) with 0.5 credit for ties."""

    pos = torch.as_tensor(positive, dtype=torch.float64).flatten()
    neg = torch.as_tensor(negative, dtype=torch.float64).flatten()
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (pos.numel() * neg.numel()))


def score_pattern_energies(
    adjacency: torch.Tensor,
    patterns: Sequence[Any],
    theta: torch.Tensor,
    *,
    degree: int,
    features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Feature-aware retained energy ``theta.T M_j theta`` for each pattern."""

    V = pattern_indicator_matrix(
        patterns, adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device
    )
    propagated = propagation_stack(adjacency, V, degree)
    if features is not None:
        features = features.to(device=adjacency.device, dtype=adjacency.dtype)
        if features.dim() == 1:
            features = features.unsqueeze(1)
    moments = feature_moment_matrices(propagated, features)
    return quadratic_scores(moments, theta)


def _mean_indicator_matrix(
    patterns: Sequence[Any], num_nodes: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Build ``U`` whose column ``j`` is ``1/|S_j|`` on the member nodes.

    Using the *mean* indicator (``1/|S_j|``) rather than the energy indicator
    (``1/sqrt(|S_j|)`` of :func:`pattern_indicator_matrix`) removes the pattern
    -size leak: ``X.T u_j`` is the size-normalized mean feature ``mean_{i in S_j} X_i``
    instead of ``sqrt(|S_j|)`` times it, so the discriminant cannot cheat on size.
    """

    if not patterns:
        raise ValueError("at least one pattern is required")
    columns = []
    for pattern in patterns:
        nodes = torch.as_tensor(pattern.node_indices, dtype=torch.long, device=device)
        nodes = torch.unique(nodes)
        nodes = nodes[(nodes >= 0) & (nodes < num_nodes)]
        if nodes.numel() == 0:
            raise ValueError(f"pattern {pattern.id!r} has no nodes in the graph")
        column = torch.zeros(num_nodes, dtype=dtype, device=device)
        column[nodes] = 1.0 / nodes.numel()
        columns.append(column)
    return torch.stack(columns, dim=1)


def feature_signatures(
    adjacency: torch.Tensor,
    patterns: Sequence[Any],
    features: torch.Tensor,
    *,
    degree: int,
    head: str = "joint",
) -> torch.Tensor:
    """Per-pattern mean-pooled feature signatures ``(m, d)``.

    * ``head="mean"`` -- ``z_j = mean_{i in S_j} X_i in R^f`` (``d = f``), the raw
      size-normalized mean feature with no propagation.
    * ``head="joint"`` -- the flattened stack
      ``z_j = [X.T u_j, X.T A_hat u_j, ..., X.T A_hat^K u_j] in R^{(K+1) f}``
      with the mean indicator ``u_j``.  A linear head on this can up-weight
      ``k=0`` (raw features) and down-weight over-smoothed depths, i.e. it
      *discovers* whether propagation helps rather than being forced into it.
    """

    if head not in ("mean", "joint"):
        raise ValueError("head must be 'mean' or 'joint'")
    features = features.to(device=adjacency.device, dtype=adjacency.dtype)
    if features.dim() == 1:
        features = features.unsqueeze(1)
    if features.shape[0] != adjacency.shape[0]:
        raise ValueError("features must have one row per graph node")

    U = _mean_indicator_matrix(
        patterns, adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device
    )
    if head == "mean":
        return (features.T @ U).T  # (m, f)
    propagated = propagation_stack(adjacency, U, degree)  # k = 0..K, each (n, m)
    stack = torch.stack([features.T @ signal for signal in propagated], dim=0)
    m = stack.shape[2]
    return stack.permute(2, 0, 1).reshape(m, -1)  # (m, (K+1) f)


def _ridge_lda_direction(
    signatures: torch.Tensor, is_alert: torch.Tensor, *, ridge: float
) -> torch.Tensor:
    """Ridge-regularized Fisher direction ``(Sigma_w + lambda I)^-1 (mu+ - mu-)``."""

    pos, neg = signatures[is_alert], signatures[~is_alert]
    gap = pos.mean(dim=0) - neg.mean(dim=0)
    within = (
        (pos - pos.mean(dim=0)).T @ (pos - pos.mean(dim=0))
        + (neg - neg.mean(dim=0)).T @ (neg - neg.mean(dim=0))
    ) / max(signatures.shape[0] - 2, 1)
    eye = torch.eye(within.shape[0], dtype=within.dtype, device=within.device)
    scale = float(torch.diagonal(within).mean().clamp_min(1e-12))
    return torch.linalg.solve(within + ridge * scale * eye, gap)


def fit_feature_discriminant(
    adjacency: torch.Tensor,
    train_patterns: Sequence[Any],
    *,
    features: torch.Tensor,
    degree: int = 8,
    head: str = "joint",
    ridge: float = 1e-2,
) -> FeatureDiscriminantResult:
    """Learn a feature-space ridge-LDA alert classifier on mean-pooled signatures.

    This is the classification head: it keeps the feature axis (so it can pick
    the discriminative direction that the scalar energy throws away) and pools
    by the mean (so it does not latch onto pattern size).  ``theta`` and the
    coarsening are left untouched -- they remain a separate *structural* tool.
    """

    labels = [str(pattern.label) for pattern in train_patterns]
    is_alert = torch.tensor(
        [label == "alert" for label in labels], device=adjacency.device
    )
    if not bool(is_alert.any()) or not bool((~is_alert).any()):
        raise ValueError(
            "feature discriminant needs both alert and normal training patterns"
        )

    signatures = feature_signatures(
        adjacency, train_patterns, features, degree=degree, head=head
    )
    center = signatures.mean(dim=0)
    centered = signatures - center
    weight = _ridge_lda_direction(centered, is_alert, ridge=ridge)
    scores = centered @ weight
    train_auc = _roc_auc(
        scores[is_alert].detach().cpu().tolist(),
        scores[~is_alert].detach().cpu().tolist(),
    )
    return FeatureDiscriminantResult(
        weight=weight,
        center=center,
        head=head,
        degree=degree,
        ridge=ridge,
        train_auc=train_auc,
        train_scores=scores.detach().cpu().tolist(),
        train_labels=labels,
    )


def score_feature_patterns(
    adjacency: torch.Tensor,
    patterns: Sequence[Any],
    discriminant: FeatureDiscriminantResult,
    *,
    features: torch.Tensor,
) -> torch.Tensor:
    """Score held-out patterns with a fitted feature discriminant (``s_j = w.T z_j``)."""

    signatures = feature_signatures(
        adjacency,
        patterns,
        features,
        degree=discriminant.degree,
        head=discriminant.head,
    )
    center = discriminant.center.to(device=signatures.device, dtype=signatures.dtype)
    weight = discriminant.weight.to(device=signatures.device, dtype=signatures.dtype)
    return (signatures - center) @ weight


def _label_separation(
    signatures: torch.Tensor, is_alert: torch.Tensor, *, ridge: float
) -> torch.Tensor:
    """Differentiable squared LDA margin between alert and normal signatures.

    ``signatures`` has shape ``(m, d)`` (one row per pattern).  Returns the
    Mahalanobis class separation

        J = (mu_+ - mu_-).T (Sigma_w + lambda I)^-1 (mu_+ - mu_-),

    the multivariate generalization of :func:`discriminative_score_ratio` and
    the exact quantity a ridge-LDA head maximizes (same regularized within-class
    scatter as :func:`_ridge_lda_direction`).  It is fully differentiable in the
    signatures, hence in ``(theta, W)``, so it can be added to the encoder's
    ``lambda_min`` objective and optimized by the same gradient step.
    """

    pos, neg = signatures[is_alert], signatures[~is_alert]
    gap = pos.mean(dim=0) - neg.mean(dim=0)
    centered_pos = pos - pos.mean(dim=0)
    centered_neg = neg - neg.mean(dim=0)
    within = (centered_pos.T @ centered_pos + centered_neg.T @ centered_neg) / max(
        signatures.shape[0] - 2, 1
    )
    eye = torch.eye(within.shape[0], dtype=within.dtype, device=within.device)
    # The ridge floor is a fixed normalization, not a trainable quantity, so the
    # scale is detached (matching :func:`_ridge_lda_direction`).
    scale = torch.diagonal(within).mean().clamp_min(1e-12).detach()
    solution = torch.linalg.solve(within + ridge * scale * eye, gap)
    return gap @ solution


def fit_joint_encoder(
    adjacency: torch.Tensor,
    train_patterns: Sequence[Any],
    *,
    features: torch.Tensor,
    degree: int = 8,
    embed_dim: int = 8,
    structural_width: int = 32,
    ridge: float = 1e-3,
    epochs: int = 400,
    learning_rate: float = 5e-2,
    seed: int = 0,
    label_patterns: Sequence[Any] | None = None,
    label_weight: float = 0.0,
    label_ridge: float = 1e-2,
    contrastive_patterns: Sequence[Any] | None = None,
    contrastive_weight: float = 0.0,
    per_hop_features: bool = False,
    retention_mode: str = "auto",
    retention_reduce: str = "min",
    retention_temp: float = 0.1,
) -> JointEncoderResult:
    """Jointly learn ``(theta, W)`` for ``Z = [g_theta(A_hat) Omega | g_theta(A_hat) X W]``.

    Maximizes ``lambda_min(G(theta, W))`` -- the retention/resolution objective
    Eq. (48), now over the *learnable feature metric* ``W W.T`` -- by projected
    Adam, with ``theta`` constrained to the unit sphere and ``W`` to unit
    Frobenius norm so the metric scale is fixed and the optimum is interior
    (otherwise ``lambda_min`` grows without bound under ``W -> c W``).  The
    closed-form GEVP no longer applies once ``W`` enters quadratically, so the
    encoder is trained by gradient.

    The objective is the retention of the *augmented* encoder: a fixed random
    structural channel ``g_theta(A_hat) Omega`` (``structural_width`` columns) is
    concatenated with the learned feature channel ``g_theta(A_hat) X W``.  This
    augments structure rather than replacing it (caveat: features can be worse
    than random) and, crucially, gives the pattern Gram a rank floor so
    ``lambda_min`` is non-zero even when ``embed_dim`` is much smaller than the
    number of training patterns -- letting ``W`` receive a useful gradient while
    staying small.  The two channels are normalized to comparable scale at init.

    Optional label supervision (``label_weight > 0``).  When ``label_patterns``
    (which must contain *both* alert and normal patterns) is supplied, the
    objective gains a supervised term and becomes the scale-balanced sum

        lambda_min(G(theta, W)) / s_lambda  +  label_weight * J(theta, W) / s_J,

    where ``J`` is the differentiable LDA margin :func:`_label_separation` on the
    *mean-pooled* signatures ``y_j = W.T (sum_k theta_k X.T A_hat^k u_j)`` -- the
    exact representation the held-out ridge-LDA head reads.  Both terms are
    divided by their init magnitude so ``label_weight`` is a clean relative
    weight; ``label_weight == 0`` recovers the pure ``lambda_min`` encoder.  This
    lets ``W`` serve coarsening (retention) and classification (label separation)
    at once instead of leaving the label signal entirely to the separate head.

    Per-hop feature maps (``per_hop_features=True``).  By default a single map
    ``W in R^{f x d}`` is shared across all propagation depths, so the feature
    channel is ``g_theta(A_hat) X W`` and the only per-hop freedom is the scalar
    ``theta_k``.  With ``per_hop_features`` the map becomes one ``W_k`` per depth,
    ``W in R^{(K+1) x f x d}``, and the feature channel is ``sum_k A_hat^k X W_k``
    -- a linear filterbank that can read a *different* feature combination at each
    smoothing radius (raw features at ``k=0``, an over-smoothed mix at deep ``k``)
    rather than the same ``X W`` everywhere.  It stays linear in ``X`` so the
    ``lambda_min(G)`` objective and the spectral story are unchanged; only the
    feature channel gains ``(K+1)x`` parameters.  ``theta`` then shapes only the
    structural channel ``g_theta(A_hat) Omega``.

    ``retention_mode`` handles the over-complete regime where the number of
    patterns ``m`` exceeds the encoder's channel dimension
    ``c = structural_width + embed_dim``.  There the ``m x m`` pattern Gram is
    rank deficient and its ``lambda_min`` is pinned at the ridge floor with no
    gradient in ``(theta, W)``.  ``"auto"`` (default) then switches the objective
    to ``lambda_min`` of the ``c x c`` channel Gram ``Y Y.T`` -- same non-zero
    spectrum, but full rank, so ``(theta, W)`` keep a live gradient.
    ``"pattern"`` forces the strict ``m x m`` Gram; ``"channel"`` forces the
    channel Gram.  For ``m <= c`` (and ``mode != "channel"``) the objective is
    byte-for-byte the original ``lambda_min(Y.T Y)`` (see
    :func:`_retention_lambda_min`).
    """

    if degree < 0:
        raise ValueError("degree must be non-negative")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if embed_dim <= 0:
        raise ValueError("embed_dim must be positive")
    if structural_width < 0:
        raise ValueError("structural_width must be non-negative")
    if label_weight < 0:
        raise ValueError("label_weight must be non-negative")
    if retention_mode not in ("auto", "pattern", "channel"):
        raise ValueError("retention_mode must be 'auto', 'pattern', or 'channel'")
    if retention_reduce not in ("min", "softmin", "mean"):
        raise ValueError("retention_reduce must be 'min', 'softmin', or 'mean'")
    if retention_temp <= 0:
        raise ValueError("retention_temp must be positive")

    device, dtype = adjacency.device, adjacency.dtype
    X = features.to(device=device, dtype=dtype)
    if X.dim() == 1:
        X = X.unsqueeze(1)
    if X.shape[0] != adjacency.shape[0]:
        raise ValueError("features must have one row per graph node")
    feature_dim = X.shape[1]
    embed = min(embed_dim, feature_dim)

    V = pattern_indicator_matrix(
        train_patterns, adjacency.shape[0], dtype=dtype, device=device
    )
    propagated = propagation_stack(adjacency, V, degree)
    # feature signatures: signature_stack[k] = X.T A_hat^k V -> (K+1, f, m); keeps
    # the feature axis so W can pick a direction (unlike the traced moments M_j).
    signature_stack = torch.stack([X.T @ signal for signal in propagated], dim=0)
    m = signature_stack.shape[2]
    labels = [str(pattern.label) for pattern in train_patterns]

    # Fixed random structural channel: struct_stack[k] = V.T A_hat^k Omega -> (K+1, m, r).
    structural_stack = None
    if structural_width > 0:
        generator_omega = torch.Generator(device=device)
        generator_omega.manual_seed(seed)
        omega = torch.randn(
            adjacency.shape[0],
            structural_width,
            dtype=dtype,
            device=device,
            generator=generator_omega,
        )
        propagated_omega = propagation_stack(adjacency, omega, degree)
        structural_stack = torch.stack(
            [V.T @ signal for signal in propagated_omega], dim=0
        )

    # Supervised label channel: mean-pooled signatures over patterns of *both*
    # classes.  label_signature_stack[k] = X.T A_hat^k U -> (K+1, f, m_lab) with
    # the mean indicator U (1/|S_j|), so W.T (sum_k theta_k . ) is exactly the
    # signature the held-out ridge-LDA head scores -- no train/eval proxy gap.
    use_labels = label_weight > 0.0 and label_patterns is not None
    label_signature_stack = None
    is_alert_label = None
    if use_labels:
        U = _mean_indicator_matrix(
            label_patterns, adjacency.shape[0], dtype=dtype, device=device
        )
        propagated_label = propagation_stack(adjacency, U, degree)
        label_signature_stack = torch.stack(
            [X.T @ signal for signal in propagated_label], dim=0
        )
        is_alert_label = torch.tensor(
            [str(pattern.label) == "alert" for pattern in label_patterns],
            device=device,
        )
        if not bool(is_alert_label.any()) or not bool((~is_alert_label).any()):
            raise ValueError(
                "label supervision needs both alert and normal patterns in "
                "label_patterns"
            )

    # Contrastive discrimination over the *joint feature channel*: push the worst
    # alert (soft lambda_min(G_+)) above the best negative (soft lambda_max(G_-)),
    # with G_pm = V_pm^T g_theta(A_hat) X W W^T X^T g_theta(A_hat) V_pm.  The
    # negatives come from `contrastive_patterns` (typically random connected sets
    # of matched size -- a size-controlled null); the positives are the alert
    # patterns already in `train_patterns`.  This gives the worst-case ratio the
    # learned feature map W, the capacity a theta-only contrastive filter lacks.
    use_contrastive = contrastive_weight > 0.0 and contrastive_patterns is not None
    pos_cols = None
    neg_signature_stack = None
    if use_contrastive:
        pos_cols = torch.tensor([label == "alert" for label in labels], device=device)
        if not bool(pos_cols.any()):
            raise ValueError("contrastive term needs alert patterns in train_patterns")
        V_neg = pattern_indicator_matrix(
            contrastive_patterns, adjacency.shape[0], dtype=dtype, device=device
        )
        propagated_neg = propagation_stack(adjacency, V_neg, degree)
        neg_signature_stack = torch.stack(
            [X.T @ signal for signal in propagated_neg], dim=0
        )  # (K+1, f, m_neg)

    eps = torch.finfo(dtype).eps
    eye_m = ridge * torch.eye(m, dtype=dtype, device=device)

    def channel_grams(
        theta: torch.Tensor, W: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if per_hop_features:
            # sum_k (X.T A_hat^k V).T W_k -> per-hop feature maps, theta absorbed.
            feature_sig = torch.einsum("kfm,kfd->dm", signature_stack, W)  # (d, m)
        else:
            feature_sig = W.T @ torch.einsum("k,kfm->fm", theta, signature_stack)  # (d, m)
        feature_gram = feature_sig.T @ feature_sig
        if structural_stack is None:
            return feature_gram, None
        structural_sig = torch.einsum("k,kmr->mr", theta, structural_stack)  # (m, r)
        return feature_gram, structural_sig @ structural_sig.T

    theta_raw = torch.nn.Parameter(torch.zeros(degree + 1, dtype=dtype, device=device))
    with torch.no_grad():
        theta_raw[-1] = 1.0
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    w_shape = (degree + 1, feature_dim, embed) if per_hop_features else (feature_dim, embed)
    W_raw = torch.nn.Parameter(
        torch.randn(*w_shape, dtype=dtype, device=device, generator=generator)
    )

    # Init-time per-channel normalization so neither term swamps the other.
    with torch.no_grad():
        init_theta = _unit(theta_raw)
        init_W = W_raw / W_raw.norm().clamp_min(eps)
        feature_gram0, structural_gram0 = channel_grams(init_theta, init_W)
        feature_scale = torch.diagonal(feature_gram0).mean().clamp_min(eps)
        structural_scale = (
            torch.diagonal(structural_gram0).mean().clamp_min(eps)
            if structural_gram0 is not None
            else None
        )

    # Channel dimension c (feature d (+ structural r)) and which Gram side the
    # retention objective uses.  When m > c the m x m pattern Gram is rank
    # starved, so the objective drops to the c x c channel Gram (same non-zero
    # spectrum, full rank, live gradient); see :func:`_retention_lambda_min`.
    channel_dim = embed + (structural_width if structural_stack is not None else 0)
    retention_side = _retention_side(channel_dim, m, retention_mode)

    def channel_signatures(theta: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """Per-channel-normalized signature matrix ``Y`` of shape ``(c, m)``.

        ``Y.T @ Y`` reproduces the normalized pattern Gram exactly (the old
        ``regularized_gram`` minus its ridge), while ``Y @ Y.T`` is the channel
        Gram with the same non-zero spectrum.
        """

        if per_hop_features:
            feature_sig = torch.einsum("kfm,kfd->dm", signature_stack, W)  # (d, m)
        else:
            feature_sig = W.T @ torch.einsum("k,kfm->fm", theta, signature_stack)
        blocks = [feature_sig / feature_scale.sqrt()]
        if structural_stack is not None:
            structural_sig = torch.einsum("k,kmr->mr", theta, structural_stack)  # (m, r)
            blocks.append((structural_sig / structural_scale.sqrt()).T)  # (r, m)
        return torch.cat(blocks, dim=0)  # (c, m)

    def retention_lambda_min(theta: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        Y = channel_signatures(theta, W)
        if retention_side == "channel":
            gram = Y @ Y.T  # (c, c)
            eye = ridge * torch.eye(gram.shape[0], dtype=dtype, device=device)
        else:
            gram = Y.T @ Y  # (m, m)
            eye = eye_m
        evals = torch.linalg.eigvalsh(0.5 * (gram + gram.T) + eye)
        return _reduce_eigs(evals, reduce=retention_reduce, temp=retention_temp)

    def label_margin(theta: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        # Mean-pooled signatures -> (m_lab, d); same feature channel as the Gram.
        if per_hop_features:
            signatures = torch.einsum("kfm,kfd->dm", label_signature_stack, W).T
        else:
            signatures = (
                W.T @ torch.einsum("k,kfm->fm", theta, label_signature_stack)
            ).T
        return _label_separation(signatures, is_alert_label, ridge=label_ridge)

    def _feature_sig(stack: torch.Tensor, theta: torch.Tensor, W: torch.Tensor):
        """Feature-channel signatures ``(d, m)`` for a signature stack ``(K+1,f,m)``."""
        if per_hop_features:
            return torch.einsum("kfm,kfd->dm", stack, W)
        return W.T @ torch.einsum("k,kfm->fm", theta, stack)

    def contrastive_ratio(theta: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        # soft lambda_min(G_+) / soft lambda_max(G_-) on the feature channel, as a
        # log-ratio (smooth, scale-free).  >0 means the worst alert out-retains the
        # best negative -- a worst-case margin discriminator with W capacity.
        f_pos = _feature_sig(signature_stack[:, :, pos_cols], theta, W)  # (d, m_pos)
        f_neg = _feature_sig(neg_signature_stack, theta, W)  # (d, m_neg)
        num = _retention_lambda_min(
            f_pos, mode=retention_mode, ridge=ridge, reduce="softmin",
            temp=retention_temp,
        )
        den = _soft_lambda_max(
            f_neg, mode=retention_mode, ridge=ridge, reduce="softmax",
            temp=retention_temp,
        )
        return torch.log(num.clamp_min(eps)) - torch.log(den.clamp_min(eps))

    with torch.no_grad():
        init_lambda = float(retention_lambda_min(init_theta, init_W))
        init_separation = (
            float(label_margin(init_theta, init_W)) if use_labels else None
        )
        init_contrastive = (
            float(contrastive_ratio(init_theta, init_W)) if use_contrastive else None
        )
    vanilla = init_lambda

    # Scale-balance the terms so ``label_weight`` / ``contrastive_weight`` are clean
    # relative weights.  With both off this leaves the objective as the raw
    # lambda_min, so the default path is byte-for-byte the original encoder.
    lambda_scale = 1.0
    separation_scale = 1.0
    contrastive_scale = 1.0
    if use_labels or use_contrastive:
        lambda_scale = abs(init_lambda) if abs(init_lambda) > eps else 1.0
    if use_labels:
        separation_scale = (
            init_separation if init_separation and init_separation > eps else 1.0
        )
    if use_contrastive:
        contrastive_scale = (
            abs(init_contrastive) if init_contrastive and abs(init_contrastive) > eps
            else 1.0
        )

    def combined_objective(theta: torch.Tensor, W: torch.Tensor):
        lam = retention_lambda_min(theta, W)
        sep = label_margin(theta, W) if use_labels else None
        con = contrastive_ratio(theta, W) if use_contrastive else None
        if not use_labels and not use_contrastive:
            return lam, lam, None, None
        total = lam / lambda_scale
        if use_labels:
            total = total + label_weight * sep / separation_scale
        if use_contrastive:
            total = total + contrastive_weight * con / contrastive_scale
        return total, lam, sep, con

    init_total = init_lambda
    if use_labels or use_contrastive:
        init_total = init_lambda / lambda_scale
        if use_labels:
            init_total += label_weight * (init_separation or 0.0) / separation_scale
        if use_contrastive:
            init_total += (
                contrastive_weight * (init_contrastive or 0.0) / contrastive_scale
            )

    optimizer = torch.optim.Adam((theta_raw, W_raw), lr=learning_rate)
    best_theta = init_theta.detach().clone()
    best_W = init_W.detach().clone()
    best_total = init_total
    best_lambda = init_lambda
    best_separation = init_separation
    best_contrastive = init_contrastive
    history: List[float] = [init_total]

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        theta = _unit(theta_raw)
        W = W_raw / W_raw.norm().clamp_min(eps)
        total, lam, sep, con = combined_objective(theta, W)
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite objective while fitting encoder")
        (-total).backward()
        optimizer.step()

        value = float(total.detach().cpu())
        history.append(value)
        if value > best_total:
            best_total = value
            best_lambda = float(lam.detach().cpu())
            best_separation = float(sep.detach().cpu()) if sep is not None else None
            best_contrastive = float(con.detach().cpu()) if con is not None else None
            best_theta = theta.detach().clone()
            best_W = W.detach().clone()

    return JointEncoderResult(
        theta=best_theta,
        feature_map=best_W,
        degree=degree,
        embed_dim=embed,
        structural_width=structural_width,
        ridge=ridge,
        seed=seed,
        objective=best_lambda,
        vanilla_objective=vanilla,
        history=history,
        train_labels=labels,
        label_weight=label_weight,
        combined_objective=best_total if (use_labels or use_contrastive) else None,
        label_separation=best_separation if use_labels else None,
        per_hop_features=per_hop_features,
        retention_side=retention_side,
        contrastive_weight=contrastive_weight,
        contrastive_ratio=best_contrastive if use_contrastive else None,
    )


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm().clamp_min(torch.finfo(vector.dtype).eps)


def _pattern_margin_scores(gram: torch.Tensor, ridge: float = 1e-10) -> torch.Tensor:
    """Per-pattern energy that remains after accounting for cross-talk.

    For a positive-definite Gram matrix this is ``1 / diag(G^-1)``: the squared
    norm of each filtered pattern after projection away from all other patterns.
    It is an individual counterpart to the collective ``lambda_min(G)`` target.
    A pseudoinverse gives a stable score if a test set contains duplicate or
    linearly dependent patterns.
    """

    gram = 0.5 * (gram + gram.T)
    stabilized = gram + ridge * torch.eye(
        gram.shape[0], dtype=gram.dtype, device=gram.device
    )
    inverse_diag = torch.diagonal(torch.linalg.pinv(stabilized)).clamp_min(ridge)
    return inverse_diag.reciprocal()


def fit_collective_sgc(
    adjacency: torch.Tensor,
    train_patterns: Sequence[Any],
    *,
    degree: int = 8,
    epochs: int = 400,
    learning_rate: float = 5e-2,
    threshold_quantile: float = 0.05,
    features: torch.Tensor | None = None,
    mode: str = "lambda_min",
    ridge: float = 1e-6,
    retention_mode: str = "auto",
    retention_reduce: str = "min",
    retention_temp: float = 0.1,
) -> SGCTrainingResult:
    """Fit unit-norm ``theta`` on training patterns.

    Two options control the Gram and the objective:

    * ``features`` -- when ``None`` the structural Gram ``G = F.T F`` is used
      (``Sigma_X = I``, isotropic/Gaussian features).  When the node-feature
      matrix ``X = graph.x`` is supplied, the feature-aware Gram
      ``G = P.T g_theta(A_hat) X X.T g_theta(A_hat) P`` is used instead.
    * ``mode`` -- ``"lambda_min"`` maximizes Eq. (48) ``lambda_min(G(theta))``
      (retain/resolve every pattern) by projected Adam on the unit sphere;
      ``"fisher"`` instead returns the top generalized eigenvector of the
      class-mean feature-moment pair ``(S_+, S_-)`` to *discriminate* alerts
      from normals (needs both classes in ``train_patterns``);
      ``"discriminative"`` trains ``theta`` by projected Adam to maximize the
      soft Fisher ratio of the per-pattern energies
      :func:`discriminative_score_ratio` (also needs both classes).

    ``retention_mode`` (``"lambda_min"`` mode only) controls the over-complete
    regime where the number of patterns ``m`` exceeds the channel rank ``r``
    (here ``r = f`` for the feature-aware Gram, ``r = N`` for the structural
    one).  ``"pattern"`` keeps the strict ``m x m`` Gram (whose ``lambda_min``
    collapses to ``0`` with no ``theta`` gradient once ``m > r``); ``"auto"``
    (default) switches to the channel-side ``r x r`` Gram ``Y Y.T`` in that
    regime so ``theta`` keeps a live gradient (see :func:`_retention_lambda_min`).
    For ``m <= r`` (and the structural encoder, ``N >> m``) ``"auto"`` is
    identical to ``"pattern"``.

    ``retention_reduce`` (with ``retention_temp``) chooses how the spectrum is
    reduced -- ``"min"`` (strict ``lambda_min``, default), ``"softmin"`` (smooth
    eigenvalue-weighted trace), or ``"mean"`` -- via :func:`_reduce_eigs`.
    """

    if degree < 0:
        raise ValueError("degree must be non-negative")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not 0.0 <= threshold_quantile <= 1.0:
        raise ValueError("threshold_quantile must be in [0, 1]")
    if mode not in ("lambda_min", "fisher", "discriminative", "contrastive_ratio"):
        raise ValueError(
            "mode must be 'lambda_min', 'fisher', 'discriminative', or "
            "'contrastive_ratio'"
        )
    if retention_mode not in ("auto", "pattern", "channel"):
        raise ValueError("retention_mode must be 'auto', 'pattern', or 'channel'")
    if retention_reduce not in ("min", "softmin", "mean"):
        raise ValueError("retention_reduce must be 'min', 'softmin', or 'mean'")
    if retention_temp <= 0:
        raise ValueError("retention_temp must be positive")

    device, dtype = adjacency.device, adjacency.dtype
    V = pattern_indicator_matrix(
        train_patterns, adjacency.shape[0], dtype=dtype, device=device
    )
    propagated = propagation_stack(adjacency, V, degree)
    if features is not None:
        features = features.to(device=device, dtype=dtype)
        if features.dim() == 1:
            features = features.unsqueeze(1)
        if features.shape[0] != adjacency.shape[0]:
            raise ValueError("features must have one row per graph node")
    labels = [str(pattern.label) for pattern in train_patterns]
    moments = feature_moment_matrices(propagated, features)

    # Channel signature matrix Y (r, m): r = f (feature-aware) or N (structural).
    m_patterns = V.shape[1]
    channel_rank = features.shape[1] if features is not None else adjacency.shape[0]
    retention_side = _retention_side(channel_rank, m_patterns, retention_mode)

    def _signatures(theta: torch.Tensor) -> torch.Tensor:
        filtered = filter_signals(propagated, theta)  # (N, m) = g_theta(A_hat) V
        if features is not None:
            filtered = features.T @ filtered  # (f, m) = Y
        return filtered

    initial_theta = torch.zeros(degree + 1, dtype=dtype, device=device)
    initial_theta[-1] = 1.0
    with torch.no_grad():
        vanilla = float(
            _retention_lambda_min(
                _signatures(initial_theta),
                mode=retention_mode,
                reduce=retention_reduce,
                temp=retention_temp,
            )
        )

    separation_ratio: float | None = None
    if mode == "fisher":
        best_theta, separation_ratio = fisher_theta(moments, labels, ridge=ridge)
        with torch.no_grad():
            best_objective = torch.linalg.eigvalsh(
                gram_matrix(propagated, best_theta, features)
            )[0].item()
        history: List[float] = [best_objective]
    elif mode == "discriminative":
        is_alert_t = torch.tensor([label == "alert" for label in labels], device=device)
        if not bool(is_alert_t.any()) or not bool((~is_alert_t).any()):
            raise ValueError(
                "discriminative objective needs both alert and normal "
                "training patterns"
            )
        raw_theta = torch.nn.Parameter(initial_theta.clone())
        optimizer = torch.optim.Adam((raw_theta,), lr=learning_rate)
        best_theta = initial_theta.clone()
        best_ratio = float("-inf")
        history = []
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            theta = _unit(raw_theta)
            scores = quadratic_scores(moments, theta)
            ratio = discriminative_score_ratio(scores, is_alert_t)
            if not torch.isfinite(ratio):
                raise FloatingPointError(
                    "non-finite discriminative ratio while fitting SGC"
                )
            (-ratio).backward()
            optimizer.step()

            value = float(ratio.detach().cpu())
            history.append(value)
            if value > best_ratio:
                best_ratio = value
                best_theta = theta.detach().clone()
        separation_ratio = best_ratio
        with torch.no_grad():
            best_objective = torch.linalg.eigvalsh(
                gram_matrix(propagated, best_theta, features)
            )[0].item()
    elif mode == "contrastive_ratio":
        # Worst-case / margin discriminator:  max_theta  lambda_min(G_+) /
        # lambda_max(G_-),  G_pm = V_pm^T g_theta(A_hat)^2 V_pm (feature-aware when
        # `features` is given).  Numerator retains every positive (no gang missed);
        # denominator suppresses every negative (no look-alike passed); ratio > 1
        # at the optimum is a hard worst-case separation with a margin.  Scale-free
        # (both extremal eigenvalues carry the filter gain), so no beta to tune.
        is_alert_t = torch.tensor(
            [label == "alert" for label in labels], device=device
        )
        if not bool(is_alert_t.any()) or not bool((~is_alert_t).any()):
            raise ValueError(
                "contrastive_ratio objective needs both alert and normal "
                "training patterns"
            )
        pos_cols, neg_cols = is_alert_t, ~is_alert_t
        eps_t = torch.finfo(dtype).eps
        raw_theta = torch.nn.Parameter(initial_theta.clone())
        optimizer = torch.optim.Adam((raw_theta,), lr=learning_rate)

        def _ratio(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            signatures = _signatures(theta)  # (r, m)
            num = _retention_lambda_min(  # soft lambda_min(G_+)
                signatures[:, pos_cols], mode=retention_mode, ridge=ridge,
                reduce="softmin", temp=retention_temp,
            )
            den = _soft_lambda_max(  # soft lambda_max(G_-)
                signatures[:, neg_cols], mode=retention_mode, ridge=ridge,
                reduce="softmax", temp=retention_temp,
            )
            return num, den

        with torch.no_grad():
            n0, d0 = _ratio(initial_theta)
            vanilla_ratio = float((n0 / d0.clamp_min(eps_t)).cpu())
        best_theta = initial_theta.clone()
        best_ratio = vanilla_ratio
        history = [vanilla_ratio]
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            theta = _unit(raw_theta)
            num, den = _ratio(theta)
            # maximize the log-ratio: smooth, scale-free, and avoids the divide
            # blowing up when the denominator is briefly tiny.
            objective = torch.log(num.clamp_min(eps_t)) - torch.log(
                den.clamp_min(eps_t)
            )
            if not torch.isfinite(objective):
                raise FloatingPointError(
                    "non-finite contrastive ratio while fitting SGC"
                )
            (-objective).backward()
            optimizer.step()
            value = float((num / den.clamp_min(eps_t)).detach().cpu())
            history.append(value)
            if value > best_ratio:
                best_ratio = value
                best_theta = theta.detach().clone()
        separation_ratio = best_ratio
        with torch.no_grad():
            best_objective = torch.linalg.eigvalsh(
                gram_matrix(propagated, best_theta, features)
            )[0].item()
    else:
        raw_theta = torch.nn.Parameter(initial_theta.clone())
        optimizer = torch.optim.Adam((raw_theta,), lr=learning_rate)
        # Keep vanilla SGC as a feasible candidate.  This means optimization
        # cannot report a result worse than the requested SGC baseline if an
        # Adam step is unhelpful for a particular AMLGenTex split.
        history = [vanilla]
        best_theta = initial_theta.clone()
        best_objective = vanilla

        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            theta = _unit(raw_theta)
            objective = _retention_lambda_min(
                _signatures(theta),
                mode=retention_mode,
                reduce=retention_reduce,
                temp=retention_temp,
            )
            if not torch.isfinite(objective):
                raise FloatingPointError("non-finite lambda_min(G) while fitting SGC")
            (-objective).backward()
            optimizer.step()

            value = float(objective.detach().cpu())
            history.append(value)
            if value > best_objective:
                best_objective = value
                best_theta = theta.detach().clone()

    with torch.no_grad():
        final_gram = gram_matrix(propagated, best_theta, features)
        train_scores = _pattern_margin_scores(final_gram)
        threshold = torch.quantile(train_scores, threshold_quantile).item()

    # Report the separation of the *decision* score -- the cross-talk-adjusted
    # margin s_j that ``threshold`` actually gates in ``detect_patterns`` -- so
    # ``auc`` matches the quantity being thresholded.  The raw retained energy
    # ``theta^T M_j theta`` is a different quantity (it ignores cross-talk); the
    # held-out histogram in the runner plots those energies separately.
    is_alert = torch.tensor(
        [label == "alert" for label in labels], device=train_scores.device
    )
    alert_scores = train_scores[is_alert].detach().cpu().tolist()
    normal_scores = train_scores[~is_alert].detach().cpu().tolist()
    auc = (
        _roc_auc(alert_scores, normal_scores)
        if alert_scores and normal_scores
        else None
    )

    return SGCTrainingResult(
        theta=best_theta,
        objective=best_objective,
        vanilla_sgc_objective=vanilla,
        train_threshold=threshold,
        degree=degree,
        epochs=epochs,
        history=history,
        train_scores=train_scores.detach().cpu().tolist(),
        train_labels=labels,
        mode=mode,
        feature_aware=features is not None,
        separation_ratio=separation_ratio,
        alert_scores=alert_scores,
        normal_scores=normal_scores,
        auc=auc,
        retention_side=retention_side if mode == "lambda_min" else "pattern",
    )


def detect_patterns(
    adjacency: torch.Tensor,
    test_patterns: Sequence[Any],
    fit: SGCTrainingResult,
    features: torch.Tensor | None = None,
) -> tuple[List[PatternDetection], Dict[str, Dict[str, Any]]]:
    """Score only held-out patterns using a theta learned from training patterns.

    A pattern is detected when its cross-talk-adjusted retained energy is at
    least the lower-quantile training threshold.  Results are grouped by the
    source label first (``alert`` or ``normal``), then pattern type.  This
    prevents a type name shared by both families from being merged in the
    detection report.

    ``features`` must be the *same* node-feature matrix ``X`` used to fit
    ``theta`` and calibrate ``fit.train_threshold``.  Otherwise the held-out
    Gram would use ``Sigma_X = I`` while the threshold was calibrated
    feature-aware (``Sigma_X = X X^T``), scoring test patterns inconsistently.
    """

    if not test_patterns:
        return [], {}

    if fit.feature_aware and features is None:
        raise ValueError(
            "fit is feature-aware (theta and threshold calibrated with X); pass "
            "the same node-feature matrix to detect_patterns via features="
        )

    V = pattern_indicator_matrix(
        test_patterns,
        adjacency.shape[0],
        dtype=adjacency.dtype,
        device=adjacency.device,
    )
    propagated = propagation_stack(adjacency, V, fit.degree)
    theta = fit.theta.to(device=adjacency.device, dtype=adjacency.dtype)
    if features is not None:
        features = features.to(device=adjacency.device, dtype=adjacency.dtype)
        if features.dim() == 1:
            features = features.unsqueeze(1)
        if features.shape[0] != adjacency.shape[0]:
            raise ValueError("features must have one row per graph node")
    gram = gram_matrix(propagated, theta, features)
    retained = torch.diagonal(gram)
    separated = _pattern_margin_scores(gram)

    detections: List[PatternDetection] = []
    grouped: Dict[str, Dict[str, List[PatternDetection]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, pattern in enumerate(test_patterns):
        score = float(separated[index].detach().cpu())
        result = PatternDetection(
            pattern_id=str(pattern.id),
            pattern_type=str(pattern.pattern_type),
            label=str(pattern.label),
            retained_energy=float(retained[index].detach().cpu()),
            separation_energy=score,
            score=score,
            detected=score >= fit.train_threshold,
        )
        detections.append(result)
        grouped[result.label][result.pattern_type].append(result)

    by_label: Dict[str, Dict[str, Any]] = {}
    for label, patterns_by_type in sorted(grouped.items()):
        type_results: Dict[str, Dict[str, float]] = {}
        all_entries: List[PatternDetection] = []
        for pattern_type, entries in sorted(patterns_by_type.items()):
            detected = sum(entry.detected for entry in entries)
            type_results[pattern_type] = {
                "detected": detected,
                "total": len(entries),
                "detection_rate": detected / len(entries),
                "mean_retained_energy": sum(entry.retained_energy for entry in entries)
                / len(entries),
                "mean_separation_energy": sum(
                    entry.separation_energy for entry in entries
                )
                / len(entries),
            }
            all_entries.extend(entries)
        detected = sum(entry.detected for entry in all_entries)
        by_label[label] = {
            "detected": detected,
            "total": len(all_entries),
            "detection_rate": detected / len(all_entries),
            "by_pattern_type": type_results,
        }
    return detections, by_label


def train_and_detect(
    graph: Any,
    train_patterns: Sequence[Any],
    test_patterns: Sequence[Any],
    **fit_kwargs: Any,
) -> tuple[SGCTrainingResult, List[PatternDetection], Dict[str, Dict[str, Any]]]:
    """Convenience entry point for AMLGenTex ``Data`` returned by the loader."""

    adjacency = normalized_adjacency(
        graph.edge_index,
        int(graph.num_nodes),
        getattr(graph, "edge_weight", None),
    )
    fit = fit_collective_sgc(adjacency, train_patterns, **fit_kwargs)
    detections, by_label = detect_patterns(
        adjacency, test_patterns, fit, features=fit_kwargs.get("features")
    )
    return fit, detections, by_label


def save_report(
    path: str | Path,
    fit: SGCTrainingResult,
    detections: Iterable[PatternDetection],
    by_label: Mapping[str, Mapping[str, Any]],
) -> None:
    """Persist a JSON report without serializing GPU tensors."""

    payload = {
        "theta": fit.theta.detach().cpu().tolist(),
        "objective_lambda_min_G": fit.objective,
        "vanilla_sgc_lambda_min_G": fit.vanilla_sgc_objective,
        "training_score_threshold": fit.train_threshold,
        "degree": fit.degree,
        "epochs": fit.epochs,
        "training_pattern_count": len(fit.train_labels),
        "training_pattern_labels": sorted(set(fit.train_labels)),
        "detection_rate_by_label": by_label,
        "patterns": [asdict(detection) for detection in detections],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def plot_diagnostics(
    directory: str | Path,
    fit: SGCTrainingResult,
    detections: Sequence[PatternDetection],
    by_label: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Path]:
    """Write plots that validate optimization and held-out detection behavior.

    The plots deliberately expose the two separate test families: alerts and
    normals.  Matplotlib is imported lazily so SGC fitting itself has only a
    PyTorch dependency.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output: Dict[str, Path] = {}

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(range(len(fit.history)), fit.history, color="tab:blue", linewidth=1.5)
    axis.axhline(
        fit.vanilla_sgc_objective,
        color="tab:gray",
        linestyle="--",
        label="vanilla SGC",
    )
    axis.set(xlabel="optimization step", ylabel=r"$\lambda_{\min}(G(\theta))$")
    axis.set_title("Eq. (48) collective SGC objective")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    output["objective_history"] = directory / "objective_history.png"
    fig.savefig(output["objective_history"], dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    theta = fit.theta.detach().cpu().tolist()
    axis.stem(range(len(theta)), theta, basefmt=" ")
    axis.set(xlabel="polynomial degree k", ylabel=r"$\theta_k$")
    axis.set_title("Learned trainable-SGC coefficients")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    output["theta"] = directory / "theta_coefficients.png"
    fig.savefig(output["theta"], dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    labels = sorted({detection.label for detection in detections})
    for label in labels:
        scores = [
            detection.score for detection in detections if detection.label == label
        ]
        if scores:
            axis.hist(scores, bins="auto", alpha=0.55, label=f"test {label}")
    for label in sorted(set(fit.train_labels)):
        scores = [
            score
            for score, train_label in zip(fit.train_scores, fit.train_labels)
            if train_label == label
        ]
        if scores:
            axis.hist(
                scores,
                bins="auto",
                histtype="step",
                linewidth=2,
                label=f"train {label}",
            )
    axis.axvline(
        fit.train_threshold,
        color="black",
        linestyle="--",
        label="training threshold",
    )
    axis.set(xlabel="cross-talk-adjusted retained energy", ylabel="pattern count")
    axis.set_title("Train-calibrated scores on held-out alert and normal patterns")
    axis.legend()
    fig.tight_layout()
    output["scores"] = directory / "train_test_score_distributions.png"
    fig.savefig(output["scores"], dpi=160)
    plt.close(fig)

    labels = [label for label in ("alert", "normal") if label in by_label]
    if not labels:
        labels = list(by_label)
    if not labels:
        return output
    fig, axes = plt.subplots(
        1, len(labels), figsize=(max(6, 4 * len(labels)), 4), squeeze=False
    )
    for axis, label in zip(axes[0], labels):
        per_type = by_label[label]["by_pattern_type"]
        types = list(per_type)
        rates = [per_type[pattern_type]["detection_rate"] for pattern_type in types]
        bars = axis.bar(
            types, rates, color="tab:red" if label == "alert" else "tab:green"
        )
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("detection rate")
        axis.set_title(f"Held-out {label} patterns")
        axis.tick_params(axis="x", rotation=30)
        for bar, pattern_type in zip(bars, types):
            metrics = per_type[pattern_type]
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{int(metrics['detected'])}/{int(metrics['total'])}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("Detection rate by pattern type and label")
    fig.tight_layout()
    output["detection_rates"] = directory / "detection_rates_by_label_and_type.png"
    fig.savefig(output["detection_rates"], dpi=160)
    plt.close(fig)
    return output
