"""Trace-ratio (Dinkelbach) solver for the collective bank: eigensolves, not epochs.

The gradient path ascends ``lambda_min(Gamma(Theta))`` with Adam for hundreds of
epochs.  The same stationary points are reachable in *tens of small
eigenproblems* by exploiting two facts about the bank parameterization.

**1. The maximin splits.**  With ``lambda_min(Gamma) = min_{U >= 0, tr U = 1}
tr(U Gamma)``, the problem is a game: an outer player picks the density matrix
``U`` (which gang direction is hardest), an inner player picks ``Theta``.  For a
*fixed* ``U`` the inner objective is a trace ratio,

    tr(U Gamma(Theta)) = tr[ (S^T G S)^{-1} S^T A S ],     A = b U b^T,

where ``b = T^T M_tau Vhat`` are the dictionary alignments and ``S = S(Theta)``
is the (channel-block-structured) matrix with ``Z = T S``.

**2. The Dinkelbach step decouples per channel.**  Linearizing the ratio at the
current value ``lam`` gives ``max_Theta tr[S^T (A - lam G) S]``, and because the
trace only sees the *diagonal* channel blocks of ``S``, this separates into ``d``
independent problems of size ``(K+1) x (K+1)``::

    max_{||theta_{h,a}||=1}  sum_{h,a} theta_{h,a}^T C_a(lam) theta_{h,a},
    C_a(lam) = A_aa - beta W_aa - lam G_aa.

For one head the maximizer is the top eigenvector of ``C_a``; for ``H`` heads,
the top ``H`` eigenvectors.  So each iteration is ``d`` symmetric eigenproblems
of size ``K+1`` (milliseconds), and **the heads come out mutually orthogonal per
channel by construction** -- head collapse is impossible here, so no diversity
penalty and no random-init symmetry breaking are needed.

The confusability enters exactly as in the pencil metric: ``W_all = sum_l W_l``
restricted to the same channel blocks, subtracted with weight ``beta``.  Its
low-rank factor ``B_all`` is reused from :mod:`src.margin_pencil`, so only the
``d`` blocks of size ``(K+1)^2`` are ever formed -- never the ``P x P`` matrix.

Everything is ``O(d (K+1)^3 + m^3)`` per iteration and independent of ``N`` after
the one-time moment accumulation, exactly like the gradient path's kernel.
"""

from __future__ import annotations

import numpy as np
import torch

from src.run_collective_bank_detection import (
    _basis_stack,
    _collective_gamma_mz,
    _filtered_bank,
    _m_apply,
    _train_gang_m_vhat,
)
from src.utils.utils import LOGGER


def _density_matrix(gamma: torch.Tensor, temperature: float) -> torch.Tensor:
    """Outer player's move: the (soft-)min density matrix of ``Gamma``.

    ``temperature = 0`` gives the hard ``u_min u_min^T`` (Danskin); ``> 0``
    spreads weight over the low end of the spectrum, which is the matrix
    analogue of the soft-min the gradient path already uses and keeps the
    iteration from chattering between near-degenerate eigendirections.
    """

    evals, evecs = torch.linalg.eigh(gamma)
    if temperature <= 0:
        w = torch.zeros_like(evals)
        w[0] = 1.0
    else:
        w = torch.softmax(-(evals - evals.min()) / max(temperature, 1e-8), dim=0)
    return (evecs * w.unsqueeze(0)) @ evecs.T


def fit_trace_ratio_bank(
    days: list,
    *,
    degree: int,
    heads: int = 1,
    iters: int = 40,
    tau: float = 0.5,
    ridge: float = 1e-3,
    basis: str = "chebyshev",
    softmin_temperature: float = 0.2,
    conf_weight: float = 0.0,
    tol: float = 1e-10,
) -> dict:
    """Solve the collective bank objective by trace-ratio iteration.

    ``days`` is ``[(label, a_hat, adjacency, train_patterns, X), ...]``.  With one
    entry this is the plain single-graph solve; with several, each iteration
    draws one graph uniformly (the same "one group per step" rule the gradient
    multi-day path uses), so the filter must satisfy every graph rather than
    memorizing one.  Per-graph blocks are precomputed once, so the drawn graph
    costs nothing extra at iteration time.

    Returns the same report shape as
    :func:`~src.run_collective_bank_detection.fit_collective_bank` so the
    downstream reporting and training plots work unchanged.
    """

    if not days:
        raise ValueError("days must be non-empty")
    dtype = days[0][4].dtype
    d = int(days[0][4].shape[1])
    K1 = degree + 1
    m = sum(len(spec[3]) for spec in days)
    rng = np.random.default_rng(0)

    # ---- one-time moment accumulation per graph (the only O(N) work) -------
    bundles = []
    for lbl, a_hat, adjacency, patterns, X in days:
        if int(X.shape[1]) != d:
            raise ValueError(
                f"group {lbl} has feature-dim {X.shape[1]} != {d}; the filter is "
                "shared so the feature dimension must match."
            )
        prop_i = _basis_stack(a_hat, X, degree, basis, tau)
        Tst = torch.stack(prop_i, dim=1)  # (N, K+1, d)
        m_prop_i = [_m_apply(a_hat, P, tau) for P in prop_i]
        MTst = torch.stack(m_prop_i, dim=1)
        m_v = _train_gang_m_vhat(a_hat, adjacency, patterns, tau)  # (N, m_i)
        G_i = torch.einsum("nka,nla->kla", Tst, MTst)
        G_i = 0.5 * (G_i + G_i.transpose(0, 1))
        b_i = torch.einsum("nka,nj->kaj", MTst, m_v)  # (K+1, d, m_i)
        bundles.append({"label": lbl, "prop": prop_i, "m_prop": m_prop_i,
                        "m_vhat": m_v, "G": G_i, "b": b_i})
    # the reported/So-far "current" graph defaults to the last group
    cur = bundles[-1]
    prop, m_prop, m_vhat = cur["prop"], cur["m_prop"], cur["m_vhat"]
    G_blk, b_blk = cur["G"], cur["b"]

    # Confusability is NOT handled on this path yet: W_all's channel blocks need
    # the un-whitened factor of W_l, which :mod:`src.margin_pencil` only keeps in
    # whitened form.  The iteration below therefore solves the pure capture
    # objective; use the gradient solver when the chi penalty matters.
    W_blk = torch.zeros(K1, K1, d, dtype=dtype)
    if conf_weight > 0.0:
        LOGGER.warning(
            "  trace-ratio solver ignores --conf-weight (capture objective only); "
            "use --collective-solver gradient for the margin objective."
        )
        conf_weight = 0.0

    # ---- initialization: the capture-only Dinkelbach step at lam = 0 -------
    theta = torch.zeros(heads, K1, d, dtype=dtype)
    A0 = torch.einsum("kaj,laj->kla", b_blk, b_blk)  # U = I/m up to scale
    for a in range(d):
        ev, evec = torch.linalg.eigh(A0[:, :, a] - conf_weight * W_blk[:, :, a])
        for h in range(heads):
            theta[h, :, a] = evec[:, -1 - min(h, K1 - 1)]

    history, energy_history, conf_history, margin_history = [], [], [], []
    best_theta, best_val = theta.clone(), -float("inf")
    prev = None
    for it in range(iters):
        if len(bundles) > 1:  # draw one graph per iteration
            cur = bundles[int(rng.integers(len(bundles)))]
            prop, m_prop, m_vhat = cur["prop"], cur["m_prop"], cur["m_vhat"]
            G_blk, b_blk = cur["G"], cur["b"]
        # --- current Gamma (N-independent given the filtered bank) ---------
        Z = _filtered_bank(prop, theta)
        m_z = _filtered_bank(m_prop, theta)
        gamma = _collective_gamma_mz(Z, m_z, m_vhat, ridge)
        lam_min = float(torch.linalg.eigvalsh(gamma)[0])
        diag = torch.diagonal(gamma).clamp(0.0, 1.0)
        history.append(lam_min)
        energy_history.append(float(diag.mean()))

        # --- outer player: hardest direction -------------------------------
        U = _density_matrix(gamma, softmin_temperature)
        lam = float(torch.einsum("ij,ji->", U, gamma))
        margin_history.append(lam)
        if lam_min > best_val:
            best_val, best_theta = lam_min, theta.clone()

        # --- inner player: Dinkelbach step, d small eigenproblems ----------
        # A_a[k,l] = sum_{j,n} b[k,a,j] U[j,n] b[l,a,n]
        A_blk = torch.einsum("kaj,jn,lan->kla", b_blk, U, b_blk)  # (K+1, K+1, d)
        A_blk = 0.5 * (A_blk + A_blk.transpose(0, 1))
        C = A_blk - conf_weight * W_blk - lam * G_blk
        new_theta = torch.empty_like(theta)
        for a in range(d):
            ev, evec = torch.linalg.eigh(C[:, :, a])
            for h in range(heads):
                new_theta[h, :, a] = evec[:, -1 - min(h, K1 - 1)]
        theta = new_theta
        conf_history.append(0.0)

        if prev is not None and abs(lam - prev) <= tol * max(abs(lam), 1e-12):
            LOGGER.info(f"    trace-ratio converged at iteration {it + 1}")
            break
        prev = lam

    # final evaluation at the best iterate, on the reported (last) group
    fin = bundles[-1]
    Z = _filtered_bank(fin["prop"], best_theta)
    m_z = _filtered_bank(fin["m_prop"], best_theta)
    gamma = _collective_gamma_mz(Z, m_z, fin["m_vhat"], ridge)
    diag = torch.diagonal(gamma).clamp(0.0, 1.0)
    return {
        "theta": best_theta,
        "init_objective": history[0] if history else 0.0,
        "objective": float(torch.linalg.eigvalsh(gamma)[0]),
        "margin": best_val,
        "confusability_init": 0.0,
        "confusability": 0.0,
        "confusability_mean": 0.0,
        "neg_objective_init": 0.0,
        "neg_objective": 0.0,
        "neg_objective_mean": 0.0,
        "history": history,
        "neg_history": [],
        "conf_history": conf_history,
        "energy_history": energy_history,
        "margin_history": margin_history,
        "ce_history": [],
        "capture_min_history": [],
        "head_sim_history": [],
        "heads": heads,
        "head_similarity": 0.0,
        "snapshots": [],
        "n_train_patterns": m,
        "train_days": [b["label"] for b in bundles],
        "capture_objective": "lambda_min (trace-ratio)",
        "conf_weight": conf_weight,
        "label_weight": 0.0,
        "iterations": len(history),
        "per_gang_capture": [float(v) for v in diag],
    }
