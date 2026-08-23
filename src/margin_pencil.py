"""Closed-form margin pencil: one learned direction per gang, no gradient steps.

Implements the single-direction margin theory.  For a rank-one target in the
FULL Chebyshev dictionary ``T = [T_0(A_hat)X, ..., T_K(A_hat)X]`` (all
``P = (K+1)d`` coefficients free -- "one direction", not one feature), capture
and confusability are Rayleigh quotients with the *same* denominator, so the
margin is a single generalized Rayleigh quotient and the optimum is an
eigenvalue problem rather than an optimization::

    C(theta)     = (theta^T b)^2 / ((Phi + tau) theta^T G theta)
    chi(theta)   = theta^T W theta / (theta^T G theta)
    D_beta(theta)= C - beta*chi
                 = theta^T (b b^T/(Phi+tau) - beta W) theta / (theta^T G theta)

with, writing ``S`` for the gang, ``s = |S|``:

    G   = T^T M_tau T                                   (P, P)
    b   = T^T M_tau v_S                                 (P,)
    Y_S = [(M_tau T_k(A_hat) x_a)_i sqrt(dtilde_i)]     (P, s),  i in S
    Q   = L_S^int + diag(d_boundary) + tau * Dtilde_S   (s, s)   local M_tau form
    P_0 = basis of {z : sum_{i in S} dtilde_i z_i = 0}  (s, s-1)
    W   = Y_S P_0 (P_0^T Q P_0)^{-1} P_0^T Y_S^T        (P, P), PSD, rank <= s-1

**Two solvers.**

``solve_gang_pencil`` -- the exact single-gang optimum

    D*(beta)     = lambda_max( G^{-1} ( b b^T/(Phi+tau) - beta W ) )
    theta*(beta) = (lambda G + beta W)^{-1} b  at lambda = D*(beta)

computed through the secular equation rather than a ``P x P`` eigensolve: in
``G``-whitened coordinates ``W`` has rank ``<= s-1``, so with
``btilde = G^{-1/2} b / sqrt(Phi+tau)``, ``Wtilde = sum_j omega_j q_j q_j^T``,
``c_j = q_j^T btilde`` and ``c_0^2`` the mass of ``btilde`` on ``ker Wtilde``,

    f(lambda) = c_0^2/lambda + sum_j c_j^2/(lambda + beta omega_j) = 1

has a unique positive root, which is ``D*(beta)``.  Cost is one whitening plus
an ``O(s^3)`` problem per gang -- no ``P``-sized eigendecomposition.

``solve_joint_pencil`` -- the ``W_all`` surrogate (metric-regularized m-gang
solve): one *shared* regularized Gram and ``m`` linear solves,

    G_beta = G + beta * sum_l W_l,      theta_j = G_beta^{-1} b_j

with the certificates the theory attaches to it: a capture floor
``C_j >= b_j^T G_beta^{-1} b_j/(Phi_j+tau)``, a joint confusability bound
``chi_{S_l}(theta_j) <= (r_j-1)/beta`` with
``r_j = theta_j^T G_beta theta_j / theta_j^T G theta_j``, and a Gershgorin
separation certificate on the direction Gram.  ``sum_l W_l = B_all B_all^T`` is
low rank, so Woodbury turns the ``P x P`` solve into a
``(sum_l (s_l-1))``-sized one.

**Limits worth checking against the rest of the pipeline.**  At ``beta = 0``
the direction is ``theta = G^{-1} b``, i.e. ``T theta = Pi^{M_tau}_{col T} v_S``
-- *exactly* the ``coarsen_target="dictionary"`` column, so ``D*(0)`` must equal
that gang's capture ceiling.  As ``beta -> infinity`` the direction converges to
the capture solve constrained to ``ker W`` (screened response flat on the gang),
with ``chi -> 0``: the projected-indicator hand-over target is *derived*, not
assumed.  Exact annihilation with positive margin needs ``(K+1) d >= s``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.run_collective_bank_detection import (
    _basis_stack,
    _collective_gamma_mz,
    _l_apply,
    _m_apply,
    _train_gang_m_vhat,
    degree_weighted_indicators,
)


# --------------------------------------------------------------------------- #
# dictionary-level objects (built once per graph)
# --------------------------------------------------------------------------- #
@dataclass
class PencilSpace:
    """Whitened dictionary: everything gang-independent, built once.

    ``G = T^T M_tau T`` is rank-revealed once and replaced by the whitening
    ``Vr Lr^{-1/2}``; all per-gang work then happens in the ``r``-dimensional
    whitened space where ``G`` is the identity, which is what makes the
    certificates below scale-free and the solves numerically clean.
    """

    propagated: list  # [phi_k(A_hat) X]
    T: torch.Tensor  # (N, P)
    MT: torch.Tensor  # (N, P) = M_tau T
    whiten: torch.Tensor  # (P, r): G^{-1/2} in the sense  Vr Lr^{-1/2}
    rank: int
    tau: float

    @property
    def dim(self) -> int:
        return int(self.T.shape[1])


def build_pencil_space(
    a_hat: torch.Tensor,
    X: torch.Tensor,
    degree: int,
    tau: float,
    *,
    basis: str = "chebyshev",
    rel_cutoff: float = 1e-10,
) -> PencilSpace:
    """Dictionary Gram ``G`` and its rank-revealing whitening."""

    propagated = _basis_stack(a_hat, X, degree, basis, tau)
    T = torch.cat(propagated, dim=1)  # (N, P)
    MT = _m_apply(a_hat, T, tau)  # (N, P)
    G = T.T @ MT
    G = 0.5 * (G + G.T)
    evals, evecs = torch.linalg.eigh(G)
    keep = evals > evals.max().clamp_min(1e-300) * rel_cutoff
    whiten = evecs[:, keep] / evals[keep].sqrt().unsqueeze(0)  # (P, r)
    return PencilSpace(
        propagated=propagated, T=T, MT=MT, whiten=whiten,
        rank=int(keep.sum()), tau=float(tau),
    )


@dataclass
class GangPencil:
    """Per-gang whitened pencil data: ``btilde`` and the low-rank ``Btilde``."""

    gang_id: str
    size: int
    phi: float
    b_tilde: torch.Tensor  # (r,)   G^{-1/2} b / sqrt(Phi + tau)
    B_tilde: torch.Tensor  # (r, s-1)  Wtilde = B_tilde B_tilde^T
    c_block: float  # ||b_tilde||^2 = capture ceiling = D*(0)


def _fluctuation_basis(dtilde_S: torch.Tensor) -> torch.Tensor:
    """Basis ``P_0`` (s, s-1) of ``{z : sum_i dtilde_i z_i = 0}`` (Householder)."""

    s = dtilde_S.numel()
    v = dtilde_S.clone()
    nrm = v.norm().clamp_min(torch.finfo(v.dtype).eps)
    e = torch.zeros_like(v)
    e[0] = nrm
    u = v - e
    if u.norm() < 1e-14:
        return torch.eye(s, dtype=v.dtype)[:, 1:]
    u = u / u.norm()
    H = torch.eye(s, dtype=v.dtype) - 2.0 * torch.outer(u, u)
    return H[:, 1:]


def build_gang_pencil(
    space: PencilSpace,
    a_hat: torch.Tensor,
    adjacency: torch.Tensor,
    pattern,
    v_S: torch.Tensor,
    phi: float,
    *,
    signal_only: bool = False,
    dense_adj: "torch.Tensor | None" = None,
) -> GangPencil:
    """Whitened ``b`` and the factor ``B`` with ``W = B B^T`` for one gang.

    ``signal_only`` returns just ``b_tilde`` with an empty ``B``: groups that
    only widen ``A_eq``'s signal span never enter ``H``, and skipping the
    fluctuation machinery avoids an ``s x s`` eigendecomposition (and the dense
    adjacency) per filler community.  ``dense_adj`` lets a caller densify once
    and share it across groups instead of per group.
    """

    tau = space.tau
    dtype = space.T.dtype
    idx = torch.as_tensor(sorted(int(i) for i in pattern.node_indices), dtype=torch.long)
    s = int(idx.numel())

    # b = T^T M_tau v_S  (M_tau symmetric, so use the cached M_tau T)
    b = space.MT.T @ v_S  # (P,)
    b_tilde = (space.whiten.T @ b) / float(np.sqrt(phi + tau))

    if s < 2 or signal_only:  # no internal fluctuations used: W = 0
        return GangPencil(str(pattern.id), s, phi, b_tilde,
                          torch.zeros(space.rank, 0, dtype=dtype),
                          float(b_tilde.pow(2).sum()))

    if dense_adj is not None:
        A = dense_adj
    else:
        A = adjacency.to_dense() if adjacency.is_sparse else adjacency
    W_SS = A[idx][:, idx].to(dtype)
    deg_all = A.sum(1).to(dtype)
    dtilde = deg_all + 1.0  # D_tilde = D(W + I)
    d_int = W_SS.sum(1)
    d_bnd = deg_all[idx] - d_int  # edges leaving S
    # local M_tau form: Q = L_S^int + diag(d_boundary) + tau * Dtilde_S
    Q = torch.diag(d_int) - W_SS + torch.diag(d_bnd) + tau * torch.diag(dtilde[idx])
    Q = 0.5 * (Q + Q.T)

    P0 = _fluctuation_basis(dtilde[idx])  # (s, s-1)
    A_q = P0.T @ Q @ P0  # (s-1, s-1), SPD
    A_q = 0.5 * (A_q + A_q.T)
    ev, evec = torch.linalg.eigh(A_q)
    keep = ev > ev.max().clamp_min(1e-300) * 1e-12
    A_inv_half = evec[:, keep] / ev[keep].sqrt().unsqueeze(0)  # (s-1, k)

    # Y_S[(k,a), i] = (M_tau T_k x_a)_i sqrt(dtilde_i)
    Y = (space.MT[idx] * dtilde[idx].sqrt().unsqueeze(1)).T  # (P, s)
    B = Y @ P0 @ A_inv_half  # (P, k):  W = B B^T
    B_tilde = space.whiten.T @ B  # (r, k)
    return GangPencil(str(pattern.id), s, phi, b_tilde, B_tilde,
                      float(b_tilde.pow(2).sum()))


# --------------------------------------------------------------------------- #
# solver 1: the exact single-gang pencil  (eq. pencil-solution)
# --------------------------------------------------------------------------- #
def _secular_root(c0_sq: float, c_sq: np.ndarray, omega: np.ndarray,
                  beta: float, hi: float, tol: float = 1e-14) -> float:
    """Unique positive root of ``c0^2/l + sum_j c_j^2/(l + beta w_j) = 1``.

    ``f`` is strictly decreasing on ``(0, hi]`` with ``f(hi) <= 1``, so plain
    bisection is both safe and fast; ``hi = ||btilde||^2 = D*(0)``.
    """

    if hi <= 0:
        return 0.0

    def f(l: float) -> float:
        out = c0_sq / l if c0_sq > 0 else 0.0
        if c_sq.size:
            out += float(np.sum(c_sq / (l + beta * omega)))
        return out

    if f(hi) >= 1.0 - 1e-15:  # beta = 0 (or W kills nothing): root is at hi
        return hi
    lo = hi * 1e-18
    if f(lo) < 1.0:  # c_0 = 0 and beta large: margin collapses to ~0
        return 0.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if f(mid) > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(hi, 1e-30):
            break
    return 0.5 * (lo + hi)


def solve_gang_pencil(gp: GangPencil, beta: float) -> dict:
    """``D*(beta)``, the whitened optimal direction, and its exact ``C``/``chi``.

    Returns ``u`` (whitened direction; un-whiten with ``space.whiten @ u`` to get
    ``theta``), the attained margin, and the two components evaluated *at the
    optimizer* -- so ``C - beta*chi == D_star`` is a free self-check.
    """

    b = gp.b_tilde
    if gp.B_tilde.shape[1] == 0:  # no fluctuation space
        u = b.clone()
        return {"D_star": gp.c_block, "u": u, "C": gp.c_block, "chi": 0.0,
                "c0_sq": gp.c_block, "n_omega": 0}

    # Wtilde = B B^T: nonzero eigenpairs from the thin SVD of B
    U, sv, _ = torch.linalg.svd(gp.B_tilde, full_matrices=False)
    omega = (sv**2)
    live = omega > omega.max().clamp_min(1e-300) * 1e-12
    U, omega = U[:, live], omega[live]
    c = U.T @ b  # (n_omega,)
    c0_sq = float((b.pow(2).sum() - c.pow(2).sum()).clamp_min(0.0))

    lam = _secular_root(c0_sq, (c**2).cpu().numpy(), omega.cpu().numpy(),
                        float(beta), gp.c_block)

    # u* propto (lambda I + beta Wtilde)^{-1} btilde
    if lam <= 0:
        u = b - U @ c  # the kernel component (margin has collapsed)
        if u.norm() < 1e-300:
            u = b.clone()
    else:
        u = (b - U @ c) / lam + U @ (c / (lam + float(beta) * omega))
    nrm = u.norm().clamp_min(torch.finfo(u.dtype).eps)
    u = u / nrm
    C = float((u @ b) ** 2)
    chi = float((gp.B_tilde.T @ u).pow(2).sum())
    return {"D_star": lam, "u": u, "C": C, "chi": chi,
            "c0_sq": c0_sq, "n_omega": int(omega.numel())}


# --------------------------------------------------------------------------- #
# solver 2: the W_all surrogate  (eq. m-solve, Theorem m-pencil)
# --------------------------------------------------------------------------- #
def solve_joint_pencil(gangs: list, beta: float) -> dict:
    """Shared regularized Gram ``G_beta = G + beta sum_l W_l``, ``m`` solves.

    In whitened coordinates ``G_beta`` becomes ``I + beta B_all B_all^T`` with
    ``B_all = [B_1 | ... | B_m]`` low rank, so Woodbury gives every direction
    exactly at the cost of one ``(sum_l (s_l-1))``-sized solve::

        (I + beta B B^T)^{-1} = I - beta B (I + beta B^T B)^{-1} B^T

    Returns the whitened directions plus every certificate of the theorem:
    capture floors (i), the ``(r_j - 1)/beta`` confusability bound (ii), the
    pairwise overlaps and the Gershgorin bound on ``lambda_min(Gamma)`` (iii).
    """

    m = len(gangs)
    dtype = gangs[0].b_tilde.dtype
    B_all = torch.cat([g.B_tilde for g in gangs], dim=1)  # (r, q)
    q = int(B_all.shape[1])

    def apply_inv(v: torch.Tensor) -> torch.Tensor:
        """``(I + beta B B^T)^{-1} v`` by Woodbury."""
        if q == 0 or beta == 0.0:
            return v
        BtV = B_all.T @ v
        core = torch.eye(q, dtype=dtype) + beta * (B_all.T @ B_all)
        return v - beta * (B_all @ torch.linalg.solve(core, BtV))

    U = []  # normalized whitened directions
    floors, ratios, chis, caps = [], [], [], []
    for g in gangs:
        u_raw = apply_inv(g.b_tilde)  # propto G_beta^{-1} b_j
        # (i) certified floor: b^T G_beta^{-1} b / (Phi+tau) = <btilde, u_raw>
        floors.append(float(g.b_tilde @ u_raw))
        nrm = u_raw.norm().clamp_min(torch.finfo(dtype).eps)
        u = u_raw / nrm
        U.append(u)
        caps.append(float((u @ g.b_tilde) ** 2))  # exact capture at theta_j
        # (ii) r_j = u^T (I + beta sum W) u / ||u||^2  (||u|| = 1 here)
        r = 1.0 + (beta * float((B_all.T @ u).pow(2).sum()) if q else 0.0)
        ratios.append(r)
        chis.append(float((g.B_tilde.T @ u).pow(2).sum()))  # own-gang chi

    Umat = torch.stack(U, dim=1)  # (r, m)
    # (iii) normalized M_tau overlaps of the directions (G -> I when whitened)
    gram = Umat.T @ Umat
    rho = gram.abs() - torch.eye(m, dtype=dtype)
    off_sum = rho.clamp_min(0.0).sum(1)
    lam_min_bound = float(min(caps) - off_sum.max()) if m else float("nan")
    # worst chi of direction j against EVERY gang's cone (the joint statement)
    chi_cross = torch.zeros(m, m, dtype=dtype)
    for j in range(m):
        for l, gl in enumerate(gangs):
            chi_cross[j, l] = (gl.B_tilde.T @ U[j]).pow(2).sum()
    return {
        "U": Umat,
        "capture": caps,
        "capture_floor": floors,
        "chi_own": chis,
        "chi_cross_max": [float(chi_cross[j].max()) for j in range(m)],
        "chi_bound": [(r - 1.0) / beta if beta > 0 else float("inf") for r in ratios],
        "r": ratios,
        "rho_max": [float(rho[j].clamp_min(0.0).max()) if m > 1 else 0.0
                    for j in range(m)],
        "lambda_min_gershgorin": lam_min_bound,
    }


# --------------------------------------------------------------------------- #
# solver 3: the collective closed form  (Theorems A and B)
# --------------------------------------------------------------------------- #
def collective_pencil_solve(gangs: list, beta: float) -> dict:
    """``Theta_beta = G_beta^{-1} Bhat``: the collective solve, with certificates.

    With ``bhat_j = b_j/sqrt(Phi_j+tau)``, ``Bhat = [bhat_1 ... bhat_m]`` and
    ``G_beta = G + beta*sum_l W_l``, the candidate target is ``Theta_beta =
    G_beta^{-1} Bhat`` -- one shared factorization and ``m`` solves, no
    eigen-ascent.  For any ``m``-column ``Theta`` the collective Gram is
    ``Gamma(Theta) = Bhat^T Theta (Theta^T G Theta)^{-1} Theta^T Bhat``.

    **Theorem A (beta = 0).** ``max_R lambda_min(Gamma(R))`` over every target
    inside the dictionary module equals ``lambda_min(N_0)``, ``N_0 = Bhat^T
    G^{-1} Bhat``, and is *attained* at ``Theta_0 = G^{-1} Bhat``: since
    ``Theta_0^T G Theta_0 = N_0``, ``Gamma(Theta_0) = N_0 N_0^{-1} N_0 = N_0``.
    So the ``lambda_min`` minimax is not approximated here, it is solved -- and
    when ``lambda_min(N_0) ~ 0`` no target whatsoever can do better (a capacity
    ceiling in the alignment geometry, curable only by features, never by the
    objective).

    **Theorem B (beta > 0).** ``N_beta <= Gamma(Theta_beta) <= N_0`` in PSD
    order, with ``N_beta = Bhat^T G_beta^{-1} Bhat``: writing
    ``Gamma(Theta_beta) = N_beta Dt^{-1} N_beta`` with
    ``Dt = Bhat^T G_beta^{-1} G G_beta^{-1} Bhat <= N_beta`` (as ``G <=
    G_beta``), congruence preserves the order.  The sandwich width
    ``lambda_min(N_0) - lambda_min(N_beta)`` is a computable, per-instance
    bound on everything an exact minimax could still recover -- monitor it and
    pick ``beta`` where it balances the confusability term.

    All quantities are returned; the whitened directions are in ``"U"``.
    """

    dtype = gangs[0].b_tilde.dtype
    m = len(gangs)
    B_t = torch.stack([g.b_tilde for g in gangs], dim=1)  # (r, m) = G^{-1/2} Bhat
    B_all = torch.cat([g.B_tilde for g in gangs], dim=1)  # W_all = B_all B_all^T
    q = int(B_all.shape[1])

    def apply_inv(V: torch.Tensor) -> torch.Tensor:
        """``(I + beta B_all B_all^T)^{-1} V`` by Woodbury (q is small)."""
        if q == 0 or beta == 0.0:
            return V
        core = torch.eye(q, dtype=dtype) + beta * (B_all.T @ B_all)
        return V - beta * (B_all @ torch.linalg.solve(core, B_all.T @ V))

    N0 = B_t.T @ B_t  # Bhat^T G^{-1} Bhat
    N0 = 0.5 * (N0 + N0.T)
    U_raw = apply_inv(B_t)  # (r, m), propto Theta_beta whitened
    N_beta = B_t.T @ U_raw
    N_beta = 0.5 * (N_beta + N_beta.T)
    Dt = U_raw.T @ U_raw  # Theta^T G Theta in whitened coords
    Dt = 0.5 * (Dt + Dt.T)
    jitter = 1e-14 * torch.diag(Dt).mean().clamp_min(1e-300)
    gamma = N_beta @ torch.linalg.solve(
        Dt + jitter * torch.eye(m, dtype=dtype), N_beta
    )
    gamma = 0.5 * (gamma + gamma.T)

    lam_N0 = float(torch.linalg.eigvalsh(N0)[0])
    lam_Nb = float(torch.linalg.eigvalsh(N_beta)[0])
    lam_G = float(torch.linalg.eigvalsh(gamma)[0])

    # per-direction confusability bound (r_j - 1)/beta, uniform over ALL gangs
    U = U_raw / U_raw.norm(dim=0, keepdim=True).clamp_min(torch.finfo(dtype).eps)
    r_j, chi_cross = [], []
    for j in range(m):
        u = U[:, j]
        r_j.append(1.0 + (beta * float((B_all.T @ u).pow(2).sum()) if q else 0.0))
        chi_cross.append(max(float((g.B_tilde.T @ u).pow(2).sum()) for g in gangs))
    # the same subspace certificate every other solver is scored on, so the
    # closed form and the A_eq pencil are read off identical statistics
    sub = subspace_report(gangs, U_raw, beta)
    return {
        "U": U,
        "chi_subspace": sub["chi_subspace"],
        "chi_subspace_max": sub["chi_subspace_max"],
        "margin": sub["margin"],
        "trace_Gamma": sub["trace_Gamma"],
        "lambda_min_N0": lam_N0,  # Theorem A ceiling (beta = 0 optimum)
        "lambda_min_N_beta": lam_Nb,  # certified lower half of the sandwich
        "lambda_min_Gamma": lam_G,  # what Theta_beta actually attains
        "optimality_gap": lam_N0 - lam_Nb,  # bound on any minimax's headroom
        "capture": [float(gamma[j, j]) for j in range(m)],
        "chi_cross_max": chi_cross,
        "chi_bound": [(r - 1.0) / beta if beta > 0 else float("inf") for r in r_j],
        "r": r_j,
        "sandwich_ok": bool(lam_Nb <= lam_G + 1e-9 and lam_G <= lam_N0 + 1e-9),
    }


def collective_pencil_theta(
    a_hat: torch.Tensor,
    adjacency: torch.Tensor,
    X: torch.Tensor,
    patterns: list,
    *,
    degree: int,
    tau: float,
    beta: float,
    basis: str = "chebyshev",
) -> tuple:
    """Dictionary coefficients ``Theta_beta`` (P, m) plus the Theorem A/B report.

    The returned ``Theta`` is a *dictionary coefficient matrix*, exactly like the
    filter bank's ``theta``: it is graph-independent, so freezing it and applying
    it to another day's dictionary is an inductive transfer.
    """

    space = build_pencil_space(a_hat, X, degree, tau, basis=basis)
    gps = pencil_gang_data(space, a_hat, adjacency, patterns)
    sol = collective_pencil_solve(gps, beta)
    theta = space.whiten @ sol["U"]  # (P, m)
    report = {k: v for k, v in sol.items() if k != "U"}
    report.update({
        "beta": beta, "gang": [g.gang_id for g in gps],
        "size": [g.size for g in gps], "c_block": [g.c_block for g in gps],
        "dictionary_dim": space.dim, "dictionary_rank": space.rank,
    })
    return theta, report


def apply_dictionary_theta(
    a_hat: torch.Tensor, X: torch.Tensor, theta: torch.Tensor, *,
    degree: int, tau: float, basis: str = "chebyshev",
) -> torch.Tensor:
    """Target ``T(graph) @ Theta`` -- applies frozen coefficients to a new graph."""

    prop = _basis_stack(a_hat, X, degree, basis, tau)
    return torch.cat(prop, dim=1) @ theta


# --------------------------------------------------------------------------- #
# targets for the coarsener
# --------------------------------------------------------------------------- #
def pencil_gang_data(
    space: PencilSpace,
    a_hat: torch.Tensor,
    adjacency: torch.Tensor,
    patterns: list,
    *,
    signal_only: bool = False,
) -> list:
    """Per-gang whitened pencil data for a list of patterns.

    The adjacency is densified once here and shared, rather than once per group.
    """

    V = degree_weighted_indicators(adjacency, patterns).to(space.T.dtype)
    phi = (V * _l_apply(a_hat, V)).sum(0).clamp_min(1e-300)
    dense = None
    if not signal_only:
        dense = adjacency.to_dense() if adjacency.is_sparse else adjacency
    return [
        build_gang_pencil(space, a_hat, adjacency, p, V[:, j], float(phi[j]),
                          signal_only=signal_only, dense_adj=dense)
        for j, p in enumerate(patterns)
    ]


def pencil_target(
    a_hat: torch.Tensor,
    adjacency: torch.Tensor,
    X: torch.Tensor,
    patterns: list,
    *,
    degree: int,
    tau: float,
    beta: float,
    mode: str = "pencil",
    basis: str = "chebyshev",
    space: "PencilSpace | None" = None,
    return_stats: bool = False,
):
    """Coarsening target of one column per gang, from the closed-form pencil.

    ``mode="pencil"``       exact per-gang ``D*(beta)`` (each gang's own optimum);
    ``mode="pencil-joint"`` the shared-``G_beta`` ``W_all`` surrogate.
    """

    space = space or build_pencil_space(a_hat, X, degree, tau, basis=basis)
    gps = pencil_gang_data(space, a_hat, adjacency, patterns)
    if mode == "pencil":
        sols = [solve_gang_pencil(g, beta) for g in gps]
        U = torch.stack([s["u"] for s in sols], dim=1)
        stats = {
            "mode": mode, "beta": beta,
            "D_star": [s["D_star"] for s in sols],
            "capture": [s["C"] for s in sols],
            "chi": [s["chi"] for s in sols],
            "c_block": [g.c_block for g in gps],
            "c0_sq": [s["c0_sq"] for s in sols],
        }
    elif mode == "pencil-joint":
        sol = solve_joint_pencil(gps, beta)
        U = sol["U"]
        stats = {"mode": mode, "beta": beta,
                 "c_block": [g.c_block for g in gps], **sol}
        stats.pop("U", None)
    else:
        raise ValueError("mode must be 'pencil' or 'pencil-joint'")

    theta = space.whiten @ U  # (P, m) un-whitened directions
    target = space.T @ theta  # (N, m) one column per gang
    stats["gang"] = [g.gang_id for g in gps]
    stats["size"] = [g.size for g in gps]
    stats["width"] = int(target.shape[1])
    stats["dictionary_dim"] = space.dim
    stats["dictionary_rank"] = space.rank
    if return_stats:
        return target, stats
    return target


# --------------------------------------------------------------------------- #
# solver 4: the A_eq signal-to-confusion generalized eigenproblem
# --------------------------------------------------------------------------- #
#
# The max-min certificate ``lambda_min(Gamma) - beta*max_j chi_j`` is the right
# *statistic* but a poor *optimizer*: ``lambda_min`` is identically zero once
# m > d, both extrema are nonsmooth at multiplicities, one noisy gang dominates
# every update, and there is no closed form.  The alternative below keeps a
# closed form while staying informative for m > d.
#
# In the whitened coordinates of :class:`PencilSpace` (where ``G`` is the
# identity, ``bar b_j = b_tilde``, ``bar H_j = B_tilde B_tilde^T``) define
#
#     A_eq = Bbar (Bbar^T Bbar + alpha I_m)^{-1} Bbar^T        (group signal)
#     H    = sum_j omega_j  bar H_j / (|S_j| - 1)              (confusability)
#
# with ``omega_j = 1/m`` by default.  ``A_eq -> `` the projector onto the
# realizable group-indicator span as ``alpha -> 0``, so unlike ``Bbar Bbar^T/m``
# it does not weight directions by how strongly or how often a group appears;
# dividing by ``|S_j| - 1`` stops large groups (whose fluctuation spaces have
# more dimensions) from owning the penalty.  The target is then the top-``d``
# generalized eigenspace
#
#     A_eq w_k = lambda_k (H + rho I) w_k,
#     W* = argmax_{W^T (H+rho I) W = I_d} tr(W^T A_eq W),
#
# i.e. each retained direction maximizes captured group energy per unit internal
# fluctuation -- a regularized Fisher discriminant with ``A_eq`` as the between-
# group signal, ``H`` as the within-group nuisance and ``lambda_k`` as the
# distinguishability of filter ``k``.  ``rho > 0`` keeps the ratio finite on
# ``ker H``.
#
# Nothing here is ever formed at size ``r``: ``A_eq`` has rank <= m, so with
# ``M = (N_0 + alpha I)^{-1}`` (``N_0 = Bbar^T Bbar``) and
# ``K = Bbar^T (H + rho I)^{-1} Bbar`` the nonzero generalized eigenpairs are
# those of the ``m x m`` symmetric ``S = M^{1/2} K M^{1/2}``:
#
#     S y = lambda y   =>   w = (H + rho I)^{-1} Bbar M^{1/2} y,
#
# and ``(H + rho I)^{-1}`` is applied by the same Woodbury identity the other
# solvers use, ``H = B_w B_w^T`` being low rank.  Cost is one ``q x q`` and one
# ``m x m`` solve -- no ``P``-sized eigendecomposition.


def _psd_pow(M: torch.Tensor, p: float, rel_cutoff: float = 1e-12) -> torch.Tensor:
    """``M^p`` for symmetric PSD ``M`` (eigenvalues below the cutoff dropped)."""

    M = 0.5 * (M + M.T)
    ev, evec = torch.linalg.eigh(M)
    keep = ev > ev.max().clamp_min(1e-300) * rel_cutoff
    return (evec[:, keep] * ev[keep].pow(p).unsqueeze(0)) @ evec[:, keep].T


def aeq_confusability_factor(
    gangs: list, weights: "torch.Tensor | None" = None
) -> torch.Tensor:
    """Factor ``B_w`` with ``H = B_w B_w^T = sum_j omega_j bar H_j/(|S_j|-1)``."""

    dtype = gangs[0].b_tilde.dtype
    m = len(gangs)
    if weights is None:
        weights = torch.full((m,), 1.0 / m, dtype=dtype)
    cols = []
    for g, w in zip(gangs, weights):
        if g.B_tilde.shape[1] == 0:
            continue
        scale = float(w) / max(g.size - 1, 1)
        cols.append(g.B_tilde * float(np.sqrt(max(scale, 0.0))))
    if not cols:
        return torch.zeros(gangs[0].b_tilde.numel(), 0, dtype=dtype)
    return torch.cat(cols, dim=1)  # (r, q)


def _apply_h_inv(B_w: torch.Tensor, rho: float, V: torch.Tensor) -> torch.Tensor:
    """``(B_w B_w^T + rho I)^{-1} V`` by Woodbury (``q`` small)."""

    q = int(B_w.shape[1])
    if q == 0:
        return V / rho
    core = rho * torch.eye(q, dtype=V.dtype) + B_w.T @ B_w
    return (V - B_w @ torch.linalg.solve(core, B_w.T @ V)) / rho


def subspace_report(gangs: list, U: torch.Tensor, beta: float) -> dict:
    """The original max-min certificate evaluated on a whitened target ``U``.

    Kept as the *evaluation statistic* for every solver (Section: "keep the
    original objective as a diagnostic"), so targets fitted by different
    objectives are scored on identical ground:

    * ``lambda_min(Gamma)`` with ``Gamma = Bhat^T Theta (Theta^T G Theta)^{-1}
      Theta^T Bhat`` -- collective capture floor;
    * ``chi_j = lambda_max(Q^T bar H_j Q)`` over the ``G``-orthonormalized target
      ``Q`` -- the eq. 40 worst-case confusability of the whole subspace, not of
      one column;
    * ``margin = lambda_min(Gamma) - beta * max_j chi_j``.
    """

    dtype = U.dtype
    if U.shape[1] == 0:
        return {"lambda_min_Gamma": 0.0, "capture": [], "chi_subspace": [],
                "chi_subspace_max": float("nan"), "margin": float("nan"),
                "trace_Gamma": 0.0}
    # rank-revealing orthonormalization: a plain QR would hand back d columns
    # even when the eigenvectors are (near-)dependent, inflating chi with
    # directions the target does not actually span
    Uo, sv, _ = torch.linalg.svd(U, full_matrices=False)
    Q = Uo[:, sv > sv.max().clamp_min(1e-300) * 1e-10]  # G-orthonormal, same span
    B_t = torch.stack([g.b_tilde for g in gangs], dim=1)  # (r, m)
    P = Q.T @ B_t  # (d, m)
    gamma = P.T @ P
    gamma = 0.5 * (gamma + gamma.T)
    lam_min = float(torch.linalg.eigvalsh(gamma)[0])
    chi = []
    for g in gangs:
        if g.B_tilde.shape[1] == 0:
            chi.append(0.0)
            continue
        sv = torch.linalg.svdvals(g.B_tilde.T @ Q)
        chi.append(float(sv[0] ** 2) if sv.numel() else 0.0)
    chi_max = max(chi) if chi else float("nan")
    return {
        "lambda_min_Gamma": lam_min,
        "capture": [float(gamma[j, j]) for j in range(len(gangs))],
        "chi_subspace": chi,
        "chi_subspace_max": chi_max,
        "margin": lam_min - beta * chi_max,
        "trace_Gamma": float(torch.diagonal(gamma).sum()),
    }


def aeq_solve(
    gangs: list,
    *,
    extra_signal: "list | None" = None,
    alpha: float = 1e-3,
    rho: float = 1e-3,
    width: int = 0,
    lambda_floor: float = 0.0,
    reweight_iters: int = 0,
    kappa: float = 5.0,
    beta: float = 0.0,
    rho_scale: str = "relative",
) -> dict:
    """Top-``d`` generalized eigenspace of ``A_eq w = lambda (H + rho I) w``.

    ``width=0`` takes ``d = m`` (one direction per signal group);
    ``lambda_floor > 0`` additionally drops directions whose distinguishability
    ``lambda_k`` falls below the floor (at least one column is always kept).

    **The width ceiling and what lifts it.**  ``A_eq`` is built from ``Bbar``,
    so its rank is the number of *signal groups* and no wider target exists:
    eigenvectors past that rank all sit at ``lambda_k = 0``, and they are not
    even determined (``H`` is low rank, so its kernel is enormous and every
    direction in it ties).  That same fact is why a target fitted on the
    training gangs alone does not transfer to held-out ones -- ``col(A_eq)`` is
    *exactly* the training groups' indicator span, and as ``alpha -> 0`` it is
    precisely the projector onto it.

    ``extra_signal`` is the lever for both: additional groups (label-free
    communities, licit components, ...) whose indicators join ``Bbar`` and widen
    the realizable span, turning the target from "isolate these m gangs" into
    "preserve community structure, of which these m gangs are examples".  They
    enter ``A_eq`` only -- ``H`` stays built from ``gangs``, since the
    confusability constraint is about the groups that must not be internally
    confused, and because the Woodbury core is ``q x q`` with
    ``q = sum_j (|S_j| - 1)``, which large filler communities would blow up.

    ``rho_scale="relative"`` (the default) reads ``rho`` as a multiple of the
    mean nonzero eigenvalue ``tr(H)/rank(H)`` rather than as an absolute ridge: the magnitude
    of ``H`` depends on the graph size, the degrees and ``tau``, so an absolute
    ridge silently swings between "no regularization" and "H ignored" across
    days.  ``rho_scale="absolute"`` uses ``rho`` as written.

    ``reweight_iters > 0`` runs the iterative reweighting that walks the mean
    penalty towards the worst case: solve, evaluate every ``chi_j``, set
    ``omega_j = softmax(kappa * chi_j)``, rebuild ``H`` and solve again.  Each
    iteration is still an eigenproblem, and ``kappa -> infinity`` concentrates
    all the weight on the most confusable group.
    """

    dtype = gangs[0].b_tilde.dtype
    signal = list(gangs) + list(extra_signal or [])
    m = len(signal)
    B_t = torch.stack([g.b_tilde for g in signal], dim=1)  # (r, m) = Bbar
    N0 = B_t.T @ B_t
    N0 = 0.5 * (N0 + N0.T)
    M_half = _psd_pow(N0 + alpha * torch.eye(m, dtype=dtype), -0.5)
    BM = B_t @ M_half  # (r, m)

    d = m if width <= 0 else min(int(width), m)
    n_conf = len(gangs)  # H is built from the confusability groups only
    weights = torch.full((n_conf,), 1.0 / n_conf, dtype=dtype)
    history = []
    U = evals = None
    r = int(B_t.shape[0])
    rho_used = float(rho)
    for it in range(int(reweight_iters) + 1):
        B_w = aeq_confusability_factor(gangs, weights)
        if rho_scale == "relative":
            # mean NONZERO eigenvalue tr(H)/rank(H): H is supported on the
            # q = sum_j (|S_j|-1) fluctuation directions, so tr(H)/r would
            # understate its scale by the (large) ambient dimension
            rank_h = max(min(int(B_w.shape[1]), r), 1)
            mean_h = float(B_w.pow(2).sum()) / rank_h  # tr(H)/rank(H)
            rho_used = float(rho) * max(mean_h, 1e-300)
            if rho_used <= 0.0:  # H = 0 (every gang a singleton)
                rho_used = float(rho)
        HinvBM = _apply_h_inv(B_w, rho_used, BM)  # (H+rho I)^{-1} Bbar M^{1/2}
        S = BM.T @ HinvBM  # M^{1/2} Bbar^T (H+rho I)^{-1} Bbar M^{1/2}
        S = 0.5 * (S + S.T)
        ev, evec = torch.linalg.eigh(S)  # ascending
        order = torch.argsort(ev, descending=True)[:d]
        evals = ev[order]
        U = HinvBM @ evec[:, order]  # (r, d) generalized eigenvectors
        U = U / U.norm(dim=0, keepdim=True).clamp_min(torch.finfo(dtype).eps)
        rep = subspace_report(gangs, U, beta)
        history.append({
            "iter": it,
            "lambda_min_Gamma": rep["lambda_min_Gamma"],
            "chi_max": rep["chi_subspace_max"],
            "margin": rep["margin"],
            "weight_max": float(weights.max()),
        })
        if it == int(reweight_iters):
            break
        chi = torch.tensor(rep["chi_subspace"], dtype=dtype)
        weights = torch.softmax(kappa * chi, dim=0)

    if lambda_floor > 0.0 and U.shape[1] > 1:
        keep = evals >= lambda_floor
        if not bool(keep.any()):
            keep[0] = True
        U, evals = U[:, keep], evals[keep]

    report = subspace_report(gangs, U, beta)
    # Theorem A ceiling on the same graph: no target inside the dictionary can
    # beat lambda_min(N_0), so the A_eq target's distance to it is its price.
    lam_N0 = float(torch.linalg.eigvalsh(N0)[0])
    report.update({
        "U": U,
        "eigenvalues": [float(v) for v in evals],
        "width": int(U.shape[1]),
        "alpha": alpha, "rho": rho, "rho_used": rho_used,
        "rho_scale": rho_scale, "beta": beta,
        "kappa": kappa, "reweight_iters": int(reweight_iters),
        "lambda_min_N0": lam_N0,
        "n_signal": m, "n_extra_signal": len(extra_signal or []),
        "capture_ceiling": [g.c_block for g in gangs],
        "reweight_history": history,
    })
    return report


def aeq_pencil_theta(
    a_hat: torch.Tensor,
    adjacency: torch.Tensor,
    X: torch.Tensor,
    patterns: list,
    *,
    signal_patterns: "list | None" = None,
    degree: int,
    tau: float,
    alpha: float = 1e-3,
    rho: float = 1e-3,
    width: int = 0,
    lambda_floor: float = 0.0,
    reweight_iters: int = 0,
    kappa: float = 5.0,
    beta: float = 0.0,
    rho_scale: str = "relative",
    basis: str = "chebyshev",
) -> tuple:
    """Dictionary coefficients ``Theta = G^{-1/2} W*`` from the ``A_eq`` pencil.

    Same contract as :func:`collective_pencil_theta`: the returned ``Theta`` is
    a graph-independent dictionary coefficient matrix, so freezing it and
    applying it to another day is an inductive transfer.
    """

    space = build_pencil_space(a_hat, X, degree, tau, basis=basis)
    gps = pencil_gang_data(space, a_hat, adjacency, patterns)
    extra = (pencil_gang_data(space, a_hat, adjacency, signal_patterns,
                              signal_only=True)
             if signal_patterns else None)
    sol = aeq_solve(gps, extra_signal=extra, alpha=alpha, rho=rho, width=width,
                    lambda_floor=lambda_floor, reweight_iters=reweight_iters,
                    kappa=kappa, beta=beta, rho_scale=rho_scale)
    theta = space.whiten @ sol["U"]  # (P, d)
    report = {k: v for k, v in sol.items() if k != "U"}
    report.update({
        "gang": [g.gang_id for g in gps], "size": [g.size for g in gps],
        "signal_size": [g.size for g in (extra or [])],
        "c_block": [g.c_block for g in gps],
        "dictionary_dim": space.dim, "dictionary_rank": space.rank,
    })
    return theta, report


# --------------------------------------------------------------------------- #
# solver-agnostic certificate: the same statistic on ANY target subspace
# --------------------------------------------------------------------------- #
def basis_subspace_report(
    a_hat: torch.Tensor,
    adjacency: torch.Tensor,
    Z: torch.Tensor,
    patterns: list,
    *,
    tau: float,
    beta: float = 0.0,
    ridge: float = 1e-10,
) -> dict:
    """``lambda_min(Gamma)``, ``max_j chi_j`` and the margin of a *node-space* target.

    :func:`subspace_report` needs whitened dictionary coordinates, so it only
    scores targets produced by the pencil solvers.  This one takes the ``(N, d)``
    basis the coarsener is actually handed -- whatever produced it, gradient bank
    included -- so every solver is judged on the identical certificate:

        Gamma  = Vhat^T M_tau Z (Z^T M_tau Z)^+ Z^T M_tau Vhat,
        chi_j  = lambda_max(B_j^T (Z^T M_tau Z)^+ B_j),
        B_j    = [(M_tau Z)|_S^T diag(sqrt(dtilde_S))] P_0 (P_0^T Q_S P_0)^{-1/2},

    with ``Q_S = L_S^int + diag(d_boundary) + tau D_tilde_S`` the exact local
    ``M_tau`` form and ``P_0`` the mean-zero fluctuation basis -- the same objects
    :func:`build_gang_pencil` builds, evaluated against ``Z`` instead of the full
    dictionary.
    """

    dtype = Z.dtype
    MZ = _m_apply(a_hat, Z, tau)  # (N, d)
    G_Z = Z.T @ MZ
    G_Z = 0.5 * (G_Z + G_Z.T)
    ev, evec = torch.linalg.eigh(G_Z)
    keep = ev > ev.max().clamp_min(1e-300) * 1e-12
    G_half_inv = evec[:, keep] / ev[keep].sqrt().unsqueeze(0)  # (d, k): G_Z^{-1/2}

    m_vhat = _train_gang_m_vhat(a_hat, adjacency, patterns, tau).to(dtype)
    gamma = _collective_gamma_mz(Z, MZ, m_vhat, ridge)
    gamma = 0.5 * (gamma + gamma.T)

    A = adjacency.to_dense() if adjacency.is_sparse else adjacency
    deg_all = A.sum(1).to(dtype)
    dtilde = deg_all + 1.0
    chi = []
    for p in patterns:
        idx = torch.as_tensor(sorted(int(i) for i in p.node_indices), dtype=torch.long)
        s = int(idx.numel())
        if s < 2:
            chi.append(0.0)
            continue
        W_SS = A[idx][:, idx].to(dtype)
        d_int = W_SS.sum(1)
        Q = (torch.diag(d_int) - W_SS + torch.diag(deg_all[idx] - d_int)
             + tau * torch.diag(dtilde[idx]))
        Q = 0.5 * (Q + Q.T)
        P0 = _fluctuation_basis(dtilde[idx])
        A_q = P0.T @ Q @ P0
        A_q = 0.5 * (A_q + A_q.T)
        eq, evq = torch.linalg.eigh(A_q)
        kq = eq > eq.max().clamp_min(1e-300) * 1e-12
        A_inv_half = evq[:, kq] / eq[kq].sqrt().unsqueeze(0)
        Y = (MZ[idx] * dtilde[idx].sqrt().unsqueeze(1)).T  # (d, s)
        B_g = G_half_inv.T @ (Y @ P0 @ A_inv_half)  # (k, .) in whitened Z coords
        sv = torch.linalg.svdvals(B_g)
        chi.append(float(sv[0] ** 2) if sv.numel() else 0.0)

    lam_min = float(torch.linalg.eigvalsh(gamma)[0])
    chi_max = max(chi) if chi else float("nan")
    diag = torch.diagonal(gamma)
    return {
        "lambda_min_Gamma": lam_min,
        "chi": chi,
        "chi_max": chi_max,
        "margin": lam_min - beta * chi_max,
        "capture": [float(v) for v in diag],
        "capture_mean": float(diag.mean()),
        "capture_min": float(diag.min()),
        "trace_Gamma": float(diag.sum()),
        "width": int(Z.shape[1]),
        "rank_Z": int(keep.sum()),
    }


def community_patterns(
    a_hat: torch.Tensor,
    X: torch.Tensor,
    *,
    n_clusters: int,
    hops: int = 3,
    min_size: int = 2,
    seed: int = 0,
) -> list:
    """Label-free community indicators to widen ``A_eq``'s signal span.

    ``A_eq``'s rank is the number of signal groups, so a target fitted on the
    ``m`` labelled groups alone can be neither wider than ``m`` nor able to hold
    a group it never saw.  Clustering the ``hops``-smoothed features gives as
    many community indicators as asked for, using no labels at all, so the
    resulting target preserves community structure generally instead of the
    training groups specifically -- the labelled groups become examples of what
    to keep rather than the whole definition of it.
    """

    from sklearn.cluster import MiniBatchKMeans

    from src.pattern_models import Pattern

    Z = X
    for _ in range(hops):  # smooth: k-means then sees communities, not nodes
        Z = a_hat @ Z
    Zn = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    labels = MiniBatchKMeans(
        n_clusters=int(n_clusters), random_state=int(seed), n_init=3,
        batch_size=4096,
    ).fit_predict(Zn.detach().cpu().numpy())
    out = []
    for c in range(int(n_clusters)):
        idx = torch.nonzero(torch.as_tensor(labels == c), as_tuple=False).flatten()
        if int(idx.numel()) >= min_size:
            out.append(Pattern(pattern_id=f"c{c}", nodes=idx, pattern_type="community"))
    return out
