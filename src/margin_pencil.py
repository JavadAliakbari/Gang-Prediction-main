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
    _l_apply,
    _m_apply,
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
) -> GangPencil:
    """Whitened ``b`` and the factor ``B`` with ``W = B B^T`` for one gang."""

    tau = space.tau
    dtype = space.T.dtype
    idx = torch.as_tensor(sorted(int(i) for i in pattern.node_indices), dtype=torch.long)
    s = int(idx.numel())

    # b = T^T M_tau v_S  (M_tau symmetric, so use the cached M_tau T)
    b = space.MT.T @ v_S  # (P,)
    b_tilde = (space.whiten.T @ b) / float(np.sqrt(phi + tau))

    if s < 2:  # no internal fluctuations: W = 0
        return GangPencil(str(pattern.id), s, phi, b_tilde,
                          torch.zeros(space.rank, 0, dtype=dtype),
                          float(b_tilde.pow(2).sum()))

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
    return {
        "U": U,
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
    space: PencilSpace, a_hat: torch.Tensor, adjacency: torch.Tensor, patterns: list
) -> list:
    """Per-gang whitened pencil data for a list of patterns."""

    V = degree_weighted_indicators(adjacency, patterns).to(space.T.dtype)
    phi = (V * _l_apply(a_hat, V)).sum(0).clamp_min(1e-300)
    return [
        build_gang_pencil(space, a_hat, adjacency, p, V[:, j], float(phi[j]))
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
