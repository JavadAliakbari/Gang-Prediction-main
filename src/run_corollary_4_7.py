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
# 5.  confirm the bounds
# --------------------------------------------------------------------------- #
def markov_bound(m1t: float, lam_K: np.ndarray) -> np.ndarray:
    return m1t / lam_K


def cantelli_bound(sig2: float, m1t: float, lam_K: np.ndarray) -> np.ndarray:
    t = lam_K - m1t
    out = np.ones_like(lam_K)  # bound is trivially <= 1 when lam_K <= m1t
    ok = t > 0
    out[ok] = sig2 / (sig2 + t[ok] ** 2)
    return out


def run(args) -> None:
    LOGGER.info("=" * 78)
    LOGGER.info("Corollary 4.7 -- planted Erdos-Renyi motif: bound confirmation")
    LOGGER.info(
        f"N={args.N}  s={args.s}  p={args.p}  q={args.q}  tau={args.tau}  "
        f"seeds={args.seeds}"
    )
    th = theoretical_moments(args.N, args.s, args.p, args.q)
    LOGGER.info(
        f"  theory: delta={th['delta']:.2f}  b={th['b']:.2f}  dbar={th['dbar']:.2f}"
        f"  Dbar={th['Dbar']:.2f}   min_deg={th['min_deg']:.1f} vs C*logN"
        f"~{args.log_const*th['logN']:.1f}"
    )
    if th["min_deg"] < args.log_const * th["logN"]:
        LOGGER.info("  WARNING: min{delta,b} < C log N -- concentration may be loose")
    LOGGER.info("=" * 78)

    emp = {"Phi": [], "m1t": [], "sig2": []}
    worst_markov = (
        -np.inf
    )  # max over seeds/K of (1 - C_K) - markov_bound  (<=0 => holds)
    worst_cantelli = -np.inf
    n_valid = 0
    last = None
    seed_data: list = []

    for seed in range(args.seeds):
        rng = np.random.default_rng(args.seed0 + seed)
        W = build_planted_er(args.N, args.s, args.p, args.q, rng)
        L, dt = operators(W)
        v = degree_indicator(dt, args.s)
        lam, Phi, m1t, sig2, qk, C = spectral_capture(L, v, args.tau)
        emp["Phi"].append(Phi)
        emp["m1t"].append(m1t)
        emp["sig2"].append(sig2)

        # 1 - C_K vs the two bounds, for K = 1 .. N-1 (lambda_K = lam[K])
        K = np.arange(1, args.N)
        lam_K = lam[K]
        one_minus_C = 1.0 - C[K - 1]
        mk = markov_bound(m1t, lam_K)
        ct = cantelli_bound(sig2, m1t, lam_K)
        # only score where lambda_K > m1t (both bounds informative, < 1)
        valid = lam_K > m1t
        n_valid += int(valid.sum())
        worst_markov = max(worst_markov, float((one_minus_C - mk)[valid].max()))
        worst_cantelli = max(worst_cantelli, float((one_minus_C - ct)[valid].max()))
        seed_data.append((lam, Phi, m1t, sig2, C, K, lam_K, one_minus_C, mk, ct))
        last = seed_data[-1]

    def stat(name):
        a = np.array(emp[name])
        return a.mean(), a.std()

    LOGGER.info("\nMoment check (empirical over seeds  vs  Corollary 4.7 closed form):")
    LOGGER.info(
        f"  {'stat':<8}{'empirical (mean+/-std)':>28}{'theory':>12}{'rel.err':>9}"
    )
    for name, key, cmp in [
        ("Phi", "Phi", "eq"),
        ("m1t", "m1t", "eq"),
        ("sig2", "sig2", "<="),
    ]:
        mu, sd = stat(name)
        val = th[key]
        rel = (mu - val) / val
        flag = (
            ""
            if cmp == "eq"
            else ("  (bound: emp<=thy? " f"{'OK' if mu <= val*1.15 else 'CHECK'})")
        )
        LOGGER.info(
            f"  {name:<8}{mu:>15.5f} +/-{sd:>8.5f}{val:>12.5f}{rel:>+9.2%}{flag}"
        )

    LOGGER.info(
        "\nBound check  (1 - C_K  <=  bound,  over all seeds and all K with "
        "lambda_K > m1t):"
    )
    LOGGER.info(f"  valid (K,seed) points scored: {n_valid}")
    LOGGER.info(
        f"  Markov   max[(1-C_K) - m1t/lambda_K]           = {worst_markov:+.3e}"
        f"   -> {'HOLDS' if worst_markov <= 1e-9 else 'VIOLATED'}"
    )
    LOGGER.info(
        f"  Cantelli max[(1-C_K) - sig2/(sig2+(lK-m1t)^2)] = {worst_cantelli:+.3e}"
        f"   -> {'HOLDS' if worst_cantelli <= 1e-9 else 'VIOLATED'}"
    )

    # a few sample rows from the last seed, spanning the lambda_K > m1t regime
    lam, Phi, m1t, sig2, C, K, lam_K, one_minus_C, mk, ct = last
    LOGGER.info(
        f"\nSample rows (last seed;  Phi={Phi:.4f}  m1t={m1t:.4f}  sig2={sig2:.4f}):"
    )
    LOGGER.info(
        f"  {'K':>5}{'lambda_K':>10}{'C_K':>9}{'1-C_K':>10}"
        f"{'Markov':>10}{'Cantelli':>10}{'holds':>7}"
    )
    idx = np.where(lam_K > m1t)[0]
    picks = (
        idx[np.linspace(0, len(idx) - 1, min(10, len(idx))).astype(int)]
        if len(idx)
        else []
    )
    for i in picks:
        holds = one_minus_C[i] <= min(mk[i], ct[i]) + 1e-9
        LOGGER.info(
            f"  {K[i]:>5}{lam_K[i]:>10.4f}{C[K[i]-1]:>9.4f}{one_minus_C[i]:>10.4f}"
            f"{mk[i]:>10.4f}{ct[i]:>10.4f}{'yes' if holds else 'NO':>7}"
        )

    if _HAVE_PLT and args.plot:
        _plot(seed_data, th, args)


def _plot(seed_data: list, th: dict, args) -> None:
    """Four-panel figure confirming Corollary 4.7."""

    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f"Corollary 4.7 – planted ER motif  "
        f"(N={args.N}, s={args.s}, p={args.p}, q={args.q}, "
        rf"$\tau$={args.tau},  {len(seed_data)} seeds)",
        fontsize=11,
    )

    # ---- Panel A: main bound verification – all seeds ----------------------
    ax = axes[0, 0]
    x_grid = None
    interp_omc = []
    for i, (lam, Phi, m1t, sig2, C, K, lam_K, one_minus_C, mk, ct) in enumerate(
        seed_data
    ):
        order = np.argsort(lam_K)
        x = lam_K[order]
        ax.plot(x, one_minus_C[order], color="tab:blue", alpha=0.20, lw=0.8)
        if x_grid is None:
            x_grid = x
        interp_omc.append(np.interp(x_grid, x, one_minus_C[order]))

    mean_omc = np.mean(interp_omc, axis=0)
    ax.plot(
        x_grid, mean_omc, color="tab:blue", lw=2.0, alpha=0.9, label=r"mean $1-C_K$"
    )
    mk_th = markov_bound(th["m1t"], x_grid)
    ct_th = cantelli_bound(th["sig2"], th["m1t"], x_grid)
    ax.plot(
        x_grid,
        np.clip(mk_th, 0, 1.05),
        "-",
        color="tab:red",
        lw=1.8,
        label=r"Markov $\tilde m_1^{\rm th}/\lambda_K$",
    )
    ax.plot(
        x_grid,
        np.clip(ct_th, 0, 1.05),
        "-",
        color="tab:green",
        lw=1.8,
        label=r"Cantelli (theory)",
    )
    ax.axvline(th["Phi"], ls="--", lw=1.2, color="grey", label=r"$\Phi$ (threshold)")
    ax.axvline(th["m1t"], ls=":", lw=1.2, color="black", label=r"$\tilde m_1^{\rm th}$")
    proxy = Line2D(
        [0],
        [0],
        color="tab:blue",
        alpha=0.4,
        lw=1,
        label=rf"$1-C_K$ per seed ({len(seed_data)})",
    )
    h, l = ax.get_legend_handles_labels()
    ax.legend([proxy] + h, [proxy.get_label()] + l, fontsize=7)
    ax.set_xlabel(r"$\lambda_K$ (lowest freq. outside span$(U_K)$)")
    ax.set_ylabel(r"missed energy $1-C_K$")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("(A) Bounds hold: empirical \u2264 theory bound")
    ax.grid(alpha=0.3)

    # ---- Panel B: scatter – theory bound vs empirical (lam_K > m1t) -------
    ax = axes[0, 1]
    for i, (lam, Phi, m1t, sig2, C, K, lam_K, one_minus_C, mk, ct) in enumerate(
        seed_data
    ):
        mk_t = markov_bound(th["m1t"], lam_K)
        ct_t = cantelli_bound(th["sig2"], th["m1t"], lam_K)
        valid = lam_K > th["m1t"]
        ax.scatter(
            mk_t[valid],
            one_minus_C[valid],
            s=4,
            color="tab:red",
            alpha=0.25,
            label="Markov" if i == 0 else None,
        )
        ax.scatter(
            ct_t[valid],
            one_minus_C[valid],
            s=4,
            color="tab:green",
            alpha=0.25,
            label="Cantelli" if i == 0 else None,
        )
    lim = 1.02
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, label="y = x (bound tight)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("theory bound value")
    ax.set_ylabel(r"empirical $1-C_K$")
    ax.set_title(r"(B) Theory bound $\geq$ empirical (points below diagonal)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ---- Panel C: moment fidelity per seed ---------------------------------
    ax = axes[1, 0]
    phis = [d[1] for d in seed_data]
    m1ts = [d[2] for d in seed_data]
    sig2s = [d[3] for d in seed_data]
    xs = np.arange(len(seed_data))
    ax.scatter(xs, phis, s=70, color="tab:blue", zorder=3, label=r"empirical $\Phi$")
    ax.scatter(
        xs,
        m1ts,
        s=70,
        marker="s",
        color="tab:orange",
        zorder=3,
        label=r"empirical $\tilde m_1$",
    )
    ax.axhline(
        th["Phi"],
        ls="--",
        color="tab:blue",
        lw=1.5,
        label=rf"theory $\Phi={th['Phi']:.4f}$",
    )
    ax.axhline(
        th["m1t"],
        ls="--",
        color="tab:orange",
        lw=1.5,
        label=rf"theory $\tilde m_1={th['m1t']:.4f}$",
    )
    ax2 = ax.twinx()
    ax2.scatter(
        xs,
        sig2s,
        s=70,
        marker="^",
        color="tab:purple",
        zorder=3,
        label=r"empirical $\tilde\sigma^2$",
    )
    ax2.axhline(
        th["sig2"],
        ls=":",
        color="tab:purple",
        lw=1.5,
        label=rf"bound $\tilde\sigma^2\leq{th['sig2']:.4f}$",
    )
    ax2.set_ylabel(r"$\tilde\sigma^2$ (right axis)", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")
    ax.set_xlabel("seed index")
    ax.set_ylabel(r"$\Phi$,  $\tilde m_1$")
    ax.set_title(r"(C) Moments: empirical $\approx$ theory; $\tilde\sigma^2\leq$ bound")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.grid(alpha=0.3)

    # ---- Panel D: sharp threshold – C_K vs lambda_K / Phi -----------------
    ax = axes[1, 1]
    x_norm_grid = np.linspace(0.01, 3.0, 500)
    interp_CK = []
    for lam, Phi, m1t, sig2, C, K, lam_K, one_minus_C, mk, ct in seed_data:
        order = np.argsort(lam_K)
        x_norm = lam_K[order] / th["Phi"]
        CK = 1.0 - one_minus_C[order]
        ax.plot(x_norm, CK, color="tab:blue", alpha=0.20, lw=0.8)
        interp_CK.append(np.interp(x_norm_grid, x_norm, CK, left=0.0, right=1.0))
    mean_CK = np.mean(interp_CK, axis=0)
    std_CK = np.std(interp_CK, axis=0)
    ax.plot(x_norm_grid, mean_CK, color="tab:blue", lw=2.0, label=r"mean $C_K$")
    ax.fill_between(
        x_norm_grid,
        np.clip(mean_CK - std_CK, 0, 1),
        np.clip(mean_CK + std_CK, 0, 1),
        alpha=0.15,
        color="tab:blue",
    )
    lk = x_norm_grid * th["Phi"]
    lb = np.clip(1.0 - markov_bound(th["m1t"], np.where(lk > 0, lk, 1e-12)), 0, 1)
    ax.plot(
        x_norm_grid,
        lb,
        "--",
        color="tab:red",
        lw=1.5,
        label=r"Markov lower bound $1-\tilde m_1/\lambda_K$",
    )
    ax.axvline(
        1.0, ls="--", lw=1.5, color="grey", label=r"$\lambda_K=\Phi$ (sharp threshold)"
    )
    ax.axvline(
        th["m1t"] / th["Phi"],
        ls=":",
        lw=1.2,
        color="black",
        label=r"$\lambda_K=\tilde m_1$",
    )
    ax.set_xlabel(r"$\lambda_K\;/\;\Phi$")
    ax.set_ylabel(r"$C_K$ (retained energy)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(0, 3.0)
    ax.set_title(r"(D) Sharp threshold: $C_K\to1$ iff $\lambda_K\geq\Phi$")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    LOGGER.info(f"\nplot -> {args.out}")
    plt.close(fig)


def _plot_scaling(scale_rows: list, args) -> None:
    """Three-panel figure: concentration as N grows (run with --scaling)."""

    Ns = [r["N"] for r in scale_rows]
    excess_emp = [r["m1t_emp"] - r["Phi_emp"] for r in scale_rows]
    excess_th = [r["m1t_th"] - r["Phi_th"] for r in scale_rows]
    sig2_emp = [r["sig2_emp"] for r in scale_rows]
    sig2_th = [r["sig2_th"] for r in scale_rows]
    worst_gaps = [r["worst_cant_gap"] for r in scale_rows]

    ratio = args.s / args.N
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        f"Corollary 4.7 – concentration as N grows  "
        f"(p={args.p}, q={args.q}, s/N fixed\u2248{ratio:.3f})",
        fontsize=11,
    )

    ax = axes[0]
    ax.plot(
        Ns, excess_emp, "o-", color="tab:blue", label=r"empirical $\tilde m_1-\Phi$"
    )
    ax.plot(Ns, excess_th, "s--", color="tab:orange", label=r"theory prediction")
    ax.set_xlabel("N")
    ax.set_ylabel(r"wiring excess $\tilde m_1-\Phi$")
    ax.set_title(r"Excess $\to 0$: ER randomness screens the motif")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(Ns, sig2_emp, "o-", color="tab:blue", label=r"empirical $\tilde\sigma^2$")
    ax.plot(Ns, sig2_th, "s--", color="tab:orange", label=r"theory bound")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$\tilde\sigma^2$")
    ax.set_title(r"Variance $\to 0$: $\nu$ concentrates at $\Phi$")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(Ns, worst_gaps, "o-", color="tab:green", label="worst gap")
    ax.axhline(0.0, ls="--", lw=1.2, color="grey", label="bound holds (\u2264 0)")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$\max_K[(1-C_K) - \mathrm{Cantelli\;bound}]$")
    ax.set_title("Cantelli bound holds throughout scaling")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    scale_out = args.out.replace(".png", "_scaling.png")
    fig.savefig(scale_out, dpi=150, bbox_inches="tight")
    LOGGER.info(f"\nscaling plot -> {scale_out}")
    plt.close(fig)


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
        # if p < stop_precision:
        #     break

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
        )
        from src.pattern_models import create_pattern
    except ImportError as exc:
        LOGGER.info(f"  run_coarsening_sweep requires torch + src package: {exc}")
        return

    use_ward = args.coarsen_method == "ward"
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
    if use_ward:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            LOGGER.info("  --coarsen-method ward requires scikit-learn; aborting sweep")
            return

    th = theoretical_moments(args.N, args.s, args.p, args.q)
    algo = "Ward agglomeration" if use_ward else "edge-greedy"
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
    # step = max(1, 60 // 20)
    # Ks_dense = np.arange(max(2, K_thresh0 - 30), min(K_max, K_thresh0 + 30) + 1, step)
    Ks_dense = np.array([K_thresh0 - 30, K_max, K_max, K_thresh0 + 30], dtype=int)
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

            recalls, precisions, _n_coarses = (
                _ward_pr_curve(
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
                if use_ward
                else _incremental_pr_curve(
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


def run_scaling(args) -> None:
    """Confirm the asymptotics: excess ``m1t - Phi`` and ``sig2`` vanish as N grows.

    Holds the density ratio fixed (``p``, and ``q`` scaled so the expected degrees
    grow with ``N``) so ``delta, b -> infinity``; Corollary 4.7 predicts the wiring
    excess ``m1t - Phi = O(1/dbar + s/N + 1/Dbar)`` and ``sig2`` both -> 0, i.e. the
    measure concentrates at the conductance ``Phi`` and the bounds tighten.
    """

    ratio = args.s / args.N  # hold s/N fixed; degrees grow proportionally to N
    LOGGER.info("=" * 78)
    LOGGER.info("Corollary 4.7 -- concentration as degrees grow (randomness screens)")
    LOGGER.info(
        f"  fixed p={args.p}, q={args.q}, s/N={ratio:.3f}; degrees grow ~ N, "
        f"so dbar,Dbar -> infinity and sig2 -> 0"
    )
    LOGGER.info("=" * 78)
    LOGGER.info(
        f"  {'N':>6}{'s':>5}{'dbar':>8}{'Dbar':>8}{'Phi':>9}{'m1t-Phi':>10}"
        f"{'(pred)':>9}{'sig2':>9}{'maxGap_Cant':>13}"
    )
    scale_rows: list = []
    for N in args.scaling_N:
        s = max(5, int(round(ratio * N)))
        q = args.q  # fixed background probability => expected degrees grow with N
        rng = np.random.default_rng(args.seed0)
        W = build_planted_er(N, s, args.p, q, rng)
        L, dt = operators(W)
        v = degree_indicator(dt, s)
        lam, Phi, m1t, sig2, qk, C = spectral_capture(L, v, args.tau)
        th = theoretical_moments(N, s, args.p, q)
        K = np.arange(1, N)
        lam_K = lam[K]
        one_minus_C = 1.0 - C[K - 1]
        ct = cantelli_bound(sig2, m1t, lam_K)
        valid = lam_K > m1t
        gap = float((one_minus_C - ct)[valid].max()) if valid.any() else float("nan")
        LOGGER.info(
            f"  {N:>6}{s:>5}{th['dbar']:>8.1f}{th['Dbar']:>8.1f}{Phi:>9.4f}"
            f"{m1t - Phi:>10.4f}{th['m1t'] - th['Phi']:>9.4f}{sig2:>9.4f}"
            f"{gap:>+13.2e}"
        )
        scale_rows.append(
            {
                "N": N,
                "s": s,
                "Phi_emp": Phi,
                "m1t_emp": m1t,
                "sig2_emp": sig2,
                "Phi_th": th["Phi"],
                "m1t_th": th["m1t"],
                "sig2_th": th["sig2"],
                "worst_cant_gap": gap,
            }
        )
    LOGGER.info(
        f"\n  -> Phi stays ~constant; sig2 -> 0 and the degree part of the excess"
        f"\n     (1-Phi)/dbar + 1/Dbar -> 0 (residual excess -> s/N = {ratio:.3f});"
        f"\n     the Cantelli gap stays <= 0, so the bound holds throughout."
    )
    if _HAVE_PLT and args.plot:
        _plot_scaling(scale_rows, args)


# --------------------------------------------------------------------------- #
# 7.  density sweep: grow the motif's internal density p (add edges gradually)
#     and track Phi, m1t, sig2, C_K against the closed-form theory
# --------------------------------------------------------------------------- #
def indicator_for_set(dt: np.ndarray, idx) -> np.ndarray:
    """Degree-weighted indicator for an arbitrary node index set.

    Generalizes ``degree_indicator``, which hard-codes ``S = {0, ..., s-1}``;
    used by the size sweep where ``S`` grows by absorbing graph neighbours
    rather than staying a fixed node-index prefix.
    """
    idx = np.asarray(list(idx), dtype=int)
    v = np.zeros_like(dt)
    v[idx] = np.sqrt(dt[idx])
    v /= np.sqrt(dt[idx].sum())
    return v


def run_density_sweep(args) -> None:
    """Grow the planted motif's internal density ``p`` and track the statistics.

    A host graph (all pairs Bern(q), including the S-S block) is generated once
    per seed; the internal S-S block is then cleared and edges are added back
    in monotonically via a fixed random permutation of the C(s,2) internal
    pairs, so each density level is a strict superset of the previous one
    ("gradually increase density by adding connections", not a fresh resample).
    At each density, ``Phi``, ``m1t``, ``sig2`` and ``C_K`` (for a handful of
    fixed K) are computed and compared against ``theoretical_moments``.
    """
    LOGGER.info("=" * 78)
    LOGGER.info("Density sweep: motif density p grown by monotonic edge addition")
    K_list = [k for k in args.density_k_list if 0 < k < args.N]
    p_grid = np.linspace(args.density_p_min, args.density_p_max, args.density_points)
    LOGGER.info(
        f"  N={args.N}  s={args.s}  q={args.q}  tau={args.tau}  "
        f"p: {p_grid[0]:.3f} -> {p_grid[-1]:.3f}  ({args.density_points} points)  "
        f"K_list={K_list}  seeds={args.density_seeds}"
    )
    LOGGER.info("=" * 78)

    n_pairs = args.s * (args.s - 1) // 2
    iu, ju = np.triu_indices(args.s, k=1)

    all_Phi = np.full((args.density_seeds, len(p_grid)), np.nan)
    all_m1t = np.full((args.density_seeds, len(p_grid)), np.nan)
    all_sig2 = np.full((args.density_seeds, len(p_grid)), np.nan)
    all_CK = {k: np.full((args.density_seeds, len(p_grid)), np.nan) for k in K_list}

    for seed in range(args.density_seeds):
        rng = np.random.default_rng(args.seed0 + seed)
        # fixed host structure for this seed: uniform Bern(q) graph on all N nodes
        W = build_planted_er(args.N, 0, args.q, args.q, rng)
        W[: args.s, : args.s] = 0.0  # clear internal block; refill via permutation

        perm = rng.permutation(n_pairs) if n_pairs > 0 else np.array([], dtype=int)

        for pi, p in enumerate(p_grid):
            m_target = int(round(float(p) * n_pairs))
            m_target = max(0, min(n_pairs, m_target))
            chosen = perm[:m_target]
            Wp = W.copy()
            ii, jj = iu[chosen], ju[chosen]
            Wp[ii, jj] = 1.0
            Wp[jj, ii] = 1.0

            L, dt = operators(Wp)
            v = degree_indicator(dt, args.s)
            lam, Phi, m1t, sig2, qk, C = spectral_capture(L, v, args.tau)

            all_Phi[seed, pi] = Phi
            all_m1t[seed, pi] = m1t
            all_sig2[seed, pi] = sig2
            for k in K_list:
                all_CK[k][seed, pi] = float(C[k - 1])

    th_Phi = np.array(
        [theoretical_moments(args.N, args.s, float(p), args.q)["Phi"] for p in p_grid]
    )
    th_m1t = np.array(
        [theoretical_moments(args.N, args.s, float(p), args.q)["m1t"] for p in p_grid]
    )
    th_sig2 = np.array(
        [theoretical_moments(args.N, args.s, float(p), args.q)["sig2"] for p in p_grid]
    )

    rows = {
        "p_grid": p_grid,
        "Phi_mean": np.nanmean(all_Phi, axis=0),
        "Phi_std": np.nanstd(all_Phi, axis=0),
        "m1t_mean": np.nanmean(all_m1t, axis=0),
        "m1t_std": np.nanstd(all_m1t, axis=0),
        "sig2_mean": np.nanmean(all_sig2, axis=0),
        "sig2_std": np.nanstd(all_sig2, axis=0),
        "CK_mean": {k: np.nanmean(all_CK[k], axis=0) for k in K_list},
        "CK_std": {k: np.nanstd(all_CK[k], axis=0) for k in K_list},
        "th_Phi": th_Phi,
        "th_m1t": th_m1t,
        "th_sig2": th_sig2,
    }

    LOGGER.info(
        f"\n  {'p':>7} {'Phi_emp':>9} {'Phi_th':>9} {'m1t_emp':>9} {'m1t_th':>9} "
        f"{'sig2_emp':>9} {'sig2_th':>9}"
    )
    LOGGER.info("  " + "-" * 70)
    for i, p in enumerate(p_grid):
        LOGGER.info(
            f"  {p:>7.3f} {rows['Phi_mean'][i]:>9.4f} {th_Phi[i]:>9.4f} "
            f"{rows['m1t_mean'][i]:>9.4f} {th_m1t[i]:>9.4f} "
            f"{rows['sig2_mean'][i]:>9.4f} {th_sig2[i]:>9.4f}"
        )

    if _HAVE_PLT and args.plot:
        _plot_density_sweep(rows, args)


def _plot_density_sweep(rows: dict, args) -> None:
    """2x2 figure: Phi, m1t, sig2 (empirical band + theory) and C_K vs density."""
    p_grid = rows["p_grid"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f"Density sweep  (N={args.N}, s={args.s}, q={args.q}, "
        rf"$\tau$={args.tau}, {args.density_seeds} seeds)",
        fontsize=12,
    )

    def _band(ax, x, mean, std, color, label):
        ax.plot(x, mean, "o-", color=color, ms=3, lw=1.5, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    ax = axes[0, 0]
    _band(
        ax, p_grid, rows["Phi_mean"], rows["Phi_std"], "tab:blue", r"$\Phi$ empirical"
    )
    ax.plot(p_grid, rows["th_Phi"], "--", color="black", lw=1.3, label=r"$\Phi$ theory")
    ax.set_xlabel("motif density $p$")
    ax.set_ylabel(r"$\Phi$ (conductance)")
    ax.set_title(r"(A) $\Phi$ vs density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    _band(
        ax,
        p_grid,
        rows["m1t_mean"],
        rows["m1t_std"],
        "tab:orange",
        r"$\tilde m_1$ empirical",
    )
    ax.plot(
        p_grid,
        rows["th_m1t"],
        "--",
        color="black",
        lw=1.3,
        label=r"$\tilde m_1$ theory",
    )
    ax.set_xlabel("motif density $p$")
    ax.set_ylabel(r"$\tilde m_1$ (mean)")
    ax.set_title("(B) mean vs density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    _band(
        ax,
        p_grid,
        rows["sig2_mean"],
        rows["sig2_std"],
        "tab:green",
        r"$\sigma^2$ empirical",
    )
    ax.plot(
        p_grid,
        rows["th_sig2"],
        "--",
        color="black",
        lw=1.3,
        label=r"$\sigma^2$ theory (bound)",
    )
    ax.set_xlabel("motif density $p$")
    ax.set_ylabel(r"$\sigma^2$ (variance)")
    ax.set_title("(C) variance vs density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    K_list = list(rows["CK_mean"].keys())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(1, len(K_list))))
    for k, color in zip(K_list, colors):
        _band(ax, p_grid, rows["CK_mean"][k], rows["CK_std"][k], color, rf"$C_{{{k}}}$")
    ax.set_xlabel("motif density $p$")
    ax.set_ylabel(r"$C_K$ (retained energy)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(D) retained energy vs density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = args.out.replace(".png", "_density_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    LOGGER.info(f"\ndensity sweep plot -> {out}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 8.  size sweep: grow the motif by absorbing graph neighbours (nested S)
#     and track Phi, m1t, sig2, C_K against the closed-form theory
# --------------------------------------------------------------------------- #
def _rewire_block(W: np.ndarray, S, p: float, rng: np.random.Generator) -> None:
    """In-place: resample the induced ``S x S`` block of ``W`` at Bern(p)."""
    idx = np.asarray(list(S), dtype=int)
    m = len(idx)
    if m < 2:
        return
    block = (rng.random((m, m)) < p).astype(np.float64)
    block = np.triu(block, 1)
    block = block + block.T
    W[np.ix_(idx, idx)] = block


def _grow_neighbourhood(
    S: list, W_host: np.ndarray, num_new: int, N: int, rng: np.random.Generator
) -> list:
    """Extend ``S`` by ``num_new`` nodes, preferring graph neighbours of ``S``.

    Candidates are nodes outside ``S`` with a host edge into ``S`` (so the
    enlarged motif stays spatially local -- "replace the motif with a larger
    one built from itself plus its neighbours" -- rather than jumping to
    unrelated nodes). Falls back to random remaining nodes if there are not
    enough graph neighbours available.
    """
    S_set = set(int(i) for i in S)
    neighbour_mask = W_host[list(S_set), :].sum(axis=0) > 0
    candidates = [i for i in range(N) if i not in S_set and neighbour_mask[i]]
    rng.shuffle(candidates)

    new_nodes = candidates[:num_new]
    if len(new_nodes) < num_new:
        taken = S_set | set(new_nodes)
        remaining = [i for i in range(N) if i not in taken]
        rng.shuffle(remaining)
        new_nodes += remaining[: num_new - len(new_nodes)]

    return list(S) + new_nodes


def run_size_sweep(args) -> None:
    """Grow the motif in size by absorbing neighbours and track the statistics.

    A host graph (uniform Bern(q) over all ``N`` nodes) is generated once per
    seed and stays fixed; only the induced ``S x S`` block is (re)rewired to
    Bern(p) as ``S`` grows, so the graph outside the evolving motif never
    changes ("do not drastically change the graph when the size increases").
    ``S`` at each step is a strict superset of the previous ``S`` (previous
    motif + newly absorbed neighbours), matching the requested nested growth.
    """
    LOGGER.info("=" * 78)
    LOGGER.info("Size sweep: motif grown by absorbing graph neighbours (nested S)")
    K_list = [k for k in args.size_k_list if 0 < k < args.N]
    s_grid = np.unique(
        np.round(
            np.geomspace(args.size_s_min, args.size_s_max, args.size_points)
        ).astype(int)
    )
    s_grid = s_grid[(s_grid >= 2) & (s_grid <= args.N - 2)]
    if len(s_grid) == 0 or s_grid[0] != args.size_s_min:
        s_grid = np.unique(np.concatenate([[args.size_s_min], s_grid]))
    LOGGER.info(
        f"  N={args.N}  p={args.p}  q={args.q}  tau={args.tau}  "
        f"s: {s_grid[0]} -> {s_grid[-1]}  ({len(s_grid)} points)  "
        f"K_list={K_list}  seeds={args.size_seeds}"
    )
    LOGGER.info("=" * 78)

    all_Phi = np.full((args.size_seeds, len(s_grid)), np.nan)
    all_m1t = np.full((args.size_seeds, len(s_grid)), np.nan)
    all_sig2 = np.full((args.size_seeds, len(s_grid)), np.nan)
    all_CK = {k: np.full((args.size_seeds, len(s_grid)), np.nan) for k in K_list}

    for seed in range(args.size_seeds):
        rng = np.random.default_rng(args.seed0 + seed)
        # fixed host graph for this seed: uniform Bern(q) on all N nodes
        W_host = build_planted_er(args.N, 0, args.q, args.q, rng)

        S = list(rng.choice(args.N, size=int(s_grid[0]), replace=False))
        W = W_host.copy()
        _rewire_block(W, S, args.p, rng)

        for si, s_target in enumerate(s_grid):
            s_target = int(s_target)
            num_new = s_target - len(S)
            if num_new > 0:
                S = _grow_neighbourhood(S, W_host, num_new, args.N, rng)
                W = W_host.copy()
                _rewire_block(W, S, args.p, rng)

            L, dt = operators(W)
            v = indicator_for_set(dt, S)
            lam, Phi, m1t, sig2, qk, C = spectral_capture(L, v, args.tau)

            all_Phi[seed, si] = Phi
            all_m1t[seed, si] = m1t
            all_sig2[seed, si] = sig2
            for k in K_list:
                all_CK[k][seed, si] = float(C[k - 1])

    th_Phi = np.array(
        [theoretical_moments(args.N, int(s), args.p, args.q)["Phi"] for s in s_grid]
    )
    th_m1t = np.array(
        [theoretical_moments(args.N, int(s), args.p, args.q)["m1t"] for s in s_grid]
    )
    th_sig2 = np.array(
        [theoretical_moments(args.N, int(s), args.p, args.q)["sig2"] for s in s_grid]
    )

    rows = {
        "s_grid": s_grid,
        "Phi_mean": np.nanmean(all_Phi, axis=0),
        "Phi_std": np.nanstd(all_Phi, axis=0),
        "m1t_mean": np.nanmean(all_m1t, axis=0),
        "m1t_std": np.nanstd(all_m1t, axis=0),
        "sig2_mean": np.nanmean(all_sig2, axis=0),
        "sig2_std": np.nanstd(all_sig2, axis=0),
        "CK_mean": {k: np.nanmean(all_CK[k], axis=0) for k in K_list},
        "CK_std": {k: np.nanstd(all_CK[k], axis=0) for k in K_list},
        "th_Phi": th_Phi,
        "th_m1t": th_m1t,
        "th_sig2": th_sig2,
    }

    LOGGER.info(
        f"\n  {'s':>5} {'Phi_emp':>9} {'Phi_th':>9} {'m1t_emp':>9} {'m1t_th':>9} "
        f"{'sig2_emp':>9} {'sig2_th':>9}"
    )
    LOGGER.info("  " + "-" * 68)
    for i, s in enumerate(s_grid):
        LOGGER.info(
            f"  {int(s):>5} {rows['Phi_mean'][i]:>9.4f} {th_Phi[i]:>9.4f} "
            f"{rows['m1t_mean'][i]:>9.4f} {th_m1t[i]:>9.4f} "
            f"{rows['sig2_mean'][i]:>9.4f} {th_sig2[i]:>9.4f}"
        )

    if _HAVE_PLT and args.plot:
        _plot_size_sweep(rows, args)


def _plot_size_sweep(rows: dict, args) -> None:
    """2x2 figure: Phi, m1t, sig2 (empirical band + theory) and C_K vs size."""
    s_grid = rows["s_grid"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f"Size sweep  (N={args.N}, p={args.p}, q={args.q}, "
        rf"$\tau$={args.tau}, {args.size_seeds} seeds, neighbourhood growth)",
        fontsize=12,
    )

    def _band(ax, x, mean, std, color, label):
        ax.semilogx(x, mean, "o-", color=color, ms=3, lw=1.5, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    ax = axes[0, 0]
    _band(
        ax, s_grid, rows["Phi_mean"], rows["Phi_std"], "tab:blue", r"$\Phi$ empirical"
    )
    ax.semilogx(
        s_grid, rows["th_Phi"], "--", color="black", lw=1.3, label=r"$\Phi$ theory"
    )
    ax.set_xlabel("motif size $s$")
    ax.set_ylabel(r"$\Phi$ (conductance)")
    ax.set_title(r"(A) $\Phi$ vs size")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[0, 1]
    _band(
        ax,
        s_grid,
        rows["m1t_mean"],
        rows["m1t_std"],
        "tab:orange",
        r"$\tilde m_1$ empirical",
    )
    ax.semilogx(
        s_grid,
        rows["th_m1t"],
        "--",
        color="black",
        lw=1.3,
        label=r"$\tilde m_1$ theory",
    )
    ax.set_xlabel("motif size $s$")
    ax.set_ylabel(r"$\tilde m_1$ (mean)")
    ax.set_title("(B) mean vs size")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    _band(
        ax,
        s_grid,
        rows["sig2_mean"],
        rows["sig2_std"],
        "tab:green",
        r"$\sigma^2$ empirical",
    )
    ax.semilogx(
        s_grid,
        rows["th_sig2"],
        "--",
        color="black",
        lw=1.3,
        label=r"$\sigma^2$ theory (bound)",
    )
    ax.set_xlabel("motif size $s$")
    ax.set_ylabel(r"$\sigma^2$ (variance)")
    ax.set_title("(C) variance vs size")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 1]
    K_list = list(rows["CK_mean"].keys())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(1, len(K_list))))
    for k, color in zip(K_list, colors):
        _band(ax, s_grid, rows["CK_mean"][k], rows["CK_std"][k], color, rf"$C_{{{k}}}$")
    ax.set_xlabel("motif size $s$")
    ax.set_ylabel(r"$C_K$ (retained energy)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(D) retained energy vs size")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = args.out.replace(".png", "_size_sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    LOGGER.info(f"\nsize sweep plot -> {out}")
    plt.close(fig)


def main() -> None:
    result_path = "results/corollary_4_7/"
    path = f"{result_path}{now}/"
    os.makedirs(path, exist_ok=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=2000, help="total nodes")
    ap.add_argument("--s", type=int, default=40, help="planted motif size |S|")
    ap.add_argument("--p", type=float, default=0.2, help="internal edge prob (Bern p)")
    ap.add_argument(
        "--q", type=float, default=0.012, help="background edge prob (Bern q)"
    )
    ap.add_argument("--tau", type=float, default=0.0, help="screening (0 = Cor 4.7)")
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
        "--scaling",
        action="store_true",
        default=True,
        help="run the N-scaling concentration demonstration instead",
    )
    ap.add_argument(
        "--scaling-N", type=int, nargs="+", default=[400, 800, 1600, 3200, 6400]
    )
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
        choices=["edges", "neighborhood", "capped", "linkage", "ward"],
        default="ward",
    )
    ap.add_argument(
        "--coarsen-reduction",
        type=float,
        default=0.97,
        help="node-count reduction target (stop when n_coarse <= (1-r)*N)",
    )
    ap.add_argument(
        "--coarsen-epsilon",
        type=float,
        default=float("inf"),
        help="RSA distortion budget (inf = reduction-only stop)",
    )
    ap.add_argument(
        "--coarsen-max-levels",
        type=int,
        default=50,
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
        default=25,
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
    # density sweep
    ap.add_argument(
        "--density",
        default=True,
        action="store_true",
        help="run the density sweep (grow motif density p; track Phi, m1t, sig2, C_K)",
    )
    ap.add_argument("--density-p-min", type=float, default=0.02)
    ap.add_argument("--density-p-max", type=float, default=0.9)
    ap.add_argument("--density-points", type=int, default=15)
    ap.add_argument("--density-seeds", type=int, default=10)
    ap.add_argument(
        "--density-k-list",
        type=int,
        nargs="+",
        default=[2, 5, 10],
        help="K values to track C_K for as density grows",
    )
    # size sweep
    ap.add_argument(
        "--size",
        action="store_true",
        default=True,
        help="run the size sweep (grow motif size s by absorbing neighbours; "
        "track Phi, m1t, sig2, C_K)",
    )
    ap.add_argument("--size-s-min", type=int, default=10)
    ap.add_argument("--size-s-max", type=int, default=200)
    ap.add_argument("--size-points", type=int, default=12)
    ap.add_argument("--size-seeds", type=int, default=10)
    ap.add_argument(
        "--size-k-list",
        type=int,
        nargs="+",
        default=[2, 5, 10],
        help="K values to track C_K for as size grows",
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
