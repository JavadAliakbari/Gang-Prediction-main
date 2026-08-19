"""
Verification of subsection "Distinguishability: capture is not enough".

Objects (screened metric M_tau = L + tau I, normalized Laplacian L = I - Ahat):
  - Gang indicator     v_S = D~^{1/2} 1_S / sqrt(vol(S))
  - Capture            C_K^tau = sum_{k<K} q_k^tau,  q_k^tau=(lam_k+tau)p_k/(Phi+tau)
  - Fluctuation space  F_S = {w supported on S, <w,v_S>=0},  dim = s-1
  - Confusability      chi_K^tau = max_{w in F_S} ||Pi_K w||_Mtau^2 / ||w||_Mtau^2
  - Margin             D_K^tau = C_K^tau - chi_K^tau
  - Internal agitation gamma_S = min_{w in F_S} w^T L w / w^T w  (= lam_min(F^T L F))

Claims tested:
  [C1] C_K, chi_K non-decreasing in K; both -> 1 at K=N; D_N=0.
  [C2] lem:tilt      m^tau(w) >= w^T L w / w^T w >= gamma_S  for all w in F_S.
  [C3] thm:window    chi_K^tau <= (lmax - gamma_S)/(lmax - lam_K)         (pointwise bound)
  [C4] thm:window    D_K^tau  >= 1 - m1tau/lam_K - (lmax-gamma_S)/(lmax-lam_K)
  [C5] thm:window    D* = 1 - (sqrt(m1tau)+sqrt(lmax-gamma_S))^2/lmax  <= max_K D_K (certified LB)
                     window condition sqrt(m1tau)+sqrt(lmax-gamma_S) < sqrt(lmax) -> D*>0
  [C6] prop:collapse ||Pi_K v_T - cos w_T Pi_K v_S||_Mtau <= sqrt(chi)*||w_T||_Mtau  for T subset S
  [C7] cor:motif-window / rem:empirical-chi : per-motif gamma_S, m1tau, max_K D at tau=0 and 0.5
  [C8] attenuation   m1tau = Phi(m1+tau)/(Phi+tau);  gamma_S is tau-independent
"""

import numpy as np
import networkx as nx
import scipy.linalg as sla
import json, sys

rng_global = np.random.default_rng(0)


# ----------------------------------------------------------------------------- graph / operators
def build_graph(motif, s, b, N_total=600, mean_deg=8, seed=0):
    rng = np.random.default_rng(seed)
    host_n = N_total - (0 if motif == "random" else s)
    p = mean_deg / (host_n - 1)
    G = nx.gnp_random_graph(host_n, p, seed=int(rng.integers(1 << 30)))
    if motif == "random":
        # no planting; gang = random s nodes of the ER graph
        S = sorted(rng.choice(host_n, size=s, replace=False).tolist())
        return G, S
    # add planted motif nodes host_n .. host_n+s-1
    base = host_n
    nodes = list(range(base, base + s))
    G.add_nodes_from(nodes)
    if motif == "clique":
        for i in range(s):
            for j in range(i + 1, s):
                G.add_edge(nodes[i], nodes[j])
        carriers = nodes[:b]  # boundary on arbitrary motif nodes
    elif motif in ("cycle", "short_cycle"):
        for i in range(s):
            G.add_edge(nodes[i], nodes[(i + 1) % s])
        carriers = nodes[:b]
    elif motif == "star":
        hub = nodes[0]
        leaves = nodes[1:]
        for lf in leaves:
            G.add_edge(hub, lf)
        carriers = leaves[:b]  # leaf-attached boundary (the bottleneck case)
    else:
        raise ValueError(motif)
    hosts = rng.choice(host_n, size=b, replace=False)
    for c, h in zip(carriers, hosts):
        G.add_edge(c, int(h))
    return G, nodes


def operators(G):
    N = G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=range(N))
    d = A.sum(1)
    dt = d + 1.0  # D~ = D + I (self loops)
    Dinv = 1.0 / np.sqrt(dt)
    Ahat = Dinv[:, None] * (A + np.eye(N)) * Dinv[None, :]
    L = np.eye(N) - Ahat
    L = 0.5 * (L + L.T)
    lam, U = np.linalg.eigh(L)
    lam = np.clip(lam, 0.0, None)
    return dict(N=N, A=A, dt=dt, L=L, lam=lam, U=U, lmax=lam[-1])


# ----------------------------------------------------------------------------- gang quantities
def gang_indicator(op, S):
    N, dt = op["N"], op["dt"]
    vol = dt[S].sum()
    v = np.zeros(N)
    v[S] = np.sqrt(dt[S])
    v /= np.sqrt(vol)
    return v, vol


def fluctuation_basis(op, S, v):
    """Orthonormal basis F (N x (s-1)) of {w supported on S, w perp v}."""
    N = op["N"]
    s = len(S)
    E = np.zeros((N, s))
    E[S, np.arange(s)] = 1.0  # coordinate columns on S
    M = np.column_stack([v, E])  # v in span(E)
    Q, _ = np.linalg.qr(M)
    return Q[:, 1:s]  # cols orthogonal to v, span F_S


def spectral_setup(op, v, F):
    lam, U = op["lam"], op["U"]
    p = (U.T @ v) ** 2  # p_k
    UtF = U.T @ F  # (N x s-1)
    return p, UtF


# ----------------------------------------------------------------------------- C, chi, gamma, D
def capture_curve(p, lam, Phi, tau):
    q = (lam + tau) * p / (Phi + tau)
    return np.cumsum(q)  # C_K for K=1..N (index K-1)


def confusability_curve(UtF, lam, tau):
    """chi_K^tau for K=1..N via running rank-1 accumulation in whitened coords."""
    d = lam + tau
    Denom = UtF.T @ (lam[:, None] * UtF) + tau * (UtF.T @ UtF)  # F^T M_tau F
    Denom = 0.5 * (Denom + Denom.T)
    Cchol = np.linalg.cholesky(Denom + 1e-12 * np.eye(Denom.shape[0]))
    Cinv = np.linalg.inv(Cchol)
    G = UtF @ Cinv.T  # whitened rows g_k
    m = G.shape[1]
    M = np.zeros((m, m))
    chi = np.empty(len(lam))
    for k in range(len(lam)):
        g = G[k]
        M += d[k] * np.outer(g, g)
        chi[k] = np.linalg.eigvalsh(M)[-1] if k >= m - 1 else np.linalg.eigvalsh(M)[-1]
    return np.clip(chi, 0.0, 1.0)


def gamma_S(UtF, lam):
    FLF = UtF.T @ (lam[:, None] * UtF)
    FLF = 0.5 * (FLF + FLF.T)
    return float(np.linalg.eigvalsh(FLF)[0])


# ----------------------------------------------------------------------------- one instance
def analyze(motif, s, b, tau_list, seed):
    G, S = build_graph(motif, s, b, seed=seed)
    op = operators(G)
    v, vol = gang_indicator(op, S)
    F = fluctuation_basis(op, S, v)
    p, UtF = spectral_setup(op, v, F)
    lam, lmax = op["lam"], op["lmax"]
    Phi = float(v @ op["L"] @ v)
    m1 = float((lam**2 * p).sum() / max((lam * p).sum(), 1e-15))  # m2/m1 = tilde m1
    cut = op["A"][np.ix_(S, [j for j in range(op["N"]) if j not in set(S)])].sum()
    gS = gamma_S(UtF, lam)
    out = dict(
        motif=motif,
        s=s,
        b=b,
        seed=seed,
        Phi=Phi,
        cut=float(cut),
        vol=float(vol),
        m1=m1,
        gamma_S=gS,
        lmax=float(lmax),
    )
    per_tau = {}
    for tau in tau_list:
        C = capture_curve(p, lam, Phi, tau)
        chi = confusability_curve(UtF, lam, tau)
        D = C - chi
        m1tau = Phi * (m1 + tau) / (Phi + tau)
        # theory bounds
        with np.errstate(divide="ignore", invalid="ignore"):
            chi_bound = np.clip((lmax - gS) / (lmax - lam), 0.0, 1.0)
            D_lb = 1 - m1tau / np.where(lam > 0, lam, np.nan) - chi_bound
        sq = np.sqrt(max(m1tau, 0)) + np.sqrt(max(lmax - gS, 0))
        Dstar = 1 - sq**2 / lmax
        window_open = sq < np.sqrt(lmax)
        kbest = int(np.nanargmax(D)) + 1
        per_tau[tau] = dict(
            C=C,
            chi=chi,
            D=D,
            m1tau=float(m1tau),
            chi_bound=chi_bound,
            D_lb=D_lb,
            Dstar=float(Dstar),
            window_open=bool(window_open),
            maxD=float(np.nanmax(D)),
            kbest=kbest,
            lam_best=float(lam[kbest - 1]),
            # bound validity flags
            chi_bound_ok=bool(np.all(chi <= chi_bound + 1e-8)),
            D_lb_ok=bool(np.all(D[1:] >= D_lb[1:] - 1e-8)),
            Dstar_ok=bool(Dstar <= np.nanmax(D) + 1e-8),
            CmonoOK=bool(np.all(np.diff(C) >= -1e-9)),
            chimonoOK=bool(np.all(np.diff(chi) >= -1e-6)),
            DN=float(D[-1]),
        )
    out["lam"] = lam
    out["tau"] = per_tau
    out["v"] = v
    out["op_L"] = op["L"]
    out["U"] = op["U"]
    out["S"] = S
    out["dt"] = op["dt"]
    out["F"] = F
    out["p"] = p
    out["UtF"] = UtF
    return out


# ----------------------------------------------------------------------------- [C2] tilt, [C6] collapse
def check_tilt(res, tau):
    lam, UtF = res["lam"], res["UtF"]
    gS = res["gamma_S"]
    rng = np.random.default_rng(1)
    ok = True
    ratios = []
    for _ in range(200):
        c = rng.standard_normal(UtF.shape[1])
        wc = UtF @ c  # spectral coeffs of w = F c
        num_L = (lam * wc**2).sum()
        den = (wc**2).sum()
        rayl = num_L / den
        mtau = (lam * (lam + tau) * wc**2).sum() / ((lam + tau) * wc**2).sum()
        ratios.append(mtau - rayl)
        if mtau < rayl - 1e-9 or rayl < gS - 1e-9:
            ok = False
    return ok, float(np.min(ratios))


def check_collapse(res, tau, K, n_sub=300):
    op_L, U, lam = res["op_L"], res["U"], res["lam"]
    v, S, dt = res["v"], res["S"], res["dt"]
    s = len(S)
    vol = dt[S].sum()
    PiK = U[:, :K] @ U[:, :K].T
    Mt_diag = lam + tau

    def Mnorm(x):
        c = U.T @ x
        return np.sqrt(np.sum(Mt_diag * c**2))

    chi = res["tau"][tau]["chi"][K - 1]
    PiK_vS = PiK @ v
    rng = np.random.default_rng(7)
    lhs = []
    rhs = []
    for _ in range(n_sub):
        k = rng.integers(1, s)  # |T| in 1..s-1
        T = list(rng.choice(S, size=k, replace=False))
        vT = np.zeros(op_L.shape[0])
        vT[T] = np.sqrt(dt[T])
        vT /= np.sqrt(dt[T].sum())
        cos = np.sqrt(dt[T].sum() / vol)
        wT = vT - cos * v
        L_ = Mnorm(PiK @ vT - cos * PiK_vS)
        R_ = np.sqrt(chi) * Mnorm(wT)
        lhs.append(L_)
        rhs.append(R_)
    lhs = np.array(lhs)
    rhs = np.array(rhs)
    return bool(np.all(lhs <= rhs + 1e-8)), lhs, rhs


# ----------------------------------------------------------------------------- driver
MOTIFS = [
    ("clique", 30, 6),
    ("star", 30, 6),
    ("cycle", 30, 6),
    ("short_cycle", 8, 6),
    ("random", 30, 6),
]
TAUS = [0.0, 0.5]
SEEDS = list(range(6))

if __name__ == "__main__":
    allres = {}
    for motif, s, b in MOTIFS:
        allres[motif] = [analyze(motif, s, b, TAUS, sd) for sd in SEEDS]
        r0 = allres[motif][0]
        print(
            f"[{motif:11s}] Phi={r0['Phi']:.4f} cut={r0['cut']:.0f} m1={r0['m1']:.4f} "
            f"gamma_S={r0['gamma_S']:.4f} lmax={r0['lmax']:.4f}"
        )

    # aggregate table
    print("\n=== per-motif aggregate over seeds (mean +/- std) ===")
    hdr = (
        f"{'motif':11s} {'Phi':>8s} {'m1':>7s} {'gammaS':>8s} "
        f"{'m1t(.5)':>8s} {'maxD t0':>14s} {'maxD t.5':>14s} {'win t0/.5':>10s}"
    )
    print(hdr)
    summary = {}
    for motif, _, _ in MOTIFS:
        R = allres[motif]
        Phi = np.array([r["Phi"] for r in R])
        m1 = np.array([r["m1"] for r in R])
        gS = np.array([r["gamma_S"] for r in R])
        m1t = np.array([r["tau"][0.5]["m1tau"] for r in R])
        mD0 = np.array([r["tau"][0.0]["maxD"] for r in R])
        mD5 = np.array([r["tau"][0.5]["maxD"] for r in R])
        kb0 = np.array([r["tau"][0.0]["kbest"] for r in R])
        kb5 = np.array([r["tau"][0.5]["kbest"] for r in R])
        w0 = np.mean([r["tau"][0.0]["window_open"] for r in R])
        w5 = np.mean([r["tau"][0.5]["window_open"] for r in R])
        print(
            f"{motif:11s} {Phi.mean():8.4f} {m1.mean():7.3f} {gS.mean():8.4f} "
            f"{m1t.mean():8.4f} {mD0.mean():6.3f}+-{mD0.std():.3f} "
            f"{mD5.mean():6.3f}+-{mD5.std():.3f} {w0:.1f}/{w5:.1f}"
        )
        summary[motif] = dict(
            Phi=Phi.mean(),
            m1=m1.mean(),
            gammaS=gS.mean(),
            m1tau05=m1t.mean(),
            maxD0=mD0.mean(),
            maxD0_sd=mD0.std(),
            maxD5=mD5.mean(),
            maxD5_sd=mD5.std(),
            kbest0=float(kb0.mean()),
            kbest5=float(kb5.mean()),
            win0=w0,
            win5=w5,
        )

    # bound validity across all motifs/taus/seeds
    print("\n=== theory-bound validity (fraction of instances holding) ===")
    for motif, _, _ in MOTIFS:
        for tau in TAUS:
            flags = [allres[motif][sd]["tau"][tau] for sd in range(len(SEEDS))]
            cb = np.mean([f["chi_bound_ok"] for f in flags])
            db = np.mean([f["D_lb_ok"] for f in flags])
            ds = np.mean([f["Dstar_ok"] for f in flags])
            cm = np.mean([f["CmonoOK"] for f in flags])
            xm = np.mean([f["chimonoOK"] for f in flags])
            print(
                f"  {motif:11s} tau={tau:>3}: chi<=bound {cb:.2f}  D>=LB {db:.2f}  "
                f"D*<=maxD {ds:.2f}  C_mono {cm:.2f}  chi_mono {xm:.2f}"
            )

    # tilt + collapse on seed 0
    print("\n=== lem:tilt and prop:collapse (seed 0) ===")
    for motif, _, _ in MOTIFS:
        r = allres[motif][0]
        for tau in TAUS:
            ok_t, minexc = check_tilt(r, tau)
            Kc = r["tau"][tau]["kbest"]
            ok_c, lhs, rhs = check_collapse(r, tau, Kc)
            print(
                f"  {motif:11s} tau={tau:>3}: tilt {'OK' if ok_t else 'FAIL'} "
                f"(min m^tau-rayl={minexc:+.2e})  collapse@K={Kc} "
                f"{'OK' if ok_c else 'FAIL'} (max lhs/rhs={np.max(lhs/(rhs+1e-12)):.3f})"
            )

    np.save(
        "results/distinguishability_verify/allres.npy",
        {
            m: [
                {k: v for k, v in r.items() if k not in ("op_L", "U")}
                for r in allres[m]
            ]
            for m in allres
        },
        allow_pickle=True,
    )
    with open("results/distinguishability_verify/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\nsaved allres.npy, summary.json")
