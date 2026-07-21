import numpy as np, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from distinguishability_check import (
    build_graph,
    operators,
    gang_indicator,
    fluctuation_basis,
    spectral_setup,
    capture_curve,
    confusability_curve,
    gamma_S,
)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 130,
        "savefig.dpi": 140,
    }
)
FIG = "results/distinguishability_verify/figs/"
MOTIFS = [
    ("clique", 30, 6, "clique"),
    ("star", 30, 6, "star (leaf-attached)"),
    ("cycle", 30, 6, "long cycle (s=30)"),
    ("short_cycle", 8, 6, "short cycle (s=8)"),
    ("random", 30, 6, "random set"),
]
COL = {
    "clique": "#1b9e77",
    "star": "#d95f02",
    "cycle": "#7570b3",
    "short_cycle": "#e7298a",
    "random": "#666666",
}
# paper's claimed maxD (rem:empirical-chi): (tau0, tau0.5)
PAPER = {
    "clique": (0.90, 0.996),
    "star": (0.48, 0.89),
    "cycle": (0.0, 0.0),
    "short_cycle": (None, 0.91),
    "random": (0.0, 0.0),
}


def curves(motif, s, b, tau, seed=0):
    G, S = build_graph(motif, s, b, seed=seed)
    op = operators(G)
    v, vol = gang_indicator(op, S)
    F = fluctuation_basis(op, S, v)
    p, UtF = spectral_setup(op, v, F)
    lam = op["lam"]
    lmax = op["lmax"]
    Phi = float(v @ op["L"] @ v)
    m1 = float((lam**2 * p).sum() / (lam * p).sum())
    gS = gamma_S(UtF, lam)
    C = capture_curve(p, lam, Phi, tau)
    chi = confusability_curve(UtF, lam, tau)
    D = C - chi
    m1tau = Phi * (m1 + tau) / (Phi + tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_bnd = np.clip((lmax - gS) / (lmax - lam), 0, 1)
    return dict(
        lam=lam,
        C=C,
        chi=chi,
        D=D,
        chi_bnd=chi_bnd,
        Phi=Phi,
        m1=m1,
        m1tau=m1tau,
        gS=gS,
        lmax=lmax,
        kb=int(np.argmax(D)) + 1,
    )


# ============================================================ FIG 1: C, chi, D vs lambda_K
fig, axes = plt.subplots(len(MOTIFS), 2, figsize=(12, 14), sharex=False)
for r, (motif, s, b, lab) in enumerate(MOTIFS):
    for c, tau in enumerate([0.0, 0.5]):
        ax = axes[r, c]
        d = curves(motif, s, b, tau)
        x = d["lam"]
        ax.plot(x, d["C"], color="#2166ac", lw=1.8, label=r"$C_K^\tau$ (capture)")
        ax.plot(
            x, d["chi"], color="#b2182b", lw=1.8, label=r"$\chi_K^\tau$ (confusability)"
        )
        ax.plot(x, d["D"], color="black", lw=2.2, label=r"$D_K^\tau=C-\chi$")
        ax.plot(
            x,
            d["chi_bnd"],
            color="#b2182b",
            ls=":",
            lw=1.2,
            label=r"$\chi$ bound $\frac{\lambda_{max}-\gamma_S}{\lambda_{max}-\lambda_K}$",
        )
        ax.axvline(d["m1tau"], color="#2166ac", ls="--", lw=1, alpha=0.7)
        ax.axvline(d["gS"], color="#7f7f7f", ls="--", lw=1, alpha=0.9)
        kb = d["kb"]
        ax.plot(x[kb - 1], d["D"][kb - 1], "k*", ms=13, zorder=5)
        ax.annotate(
            f"max $D$={d['D'][kb-1]:.2f}\n@K={kb}",
            (x[kb - 1], d["D"][kb - 1]),
            textcoords="offset points",
            xytext=(14, -4),
            fontsize=8.5,
        )
        ax.axhline(0, color="k", lw=0.6, alpha=0.5)
        ax.set_xlim(-0.02, d["lmax"] + 0.02)
        ax.set_ylim(-0.55, 1.05)
        ttl = f"{lab}  |  " + r"$\tau$=" + f"{tau}"
        ttl += f"\n$\\Phi$={d['Phi']:.3f}, $\\tilde m_1^\\tau$={d['m1tau']:.3f}, $\\gamma_S$={d['gS']:.3f}"
        ax.set_title(ttl, fontsize=9)
        if r == len(MOTIFS) - 1:
            ax.set_xlabel(r"$\lambda_K$ (spectral cutoff)")
        if c == 0:
            ax.set_ylabel("energy fraction")
        if r == 0 and c == 0:
            ax.legend(loc="center right", fontsize=7.5, framealpha=0.9)
# annotate the vlines meaning once
fig.text(
    0.5,
    0.005,
    r"blue dashed $=\tilde m_1^\tau$ (capture edge)   |   grey dashed $=\gamma_S$ (agitation edge).  "
    r"Certified window: $\tilde m_1^\tau\lesssim\lambda_K\lesssim\gamma_S$",
    ha="center",
    fontsize=9,
)
fig.suptitle(
    "Distinguishability decomposition: capture $C$, confusability $\\chi$, margin $D=C-\\chi$   (seed 0)",
    fontsize=13,
    y=0.998,
)
fig.tight_layout(rect=[0, 0.02, 1, 0.99])
fig.savefig(FIG + "fig1_curves.png")
plt.close(fig)
print("fig1 done")

# ============================================================ FIG 2: maxD bar vs paper
SEEDS = range(6)
mD = {m: {t: [] for t in [0.0, 0.5]} for m, _, _, _ in MOTIFS}
for motif, s, b, _ in MOTIFS:
    for sd in SEEDS:
        for t in [0.0, 0.5]:
            mD[motif][t].append(curves(motif, s, b, t, seed=sd)["D"].max())
fig, ax = plt.subplots(figsize=(10, 5))
xs = np.arange(len(MOTIFS))
w = 0.36
for i, t in enumerate([0.0, 0.5]):
    means = [np.mean(mD[m][t]) for m, _, _, _ in MOTIFS]
    sds = [np.std(mD[m][t]) for m, _, _, _ in MOTIFS]
    ax.bar(
        xs + (i - 0.5) * w,
        means,
        w,
        yerr=sds,
        capsize=3,
        color=["#9ecae1", "#3182bd"][i],
        label=f"sim $\\tau$={t}",
    )
for j, (m, _, _, _) in enumerate(MOTIFS):
    for i, t in enumerate([0.0, 0.5]):
        pv = PAPER[m][i]
        if pv is not None:
            ax.plot(xs[j] + (i - 0.5) * w, pv, "rD", ms=8, zorder=6)
ax.set_xticks(xs)
ax.set_xticklabels([l for _, _, _, l in MOTIFS], rotation=12, ha="right")
ax.set_ylabel(r"$\max_K D_K^\tau$")
ax.axhline(0, color="k", lw=0.6)
ax.set_title(
    "Peak distinguishability margin: simulation (bars, mean$\\pm$sd over 6 hosts) vs paper (red diamonds)"
)
ax.legend(loc="upper right")
ax.text(
    0.99,
    0.02,
    "red = paper rem:empirical-chi claim",
    transform=ax.transAxes,
    ha="right",
    fontsize=8,
    style="italic",
)
fig.tight_layout()
fig.savefig(FIG + "fig2_maxD_vs_paper.png")
plt.close(fig)
print("fig2 done")

# ============================================================ FIG 3: window intervals [m1tau, gammaS]
fig, ax = plt.subplots(figsize=(10, 5.2))
yl = []
for j, (motif, s, b, lab) in enumerate(MOTIFS):
    for i, tau in enumerate([0.0, 0.5]):
        d = curves(motif, s, b, tau)
        y = j * 2.4 + (0.0 if i == 0 else 1.0)
        lo, hi = d["m1tau"], d["gS"]
        open_ = lo < hi
        ax.plot([0, d["lmax"]], [y, y], color="#dddddd", lw=6, zorder=1)
        if open_:
            ax.plot(
                [lo, hi],
                [y, y],
                color=COL[motif],
                lw=9,
                zorder=2,
                solid_capstyle="butt",
            )
        ax.plot(
            lo, y, "|", color="#2166ac", ms=16, mew=2.5, zorder=3
        )  # m1tau capture edge
        ax.plot(
            hi, y, "|", color="black", ms=16, mew=2.5, zorder=3
        )  # gammaS agitation edge
        ax.text(-0.02, y, f"{lab} $\\tau$={tau}", ha="right", va="center", fontsize=8.5)
        ax.text(
            d["lmax"] + 0.03,
            y,
            ("open" if open_ else "closed") + f"  maxD={d['D'].max():.2f}",
            va="center",
            fontsize=8,
            color=(COL[motif] if open_ else "#999"),
        )
ax.set_yticks([])
ax.set_xlim(-0.32, d["lmax"] + 0.32)
ax.set_xlabel(r"$\lambda$")
ax.set_title(
    "Detection window $[\\tilde m_1^\\tau,\\ \\gamma_S]$ (cor:motif-window)\n"
    "blue tick $=\\tilde m_1^\\tau$ capture edge, black tick $=\\gamma_S$ agitation edge; "
    "colored bar = open interval"
)
fig.tight_layout()
fig.savefig(FIG + "fig3_windows.png")
plt.close(fig)
print("fig3 done")

# ============================================================ FIG 4: tau sweep of maxD
taus = np.linspace(0, 2.0, 21)
fig, ax = plt.subplots(figsize=(9, 5.2))
for motif, s, b, lab in MOTIFS:
    means = []
    for t in taus:
        vals = [curves(motif, s, b, t, seed=sd)["D"].max() for sd in range(4)]
        means.append(np.mean(vals))
    ax.plot(taus, means, "-o", ms=3.5, color=COL[motif], label=lab)
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel(r"screening $\tau$")
ax.set_ylabel(r"$\max_K D_K^\tau$")
ax.set_title(
    r"Screening sweep: peak margin vs $\tau$ (mean over 4 hosts)."
    "\nstar & short-cycle lift with $\\tau$; long cycle gains a K=2 spike; clique saturates near 1"
)
ax.legend(fontsize=8.5)
ax.set_ylim(-0.1, 1.02)
fig.tight_layout()
fig.savefig(FIG + "fig4_tau_sweep.png")
plt.close(fig)
print("fig4 done")

# ============================================================ FIG 5: prop:collapse scatter
from distinguishability_check import analyze, check_collapse

fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
for ax, (motif, s, b, lab) in zip(axes, [MOTIFS[0], MOTIFS[1], MOTIFS[2]]):
    r = analyze(motif, s, b, [0.5], seed=0)
    Kc = r["tau"][0.5]["kbest"]
    ok, lhs, rhs = check_collapse(r, 0.5, Kc, n_sub=400)
    mx = max(lhs.max(), rhs.max()) * 1.05
    ax.scatter(rhs, lhs, s=10, color=COL[motif], alpha=0.55, edgecolor="none")
    ax.plot([0, mx], [0, mx], "k--", lw=1, label="y=x (bound tight)")
    ax.set_xlim(0, mx)
    ax.set_ylim(0, mx)
    ax.set_xlabel(r"RHS $\sqrt{\chi_K^\tau}\,\|w_T\|_{M_\tau}$")
    ax.set_ylabel(r"LHS $\|\Pi_K v_T-\cos\omega_T\Pi_K v_S\|_{M_\tau}$")
    ax.set_title(
        f"{lab}\nprop:collapse  ({'HOLDS' if ok else 'VIOLATED'}: all below y=x)",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper left")
fig.suptitle(
    r"prop:collapse verification ($\tau$=0.5, K=$K^\star$): sampled subsets $T\subseteq S$",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(FIG + "fig5_collapse.png")
plt.close(fig)
print("fig5 done")
print("ALL FIGURES SAVED to", FIG)
