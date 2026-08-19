import numpy as np, scipy.linalg as sla
from distinguishability_check import build_graph, operators, gang_indicator, fluctuation_basis, spectral_setup, gamma_S

def chi_direct(UtF, lam, tau, K):
    """independent solve: chi_K = max gen-eig (F^T M_tau Pi_K F, F^T M_tau F)."""
    d = lam + tau
    A = UtF[:K]
    Numer = A.T @ (d[:K, None] * A)
    Denom = UtF.T @ (d[:, None] * UtF)
    Numer = 0.5*(Numer+Numer.T); Denom = 0.5*(Denom+Denom.T)
    w = sla.eigh(Numer, Denom, eigvals_only=True)
    return float(w[-1])

def gamma_local(op, S):
    """gamma_S via the def:agitation local form: min gen-eig(L_S^int+diag(dpart), Dtilde_S)."""
    A = op["A"]; N = op["N"]; dt = op["dt"]
    Sset = set(S); s = len(S)
    Asub = A[np.ix_(S, S)]
    dint = Asub.sum(1)
    dpart = np.array([A[i, [j for j in range(N) if j not in Sset]].sum() for i in S])
    Lint = np.diag(dint) - Asub
    Num = Lint + np.diag(dpart)
    Den = np.diag(dt[S])
    # restrict to constraint sum dt_i z_i = 0
    c = np.sqrt(dt[S]); c = c/np.linalg.norm(c)   # direction of v_S in z-Dtilde^{1/2}? build null space of dt^T
    # constraint vector is dt[S]; basis of its orthogonal complement:
    g = dt[S]/np.linalg.norm(dt[S])
    Qc = np.eye(s) - np.outer(g, g)
    U0, sv, _ = np.linalg.svd(Qc)
    B = U0[:, :s-1]                                # basis of {z: dt.z=0}
    Nb = B.T @ Num @ B; Db = B.T @ Den @ B
    w = sla.eigh(0.5*(Nb+Nb.T), 0.5*(Db+Db.T), eigvals_only=True)
    return float(w[0])

for motif, s, b in [("cycle",30,6), ("short_cycle",8,6), ("star",30,6)]:
    G,S = build_graph(motif,s,b,seed=0); op=operators(G)
    v,vol = gang_indicator(op,S); F=fluctuation_basis(op,S,v)
    p,UtF = spectral_setup(op,v,F); lam=op["lam"]; lmax=op["lmax"]
    Phi=float(v@op["L"]@v); m1=float((lam**2*p).sum()/(lam*p).sum())
    gS_spec = gamma_S(UtF,lam); gS_loc = gamma_local(op,S)
    gS_formula = (2/3)*(1-np.cos(2*np.pi/s)) if "cycle" in motif else None
    print(f"\n### {motif} s={s} b={b}: Phi={Phi:.4f} m1={m1:.4f} lmax={lmax:.4f}")
    print(f"    gamma_S spectral={gS_spec:.4f}  local-form={gS_loc:.4f}  cycle-formula={gS_formula}")
    for tau in [0.0,0.5]:
        d=lam+tau; q=d*p/(Phi+tau); C=np.cumsum(q)
        # chi via running (as in main) vs direct at a few K
        Denom=UtF.T@(lam[:,None]*UtF)+tau*(UtF.T@UtF)
        Ch=np.linalg.cholesky(Denom+1e-12*np.eye(Denom.shape[0])); Ci=np.linalg.inv(Ch)
        Gc=UtF@Ci.T; M=np.zeros((Gc.shape[1],)*2); chi=np.empty(len(lam))
        for k in range(len(lam)):
            g=Gc[k]; M+=d[k]*np.outer(g,g); chi[k]=np.linalg.eigvalsh(M)[-1]
        chi=np.clip(chi,0,1); D=C-chi; kb=int(np.argmax(D))+1
        m1tau=Phi*(m1+tau)/(Phi+tau)
        # cross check chi at kb
        chk=chi_direct(UtF,lam,tau,kb)
        print(f"  tau={tau}: maxD={D.max():.3f} @K={kb} (lamK={lam[kb-1]:.3f})  "
              f"C={C[kb-1]:.3f} chi={chi[kb-1]:.3f} chi_direct={chk:.3f}  m1tau={m1tau:.4f}")
        # window certificate
        sq=np.sqrt(m1tau)+np.sqrt(lmax-gS_spec)
        print(f"          window: m1tau={m1tau:.3f} gamma_S={gS_spec:.3f} -> sqrt-sum={sq:.3f} "
              f"vs sqrt(lmax)={np.sqrt(lmax):.3f}  {'OPEN' if sq<np.sqrt(lmax) else 'CLOSED'};  "
              f"D*={1-sq**2/lmax:+.3f}")
        # where is D max: dump a few points
        idx=[max(1,kb-2),kb,min(len(lam),kb+2), int(0.5*len(lam)), len(lam)]
        for K in idx:
            print(f"            K={K:4d} lamK={lam[K-1]:.3f} C={C[K-1]:.3f} chi={chi[K-1]:.3f} D={D[K-1]:+.3f}")
