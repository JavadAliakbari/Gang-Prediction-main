import numpy as np
from scipy.sparse import csr_matrix

rng = np.random.default_rng(2)
N, s, p, q = 2000, 60, 0.25, 0.04  # degree-matched replica (delta=7.8, b~46, Phi~0.84)
prob = np.full((N, N), q)
prob[:s, :s] = p
up = rng.random((N, N)) < prob
up = np.triu(up, 1)
W = (up + up.T).astype(float)
Wt = W + np.eye(N)
dt = Wt.sum(1)
Ah = Wt / np.sqrt(np.outer(dt, dt))
L = np.eye(N) - Ah
lam, U = np.linalg.eigh(L)
lam = np.clip(lam, 0, None)
K = 60
A = U[:, 1:K] * (lam[1:K] ** -0.5)[None, :]  # Loukas B0 = U_K Lam^{+1/2} rows


def best_f1(labels):
    best = (0, 0, 0)
    for c in np.unique(labels):
        m = labels == c
        tp = (m[:s]).sum()
        pr = tp / m.sum()
        rc = tp / s
        f1 = 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0
        if f1 > best[0]:
            best = (f1, pr, rc)
    return best


# ---- contiguity-constrained Ward ----
from sklearn.cluster import AgglomerativeClustering

conn = csr_matrix(W)
print("n_clusters |  Ward (F1, prec, rec)  |  edge-greedy (F1, prec, rec)")


# ---- edge-matching greedy baseline (level-by-level, recomputed costs) ----
def edge_greedy(n_target):
    lab = np.arange(N)
    Wc = W.copy()
    B = A.copy()
    sizes = np.ones(N)
    while True:
        n = Wc.shape[0]
        if n <= n_target:
            break
        ii, jj = np.nonzero(np.triu(Wc, 1))
        if len(ii) == 0:
            break
        cost = ((B[ii] - B[jj]) ** 2).sum(1)
        order = np.argsort(cost)
        used = np.zeros(n, bool)
        grp = -np.ones(n, int)
        g = 0
        for e in order:
            a, b = ii[e], jj[e]
            if not used[a] and not used[b]:
                used[a] = used[b] = True
                grp[a] = grp[b] = g
                g += 1
                if n - (used.sum() - g) <= n_target:
                    break
        for v in range(n):
            if grp[v] < 0:
                grp[v] = g
                g += 1
        # merge
        P = np.zeros((g, n))
        P[grp, np.arange(n)] = 1
        sz = P @ sizes
        Bn = (P @ (B * sizes[:, None])) / sz[:, None]
        Wn = P @ Wc @ P.T
        np.fill_diagonal(Wn, 0)
        lab = grp[lab]
        Wc, B, sizes = Wn, Bn, sz
        if g >= n:
            break
    return lab


for n in [300, 150, 75, 40, 20, 10]:
    ward = AgglomerativeClustering(n_clusters=n, connectivity=conn, linkage="ward").fit(
        A
    )
    f1w, pw, rw = best_f1(ward.labels_)
    f1g, pg, rg = best_f1(edge_greedy(n))
    print(
        f"{n:>10} |  {f1w:.3f}  {pw:.3f}  {rw:.3f}   |  {f1g:.3f}  {pg:.3f}  {rg:.3f}"
    )
