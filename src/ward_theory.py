import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering


def _exact_rsa_epsilon(
    a0,
    laplacian,
    original_to_supernode,
) -> float:
    """Exact restricted-spectral-approximation constant of a coarsening.

    The RSA definition (Loukas 2019, Def. 2) is the smallest ``epsilon`` with
    ``||x - Pi x||_L <= epsilon ||x||_L`` for every ``x`` in the target subspace
    ``R``, where ``Pi = P^+ P`` is the block-averaging projection onto vectors
    that are constant on each supernode.  This is the *exact* worst-case
    distortion of the cumulative coarsening -- not the looser per-level product
    bound ``prod_l (1 + sigma_l) - 1``.

    With ``a0`` an ``L``-orthonormal basis of ``R`` (``a0^T L a0 = I``), any
    ``x = a0 c`` has ``||x||_L = ||c||``, so

        epsilon^2 = max_c (c^T Y^T L Y c) / (c^T c) = lambda_max(Y^T L Y),

    with ``Y = (I - Pi) a0`` -- each row of ``a0`` minus its supernode mean.
    ``laplacian`` and ``a0`` are those of the *original* graph; only the
    partition ``original_to_supernode`` changes across levels.
    """

    n_super = int(np.max(original_to_supernode)) + 1
    counts = (
        np.bincount(original_to_supernode, minlength=n_super)
        .astype(a0.dtype)
        .clip(min=1.0)[:, None]
    )
    sums = np.zeros((n_super, a0.shape[1]), dtype=a0.dtype)
    for i in range(n_super):
        sums[i] = a0[original_to_supernode == i].sum(0)
    residual = a0 - (sums / counts)[original_to_supernode]  # (I - Pi) a0
    gram = residual.T @ (laplacian @ residual)
    gram = 0.5 * (gram + gram.T)
    top = np.linalg.eigvalsh(gram)[-1].clip(min=0.0)
    return float(np.sqrt(top))


def build(N, s, p, q, seed):
    rng = np.random.default_rng(seed)
    prob = np.full((N, N), q)
    prob[:s, :s] = p
    up = rng.random((N, N)) < prob
    up = np.triu(up, 1)
    W = (up + up.T).astype(float)
    return W


def analyze(N, s, p, q, K, tag, n_grid):
    W = build(N, s, p, q, 3)
    Wt = W + np.eye(N)
    dt = Wt.sum(1)
    Ah = Wt / np.sqrt(np.outer(dt, dt))
    L = np.eye(N) - Ah
    d = K - 1
    lam, U = np.linalg.eigh(L)
    lam = np.clip(lam, 1e-12, None)
    lmax = lam[-1]
    v = np.zeros(N)
    v[:s] = np.sqrt(dt[:s])
    v /= np.linalg.norm(v)
    pk = (U.T @ v) ** 2
    Phi = float(pk @ lam)
    A = U[:, 1:K] * (lam[1:K] ** -0.5)[None, :]
    # --- theory quantities ---
    rK = float((pk[1:K] / lam[1:K]).sum())  # truncated resistance of v_S
    R2_pred = rK / s
    nu_pred = float((1.0 / lam[1:K]).sum()) / N
    # --- measured geometry ---
    cS = A[:s].mean(0)
    cH = A[s:].mean(0)
    R2_meas = float(((cS - cH) ** 2).sum())
    nu_meas = float(((A[s:] - cH) ** 2).sum(1).mean())
    Z1 = np.sqrt(R2_pred) / (2 * np.sqrt(nu_pred))
    print(f"\n=== {tag}: Phi={Phi:.3f}  K={K}")
    print(
        f"  R^2: pred {R2_pred:.4f}  meas {R2_meas:.4f}   nu: pred {nu_pred:.4f}  meas {nu_meas:.4f}   Z1={Z1:.2f}"
    )
    print(
        f"  Ward frontier N*rK/(4*s*trK) = {N*rK/(4*s*nu_pred*N):.2f}  (>1 => detectable)"
    )
    # --- Ward runs: F1 and Frobenius cost vs n ---
    conn = csr_matrix(W)
    normA = float((A**2).sum())
    for n in n_grid:
        lab = (
            AgglomerativeClustering(n_clusters=n, connectivity=conn, linkage="ward")
            .fit(A)
            .labels_
        )
        # frobenius cost of partition
        F = 0.0
        for c in np.unique(lab):
            m = lab == c
            F += ((A[m] - A[m].mean(0)) ** 2).sum()
        best = (0, 0, 0)
        for c in np.unique(lab):
            m = lab == c
            tp = m[:s].sum()
            pr = tp / m.sum()
            rc = tp / s
            f1 = 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0
            if f1 > best[0]:
                best = (f1, pr, rc)

        eps__measured = _exact_rsa_epsilon(A, L, lab)
        F_pred = nu_pred * N * n ** (-2.0 / d)  # a-priori Zador (c_d=1)
        # F_pred = (N - n) * nu_pred
        eps_cert = np.sqrt(
            lmax * F_pred / d
        )  # certified: eps <= sqrt(lmax*F)/|A| scale
        print(
            f"  n={n:4d}: F1={best[0]:.3f} (p={best[1]:.2f} r={best[2]:.2f})  F={F:.2f} vs pred {F_pred:.2f}  eps_cert~{eps_cert:.3f} vs eps_meas~{eps__measured:.3f}"
        )


# EASY regime: window open (Phi<1/2)
analyze(
    1200, 40, 0.5, 0.008, K=5, tag="easy (p=0.5,q=0.008)", n_grid=[300, 100, 40, 15]
)
# HARD regime replica
analyze(1200, 40, 0.2, 0.04, K=60, tag="hard (user replica)", n_grid=[75, 40])
