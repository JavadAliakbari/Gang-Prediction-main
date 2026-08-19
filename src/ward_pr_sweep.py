"""Full incremental Ward-order sweep: every metric at every coarsening level.

Instead of cutting the Ward tree at a geometric schedule of cluster counts and
picking one stop (``ward_tree_coarsen``), this walks the **entire merge order** the
Ward clustering produces -- from ``n`` singletons down to 2 clusters -- and records
recall / precision / F1 / Jaccard / detection-rate and the coarsening distortion
``epsilon`` at *every* level.  That is exactly an ``epsilon``-sweep from 0 to 1 with
all intermediate results, from which the stop-criterion metrics, the
precision-recall curve (+ its AUC), and precision/recall-vs-epsilon (with the best
F1 marked) all follow.

Efficiency.  Building the labels afresh at each cut is ``O(n)`` per cut, i.e.
``O(n^2)`` over the sweep.  Instead we replay the merges once through a union-find,
and maintain -- incrementally -- for every community its dominant supernode and the
counts that recall/precision/F1/Jaccard need.  A merge only touches the communities
present in the two merged supernodes, and only those communities' running metric
sums are updated, so the whole sweep costs ``O(sum_i |S_i|)`` plus the one-off Ward
build, *not* ``O(n * #communities)``.  A trajectory row is emitted only at levels
where some community's assignment actually changes (background-only merges leave the
metrics untouched), so the trajectory is compact and lossless for the metrics.

``epsilon`` here is the normalized cumulative Ward distortion
``sqrt(sum_{merged} d / sum_all d)`` in ``[0, 1]`` -- the fraction of the target's
energy the coarsening has destroyed so far (the tight, additive per-merge RSA cost;
Loukas' product bound is a looser form).  It is monotone because Ward's merge
distances are non-decreasing.
"""

from __future__ import annotations

import numpy as np


def _jaccard(recall: float, precision: float) -> float:
    if recall <= 1e-12 or precision <= 1e-12:
        return 0.0
    return 1.0 / (1.0 / recall + 1.0 / precision - 1.0)


def ward_order(adjacency, basis, tau, laplacian="symmetric"):
    """Build the connectivity-constrained Ward tree.

    Returns ``(children, distances, a0, metric)`` where ``a0`` is the
    ``M_tau``-orthonormal target basis the Ward objective acts on (``a0^T M_tau
    a0 = I``) and ``metric`` is the sparse screened metric ``M_tau`` -- both are
    what :func:`~src.loukas_sgc_detection._exact_rsa_epsilon` needs to score any
    partition's exact RSA distortion.
    """

    from scipy.sparse import csr_matrix
    from sklearn.cluster import AgglomerativeClustering

    from src.loukas_sgc_detection import (
        _l_orthonormalize,
        _laplacian,
        _normalized_laplacian,
        _screened_metric,
    )

    base = (
        _laplacian if laplacian in ("combinatorial", "comb") else _normalized_laplacian
    )
    metric = _screened_metric(base(adjacency), tau)
    a0 = _l_orthonormalize(basis, metric)
    if a0.shape[1] == 0:
        raise ValueError("degenerate target subspace (no positive-energy direction)")
    A = a0.detach().cpu().numpy()
    n = int(adjacency.shape[0])
    idx = adjacency.coalesce().indices().cpu().numpy()
    off = idx[0] != idx[1]
    conn = csr_matrix(
        (np.ones(int(off.sum())), (idx[0][off], idx[1][off])), shape=(n, n)
    )
    model = AgglomerativeClustering(
        n_clusters=2,
        linkage="ward",
        connectivity=conn,
        compute_full_tree=True,
        compute_distances=True,
    ).fit(A)
    return np.asarray(model.children_), np.asarray(model.distances_), a0, metric


def sweep_metrics(children, distances, n, gang_sets, threshold):
    """Incremental replay of the Ward merges -> full per-level metric trajectory.

    Returns a list of dicts (one per level where the metrics change), each with
    ``n_coarse, epsilon, mean_recall, mean_precision, mean_f1, mean_jaccard,
    det_rate``.
    """

    G = len(gang_sets)
    if G == 0:
        return []
    gsize = np.array([len(s) for s in gang_sets], dtype=np.float64)

    parent = np.arange(2 * n - 1, dtype=np.int64)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    size = np.ones(2 * n - 1, dtype=np.int64)  # node count per supernode
    cnt = [dict() for _ in range(G)]  # gang g: root -> #g-nodes there
    best_cnt = np.ones(G, dtype=np.int64)  # dominant supernode's g-count
    best_root = np.zeros(G, dtype=np.int64)  # dominant supernode id
    gangs_in: list = [None] * (2 * n - 1)  # root -> set of gang ids

    for g, S in enumerate(gang_sets):
        for v in S:
            v = int(v)
            cnt[g][v] = 1
            if gangs_in[v] is None:
                gangs_in[v] = {g}
            else:
                gangs_in[v].add(g)
        best_root[g] = int(S[0])

    # per-gang current metric tuple + running sums
    def metrics_of(g):
        bc = best_cnt[g]
        r = bc / gsize[g]
        p = bc / size[best_root[g]]
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        j = _jaccard(r, p)
        d = 1.0 if (r > threshold and p > threshold) else 0.0
        return (r, p, f, j, d)

    cur = np.array([metrics_of(g) for g in range(G)], dtype=np.float64)  # (G,5)
    s = cur.sum(0)  # [sum_recall, sum_prec, sum_f1, sum_jac, sum_det]

    total = float(distances.sum()) or 1.0
    cum = 0.0
    traj = [
        {
            "n_coarse": n,
            "epsilon": 0.0,
            "mean_recall": s[0] / G,
            "mean_precision": s[1] / G,
            "mean_f1": s[2] / G,
            "mean_jaccard": s[3] / G,
            "det_rate": s[4] / G,
        }
    ]

    for i in range(len(children)):
        a, b = int(children[i, 0]), int(children[i, 1])
        node = n + i
        ra, rb = find(a), find(b)
        cum += float(distances[i])
        parent[ra] = node
        parent[rb] = node
        size[node] = size[ra] + size[rb]

        ga, gb = gangs_in[ra], gangs_in[rb]
        touched = (ga or set()) | (gb or set()) if (ga or gb) else None
        gangs_in[node] = touched
        gangs_in[ra] = gangs_in[rb] = None

        changed = False
        if touched:
            for g in touched:
                ca = cnt[g].pop(ra, 0)
                cb = cnt[g].pop(rb, 0)
                nc = ca + cb
                if nc:
                    cnt[g][node] = nc
                if nc >= best_cnt[g]:  # only the merged supernode grew
                    best_cnt[g] = nc
                    best_root[g] = node
                new = metrics_of(g)
                old = cur[g]
                if new != tuple(old):
                    s += np.subtract(new, old)
                    cur[g] = new
                    changed = True

        if changed or i == len(children) - 1:
            traj.append(
                {
                    "n_coarse": n - (i + 1),
                    "epsilon": float(np.sqrt(cum / total)),
                    "mean_recall": s[0] / G,
                    "mean_precision": s[1] / G,
                    "mean_f1": s[2] / G,
                    "mean_jaccard": s[3] / G,
                    "det_rate": s[4] / G,
                }
            )
    return traj


# --------------------------------------------------------------------------- #
# Exact RSA epsilon on an adaptive, budgeted set of levels.
#
# The sweep's default x-axis is the cumulative Ward distortion -- monotone and
# free, but only the additive/aggregate form of the RSA constant, not the exact
# worst-case ``sqrt(lambda_max(Y^T M_tau Y))``.  Computing the exact constant at
# every one of ``n-1`` levels means an eigendecomposition per level (infeasible at
# 334k scale).  Instead we sample the exact constant at a *fixed budget* of levels,
# placed adaptively so they come out as close to uniformly spaced *in epsilon* as
# the budget allows: anchor both endpoints, then always spend the next evaluation
# on the interval whose endpoints currently straddle the largest epsilon gap,
# splitting it at its middle level.  (This is the greedy largest-gap form of "divide
# each side in proportion to its epsilon span" -- it drives the max unresolved gap
# down fastest.)  epsilon(level) is monotone, so linear interpolation between the
# sampled levels calibrates an exact epsilon for every level in the trajectory.
# --------------------------------------------------------------------------- #
def _labels_at(children, n, t):
    """Contiguous supernode labels (torch long, length n) after the first t merges."""

    import torch

    parent = np.arange(n + t, dtype=np.int64)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for i in range(t):
        a, b = int(children[i, 0]), int(children[i, 1])
        parent[find(a)] = n + i
        parent[find(b)] = n + i
    roots = np.fromiter((find(v) for v in range(n)), dtype=np.int64, count=n)
    _, inv = np.unique(roots, return_inverse=True)
    return torch.as_tensor(inv, dtype=torch.long)


def adaptive_exact_epsilon(children, n, a0, metric, *, budget=40, min_gap=1e-3):
    """Exact RSA epsilon at up to ``budget`` adaptively-placed Ward levels.

    Returns ``(levels, eps)`` sorted by merge count ``t`` (number of merges applied,
    so ``n_coarse = n - t``); both endpoints ``t=0`` (eps 0, free) and ``t=M`` (fully
    merged) are always included.  ``levels`` are counts of merges; convert to
    ``n_coarse`` with ``n - t``.
    """

    import heapq

    from src.loukas_sgc_detection import _exact_rsa_epsilon

    M = len(children)
    cache = {0: 0.0}  # t=0: identity coarsening, zero distortion (free anchor)

    def eps_at(t):
        if t in cache:
            return cache[t]
        e = _exact_rsa_epsilon(a0, metric, _labels_at(children, n, t))
        cache[t] = e
        return e

    e_hi = eps_at(M)  # far anchor -- one solve
    used = 1
    heap = []  # min-heap on negative epsilon gap

    def push(l, r, el, er):
        if r - l >= 2:  # room for an interior level
            heapq.heappush(heap, (-(er - el), l, r, el, er))

    push(0, M, cache[0], e_hi)
    while used < budget and heap:
        neg, l, r, el, er = heapq.heappop(heap)
        if -neg <= min_gap:  # largest remaining gap is negligible -> uniform enough
            break
        m = (l + r) // 2
        em = eps_at(m)
        used += 1
        push(l, m, el, em)
        push(m, r, em, er)

    ts = sorted(cache)
    return np.array(ts, dtype=np.int64), np.array([cache[t] for t in ts], dtype=float)


def calibrate_epsilon(traj, levels, eps, n):
    """Add an ``epsilon_exact`` field to every trajectory row by monotone interpolation.

    ``levels``/``eps`` come from :func:`adaptive_exact_epsilon`.  Each row's merge
    count ``t = n - n_coarse`` is mapped through the sampled exact curve (linear in
    ``t``; the curve is monotone so the interpolant is too).
    """

    for row in traj:
        t = n - row["n_coarse"]
        row["epsilon_exact"] = float(np.interp(t, levels, eps))
    return traj


def ward_pr_sweep(
    adjacency, basis, gang_sets, *, tau=0.5, laplacian="symmetric", threshold=0.51
):
    """Convenience: build the Ward order and run the full incremental metric sweep."""

    children, distances, _, _ = ward_order(adjacency, basis, tau, laplacian)
    return sweep_metrics(
        children, distances, int(adjacency.shape[0]), gang_sets, threshold
    )


# --------------------------------------------------------------------------- #
def pr_auc(recall, precision):
    """Area under the precision-recall curve (trapezoidal, sorted by recall)."""

    r = np.asarray(recall, dtype=float)
    p = np.asarray(precision, dtype=float)
    order = np.argsort(r)
    return float(np.trapezoid(p[order], r[order]))


def summarize_sweep(traj, eps_budget=None, eps_key="epsilon"):
    """Best-F1 stop + (optional) the level at a given epsilon budget + PR-AUC.

    ``eps_key`` selects which epsilon axis the budget stop is read on -- the free
    cumulative ``"epsilon"`` or the calibrated ``"epsilon_exact"``.
    """

    keys = (
        "epsilon",
        "n_coarse",
        "mean_recall",
        "mean_precision",
        "mean_f1",
        "mean_jaccard",
        "det_rate",
    )
    if eps_key != "epsilon" and eps_key in traj[0]:
        keys = (eps_key,) + keys
    df_eps = np.array([t[eps_key] for t in traj])
    f1 = np.array([t["mean_f1"] for t in traj])
    best = traj[int(f1.argmax())]
    out = {
        "pr_auc": pr_auc(
            [t["mean_recall"] for t in traj], [t["mean_precision"] for t in traj]
        ),
        "best_f1": {k: best[k] for k in keys},
    }
    if eps_budget is not None:
        # stop criterion: coarsen as long as distortion stays within budget,
        # i.e. the coarsest recorded level with epsilon <= budget
        within = np.where(df_eps <= eps_budget + 1e-12)[0]
        j = int(within[-1]) if len(within) else 0
        at = traj[j]
        out["at_epsilon"] = {"epsilon_budget": eps_budget, **{k: at[k] for k in keys}}
    return out


def plot_sweep(
    traj, tag, path, eps_budget=None, eps_key="epsilon", exact_checkpoints=None
):
    """PR curve (+AUC, best-F1) and precision/recall/F1/Jaccard vs epsilon.

    ``eps_key`` picks the epsilon axis (``"epsilon"`` = free cumulative distortion,
    ``"epsilon_exact"`` = the calibrated exact RSA constant).  ``exact_checkpoints``,
    if given as ``(n_coarse_array, eps_array)``, marks on the vs-epsilon panel the
    levels where the exact RSA constant was actually evaluated.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exact = eps_key != "epsilon" and eps_key in traj[0]
    eps = np.array([t[eps_key] for t in traj])
    rec = np.array([t["mean_recall"] for t in traj])
    prec = np.array([t["mean_precision"] for t in traj])
    f1 = np.array([t["mean_f1"] for t in traj])
    jac = np.array([t["mean_jaccard"] for t in traj])
    det = np.array([t["det_rate"] for t in traj])
    bi = int(f1.argmax())
    auc = pr_auc(rec, prec)
    axis_label = (
        "epsilon (exact RSA constant, adaptively calibrated)"
        if exact
        else "epsilon (normalized coarsening distortion, 0→1)"
    )

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))

    order = np.argsort(rec)
    a1.plot(rec[order], prec[order], "-", color="tab:blue", lw=1.5)
    a1.scatter(rec, prec, s=8, c=eps, cmap="viridis", zorder=3)
    a1.scatter(
        [rec[bi]],
        [prec[bi]],
        s=140,
        marker="*",
        color="tab:red",
        zorder=4,
        label=f"best F1={f1[bi]:.3f}",
    )
    a1.set_xlabel("mean recall")
    a1.set_ylabel("mean precision")
    a1.set_xlim(0, 1.02)
    a1.set_ylim(0, 1.02)
    a1.set_title(f"Precision-Recall over the Ward sweep — {tag}\nPR-AUC = {auc:.3f}")
    a1.legend(loc="lower left")
    a1.grid(alpha=0.3)

    a2.plot(eps, prec, "-", color="tab:orange", label="precision")
    a2.plot(eps, rec, "-", color="tab:blue", label="recall")
    a2.plot(eps, f1, "-", color="tab:green", label="F1")
    a2.plot(eps, jac, "--", color="tab:purple", alpha=0.7, label="Jaccard")
    a2.plot(eps, det, ":", color="tab:gray", alpha=0.7, label="detection rate")
    a2.axvline(
        eps[bi], color="tab:red", ls="--", alpha=0.8, label=f"best F1 @ ε={eps[bi]:.3f}"
    )
    if exact and exact_checkpoints is not None:
        _, ck_eps = exact_checkpoints
        a2.plot(
            ck_eps,
            np.zeros_like(ck_eps),
            "k|",
            ms=10,
            alpha=0.7,
            label=f"exact-ε samples ({len(ck_eps)})",
        )
    if eps_budget is not None:
        a2.axvline(
            eps_budget, color="k", ls=":", alpha=0.6, label=f"budget ε={eps_budget:g}"
        )
    a2.set_xlabel(axis_label)
    a2.set_ylabel("metric")
    a2.set_xlim(0, max(1.0, float(eps.max())))
    a2.set_ylim(0, 1.02)
    a2.set_title("Metrics vs epsilon (best-F1 stop marked)")
    a2.legend(fontsize=8, loc="upper right")
    a2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return auc, bi
