"""Ward-style agglomeration whose *mass* comes from the graph, not the node count.

Standard Ward merges the adjacent pair minimising

    Delta(C1, C2) = |C1||C2| / (|C1|+|C2|) * || abar_C1 - abar_C2 ||^2,

i.e. the mass factor is the **cardinality**.  On a coarsened *weighted* graph that
is arguably the wrong notion of size: a supernode's real mass is how much structure
it has absorbed -- its **self-loop (internal) weight** -- not how many nodes it
happens to contain.  This module implements the agglomeration with a configurable
mass:

* ``cardinality`` -- ``m_i = |C_i|``            (standard Ward; the baseline)
* ``selfloop``    -- ``m_i = W_ii + floor``     (internal weight = density mass)
* ``volume``      -- ``m_i = W_ii + cut_i + floor``  (total volume ``d_i``)

The coarse graph is maintained by the Laplacian-consistent ``W_c = S^T W S`` rule, so
**cut and density are both preserved** exactly as in :func:`_reduce_adjacency`:

    selfw[s] = selfw[i] + selfw[j] + 2 * w_ij      (internal weight; density)
    w[s, k]  = w[i, k] + w[j, k]                   (cut to every other supernode)

``floor`` (default 1.0) keeps singletons from having zero mass at level 0, where no
self-loops exist yet -- without it every first merge would cost exactly 0.

Merges are restricted to *adjacent* pairs, so every contraction set stays connected.
The merge order is returned as an sklearn-format ``children`` array (row ``i`` = the
pair merged to form node ``n+i``), so the existing tree-cut machinery
(:func:`_labels_from_tree`, epsilon/F1 stopping) applies unchanged.
"""

from __future__ import annotations

import heapq

import numpy as np
import torch


def weighted_ward_tree(W: torch.Tensor, A: torch.Tensor, *, mass_mode: str = "selfloop",
                       floor: float = 1.0) -> np.ndarray:
    """Greedy connectivity-constrained agglomeration; returns sklearn-style children.

    ``W`` is the (sparse, possibly self-looped) graph and ``A`` the ``M_tau``-orthonormal
    target rows the Ward objective acts on.
    """

    W = W.coalesce()
    n = int(W.shape[0])
    cap = 2 * n - 1
    idx = W.indices().cpu().numpy()
    val = W.values().cpu().numpy().astype(np.float64)
    d = int(A.shape[1])

    cent = np.zeros((cap, d), dtype=np.float64)
    cent[:n] = A.detach().cpu().numpy().astype(np.float64)
    selfw = np.zeros(cap)
    cut_tot = np.zeros(cap)
    card = np.zeros(cap)
    card[:n] = 1.0
    nbr: list = [dict() for _ in range(cap)]
    for a, b, w in zip(idx[0], idx[1], val):
        a, b = int(a), int(b)
        if a == b:
            selfw[a] += w
        else:
            nbr[a][b] = nbr[a].get(b, 0.0) + w
    for i in range(n):
        cut_tot[i] = sum(nbr[i].values())

    alive = np.zeros(cap, dtype=bool)
    alive[:n] = True

    def mass(i: int) -> float:
        if mass_mode == "cardinality":
            return card[i]
        if mass_mode == "volume":
            return selfw[i] + cut_tot[i] + floor
        return selfw[i] + floor  # selfloop

    def cost(i: int, j: int) -> float:
        mi, mj = mass(i), mass(j)
        diff = cent[i] - cent[j]
        return (mi * mj) / (mi + mj) * float(diff @ diff)

    heap: list = []
    for i in range(n):
        for j in nbr[i]:
            if i < j:
                heapq.heappush(heap, (cost(i, j), i, j))

    children: list = []
    nxt = n
    while heap and nxt < cap:
        c, i, j = heapq.heappop(heap)
        # lazy deletion: skip dead pairs or pairs that are no longer adjacent.
        # A live pair's cost never goes stale -- only *new* supernodes get a fresh
        # centroid/mass, and old nodes' centroids never change.
        if not (alive[i] and alive[j]) or j not in nbr[i]:
            continue
        w_ij = nbr[i][j]
        mi, mj = mass(i), mass(j)
        cent[nxt] = (mi * cent[i] + mj * cent[j]) / (mi + mj)  # mass-weighted centroid
        selfw[nxt] = selfw[i] + selfw[j] + 2.0 * w_ij  # density preserved
        card[nxt] = card[i] + card[j]

        nb: dict = {}
        for k, w in nbr[i].items():
            if k != j and alive[k]:
                nb[k] = nb.get(k, 0.0) + w
        for k, w in nbr[j].items():
            if k != i and alive[k]:
                nb[k] = nb.get(k, 0.0) + w
        nbr[nxt] = nb
        cut_tot[nxt] = sum(nb.values())
        for k, w in nb.items():  # cut preserved: weights to k add
            nbr[k].pop(i, None)
            nbr[k].pop(j, None)
            nbr[k][nxt] = w

        alive[i] = alive[j] = False
        alive[nxt] = True
        children.append((i, j))
        for k in nb:
            heapq.heappush(heap, (cost(nxt, k), min(nxt, k), max(nxt, k)))
        nxt += 1

    return np.asarray(children, dtype=np.int64)
