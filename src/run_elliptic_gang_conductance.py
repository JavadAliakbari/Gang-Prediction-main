"""Do illicit accounts form dense, low-conductance motifs in Elliptic++ Actors?

Motivation
----------
The accompanying analysis ("Graph Coarsening for Gang Detection") argues that
AML *gangs* are low-conductance vertex sets: internally connected but weakly
attached to the rest of the graph. Conductance is the single quantity that
controls whether a gang lives in the low-frequency Laplacian subspace and can
therefore be recovered by (spectral / SGC) coarsening.

For a vertex set S in a weighted graph with adjacency W, degree d = W·1:

    vol(S)  = sum_{i in S} d_i
    cut(S)  = sum_{i in S, j not in S} W_ij
    Phi(S)  = cut(S) / min(vol(S), vol(V\S))      (normalized conductance)
    phi(S)  = cut(S) / |S|                          (boundary-per-node proxy, = m1)

This script tests the hypothesis on the Elliptic++ Actors (wallet-address)
dataset: we define illicit *gangs* as the connected components (size >= 2) of
the illicit-induced subgraph, then compare their conductance against
  (a) licit connected components, and
  (b) size-matched random node sets and random *connected* sets,
all measured in the full transaction graph of the chosen day window.

Edge weights
------------
AddrAddr_edgelist.csv carries only address pairs (no per-transaction BTC
amount). The richest "transactions as weights" signal available is the
*multiplicity* of an address-address pair (how many transaction records connect
them). We therefore report two graphs in parallel:
  - unweighted  (W_ij = 1)
  - weighted    (W_ij = number of transaction records between i and j)

Run
---
    conda activate FedStruct   # python with numpy/scipy/pandas
    python -m src.GangPrediction.run_elliptic_gang_conductance \
        --day-start 24 --day-end 26 --out results/elliptic_conductance
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(
    data_dir: Path, day_start: int, day_end: int
) -> Tuple[csr_matrix, csr_matrix, np.ndarray, pd.DataFrame]:
    """Build symmetric unweighted and weighted adjacency for the day window.

    Returns
    -------
    A_unw : csr_matrix (N, N)  symmetric 0/1 adjacency
    A_w   : csr_matrix (N, N)  symmetric weighted adjacency (edge multiplicity)
    cls   : (N,) int array of class labels (1 illicit, 2 licit, 3 unknown)
    nodes_df : the per-node frame (address, Time step, class)
    """
    print(f"  Reading features (time steps {day_start}-{day_end}) …")
    feat = pd.read_csv(
        data_dir / "wallets_features.csv",
        usecols=["address", "Time step"],
        dtype={"address": str, "Time step": "int32"},
    )
    feat = feat[feat["Time step"].between(day_start, day_end)]
    feat = feat.drop_duplicates(subset=["address"], keep="last").copy()

    print("  Reading classes …")
    classes = pd.read_csv(
        data_dir / "wallets_classes.csv", dtype={"address": str}
    )
    nodes_df = feat.merge(classes, on="address", how="left")
    nodes_df["class"] = nodes_df["class"].fillna(3).astype(int)

    node_to_index = {a: i for i, a in enumerate(nodes_df["address"].values)}
    N = len(nodes_df)
    print(f"  Nodes: {N:,}")

    print("  Reading edge list …")
    edg = pd.read_csv(
        data_dir / "AddrAddr_edgelist.csv",
        dtype={"input_address": str, "output_address": str},
    )
    nodeset = set(node_to_index)
    mask = edg["input_address"].isin(nodeset) & edg["output_address"].isin(nodeset)
    edg = edg[mask]
    s = edg["input_address"].map(node_to_index).to_numpy(np.int64)
    d = edg["output_address"].map(node_to_index).to_numpy(np.int64)
    print(f"  Directed edge records in window: {len(s):,}")

    # Symmetrize: stack both directions, then sum duplicate pairs -> multiplicity.
    row = np.r_[s, d]
    col = np.r_[d, s]
    val = np.ones(len(row), dtype=np.float64)
    A_w = csr_matrix((val, (row, col)), shape=(N, N))
    A_w.sum_duplicates()  # weighted: W_ij = #records between i and j (both dirs)
    A_w.setdiag(0)
    A_w.eliminate_zeros()

    A_unw = A_w.copy()
    A_unw.data[:] = 1.0  # unweighted: presence only

    cls = nodes_df["class"].to_numpy()
    print(
        f"  Undirected edges: {A_unw.nnz // 2:,}  |  "
        f"illicit={int((cls==1).sum()):,} licit={int((cls==2).sum()):,} "
        f"unknown={int((cls==3).sum()):,}"
    )
    return A_unw, A_w, cls, nodes_df


# ---------------------------------------------------------------------------
# Conductance metrics
# ---------------------------------------------------------------------------


def set_metrics(A: csr_matrix, deg: np.ndarray, total_vol: float, S: np.ndarray) -> Dict:
    """Conductance / density metrics for vertex set S on weighted graph A.

    deg = A·1 (precomputed), total_vol = deg.sum().
    """
    s = len(S)
    sub = A[S][:, S]
    internal_vol = float(sub.sum())          # counts each internal edge twice
    internal_edges_w = internal_vol / 2.0
    vol_S = float(deg[S].sum())
    cut = vol_S - internal_vol               # weight of boundary edges
    vol_comp = total_vol - vol_S
    denom = min(vol_S, vol_comp)
    phi_norm = cut / denom if denom > 0 else np.nan      # normalized conductance
    phi_proxy = cut / s                                   # boundary-per-node (m1)
    # internal density uses the *number* of internal edges (unweighted count of
    # the submatrix nonzeros / 2); for the weighted graph this still reflects wiring
    internal_edges_cnt = sub.nnz / 2.0
    max_edges = s * (s - 1) / 2.0
    density = internal_edges_cnt / max_edges if max_edges > 0 else np.nan
    return dict(
        size=s,
        cut=cut,
        vol=vol_S,
        internal_edges=internal_edges_cnt,
        internal_weight=internal_edges_w,
        density=density,
        avg_internal_deg=2 * internal_edges_cnt / s,
        phi_norm=phi_norm,
        phi_proxy=phi_proxy,
    )


def connected_components_sets(
    A: csr_matrix, member_idx: np.ndarray, min_size: int = 2
) -> List[np.ndarray]:
    """Connected components (size >= min_size) of the subgraph induced by member_idx.

    Returned node ids are in the original (global) node space.
    """
    if len(member_idx) == 0:
        return []
    sub = A[member_idx][:, member_idx]
    n_comp, lab = connected_components(sub, directed=False)
    out = []
    for c in range(n_comp):
        m = lab == c
        if m.sum() >= min_size:
            out.append(member_idx[m])
    return out


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def random_set(N: int, size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(N, size=size, replace=False)


def random_connected_set(
    A: csr_matrix, size: int, rng: np.random.Generator, max_tries: int = 20
) -> np.ndarray:
    """Grow a connected node set of `size` by randomized BFS from a random seed.

    Falls back to whatever it reached if the seed's component is smaller.
    """
    indptr, indices = A.indptr, A.indices
    for _ in range(max_tries):
        seed = int(rng.integers(N := A.shape[0]))
        if indptr[seed + 1] == indptr[seed]:
            continue  # isolated node
        visited = {seed}
        frontier = [seed]
        while frontier and len(visited) < size:
            u = frontier.pop(rng.integers(len(frontier)) if len(frontier) > 1 else 0)
            nbrs = indices[indptr[u] : indptr[u + 1]]
            rng.shuffle(nbrs := nbrs.copy())
            for v in nbrs:
                if v not in visited:
                    visited.add(int(v))
                    frontier.append(int(v))
                    if len(visited) >= size:
                        break
        if len(visited) >= 2:
            return np.fromiter(visited, dtype=np.int64)
    return np.fromiter(visited, dtype=np.int64)


def baseline_for_sizes(
    A: csr_matrix,
    deg: np.ndarray,
    total_vol: float,
    sizes: List[int],
    kind: str,
    rng: np.random.Generator,
    samples_per_size: int = 50,
) -> pd.DataFrame:
    """Compute metrics for `samples_per_size` random sets matched to each size."""
    N = A.shape[0]
    rows = []
    for sz in sizes:
        for _ in range(samples_per_size):
            if kind == "random":
                S = random_set(N, sz, rng)
            elif kind == "random_connected":
                S = random_connected_set(A, sz, rng)
            else:
                raise ValueError(kind)
            m = set_metrics(A, deg, total_vol, S)
            m["group"] = kind
            m["target_size"] = sz
            rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarize(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby("group")[col]
    return pd.DataFrame(
        {
            "n": g.size(),
            "mean": g.mean(),
            "median": g.median(),
            "p25": g.quantile(0.25),
            "p75": g.quantile(0.75),
        }
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--day-start", type=int, default=24)
    ap.add_argument("--day-end", type=int, default=26)
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--samples-per-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/elliptic_conductance", type=Path)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"=== Elliptic++ gang conductance | days {args.day_start}-{args.day_end} ===")
    A_unw, A_w, cls, nodes_df = build_graph(
        args.data_dir, args.day_start, args.day_end
    )
    N = A_unw.shape[0]

    illicit_idx = np.where(cls == 1)[0]
    licit_idx = np.where(cls == 2)[0]

    illicit_gangs = connected_components_sets(A_unw, illicit_idx, args.min_gang_size)
    licit_comps = connected_components_sets(A_unw, licit_idx, args.min_gang_size)
    print(
        f"\n  Illicit gangs (CC>= {args.min_gang_size}): {len(illicit_gangs)}  |  "
        f"Licit components: {len(licit_comps)}"
    )

    gang_sizes = sorted(len(g) for g in illicit_gangs)

    # Run both unweighted and weighted graphs.
    for tag, A in [("unweighted", A_unw), ("weighted", A_w)]:
        print(f"\n----- {tag} graph -----")
        deg = np.asarray(A.sum(axis=1)).ravel()
        total_vol = float(deg.sum())

        rows = []
        for S in illicit_gangs:
            m = set_metrics(A, deg, total_vol, S)
            m["group"] = "illicit_gang"
            m["target_size"] = m["size"]
            rows.append(m)
        for S in licit_comps:
            m = set_metrics(A, deg, total_vol, S)
            m["group"] = "licit_comp"
            m["target_size"] = m["size"]
            rows.append(m)
        df_real = pd.DataFrame(rows)

        df_rand = baseline_for_sizes(
            A, deg, total_vol, gang_sizes, "random", rng, args.samples_per_size
        )
        df_rconn = baseline_for_sizes(
            A, deg, total_vol, gang_sizes, "random_connected", rng,
            args.samples_per_size,
        )
        df = pd.concat([df_real, df_rand, df_rconn], ignore_index=True)
        df["graph"] = tag
        df["day_start"] = args.day_start
        df["day_end"] = args.day_end

        out_csv = args.out / f"conductance_{tag}_d{args.day_start}-{args.day_end}.csv"
        df.to_csv(out_csv, index=False)

        print(f"\n  Normalized conductance  Phi(S) = cut/min(vol(S),vol(V\\S)):")
        print(summarize(df, "phi_norm").to_string())
        print(f"\n  Boundary-per-node proxy  phi(S) = cut/|S|  (= m1):")
        print(summarize(df, "phi_proxy").to_string())
        print(f"\n  Internal density  (2 E_in / s(s-1)):")
        print(summarize(df, "density").to_string())
        print(f"\n  saved -> {out_csv}")

    # Save a gang-level table (unweighted) for inspection.
    deg = np.asarray(A_unw.sum(axis=1)).ravel()
    tot = float(deg.sum())
    gtab = pd.DataFrame(
        [
            {**set_metrics(A_unw, deg, tot, S), "gang_id": i}
            for i, S in enumerate(illicit_gangs)
        ]
    ).sort_values("size", ascending=False)
    gtab.to_csv(args.out / f"illicit_gangs_d{args.day_start}-{args.day_end}.csv", index=False)
    print(f"\n  Per-gang table -> {args.out}/illicit_gangs_d{args.day_start}-{args.day_end}.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
