"""Gang detection on the GADBench homogeneous fraud graphs (arXiv:2312.06441).

Applies the same spectral-coarsening gang detector built for Elliptic++ to the
node-level anomaly graphs from the benchmark:

  * Amazon    -- 11,944 nodes,  ~4.40M edges, 25 feats,  821 fraud (6.87%)
  * T-Finance -- 39,357 nodes, ~21.22M edges, 10 feats,        (4.58%)
  * T-Social  -- 5,781,065 nodes, ~73.1M edges, 10 feats,      (3.01%)

All three are *homogeneous* graphs with a binary anomaly label.  We define a
*gang* exactly as before -- a connected component (size >= 2) of the
anomaly-induced subgraph -- then run the **structural** and **joint** encoders
(no Laplacian: too expensive) as the Loukas RSA coarsening target and report how
many gangs collapse into a single super-node (recall>thr AND precision>thr).

NOTE on structure.  These are dense node-anomaly graphs, not sparse transaction
graphs: the anomalies tend to form *one* large connected blob rather than many
disjoint motifs (e.g. Amazon: a single 806-node component + isolated singletons).
So the connected-component gang definition yields very few gangs here -- itself
the finding that AML transaction graphs and dense node-anomaly graphs have very
different "gang" structure.  ``--gang-louvain`` optionally subdivides the blob
into communities for a finer, more informative multi-gang detection rate.

Data
----
Amazon downloads itself (DGL FraudAmazon.zip -> Amazon.mat, scipy.io).  T-Finance
and T-Social are DGL ``.bin`` graphs (Tang et al. 2022); pass ``--dgl-path`` to a
local ``tfinance``/``tsocial`` saved with ``dgl.save_graphs`` (requires dgl).

Run::

    python -m src.run_graph_fraud_gang_detection --dataset amazon \
        --coarsening-method capped --epsilon 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix, triu
from scipy.sparse.csgraph import connected_components

from src.run_elliptic_gang_conductance import random_connected_set
from src.pattern_models import create_pattern
from src.run_elliptic_gang_detection import (
    coarsen_and_count,
    make_patterns,
    prepare_subspace,
    random_connected_baseline,
    set_costs,
    split_train_test,
)
from src.loukas_sgc_detection import (
    build_sgc_subspace,
    build_joint_subspace,
    evaluate_loukas_patterns,
    graph_operators,
    loukas_coarsen_pytorch,
)
from src.utils.utils import *


def detection_ceiling(adjacency, basis, eval_patterns, labels, args, eps_grid):
    """Best detection rate a subspace reaches over an epsilon sweep (per-gang union).

    Runs the label-free coarsening at several epsilon budgets and marks a gang
    detected if it clears (recall>thr AND precision>thr) at *any* resolution --
    the ceiling a perfect per-gang stop would reach.  History-free (only the final
    super-node map per run is used), so it works with the stock coarsener.
    """

    gangs = [p for p in eval_patterns if p.label == "alert"]
    ever = [False] * len(gangs)
    for eps in eps_grid:
        co = loukas_coarsen_pytorch(
            adjacency,
            basis,
            reduction=0.999,
            epsilon=eps,
            max_levels=args.ceiling_max_levels,
            method=args.coarsening_method,
            max_contraction_size=args.max_contraction_size,
            max_cluster_size=args.linkage_max_size,
        )
        results, _ = evaluate_loukas_patterns(
            gangs, co.node_to_supernode, labels, threshold=args.threshold
        )
        for i, res in enumerate(results):
            ever[i] = ever[i] or res.detected
    return int(sum(ever)), len(gangs)


from src.sgc_detection import (
    apply_feature_channel,
    apply_graph_filter,
    fit_collective_sgc,
    fit_joint_encoder,
)


def _jaccard(recall: float, precision: float) -> float:
    """Jaccard |S∩D|/|S∪D| of a community S vs its best-match super-node D.

    With recall=|S∩D|/|S| and precision=|S∩D|/|D|, Jaccard = 1/(1/r + 1/p - 1).
    """
    if recall <= 1e-12 or precision <= 1e-12:
        return 0.0
    return 1.0 / (1.0 / recall + 1.0 / precision - 1.0)


def coarsen_and_metrics(
    name, basis, *, adjacency, labels, eval_gangs, top5_gangs, args
):
    """One coarsening -> detection rate + per-community best-match F1 / Jaccard.

    For each ground-truth community we take its best-match super-node (the one
    holding the plurality of its nodes) and score recall=|S∩D|/|S|,
    precision=|S∩D|/|D|, F1=2PR/(P+R), Jaccard.  Mean F1/Jaccard over the
    communities is the standard community-recovery metric (ground-truth side).
    """

    if args.coarsening_method == "ward-tree":
        # ward-tree is the full-tree variant handled by ward_tree_coarsen (the same
        # routing CollectiveBankDetector.coarsen uses); loukas_coarsen_pytorch does
        # NOT know it.  epsilon-stop keeps the cut label-free; f1-stop peeks at
        # eval_gangs' labels (optimistic).
        import math

        from src.run_collective_bank_detection import ward_tree_coarsen

        co, _ = ward_tree_coarsen(
            adjacency,
            basis,
            eval_gangs,
            labels,
            tau=args.coarsening_tau,
            laplacian="symmetric",
            threshold=args.threshold,
            stop=args.ward_stop,
            epsilon_budget=(args.epsilon if args.epsilon is not None else math.inf),
            num_cuts=args.ward_num_cuts,
        )
    else:
        co = loukas_coarsen_pytorch(
            adjacency,
            basis,
            reduction=args.reduction,
            epsilon=args.epsilon,
            max_levels=args.max_levels,
            method=args.coarsening_method,
            max_contraction_size=args.max_contraction_size,
            max_cluster_size=args.linkage_max_size,
            epsilon_ramp_levels=(args.max_levels if args.epsilon_ramp else None),
        )
    n2s = co.node_to_supernode

    def per_set(patterns):
        res, _ = evaluate_loukas_patterns(
            patterns, n2s, labels, threshold=args.threshold
        )
        f1 = np.array([r.f1 for r in res])
        jac = np.array([_jaccard(r.recall, r.precision) for r in res])
        det = np.array([r.detected for r in res])
        rec = np.array([r.recall for r in res])
        prec = np.array([r.precision for r in res])
        return f1, jac, det, rec, prec

    f1a, jaca, deta, reca, preca = per_set(eval_gangs)
    out = {
        "encoder": name,
        "n_coarse": co.n_coarse,
        "epsilon_actual": co.epsilon,
        "n_gangs": len(eval_gangs),
        "detected": int(deta.sum()),
        "det_rate": float(deta.mean()),
        "gang_mean_recall": float(reca.mean()),
        "gang_mean_precision": float(preca.mean()),
        "meanF1_all": float(f1a.mean()),
        "meanJac_all": float(jaca.mean()),
        "mean_recall_all": float(reca.mean()),
        "mean_prec_all": float(preca.mean()),
        # per-community outcome (reused by community_diagnostics; stripped from JSON)
        "_per_gang": {
            "detected": deta.astype(int).tolist(),
            "f1": f1a.tolist(),
            "recall": reca.tolist(),
            "precision": preca.tolist(),
        },
    }
    if top5_gangs:
        f1t, jact, _, rect, prect = per_set(top5_gangs)
        out.update(
            {
                "top5_sizes": [p.num_nodes for p in top5_gangs],
                "top5_F1": [round(float(x), 3) for x in f1t],
                "top5_Jac": [round(float(x), 3) for x in jact],
                "meanF1_top5": float(f1t.mean()),
                "meanJac_top5": float(jact.mean()),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Per-community diagnostics (Elliptic-style) + the overlap dimension
# ---------------------------------------------------------------------------


def membership_counts(communities, n):
    """Per-node count of how many communities each node belongs to (overlap)."""

    m = np.zeros(n, dtype=np.int64)
    for c in communities:
        m[np.asarray(c, dtype=np.int64)] += 1
    return m


def _chunked_gang_moments(a_hat, adjacency, gangs, chunk=100):
    """(Phi, mbar1) per gang -- conductance and boundary-edge mean -- memory-bounded.

    Chunked so the ``(N, m)`` degree-weighted indicator matrix never materializes at
    full width on the large community graphs.
    """

    from src.analyze_missed_gangs import gang_moments

    phis, mbars = [], []
    for i in range(0, len(gangs), chunk):
        p, mb = gang_moments(a_hat, adjacency, gangs[i : i + chunk])
        phis.append(p)
        mbars.append(mb)
    return torch.cat(phis).numpy(), torch.cat(mbars).numpy()


def community_diagnostics(
    name, per_gang, gang_sets, a_hat, adjacency, A_scipy, membership, args, out
):
    """Elliptic-style detected-vs-missed diagnostics + the overlap dimension.

    Records, per evaluated community: size, conductance ``Phi``, boundary-edge mean
    ``mbar1``, internal density, and overlap (mean / max number of communities its
    nodes belong to).  ``per_gang`` is the detection outcome already computed by
    :func:`coarsen_and_metrics` (no re-coarsening).  The overlap columns are the key
    addition over Elliptic: a node in many communities cannot land in the right
    supernode for all of them under a hard partition, so heavy overlap should
    predict misses.
    """

    import pandas as pd

    gangs = make_patterns(gang_sets, "alert", "gang", "d")
    Phi, mbar1 = _chunked_gang_moments(a_hat, adjacency, gangs)
    rows = []
    for i, S in enumerate(gang_sets):
        S = np.asarray(S, dtype=np.int64)
        s = len(S)
        internal_dir = int(A_scipy[S][:, S].nnz)  # directed internal entries = 2*E_int
        density = internal_dir / (s * (s - 1)) if s > 1 else 0.0
        mm = membership[S]
        rows.append(
            {
                "community": i,
                "size": s,
                "Phi": float(Phi[i]),
                "mbar1": float(mbar1[i]),
                "density": density,
                "overlap_mean": float(mm.mean()),
                "overlap_max": int(mm.max()),
                "detected": int(per_gang["detected"][i]),
                "f1": float(per_gang["f1"][i]),
                "recall": float(per_gang["recall"][i]),
                "precision": float(per_gang["precision"][i]),
            }
        )
    df = pd.DataFrame(rows)
    tag = f"{args.dataset}_{name}"
    df.to_csv(out / f"community_diag_{tag}.csv", index=False)
    _plot_community_diag(df, membership, tag, out / f"community_diag_{tag}.png")

    # console summary: median detected vs missed + single-feature AUC
    LOGGER.info(
        f"community diagnostics [{name}]: detected "
        f"{int(df.detected.sum())}/{len(df)} ({df.detected.mean():.1%})"
    )
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        roc_auc_score = None
    for f in ("size", "Phi", "density", "mbar1", "overlap_mean"):
        dm = df[df.detected == 1][f].median()
        mm_ = df[df.detected == 0][f].median()
        a = (
            float(roc_auc_score(df.detected, df[f]))
            if roc_auc_score is not None and df.detected.nunique() > 1
            else float("nan")
        )
        LOGGER.info(
            f"    {f:<13} detected_med={dm:>9.3g}  missed_med={mm_:>9.3g}  AUC={a:.2f}"
        )
    return df


def _plot_community_diag(df, membership, tag, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        roc_auc_score = None

    def auc(col):
        if roc_auc_score is None or df.detected.nunique() < 2:
            return float("nan")
        try:
            return roc_auc_score(df.detected, df[col])
        except Exception:
            return float("nan")

    feats = [
        ("size", True),
        ("Phi", False),
        ("density", False),
        ("mbar1", True),
        ("overlap_mean", False),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(19, 9))
    for ax, (f, logy) in zip(axes.ravel()[:5], feats):
        d = df[df.detected == 1][f].dropna()
        m = df[df.detected == 0][f].dropna()
        ax.boxplot([d, m], tick_labels=["detected", "missed"], showfliers=False)
        ax.scatter(
            np.random.normal(1, 0.05, len(d)), d, s=7, alpha=0.35, color="tab:green"
        )
        ax.scatter(
            np.random.normal(2, 0.05, len(m)), m, s=7, alpha=0.35, color="tab:red"
        )
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"{f}   AUC={auc(f):.2f}")
        ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    sc = ax.scatter(
        df["size"], df.f1, c=df.overlap_mean, cmap="viridis", s=14, alpha=0.75
    )
    ax.set_xscale("log")
    ax.set_xlabel("community size")
    ax.set_ylabel("F1")
    ax.set_title("size vs F1 (color = overlap)")
    fig.colorbar(sc, ax=ax, label="mean #memberships/node")
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    for lab, sub, c in [
        ("detected", df[df.detected == 1], "tab:green"),
        ("missed", df[df.detected == 0], "tab:red"),
    ]:
        ax.scatter(sub.overlap_mean, sub.f1, s=12, alpha=0.6, color=c, label=lab)
    ax.set_xlabel("mean #communities per node (overlap)")
    ax.set_ylabel("F1")
    ax.set_title("overlap vs F1 (does sharing hurt recovery?)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 3]
    mm = membership[membership > 0]
    hi = int(min(mm.max(), 20))
    ax.hist(mm, bins=range(1, hi + 2), color="tab:gray", alpha=0.85)
    ax.set_xlabel("#communities a node belongs to")
    ax.set_ylabel("nodes")
    ax.set_title(
        f"overlap structure (factor {mm.sum() / len(mm):.2f}x, "
        f">1 community: {(mm > 1).mean():.0%})"
    )
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Community-recovery diagnostics — {tag}   "
        f"(detected {int(df.detected.sum())}/{len(df)} = {df.detected.mean():.0%})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dataset loading -> (A_unw csr, X float64, y int{0,1})
# ---------------------------------------------------------------------------


def _standardize(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = X.mean(0), X.std(0)
    keep = sd > 1e-12
    return (X[:, keep] - mu[keep]) / sd[keep]


def load_amazon(data_dir: Path):
    """DGL FraudAmazon ``Amazon.mat`` -- the homogeneous ('homo') graph."""

    import scipy.io as sio

    mat_path = data_dir / "Amazon.mat"
    if not mat_path.exists():
        import urllib.request
        import zipfile

        data_dir.mkdir(parents=True, exist_ok=True)
        zpath = data_dir / "FraudAmazon.zip"
        LOGGER.info("  downloading FraudAmazon.zip …")
        urllib.request.urlretrieve("https://data.dgl.ai/dataset/FraudAmazon.zip", zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(data_dir)
    m = sio.loadmat(mat_path)
    A = m["homo"].tocsr().astype(np.float64)
    A.data[:] = 1.0
    A.setdiag(0)
    A.eliminate_zeros()
    X = _standardize(np.asarray(m["features"].todense(), dtype=np.float64))
    y = m["label"].ravel().astype(np.int64)
    return A, X, y


def load_dgl_bin(dgl_path: Path):
    """T-Finance / T-Social: a homogeneous DGL graph saved with save_graphs."""

    import dgl  # noqa: F401

    graphs, _ = dgl.load_graphs(str(dgl_path))
    g = graphs[0]
    src, dst = (t.numpy() for t in g.edges())
    n = g.num_nodes()
    A = coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
    A = A + A.T
    A.data[:] = 1.0
    A.setdiag(0)
    A.eliminate_zeros()
    feat_key = "feature" if "feature" in g.ndata else "feat"
    X = _standardize(g.ndata[feat_key].numpy().astype(np.float64))
    y = g.ndata["label"].numpy().astype(np.int64)
    return A, X, y


def load_snap_community(data_dir: Path, name: str):
    """Any SNAP ground-truth-community graph (amazon / dblp / youtube / lj / orkut).

    Downloads ``com-<name>.ungraph.txt`` + ``com-<name>.top5000.cmty.txt`` and
    returns ``(A, communities)`` -- a sparse 0/1 adjacency and the community node
    sets (the 'gangs').  No node features: structural-only.
    """

    import gzip
    import urllib.request

    data_dir = Path(data_dir)
    base = "https://snap.stanford.edu/data/bigdata/communities/"
    g_txt = data_dir / f"com-{name}.ungraph.txt"
    c_txt = data_dir / f"com-{name}.top5000.cmty.txt"
    for fname in (f"com-{name}.ungraph.txt", f"com-{name}.top5000.cmty.txt"):
        if not (data_dir / fname).exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"  downloading {fname} …")
            urllib.request.urlretrieve(base + fname + ".gz", data_dir / (fname + ".gz"))
            with gzip.open(data_dir / (fname + ".gz"), "rb") as fz:
                (data_dir / fname).write_bytes(fz.read())

    edges = np.loadtxt(g_txt, dtype=np.int64, comments="#")
    ids = np.unique(edges)
    remap = {int(v): i for i, v in enumerate(ids)}
    r = np.fromiter((remap[int(a)] for a in edges[:, 0]), dtype=np.int64)
    c = np.fromiter((remap[int(b)] for b in edges[:, 1]), dtype=np.int64)
    n = len(ids)
    A = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    A = (A + A.T).tocsr()
    A.data[:] = 1.0
    A.setdiag(0)
    A.eliminate_zeros()
    communities = []
    for line in open(c_txt):
        nodes = [remap[int(x)] for x in line.split() if int(x) in remap]
        if len(nodes) >= 2:
            communities.append(np.asarray(nodes, dtype=np.int64))
    return A, communities


def load_dataset(args):
    if args.dataset == "amazon":
        return load_amazon(args.data_dir)
    if args.dataset in ("tfinance", "tsocial"):
        if not args.dgl_path:
            raise ValueError(
                f"{args.dataset} needs --dgl-path to a DGL .bin graph (requires dgl)"
            )
        return load_dgl_bin(Path(args.dgl_path))
    raise ValueError(args.dataset)


# ---------------------------------------------------------------------------
# Gangs = connected components of the anomaly-induced subgraph
# ---------------------------------------------------------------------------


def anomaly_gangs(A: csr_matrix, y: np.ndarray, min_size: int, louvain: bool):
    """Gang node-sets among the anomalies.

    Default: connected components (size >= ``min_size``) of the anomaly subgraph.
    ``louvain``: subdivide each component into communities (greedy modularity) so
    the dense single-blob graphs yield a finer, more informative gang set.
    """

    anom = np.where(y == 1)[0]
    sub = A[anom][:, anom]
    n_comp, lab = connected_components(sub, directed=False)
    comps = [anom[lab == c] for c in range(n_comp)]
    comps = [c for c in comps if len(c) >= min_size]
    if not louvain:
        return comps

    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities

    gangs = []
    for comp in comps:
        if len(comp) <= 8:
            gangs.append(comp)
            continue
        idx = {int(v): k for k, v in enumerate(comp)}
        s = A[comp][:, comp].tocoo()
        g = nx.Graph()
        g.add_nodes_from(range(len(comp)))
        g.add_edges_from(
            (i, j) for i, j in zip(s.row.tolist(), s.col.tolist()) if i < j
        )
        for community in greedy_modularity_communities(g):
            members = comp[list(community)]
            if len(members) >= min_size:
                gangs.append(members)
    return gangs


def build_graph_obj(A: csr_matrix, X, y: np.ndarray):
    up = triu(A, k=1).tocoo()
    edge_index = torch.tensor(np.vstack([up.row, up.col]), dtype=torch.long)
    return SimpleNamespace(
        edge_index=edge_index,
        edge_weight=torch.ones(edge_index.shape[1], dtype=torch.float64),
        num_nodes=int(A.shape[0]),
        x=None if X is None else torch.from_numpy(X).to(torch.float64),
        y=torch.tensor(y, dtype=torch.long),
    )


def fit_bank_basis(graph, normalized, adjacency, Xf, gang_tr, args):
    """Target subspace ``R = span(g_Theta(A_hat) X)`` from the modular collective bank.

    The dataset-agnostic :class:`CollectiveBankDetector` (the same algorithm used on
    Elliptic++) learns the Chebyshev filter on the training gangs; its target
    subspace is returned as just another encoder basis, so the identical coarsener /
    community-recovery scoring compares it against ``structural`` / ``joint``.

    ``X`` for the bank: the real node features when present (optionally with a random
    structural channel of width ``--bank-random-width`` concatenated), else a purely
    random structural range-finder -- the structural-only mode that captured the
    Elliptic gangs *better* than the real features.
    """

    from src.collective_detector import (
        CollectiveBankDetector,
        DetectorConfig,
        GraphData,
    )
    from src.run_elliptic_gang_detection import random_structural_features

    n = graph.num_nodes
    w = args.bank_random_width
    if Xf is None:
        Xbank = random_structural_features(n, w, args.seed)
    elif w > 0:
        Xr = random_structural_features(n, w, args.seed).to(Xf.dtype)
        Xbank = torch.cat([Xf, Xr], dim=1)  # real features + structural channel
    else:
        Xbank = Xf
    # GraphData built directly on the already-computed operators (no re-derivation
    # of graph_operators on the large graph)
    data = GraphData(
        edge_index=graph.edge_index,
        a_hat=normalized,
        adjacency=adjacency,
        X=Xbank.to(device=normalized.device, dtype=normalized.dtype),
        y=graph.y,
    )
    if len(gang_tr) > data.feature_dim:
        LOGGER.info(
            f"  WARNING: bank #train-gangs ({len(gang_tr)}) > feature-dim "
            f"({data.feature_dim}): capacity threshold -> lambda_min ~ 0. "
            "Raise --bank-random-width or lower --max-retain."
        )
    cfg = DetectorConfig(
        degree=args.bank_degree,
        basis=args.bank_basis,
        tau=args.bank_tau,
        epochs=args.bank_epochs,
        learning_rate=args.learning_rate,
        ridge=args.ridge,
        optimizer=args.bank_optimizer,
        softmin_temperature=args.bank_softmin_temperature,
        capture_objective=args.bank_capture_objective,
        conf_weight=args.bank_conf_weight,
        conf_reduce=args.bank_conf_reduce,
        conf_delta=args.bank_conf_delta,
        conf_halo_hops=args.bank_conf_halo_hops,
        coarsen_target=args.bank_coarsen_target,
        structural_width=args.bank_structural_width,
        seed=args.seed,
    )
    det = CollectiveBankDetector(cfg)
    LOGGER.info(
        f"  Fitting collective bank (basis={cfg.basis}, K={cfg.degree}, tau={cfg.tau}, "
        f"d={data.feature_dim}, opt={cfg.optimizer}, objective={cfg.capture_objective}, "
        f"beta={cfg.conf_weight}, delta={cfg.conf_delta}, target={cfg.coarsen_target}) …"
    )
    det.fit(data, gang_tr)
    fi = det.fit_info_
    LOGGER.info(
        f"    capture ({cfg.capture_objective}) / lambda_min: "
        f"{fi['init_objective']:.4g} -> {fi['objective']:.4g}"
    )
    return det.target_subspace(data, gang_tr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        choices=[
            "amazon",
            "tfinance",
            "tsocial",
            "comamazon",
            "comdblp",
            "comyoutube",
            "comlj",
            "comorkut",
        ],
        default="amazon",
    )
    ap.add_argument("--data-dir", default="data/graph_fraud", type=Path)
    ap.add_argument("--comamazon-dir", default="data/community", type=Path)
    ap.add_argument(
        "--dgl-path",
        default=None,
        help="path to tfinance/tsocial DGL .bin (requires dgl)",
    )
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument(
        "--gang-louvain",
        action="store_true",
        default=False,
        help="subdivide the anomaly blob into communities for a finer "
        "multi-gang detection rate",
    )
    ap.add_argument(
        "--max-gangs",
        type=int,
        default=5000,
        help="cap on #gangs (communities) to evaluate detection over",
    )
    ap.add_argument(
        "--max-retain",
        type=int,
        default=40,
        help="cap on gangs/normals used to *fit* theta (keeps the "
        "N x m propagation tractable on the large com-Amazon graph)",
    )
    ap.add_argument("--max-normal-patterns", type=int, default=100)
    ap.add_argument("--train-ratio", type=float, default=0.5)
    ap.add_argument("--degree", type=int, default=16)
    ap.add_argument("--structural-width", type=int, default=32)
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.02)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--reduction", type=float, default=0.99)
    ap.add_argument("--epsilon", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--max-levels", type=int, default=5)
    ap.add_argument("--max-contraction-size", type=int, default=4)
    ap.add_argument("--linkage-max-size", type=int, default=8)
    ap.add_argument("--epsilon-ramp", action="store_true", default=False)
    ap.add_argument(
        "--coarsening-method",
        choices=[
            "edges",
            "neighborhood",
            "capped",
            "star",
            "kmeans",
            "linkage",
            "ward",
            "ward-tree",
        ],
        default="ward-tree",
        help="'edges' scales best on the ~50k-node graph; 'ward-tree' builds the "
        "full Ward tree (heavier) and stops per --ward-stop.  Note: 'ward' is the "
        "base coarsener's ward-partition; 'ward-tree' is the full-tree variant that "
        "cuts by --ward-stop.",
    )
    ap.add_argument(
        "--ward-stop",
        choices=["epsilon", "f1"],
        default="f1",
        help="for --coarsening-method ward-tree: 'epsilon' cuts at the RSA budget "
        "--epsilon (label-free); 'f1' cuts at the best mean F1 over the evaluated "
        "gangs (peeks at labels -- optimistic).",
    )
    ap.add_argument("--ward-num-cuts", type=int, default=200)
    ap.add_argument(
        "--coarsening-tau",
        type=float,
        default=0.5,
        help="screened metric M_tau for ward-tree coarsening (the RSA distortion "
        "and the tree embedding are measured in it).",
    )
    ap.add_argument("--ceiling-max-levels", type=int, default=40)
    ap.add_argument(
        "--detection-ceiling",
        action="store_true",
        default=False,
        help="also report each subspace's best detection rate over an "
        "epsilon sweep (slow on dense graphs)",
    )
    ap.add_argument(
        "--pr-sweep",
        action="store_true",
        default=True,
        help="walk the full Ward merge order (finest -> 2 clusters), recording "
        "recall/precision/f1/jaccard/detection and epsilon at every level; "
        "emits a per-level CSV, a precision-recall curve (+AUC) and "
        "metrics-vs-epsilon plot (best-F1 stop marked) per encoder.",
    )
    ap.add_argument(
        "--pr-sweep-exact-budget",
        type=int,
        default=200,
        help="if >0, replace the cumulative-distortion epsilon axis with the exact "
        "RSA constant, computed at this many adaptively-placed levels (endpoints "
        "anchored, then the largest epsilon gap split each step) and interpolated "
        "to every level. 0 = free cumulative-distortion axis.",
    )
    ap.add_argument("--joint-contrastive-weight", type=float, default=0.0)
    # --- collective-bank encoder (the modular CollectiveBankDetector algorithm) ---
    ap.add_argument(
        "--use-bank",
        action="store_true",
        default=True,
        help="also fit the modular collective filter-bank (Chebyshev bank + "
        "confusability) and add its target subspace as an encoder, scored by the "
        "same coarsener/metrics as structural/joint.",
    )
    ap.add_argument(
        "--bank-only",
        action="store_true",
        default=True,
        help="skip the legacy structural/joint encoders; run only the collective "
        "bank (implies --use-bank).",
    )
    ap.add_argument(
        "--bank-degree", type=int, default=24, help="bank polynomial degree K"
    )
    ap.add_argument(
        "--bank-basis",
        choices=["chebyshev", "monomial", "lanczos"],
        default="chebyshev",
        help="polynomial basis of the bank: 'chebyshev' (well-conditioned Gram, "
        "eq. 30) or 'monomial' (legacy A_hat^k; same span, worse conditioning).",
    )
    ap.add_argument(
        "--bank-tau",
        type=float,
        default=0.5,
        help="screened metric M_tau = L + tau I for the bank (0 = L_sym)",
    )
    ap.add_argument("--bank-epochs", type=int, default=500)
    ap.add_argument(
        "--bank-optimizer",
        choices=["projected", "riemannian"],
        default="riemannian",
        help="how ||theta^(a)||=1 is enforced: 'riemannian' (Adam on the sphere) or "
        "'projected' (normalize-in-forward).",
    )
    ap.add_argument(
        "--bank-capture-objective",
        choices=["lambda_min", "trace", "softmin_diag"],
        default="lambda_min",
        help="what the bank ascends: 'lambda_min' (capture + cross-gang separation, "
        "m>d capacity wall) | 'trace' (mean per-gang capture) | 'softmin_diag' "
        "(worst gang's capture; no separation term, no capacity wall).",
    )
    ap.add_argument(
        "--bank-softmin-temperature",
        type=float,
        default=0.2,
        help="soft-min temperature for the capture objective (0 = hard min).",
    )
    ap.add_argument(
        "--bank-conf-weight",
        type=float,
        default=10.0,
        help="confusability penalty beta for the bank (0 = off; slow on "
        "large fitting communities since it is O(s^3) per gang).",
    )
    ap.add_argument(
        "--bank-conf-reduce",
        choices=["max", "mean"],
        default="mean",
        help="pool per-gang confusability by 'max' (eq. 40) or 'mean'.",
    )
    ap.add_argument(
        "--bank-conf-delta",
        type=float,
        default=0.0,
        help="delta-leaky confusability cone (0 = hard chi^tau; >0 penalizes "
        "confusers leaking into the one-hop halo).",
    )
    ap.add_argument("--bank-conf-halo-hops", type=int, default=1)
    ap.add_argument(
        "--bank-coarsen-target",
        choices=["bank", "indicators"],
        default="bank",
        help="R handed to the coarsener: 'bank' = full span(Z) (d columns) | "
        "'indicators' = M_tau-projected train-gang indicators (m columns).",
    )
    ap.add_argument(
        "--bank-structural-width",
        type=int,
        default=0,
        help="if >0, concatenate a class-agnostic low-frequency structural channel "
        "g_theta_bar(A_hat) Omega of this width to the coarsening target.",
    )
    ap.add_argument(
        "--bank-random-width",
        type=int,
        default=64,
        help="width of the random structural feature channel for the bank (the whole "
        "X when the dataset is structural-only, else concatenated to the real feats).",
    )
    ap.add_argument(
        "--diagnostics",
        action="store_true",
        default=True,
        help="write per-community diagnostic plots + CSV (Elliptic-style "
        "detected-vs-missed by size/conductance/density/mbar1/overlap, plus the "
        "overlap-structure panel these ground-truth-community datasets need).",
    )
    ap.add_argument("--seed", type=int, default=0)
    out_dir = Path(f"results/graph_fraud_gang_detection/{now}/")
    ap.add_argument("--out", default=out_dir, type=Path)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    SNAP_COMMUNITY = {
        "comamazon": "amazon",
        "comdblp": "dblp",
        "comyoutube": "youtube",
        "comlj": "lj",
        "comorkut": "orkut",
    }
    LOGGER.info(f"=== {args.dataset} gang detection ===")
    all_communities = None
    if args.dataset in SNAP_COMMUNITY:
        A, communities = load_snap_community(
            args.comamazon_dir, SNAP_COMMUNITY[args.dataset]
        )
        X = None
        y = np.zeros(A.shape[0], dtype=np.int64)  # detection uses geometry, not labels
        all_communities = (
            communities  # keep full set so top-5 largest is the *real* top-5
        )
        rng.shuffle(communities)
        gang_sets = sorted(communities[: args.max_gangs], key=len, reverse=True)
        LOGGER.info(
            f"  N={A.shape[0]:,}  undirected_edges={A.nnz // 2:,}  "
            f"feat=None (structural-only)  total communities in file="
            f"{len(communities):,}  evaluating={len(gang_sets)}"
        )
    else:
        A, X, y = load_dataset(args)
        LOGGER.info(
            f"  N={A.shape[0]:,}  undirected_edges={A.nnz // 2:,}  "
            f"feat={'None' if X is None else X.shape[1]}  "
            f"anomalies={int((y == 1).sum()):,} ({(y == 1).mean():.2%})"
        )
        gang_sets = anomaly_gangs(A, y, args.min_gang_size, args.gang_louvain)
    structural_only = X is None
    # the genuine 5 largest communities (from the full set, not just the sample)
    top5_sets = sorted(all_communities or gang_sets, key=len, reverse=True)[:5]

    # Normals = random *connected* sets, size-matched to the gangs (the size-
    # controlled null / contrast).  Drawn from benign nodes when a benign class
    # exists, else from the whole graph (com-Amazon has no anomaly/benign split).
    gsizes = [len(c) for c in gang_sets] or [3]
    n_normal = min(args.max_normal_patterns, max(len(gang_sets), 10))
    sizes_for_normals = np.asarray(gsizes)[rng.integers(0, len(gsizes), size=n_normal)]
    if not structural_only and int((y == 0).sum()) > 0:
        pool = np.where(y == 0)[0]
        sub = A[pool][:, pool].tocsr()
        benign_sets = [
            pool[np.asarray(random_connected_set(sub, int(sz), rng), dtype=np.int64)]
            for sz in sizes_for_normals
        ]
    else:
        benign_sets = [
            np.asarray(random_connected_set(A, int(sz), rng), dtype=np.int64)
            for sz in sizes_for_normals
        ]

    gangs = make_patterns(gang_sets, "alert", "gang", "g")
    normals = make_patterns(benign_sets, "normal", "normal", "n")
    LOGGER.info(
        f"  gangs={len(gangs)}  normals={len(normals)}  "
        f"gang sizes(top): {sorted((p.num_nodes for p in gangs), reverse=True)[:15]}"
    )
    if not gangs:
        LOGGER.info("  no gangs -- stop.")
        return

    gang_tr, _ = split_train_test(gangs, args.train_ratio, rng)
    norm_tr, norm_te = split_train_test(normals, args.train_ratio, rng)
    if not gang_tr:
        gang_tr = gangs
    if not norm_tr:
        norm_tr, norm_te = (
            normals[: max(1, len(normals) // 2)],
            normals[max(1, len(normals) // 2) :],
        )
    # Cap the *fitting* sets so the (N x m) propagation stack stays in memory on the
    # large com-Amazon graph; detection is still evaluated over all gangs.
    gang_tr, norm_tr = gang_tr[: args.max_retain], norm_tr[: args.max_retain]
    retain = gang_tr + norm_tr
    eval_patterns = gangs + norm_te
    LOGGER.info(
        f"  fitting theta on {len(gang_tr)} gangs + {len(norm_tr)} normals; "
        f"evaluating detection over all {len(gangs)} gangs"
    )

    graph = build_graph_obj(A, X, y)
    normalized, adjacency = graph_operators(graph)
    Xf = (
        None
        if graph.x is None
        else graph.x.to(device=normalized.device, dtype=normalized.dtype)
    )
    total_width = args.structural_width + args.embed_dim

    common = dict(
        degree=args.degree, epochs=args.epochs, learning_rate=args.learning_rate
    )

    use_bank = args.use_bank or args.bank_only
    encoders = []

    if not args.bank_only:
        LOGGER.info("Fitting structural encoder …")
        structural_fit = fit_collective_sgc(
            normalized, retain, features=None, mode="lambda_min", **common
        )
        structural_basis = build_sgc_subspace(
            normalized, structural_fit.theta, None, width=total_width, seed=args.seed
        )
        encoders.append(("structural", structural_basis))

    if use_bank:
        LOGGER.info("Fitting collective-bank encoder …")
        bank_basis = fit_bank_basis(graph, normalized, adjacency, Xf, gang_tr, args)
        encoders.append(("collective-bank", bank_basis))

    if not structural_only and not args.bank_only:
        LOGGER.info("Fitting joint encoder (theta, W) …")
        contrastive_negs = None
        if args.joint_contrastive_weight > 0:
            contrastive_negs = [
                create_pattern(
                    f"rneg{i}",
                    [int(v) for v in random_connected_set(A, p.num_nodes, rng)],
                    "random_neg",
                    "normal",
                )
                for i, p in enumerate(gang_tr)
            ]
        joint = fit_joint_encoder(
            normalized,
            retain,
            features=Xf,
            degree=args.degree,
            embed_dim=args.embed_dim,
            structural_width=args.structural_width,
            ridge=args.ridge,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            label_patterns=gang_tr + norm_tr,
            label_weight=1.0,
            contrastive_patterns=contrastive_negs,
            contrastive_weight=args.joint_contrastive_weight,
        )
        joint_basis = build_joint_subspace(
            normalized,
            joint.theta,
            Xf,
            joint.feature_map,
            structural_width=args.structural_width,
            seed=args.seed,
            per_hop=joint.per_hop_features,
        )
        encoders.append(("joint", joint_basis))
    top5_patterns = make_patterns(top5_sets, "alert", "gang", "t")
    eps_desc = "inf" if args.epsilon == float("inf") else f"{args.epsilon:g}"
    LOGGER.info(
        f"Coarsening (method={args.coarsening_method}, epsilon={eps_desc}, "
        f"reduction-cap={args.reduction:.0%}) and scoring community recovery …"
    )
    rows = [
        coarsen_and_metrics(
            name,
            basis,
            adjacency=adjacency,
            labels=graph.y,
            eval_gangs=gangs,
            top5_gangs=top5_patterns,
            args=args,
        )
        for name, basis in encoders
    ]

    out_json = args.out / f"gang_detection_{args.dataset}.json"
    out_json.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "n_nodes": graph.num_nodes,
                "n_edges": int(graph.edge_index.shape[1]),
                "n_gangs": len(gangs),
                "epsilon": args.epsilon,
                "coarsening_method": args.coarsening_method,
                # strip the internal per-community arrays (kept in the diagnostics CSV)
                "encoders": [
                    {k: v for k, v in r.items() if not k.startswith("_")} for r in rows
                ],
            },
            indent=2,
        )
        + "\n"
    )

    # --- per-community diagnostics (reuse each encoder's coarsening; no re-coarsen) --
    if args.diagnostics:
        overlap_src = all_communities if all_communities is not None else gang_sets
        membership = membership_counts(overlap_src, graph.num_nodes)
        for r in rows:
            community_diagnostics(
                r["encoder"],
                r["_per_gang"],
                gang_sets,
                normalized,
                adjacency,
                A,
                membership,
                args,
                args.out,
            )

    # --- full incremental Ward PR-sweep (every metric at every coarsening level) --
    if args.pr_sweep:
        import pandas as pd

        from src.ward_pr_sweep import (
            adaptive_exact_epsilon,
            calibrate_epsilon,
            plot_sweep,
            summarize_sweep,
            sweep_metrics,
            ward_order,
        )

        gsets_idx = [list(map(int, np.asarray(list(s)))) for s in gang_sets]
        lap = "symmetric"
        budget = args.pr_sweep_exact_budget
        eps_budget = None if args.epsilon == float("inf") else args.epsilon
        LOGGER.info("\n" + "=" * 92)
        LOGGER.info(
            "FULL WARD PR-SWEEP  (finest -> 2 clusters; all intermediate metrics)"
        )
        if budget > 0:
            LOGGER.info(
                f"  epsilon axis: EXACT RSA constant, {budget} adaptive samples/encoder"
            )
        for name, basis in encoders:
            children, distances, a0, metric = ward_order(
                adjacency, basis, args.coarsening_tau, laplacian=lap
            )
            traj = sweep_metrics(
                children, distances, graph.num_nodes, gsets_idx, args.threshold
            )
            eps_key, checkpoints = "epsilon", None
            if budget > 0:
                lv, ex = adaptive_exact_epsilon(
                    children, graph.num_nodes, a0, metric, budget=budget
                )
                calibrate_epsilon(traj, lv, ex, graph.num_nodes)
                eps_key = "epsilon_exact"
                checkpoints = (graph.num_nodes - lv, ex)
            summ = summarize_sweep(traj, eps_budget=eps_budget, eps_key=eps_key)
            tag = f"{args.dataset}:{name}"
            pd.DataFrame(traj).to_csv(
                args.out / f"pr_sweep_{args.dataset}_{name}.csv", index=False
            )
            auc, bi = plot_sweep(
                traj,
                tag,
                args.out / f"pr_sweep_{args.dataset}_{name}.png",
                eps_budget=eps_budget,
                eps_key=eps_key,
                exact_checkpoints=checkpoints,
            )
            b = summ["best_f1"]
            LOGGER.info(f"[{name}]  levels={len(traj)}  PR-AUC={auc:.3f}")
            LOGGER.info(
                f"    best-F1 stop: eps={b[eps_key]:.3f}  n_coarse={b['n_coarse']}  "
                f"R={b['mean_recall']:.3f}  P={b['mean_precision']:.3f}  "
                f"F1={b['mean_f1']:.3f}  J={b['mean_jaccard']:.3f}  det={b['det_rate']:.1%}"
            )
            if "at_epsilon" in summ:
                a = summ["at_epsilon"]
                LOGGER.info(
                    f"    at budget eps<={a['epsilon_budget']:g} (eps={a[eps_key]:.3f}, "
                    f"n_coarse={a['n_coarse']}): R={a['mean_recall']:.3f}  "
                    f"P={a['mean_precision']:.3f}  F1={a['mean_f1']:.3f}  "
                    f"det={a['det_rate']:.1%}"
                )
        LOGGER.info(f"PR-sweep CSVs + plots -> {args.out}")

    LOGGER.info("\n" + "=" * 92)
    LOGGER.info(
        f"COMMUNITY RECOVERY — {args.dataset}  (N={graph.num_nodes:,}, "
        f"E={graph.edge_index.shape[1]:,}, epsilon={eps_desc})"
    )
    LOGGER.info("=" * 92)
    hdr = (
        f"{'encoder':<11}{'n_coarse':>10}{'#comm':>7}"
        f"{'det_rate':>9}{'meanF1(all)':>13}{'meanJac(all)':>13}"
        f"{'meanF1(top5)':>13}{'meanJac(top5)':>14}"
        f"{'recall':>8}{'prec':>8}"
    )
    LOGGER.info(hdr)
    LOGGER.info("-" * len(hdr))
    for r in rows:
        LOGGER.info(
            f"{r['encoder']:<11}{r['n_coarse']:>10,}{r['n_gangs']:>7}"
            f"{r['det_rate']:>9.1%}{r['meanF1_all']:>13.3f}{r['meanJac_all']:>13.3f}"
            f"{r.get('meanF1_top5', float('nan')):>13.3f}"
            f"{r.get('meanJac_top5', float('nan')):>14.3f}"
            f"{r['gang_mean_recall']:>8.3f}{r['gang_mean_precision']:>8.3f}"
        )
    LOGGER.info("Per top-5-largest community (sizes -> F1):")
    for r in rows:
        pairs = ", ".join(
            f"{s}->{f:.2f}"
            for s, f in zip(r.get("top5_sizes", []), r.get("top5_F1", []))
        )
        LOGGER.info(f"    {r['encoder']:<11} {pairs}")
    LOGGER.info(
        "Read: meanF1/meanJac are the standard community-recovery metric "
        "(each ground-truth community vs its best-match super-node, averaged). "
        "'all' = the evaluated sample; 'top5' = the 5 largest communities in the "
        "full set (the hard cases the literature's 5-largest protocol uses). "
        "det_rate = fraction with recall>thr AND precision>thr."
    )
    LOGGER.info(f"JSON report: {out_json}")


if __name__ == "__main__":
    main()
