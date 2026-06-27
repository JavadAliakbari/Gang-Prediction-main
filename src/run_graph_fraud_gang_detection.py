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
        print("  downloading FraudAmazon.zip …")
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
            print(f"  downloading {fname} …")
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
        default="comamazon",
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
        default=500,
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
    # encoder / coarsening (mirror run_elliptic_gang_detection)
    ap.add_argument("--degree", type=int, default=16)
    ap.add_argument("--structural-width", type=int, default=32)
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--learning-rate", type=float, default=0.01)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--reduction", type=float, default=0.99)
    ap.add_argument("--epsilon", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--max-levels", type=int, default=5)
    ap.add_argument("--max-contraction-size", type=int, default=4)
    ap.add_argument("--linkage-max-size", type=int, default=8)
    ap.add_argument("--epsilon-ramp", action="store_true", default=False)
    ap.add_argument(
        "--coarsening-method", choices=["edges", "capped", "linkage"], default="edges"
    )
    ap.add_argument("--ceiling-max-levels", type=int, default=40)
    ap.add_argument(
        "--detection-ceiling",
        action="store_true",
        default=False,
        help="also report each subspace's best detection rate over an "
        "epsilon sweep (slow on dense graphs)",
    )
    ap.add_argument("--joint-contrastive-weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/graph_fraud_gang_detection", type=Path)
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
    print(f"=== {args.dataset} gang detection ===")
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
        print(
            f"  N={A.shape[0]:,}  undirected_edges={A.nnz // 2:,}  "
            f"feat=None (structural-only)  total communities in file="
            f"{len(communities):,}  evaluating={len(gang_sets)}"
        )
    else:
        A, X, y = load_dataset(args)
        print(
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
    print(
        f"  gangs={len(gangs)}  normals={len(normals)}  "
        f"gang sizes(top): {sorted((p.num_nodes for p in gangs), reverse=True)[:15]}"
    )
    if not gangs:
        print("  no gangs -- stop.")
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
    print(
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

    print("\n  Fitting structural encoder …")
    structural_fit = fit_collective_sgc(
        normalized, retain, features=None, mode="lambda_min", **common
    )
    structural_basis = build_sgc_subspace(
        normalized, structural_fit.theta, None, width=total_width, seed=args.seed
    )
    encoders = [("structural", structural_basis)]

    if not structural_only:
        print("  Fitting joint encoder (theta, W) …")
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
    print(
        f"\n  Coarsening (method={args.coarsening_method}, epsilon={eps_desc}, "
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
                "encoders": rows,
            },
            indent=2,
        )
        + "\n"
    )

    print("\n" + "=" * 92)
    print(
        f"COMMUNITY RECOVERY — {args.dataset}  (N={graph.num_nodes:,}, "
        f"E={graph.edge_index.shape[1]:,}, epsilon={eps_desc})"
    )
    print("=" * 92)
    hdr = (
        f"{'encoder':<11}{'n_coarse':>10}{'#comm':>7}"
        f"{'det_rate':>9}{'meanF1(all)':>13}{'meanJac(all)':>13}"
        f"{'meanF1(top5)':>13}{'meanJac(top5)':>14}"
        f"{'recall':>8}{'prec':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['encoder']:<11}{r['n_coarse']:>10,}{r['n_gangs']:>7}"
            f"{r['det_rate']:>9.1%}{r['meanF1_all']:>13.3f}{r['meanJac_all']:>13.3f}"
            f"{r.get('meanF1_top5', float('nan')):>13.3f}"
            f"{r.get('meanJac_top5', float('nan')):>14.3f}"
            f"{r['gang_mean_recall']:>8.3f}{r['gang_mean_precision']:>8.3f}"
        )
    print("\n  Per top-5-largest community (sizes -> F1):")
    for r in rows:
        pairs = ", ".join(
            f"{s}->{f:.2f}"
            for s, f in zip(r.get("top5_sizes", []), r.get("top5_F1", []))
        )
        print(f"    {r['encoder']:<11} {pairs}")
    print(
        "\n  Read: meanF1/meanJac are the standard community-recovery metric "
        "(each ground-truth community vs its best-match super-node, averaged). "
        "'all' = the evaluated sample; 'top5' = the 5 largest communities in the "
        "full set (the hard cases the literature's 5-largest protocol uses). "
        "det_rate = fraction with recall>thr AND precision>thr."
    )
    print(f"\nJSON report: {out_json}")


if __name__ == "__main__":
    main()
