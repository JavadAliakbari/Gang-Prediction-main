r"""Ward vs Smooth Dual Ward on *synthetic* graphs with planted motifs.

Elliptic++ confounds several things at once: an extremely heavy-tailed degree
distribution, gangs of wildly different sizes (2 to 833 nodes), and unknown
ground truth outside the labelled wallets.  This driver reruns the identical
optimal-``epsilon`` comparison of :mod:`src.run_dual_ward_epsilon_sweep` on a
graph where every one of those factors is a knob:

* Erdos-Renyi background (``--avg-degree``) -> a *homogeneous* degree
  distribution, the regime where degree-volume and cardinality weighting agree;
* planted motifs of controlled size (``--motif-size-min/max``) and type
  (``--motif-types``) -> the gangs, with exactly known membership;
* ``--motif-conductance`` -> how strongly each gang is wired into the host.

The point is the **ablation ladder** the sweep reports:

    ward           uniform (cardinality) Ward -- its block-averaging projector
                   matches the one the exact-RSA epsilon and the detection
                   criterion use;
    ward-degree    the same Ward code with degree node weights -- changes ONLY
                   the weighting;
    dual-ward      degree weighting *and* the dual embedding M_tau U_tau *and*
                   the m_tau normalization.

If ``ward-degree`` tracks ``dual-ward`` on Elliptic++ but ``ward`` beats both,
the gap is the degree weighting (a heavy-tail effect) rather than anything about
the dual score.  If instead ``ward-degree`` tracks ``ward``, the dual score
itself is responsible.  On a homogeneous synthetic graph the two weightings
nearly coincide, so the arms should converge -- which is exactly the control.

Run::

    conda activate FedStruct
    python -m src.run_synthetic_dual_ward_sweep --num-nodes 4000 --num-motifs 40
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import Data

from src.loukas_sgc_detection import _degrees, graph_operators
from src.pattern_models import create_pattern, make_patterns
from src.run_collective_bank_detection import _motif_edges, make_negative_sampler
from src.run_dual_ward_epsilon_sweep import add_sweep_args, run_sweep
from src.run_elliptic_gang_detection import build_arg_parser, fit_encoders


def _background_edges(args, rng) -> set:
    """Host-graph edges: homogeneous ``er`` or heavy-tailed ``ba``.

    The background degree distribution is the whole point of this driver.  Smooth
    Dual Ward weights clusters by *degree volume* while the exact-RSA epsilon and
    the Pattern-model detection criterion average *uniformly* over nodes; those
    two agree only when degrees are homogeneous.  ``er`` is that agreeing regime;
    ``ba`` (preferential attachment) reproduces the heavy tail of a real
    transaction graph, where a single hub can carry more volume than hundreds of
    leaves.
    """

    n = int(args.num_nodes)
    edges: set = set()
    if args.background == "er":
        n_edges = int(n * args.avg_degree / 2)
        src = rng.integers(0, n, size=n_edges)
        dst = rng.integers(0, n, size=n_edges)
        for u, v in zip(src.tolist(), dst.tolist()):
            if u != v:
                edges.add((min(u, v), max(u, v)))
        return edges

    # Barabasi-Albert: each new node attaches to ``m`` existing nodes chosen with
    # probability proportional to current degree -> P(k) ~ k^-3.
    m = max(1, int(round(args.avg_degree / 2)))
    targets = list(range(m))
    repeated = list(range(m))  # degree-proportional urn
    for new in range(m, n):
        chosen: set = set()
        while len(chosen) < m:
            chosen.add(int(repeated[rng.integers(0, len(repeated))]))
        for t in chosen:
            edges.add((min(new, t), max(new, t)))
            repeated.append(t)
        repeated.extend([new] * m)
    return edges


def build_motif_graph(args) -> tuple:
    """Background (ER or BA) + disjoint planted motifs of mixed size and type."""

    rng = np.random.default_rng(args.seed)
    n = int(args.num_nodes)
    motif_types = [t.strip() for t in args.motif_types.split(",") if t.strip()]

    sizes = rng.integers(args.motif_size_min, args.motif_size_max + 1, args.num_motifs)
    kinds = [
        motif_types[int(i)] for i in rng.integers(0, len(motif_types), args.num_motifs)
    ]
    if int(sizes.sum()) > n:
        raise ValueError("sum of motif sizes exceeds --num-nodes")
    perm = rng.permutation(n)
    offs = np.concatenate([[0], np.cumsum(sizes)])
    blocks = [perm[offs[m] : offs[m + 1]].tolist() for m in range(args.num_motifs)]

    edges = _background_edges(args, rng)

    y = np.zeros(n, dtype=np.int64)
    patterns = []
    for m, nodes in enumerate(blocks):
        for u, v in _motif_edges(nodes, kinds[m], density=args.motif_density, rng=rng):
            edges.add((min(u, v), max(u, v)))
        y[nodes] = 1
        p = create_pattern(f"{kinds[m]}_{m}", nodes, kinds[m], label="alert")
        p.motif_kind, p.motif_size = kinds[m], len(nodes)
        patterns.append(p)

    # optional conductance tuning: wire each motif to the host until it hits Phi
    if args.motif_conductance is not None and args.motif_conductance >= 0.0:
        motif_of = -np.ones(n, dtype=np.int64)
        for m, nodes in enumerate(blocks):
            motif_of[np.asarray(nodes)] = m
        host = np.nonzero(motif_of < 0)[0]
        cut = np.zeros(args.num_motifs, dtype=np.int64)
        vol = np.zeros(args.num_motifs, dtype=np.int64)
        for u, v in edges:
            mu, mv = motif_of[u], motif_of[v]
            if mu >= 0:
                vol[mu] += 1
            if mv >= 0:
                vol[mv] += 1
            if mu != mv:
                if mu >= 0:
                    cut[mu] += 1
                if mv >= 0:
                    cut[mv] += 1
        for m, nodes in enumerate(blocks):
            k = int(
                round(
                    (args.motif_conductance * vol[m] - cut[m])
                    / (1.0 - args.motif_conductance)
                )
            )
            arr, added, tries = np.asarray(nodes), 0, 0
            while added < k and tries < 20 * abs(k) + 100:
                tries += 1
                e = (int(rng.choice(arr)), int(rng.choice(host)))
                e = (min(e), max(e))
                if e in edges:
                    continue
                edges.add(e)
                added += 1

    arr = np.array(sorted(edges), dtype=np.int64).T
    edge_index = torch.from_numpy(np.concatenate([arr, arr[::-1]], axis=1)).long()
    gen = torch.Generator().manual_seed(args.seed)
    X = torch.randn(n, args.feature_dim, dtype=torch.float64, generator=gen)
    X = (X - X.mean(0, keepdim=True)) / X.std(0, keepdim=True).clamp_min(1e-8)
    graph = Data(x=X, edge_index=edge_index, y=torch.from_numpy(y), num_nodes=n)
    return graph, patterns


def build_synthetic_data(args) -> SimpleNamespace:
    """Synthetic graph + gang/normal patterns in the shape ``fit_encoders`` wants."""

    graph, patterns = build_motif_graph(args)
    graph.edge_weight = torch.ones(graph.edge_index.shape[1], dtype=torch.float64)
    n = int(graph.num_nodes)

    normalized, adjacency = graph_operators(graph)
    cls = np.where(graph.y.numpy() == 1, 1, 2)  # 1 = motif (illicit), 2 = host

    gangs = list(patterns)
    # 'normal' contrast sets: random connected background blobs of matched size,
    # sampled away from every motif -- the synthetic stand-in for licit CCs.
    sampler = make_negative_sampler(
        adjacency.coalesce().indices(),
        n,
        num_sets=args.max_normal_patterns,
        size_min=max(2, args.motif_size_min),
        size_max=max(3, args.motif_size_max),
        avoid=sorted({int(v) for p in gangs for v in p.node_indices}),
        rng=np.random.default_rng(args.seed + 4242),
    )
    normals = make_patterns(sampler(), "normal", "normal", "n")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(gangs))
    cut = max(1, int(round(args.train_ratio * len(gangs))))
    gang_train = [gangs[int(i)] for i in order[:cut]]
    order_n = rng.permutation(len(normals))
    cut_n = max(1, int(round(args.train_ratio * len(normals))))
    normal_train = [normals[int(i)] for i in order_n[:cut_n]]
    normal_test = [normals[int(i)] for i in order_n[cut_n:]]

    sizes = sorted((p.num_nodes for p in gangs), reverse=True)
    print(
        f"=== SYNTHETIC gang detection | background={args.background} N={n:,} "
        f"E={graph.edge_index.shape[1] // 2:,} | {len(gangs)} motifs ==="
    )
    print(f"  motif sizes: {sizes}")
    print(
        f"  retain (train): {len(gang_train)} motifs + {len(normal_train)} normal; "
        f"evaluating on all {len(gangs)} motifs"
    )
    deg = _degrees(adjacency).numpy()
    print(
        f"  degree distribution: mean={deg.mean():.1f} median={np.median(deg):.1f} "
        f"max={deg.max():.0f} (max/mean={deg.max() / max(deg.mean(), 1e-9):.1f}x, "
        f"top-1% share of total volume={np.sort(deg)[::-1][: max(1, n // 100)].sum() / deg.sum():.1%})"
    )

    return SimpleNamespace(
        A_unw=None,
        cls=cls,
        graph=graph,
        normalized=normalized,
        adjacency=adjacency,
        gang_sets=[list(map(int, p.node_indices)) for p in gangs],
        gangs=gangs,
        normals=normals,
        gang_train=gang_train,
        normal_train=normal_train,
        normal_test=normal_test,
        retain=gang_train + normal_train,
        eval_patterns=gangs + normal_test,
    )


def main() -> None:
    ap = build_arg_parser()
    add_sweep_args(ap)
    ap.add_argument("--num-nodes", type=int, default=4000)
    ap.add_argument("--num-motifs", type=int, default=40)
    ap.add_argument("--motif-size-min", type=int, default=5)
    ap.add_argument("--motif-size-max", type=int, default=25)
    ap.add_argument(
        "--background",
        choices=["er", "ba"],
        default="er",
        help="host graph: 'er' = homogeneous Erdos-Renyi (degree weighting and "
        "uniform weighting agree); 'ba' = Barabasi-Albert preferential "
        "attachment, a heavy tail like a real transaction graph (they disagree)",
    )
    ap.add_argument(
        "--motif-types",
        type=str,
        default="clique,cycle,star,random",
        help="comma list drawn from {clique,cycle,star,random}",
    )
    ap.add_argument("--motif-density", type=float, default=0.6)
    ap.add_argument(
        "--motif-conductance",
        type=float,
        default=0.2,
        help=">=0 wires each motif to hosts to hit this Phi (-1 = as planted); "
        "low Phi = a well-separated gang the coarsener can actually isolate",
    )
    ap.add_argument("--avg-degree", type=float, default=6.0)
    ap.add_argument("--feature-dim", type=int, default=32)
    ap.add_argument(
        "--sweep-out",
        default="results/synthetic_dual_ward/epsilon_sweep.json",
        type=Path,
    )
    args = ap.parse_args()

    os.makedirs(args.sweep_out.parent, exist_ok=True)
    torch.manual_seed(args.seed)

    data = build_synthetic_data(args)
    encoders = fit_encoders(args, data)
    run_sweep(
        args,
        data,
        encoders,
        title=(
            f"SYNTHETIC ER+motifs (N={int(data.adjacency.shape[0]):,}, "
            f"{len(data.gangs)} motifs, avg_degree={args.avg_degree:g})"
        ),
    )


if __name__ == "__main__":
    main()
