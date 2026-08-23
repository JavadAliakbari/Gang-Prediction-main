r"""Ward vs Smooth Dual Ward at each coarsener's *own* optimal RSA level.

The headline table in :mod:`src.run_elliptic_gang_detection` fixes one operating
point (``--reduction 0.8``) and compares coarseners there.  That is not a fair
verdict: each scoring rule has its own sweet spot on the granularity axis, and a
rule that needs coarser (or finer) super-nodes to isolate a gang is penalised for
being measured at somebody else's level.

This script removes that confound.  Every coarsener builds its **full merge
hierarchy once**, then the same tree is re-cut at a grid of cluster counts.  For
each cut we record

* ``epsilon`` -- the *exact* restricted-spectral-approximation constant of that
  partition (:func:`src.loukas_sgc_detection._exact_rsa_epsilon`), i.e. the real
  worst-case distortion of the target subspace, not a per-level bound;
* the Pattern-model detection rate over all gangs, plus mean recall/precision;
* the licit-component false-positive count and the largest super-node.

We then report each coarsener at ``epsilon*`` -- the cut maximising the gang
detection rate.  Because the hierarchy is built once and only re-cut, the whole
curve costs barely more than a single coarsening.

The ``max_cluster`` column is the direct answer to "does Ward chain too?".  Both
coarseners here run through the *same* heap-based agglomeration skeleton, so any
difference in super-node size distribution comes from the merge score alone.

Run::

    conda activate FedStruct
    python -m src.run_dual_ward_epsilon_sweep --dual-ward-max-size 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from src.loukas_sgc_detection import (
    _degrees,
    _exact_rsa_epsilon,
    _l_orthonormalize,
    _laplacian,
    custom_ward,
    evaluate_loukas_patterns,
)
from src.run_elliptic_gang_detection import (
    build_arg_parser,
    fit_encoders,
    prepare_dataset,
)
from src.minimax_coarsen import minimax_coarsen
from src.smooth_dual_ward import (
    exact_rsa_epsilon,
    m_orthonormal_basis,
    screened_operators,
    smooth_dual_ward,
)


def cut_grid(n: int, reductions) -> list:
    """Cluster counts to evaluate, descending (finest first)."""

    ks = sorted({max(1, int(round((1.0 - r) * n))) for r in reductions}, reverse=True)
    return ks


def score_cut(
    labels,
    *,
    a0,
    l0,
    rsa,
    eval_patterns,
    node_labels,
    threshold,
):
    r"""RSA constants + Pattern-model detection for one partition.

    Two distortion numbers are recorded, because they are *not* the same object:

    ``epsilon`` -- the screened, degree-weighted constant the paper defines,
    ``sqrt(lambda_max(U_t^T (I - Pi_P) M_tau (I - Pi_P) U_t))`` with ``Pi_P`` the
    degree-weighted block-averaging projector (:func:`exact_rsa_epsilon`).  This
    is the number every arm must be judged by: it is fixed by ``M_tau`` alone and
    is therefore identical across coarseners, so it is a fair common axis.

    ``epsilon_unif`` -- the legacy uniform/combinatorial constant
    (:func:`~src.loukas_sgc_detection._exact_rsa_epsilon`), kept only for
    continuity with earlier runs.  It uses a *different* metric and a *different*
    projector, so mixing the two across arms would confound the comparison.
    """

    groups = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    _, groups = torch.unique(groups, sorted=True, return_inverse=True)
    lab = groups.numpy()
    U_tau, M_tau, d_tilde = rsa
    epsilon = exact_rsa_epsilon(U_tau, M_tau, d_tilde, lab)
    epsilon_unif = _exact_rsa_epsilon(a0, l0, groups)
    _, by_label = evaluate_loukas_patterns(
        eval_patterns, groups, node_labels, threshold=threshold
    )
    gang = by_label.get("alert", {})
    normal = by_label.get("normal", {})
    sizes = np.bincount(lab)
    detected = int(gang.get("detected", 0))
    total = max(int(gang.get("total", 0)), 1)
    fp = int(normal.get("detected", 0))
    n_normal = max(int(normal.get("total", 0)), 1)
    tpr, fpr = detected / total, fp / n_normal
    return {
        "n_coarse": int(groups.max().item()) + 1,
        "epsilon": float(epsilon),
        "epsilon_unif": float(epsilon_unif),
        "gangs_detected": detected,
        "gangs_total": int(gang.get("total", 0)),
        "detection_rate": float(gang.get("detection_rate", 0.0)),
        "mean_recall": float(gang.get("mean_recall", 0.0)),
        "mean_precision": float(gang.get("mean_precision", 0.0)),
        "normal_detected": fp,
        "normal_total": int(normal.get("total", 0)),
        # selectivity: collapsing gangs is only useful if licit controls survive
        "youden_j": float(tpr - fpr),
        "max_cluster": int(sizes.max()),
    }


def run_sweep(args, data, encoders, *, title: str) -> list:
    r"""Build every coarsener's tree once, re-cut it, and report each optimum.

    Coarsener arms:

    * ``ward`` -- uniform (cardinality) Ward on the ``L``-orthonormal embedding.
      This is the arm whose implied block-averaging projector *matches* the one
      :func:`~src.loukas_sgc_detection._exact_rsa_epsilon` and the Pattern-model
      detection criterion use.
    * ``ward-degree`` -- the *same* Ward code with ``node_weights = deg``.  It
      isolates the effect of **degree weighting alone**: it shares Ward's
      embedding and score shape but adopts Smooth Dual Ward's volume weighting.
    * ``dual-ward a=..`` -- Smooth Dual Ward, which changes *both* the weighting
      (degree volumes) and the geometry (dual embedding ``M_tau U_tau`` plus the
      ``m_tau`` normalization).

    Comparing ``ward`` -> ``ward-degree`` -> ``dual-ward`` therefore attributes
    any gap to the weighting or to the dual score, rather than confounding them.
    """

    adjacency = data.adjacency
    n = int(adjacency.shape[0])
    node_labels = data.graph.y
    eval_patterns = data.eval_patterns

    # scipy view of the same adjacency the torch coarseners use
    idx = adjacency.indices().cpu().numpy()
    val = adjacency.values().cpu().numpy()
    W = sp.coo_matrix((val, (idx[0], idx[1])), shape=(n, n)).tocsr()

    l0 = _laplacian(adjacency)
    degrees = _degrees(adjacency)
    reductions = [float(x) for x in args.sweep_reductions.split(",") if x.strip()]
    ks = cut_grid(n, reductions)
    dual_alphas = [
        float(a) for a in str(args.dual_ward_alphas).split(",") if a.strip() != ""
    ]
    embeddings = [
        e.strip() for e in str(args.dual_ward_embeddings).split(",") if e.strip()
    ]
    dual_taus = [
        float(t) for t in str(args.dual_ward_taus).split(",") if t.strip() != ""
    ] or [args.dual_ward_tau]
    cap = int(args.dual_ward_max_size)
    rsa_tau = float(args.rsa_tau if args.rsa_tau > 0 else args.dual_ward_tau)

    print(
        f"\n  Sweep: {len(ks)} cuts per tree {ks}\n"
        f"  dual-ward embeddings={embeddings} taus={dual_taus} "
        f"max_cluster_size={'uncapped' if cap == 0 else cap}\n"
        f"  RSA axis: screened degree-weighted epsilon at tau={rsa_tau:g} "
        "(identical for every arm)"
    )

    # One screened metric, fixed for the whole sweep, so 'eps*' is a common axis.
    _, d_tilde, _, M_tau = screened_operators(W, rsa_tau)

    rows = []
    for name, basis in encoders:
        a0 = _l_orthonormalize(basis, l0)
        Z = basis.detach().cpu().to(torch.float64).numpy()
        U_tau, _ = m_orthonormal_basis(Z, M_tau)
        rsa = (U_tau, M_tau, d_tilde)

        trees = []
        t0 = time.time()
        ward = custom_ward(a0, adjacency, full_tree=True)
        print(f"    {name:<16} ward tree built in {time.time() - t0:.0f}s", flush=True)
        trees.append(("ward", lambda k, _t=ward: _t.labels_at(k).numpy()))

        if args.ward_degree_arm:
            t0 = time.time()
            wardd = custom_ward(a0, adjacency, full_tree=True, node_weights=degrees)
            print(
                f"    {name:<16} ward-degree tree built in {time.time() - t0:.0f}s",
                flush=True,
            )
            trees.append(("ward-degree", lambda k, _t=wardd: _t.labels_at(k).numpy()))

        if args.minimax_arm:
            t0 = time.time()
            mm = minimax_coarsen(
                W,
                Z,
                tau=rsa_tau,
                n_clusters=min(ks),
                score_mode=args.minimax_score_mode,
                build_full_tree=True,
                max_rescore=args.minimax_rescore,
            )
            print(
                f"    {name:<16} minimax tree built in {time.time() - t0:.0f}s",
                flush=True,
            )
            trees.append(("minimax", lambda k, _t=mm: _t.labels_at(k)))

        for alpha in dual_alphas:
            for embedding in embeddings:
                for dtau in dual_taus:
                    t0 = time.time()
                    tree = smooth_dual_ward(
                        W,
                        Z,
                        tau=dtau,
                        alpha=alpha,
                        build_full_tree=True,
                        max_cluster_size=cap,
                        embedding=embedding,
                    )
                    tag = f"dw-{embedding[:4]} a={alpha:g}"
                    if len(dual_taus) > 1:
                        tag += f" t={dtau:g}"
                    print(
                        f"    {name:<16} {tag} tree built in "
                        f"{time.time() - t0:.0f}s",
                        flush=True,
                    )
                    trees.append((tag, lambda k, _t=tree: _t.labels_at(k)))

        for variant, cut in trees:
            for k in ks:
                row = score_cut(
                    cut(k),
                    a0=a0,
                    l0=l0,
                    rsa=rsa,
                    eval_patterns=eval_patterns,
                    node_labels=node_labels,
                    threshold=args.threshold,
                )
                row.update({"encoder": name, "variant": variant, "k_requested": k})
                rows.append(row)
            best = max(
                (r for r in rows if r["encoder"] == name and r["variant"] == variant),
                key=lambda r: (r["detection_rate"], -r["epsilon"]),
            )
            print(
                f"      {variant:<16} best {best['gangs_detected']:>3}/"
                f"{best['gangs_total']:<3} at eps*={best['epsilon']:.3f} "
                f"(n_coarse={best['n_coarse']:,})",
                flush=True,
            )

    args.sweep_out.write_text(json.dumps(rows, indent=2) + "\n")
    report_sweep(rows, encoders, n=n, title=title, out=args.sweep_out)
    return rows


def report_sweep(rows, encoders, *, n: int, title: str, out) -> None:
    """Print the per-optimum table and the full detection-vs-epsilon curves."""

    print("\n" + "=" * 104)
    print(
        f"OPTIMAL-EPSILON COMPARISON — {title}; each row is the cut maximising "
        "detection"
    )
    print("=" * 104)
    hdr = (
        f"{'encoder':<16} {'coarsener':<18} {'eps*':>7} {'n_coarse':>9} "
        f"{'detected':>10} {'det_rate':>9} {'recall':>8} {'prec':>7} "
        f"{'norm_FP':>8} {'J':>7} {'max_cl':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    last = None
    for name, _ in encoders:
        if last is not None:
            print("-" * len(hdr))
        last = name
        variants = []
        for r in rows:
            if r["encoder"] == name and r["variant"] not in variants:
                variants.append(r["variant"])
        for variant in variants:
            best = max(
                (r for r in rows if r["encoder"] == name and r["variant"] == variant),
                key=lambda r: (r["detection_rate"], -r["epsilon"]),
            )
            print(
                f"{name:<16} {variant:<18} {best['epsilon']:>7.3f} "
                f"{best['n_coarse']:>9,} "
                f"{best['gangs_detected']:>4}/{best['gangs_total']:<5} "
                f"{best['detection_rate']:>8.1%} {best['mean_recall']:>8.3f} "
                f"{best['mean_precision']:>7.3f} "
                f"{best['normal_detected']:>3}/{best['normal_total']:<4} "
                f"{best['youden_j']:>7.3f} "
                f"{best['max_cluster']:>7,}"
            )

    print("\n" + "=" * 104)
    print("FULL CURVES — detection rate vs exact RSA epsilon at every evaluated cut")
    print("=" * 104)
    for name, _ in encoders:
        print(f"\n  encoder: {name}")
        variants = []
        for r in rows:
            if r["encoder"] == name and r["variant"] not in variants:
                variants.append(r["variant"])
        for variant in variants:
            curve = [
                r for r in rows if r["encoder"] == name and r["variant"] == variant
            ]
            cells = " ".join(
                f"{r['n_coarse'] / n:.2f}/{r['epsilon']:.2f}:{r['detection_rate']:.0%}"
                for r in curve
            )
            print(f"    {variant:<16} {cells}")
    print(
        "\n  cell format: (n_coarse/N)/epsilon:detection_rate — smaller n_coarse/N "
        "is coarser; epsilon is the exact RSA distortion of that partition."
    )
    print(
        "  'J' = detection_rate - normal_FP_rate (Youden): a gang collapsing into "
        "one super-node only counts if licit controls do NOT, so detection alone "
        "overstates a coarsener that simply merges everything."
    )
    print(f"\nJSON: {out}")


def add_sweep_args(ap):
    """Sweep-specific CLI flags, shared with the synthetic driver."""

    ap.add_argument(
        "--sweep-reductions",
        type=str,
        default="0.3,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95",
        help="comma-separated reduction levels (1 - n_coarse/N) to evaluate; "
        "each is one horizontal cut of the *same* merge tree",
    )
    ap.add_argument(
        "--ward-degree-arm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include degree-weighted Ward, the ablation that isolates Smooth "
        "Dual Ward's volume weighting from its dual score",
    )
    ap.add_argument(
        "--dual-ward-embeddings",
        type=str,
        default="dual",
        help="comma list from {dual,primal}. 'dual' scores merges against "
        "M_tau U_tau (the spec); 'primal' scores against U_tau, removing the "
        "M_tau high-pass so the telescoped objective matches the RSA constant "
        "that is actually evaluated",
    )
    ap.add_argument(
        "--dual-ward-taus",
        type=str,
        default="",
        help="optional comma list of dual-metric tau values to sweep (empty = "
        "just --dual-ward-tau). Larger tau makes M_tau -> tau I, so the dual "
        "embedding continuously approaches the primal one; the relative "
        "high-pass strength is bounded by sqrt((lambda_max + tau)/tau)",
    )
    ap.add_argument(
        "--rsa-tau",
        type=float,
        default=0.0,
        help="tau of the screened metric used to REPORT epsilon (0 = use "
        "--dual-ward-tau). Held fixed across every arm so eps* is a common axis, "
        "independent of the tau a given dual-ward tree was built with",
    )
    ap.add_argument(
        "--minimax-arm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include the minimax coarsener (:mod:`src.minimax_coarsen`), which "
        "greedily minimizes lambda_max(H_P) -- the RSA constant itself -- instead "
        "of a Frobenius surrogate",
    )
    ap.add_argument(
        "--minimax-score-mode",
        choices=["exact", "ritz", "rayleigh"],
        default="exact",
        help="candidate scoring for the minimax arm. The cheap 'ritz'/'rayleigh' "
        "lower bounds degenerate into one giant block and must not drive the "
        "decision (see the module docstring)",
    )
    ap.add_argument("--minimax-rescore", type=int, default=8)
    return ap


def main() -> None:
    ap = build_arg_parser()
    add_sweep_args(ap)
    ap.add_argument(
        "--sweep-out",
        default="results/elliptic_gang_detection/epsilon_sweep.json",
        type=Path,
    )
    args = ap.parse_args()

    os.makedirs(args.sweep_out.parent, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    data = prepare_dataset(args, rng)
    encoders = fit_encoders(args, data)
    run_sweep(
        args,
        data,
        encoders,
        title=(
            f"Elliptic++ days {args.day_start}-{args.day_end} "
            f"(N={int(data.adjacency.shape[0]):,}, {len(data.gangs)} gangs)"
        ),
    )


if __name__ == "__main__":
    main()
