"""Does a wider target actually buy a better *margin*?  C, chi and D per target.

Capture ``C`` is monotone in the target: adding columns can only raise it.  So a
multi-head span *must* capture more than a single-head span -- yet it does not
detect proportionally better.  The theory says why: detection is governed by the
**margin** ``D = C - chi`` (Definition 4.6 / Theorem 4.10), and a wider target
also retains more of each gang's internal *fluctuation* space, raising ``chi``.
A target that grows both terms equally buys nothing, and pays twice: once in
confusability, once in RSA budget (more columns to preserve -> the coarsening
must stop earlier at a fixed epsilon, Prop 6.14).

For each target this script computes, per gang and exactly (no training):

* ``C_S``   -- retained M_tau-energy of the gang indicator (rank-revealing);
* ``chi_S`` -- the hard (delta=0) confusability: the largest M_tau-principal
  cosine between the target and the fluctuation space
  ``F_S = {w : supp(w) subset S, <w, v_S>_2 = 0}``, as a generalized eigenproblem
  of size ``s-1``;
* ``D_S = C_S - chi_S`` -- the distinguishability margin;
* the *width* of the target and the Ward stop it forces at a fixed epsilon.

Run::

    conda activate FedStruct
    python -m src.run_margin_vs_width --day-start 25 --day-end 25
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import torch

from src.collective_detector import CollectiveBankDetector, DetectorConfig, GraphData
from src.run_capture_ceiling import subspace_captures
from src.run_collective_bank_detection import _l_apply, _m_apply, degree_weighted_indicators
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import (
    build_torch_graph,
    load_node_features,
    make_patterns,
    random_structural_features,
    split_train_test,
)
from src.utils.utils import LOGGER, now


def m_orthonormal(a_hat: torch.Tensor, R: torch.Tensor, tau: float) -> torch.Tensor:
    """``M_tau``-orthonormal basis of ``span(R)`` (rank-revealing)."""

    MR = _m_apply(a_hat, R, tau)
    G = 0.5 * (R.T @ MR + (R.T @ MR).T)
    evals, evecs = torch.linalg.eigh(G)
    keep = evals > evals.max().clamp_min(1e-300) * 1e-10
    return R @ (evecs[:, keep] / evals[keep].sqrt().unsqueeze(0))


def gang_confusability(
    a_hat: torch.Tensor, Q: torch.Tensor, nodes: list, v_S: torch.Tensor, tau: float
) -> float:
    """Hard confusability ``chi_S`` of the target with ``M_tau``-orthonormal basis ``Q``.

    ``chi_S = max_{w in F_S} ||Pi_R w||^2_M / ||w||^2_M`` with ``F_S`` the
    fluctuations supported on ``S`` and ``l2``-orthogonal to ``v_S`` -- a
    generalized symmetric eigenproblem of size ``s-1``.
    """

    idx = torch.as_tensor(sorted(int(v) for v in nodes), dtype=torch.long)
    s = len(idx)
    if s < 2:
        return 0.0
    # basis of {z in R^s : <z, v_S|_S> = 0} via a Householder reflection
    v = v_S[idx]
    nrm = v.norm().clamp_min(torch.finfo(v.dtype).eps)
    e = torch.zeros_like(v)
    e[0] = nrm
    u = v - e
    if u.norm() < 1e-14:
        B_s = torch.eye(s, dtype=v.dtype)[:, 1:]
    else:
        u = u / u.norm()
        H = torch.eye(s, dtype=v.dtype) - 2.0 * torch.outer(u, u)
        B_s = H[:, 1:]  # columns span the orthogonal complement of v within S
    B = torch.zeros(a_hat.shape[0], s - 1, dtype=v.dtype)
    B[idx] = B_s

    MB = _m_apply(a_hat, B, tau)
    G_B = 0.5 * (B.T @ MB + (B.T @ MB).T)  # ||w||^2_M
    P = Q.T @ MB  # (q, s-1): M-inner products with the orthonormal target
    A = P.T @ P  # ||Pi_R w||^2_M
    A = 0.5 * (A + A.T)
    jitter = 1e-12 * torch.diag(G_B).mean().clamp_min(1e-300)
    try:
        L = torch.linalg.cholesky(
            G_B + jitter * torch.eye(s - 1, dtype=G_B.dtype)
        )
        C = torch.cholesky_solve(A, L)
        lam = torch.linalg.eigvals(C).real.max()
    except Exception:
        lam = torch.linalg.eigvalsh(A)[-1] / torch.linalg.eigvalsh(G_B)[-1].clamp_min(1e-30)
    return float(torch.clamp(lam, 0.0, 1.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--day-start", type=int, default=25)
    ap.add_argument("--day-end", type=int, default=25)
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--train-ratio", type=float, default=0.6)
    ap.add_argument("--feature-mode", default="wallet+random",
                    choices=["wallet", "random", "wallet+random"])
    ap.add_argument("--random-width", type=int, default=64)
    ap.add_argument("--degree", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--epsilon", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=Path(f"results/margin_vs_width/{now}/"), type=Path)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    LOGGER.info(f"=== margin vs target width | day {args.day_start}-{args.day_end} ===")
    A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, args.day_start, args.day_end)
    if args.feature_mode in ("wallet", "wallet+random"):
        X = load_node_features(args.data_dir, nodes_df, args.day_start, args.day_end)
        if args.feature_mode == "wallet+random":
            X = torch.cat([X, random_structural_features(
                int(A_unw.shape[0]), args.random_width, args.seed)], dim=1)
    else:
        X = random_structural_features(int(A_unw.shape[0]), args.random_width, args.seed)
    graph = build_torch_graph(A_w, A_unw, cls, X, weighted=False)
    data = GraphData.from_graph(graph)
    gang_sets = connected_components_sets(
        A_unw, np.where(cls == 1)[0], args.min_gang_size
    )
    gangs = make_patterns(gang_sets, "alert", "gang", "g")
    rng = np.random.default_rng(args.seed)
    gang_train, _ = split_train_test(gangs, args.train_ratio, rng)
    train_ids = {p.id for p in gang_train}
    LOGGER.info(f"  {len(gangs)} gangs ({len(gang_train)} train), d={data.feature_dim}")

    V = degree_weighted_indicators(data.adjacency, gangs).to(data.a_hat.dtype)
    phi = (V * _l_apply(data.a_hat, V)).sum(0).clamp_min(1e-300)

    base = DetectorConfig(
        degree=args.degree, basis="chebyshev", tau=args.tau, epochs=args.epochs,
        learning_rate=0.02, ridge=1e-3, optimizer="riemannian",
        capture_objective="lambda_min", conf_weight=10.0, conf_reduce="mean",
        heads=args.heads, epsilon=args.epsilon, seed=args.seed,
    )
    from dataclasses import replace

    det = CollectiveBankDetector(base).fit(data, gang_train)

    targets = {}
    for name in ("bank", "multihead-warm", "multihead"):
        det.config = replace(base, coarsen_target=name)
        det.heads_state_ = None  # rebuild heads per target
        targets[name] = det.target_subspace(data, gang_train)
        LOGGER.info(f"  target [{name}]: {targets[name].shape[1]} columns")

    rows = []
    for name, R in targets.items():
        cap, q = subspace_captures(data.a_hat, R, V, phi, args.tau)
        Q = m_orthonormal(data.a_hat, R, args.tau)
        LOGGER.info(f"  [{name}] computing chi for {len(gangs)} gangs "
                    f"(rank {Q.shape[1]}) ...")
        for gi, p in enumerate(gangs):
            chi = gang_confusability(
                data.a_hat, Q, list(p.node_indices), V[:, gi], args.tau
            )
            rows.append({
                "target": name, "width": int(R.shape[1]), "rank": int(Q.shape[1]),
                "gang": p.id, "size": p.num_nodes, "train": p.id in train_ids,
                "C": float(cap[gi]), "chi": chi, "D": float(cap[gi]) - chi,
            })
        # what the fixed epsilon buys with this width
        det.config = replace(base, coarsen_target=name, ward_stop="epsilon")
        det.heads_state_ = None
        co, _ = det.coarsen(data, R, gang_train)
        for r in rows:
            if r["target"] == name:
                r["n_coarse"] = int(co.n_coarse)
                r["epsilon"] = float(co.epsilon)
        LOGGER.info(f"    -> ward stop at n_coarse={co.n_coarse:,} "
                    f"(eps={co.epsilon:.3f}) for eps budget {args.epsilon}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "margin_vs_width.csv", index=False)

    LOGGER.info("\n" + "=" * 92)
    LOGGER.info("CAPTURE vs CONFUSABILITY vs MARGIN, by target width")
    LOGGER.info("=" * 92)
    hdr = (f"  {'target':<16}{'cols':>6}{'rank':>7}{'median C':>11}{'median chi':>12}"
           f"{'median D':>11}{'n_coarse':>10}")
    LOGGER.info(hdr)
    LOGGER.info("  " + "-" * (len(hdr) - 2))
    summary = {}
    for name in targets:
        s = df[df.target == name]
        summary[name] = {
            "width": int(s.width.iloc[0]), "rank": int(s["rank"].iloc[0]),
            "median_C": float(s.C.median()), "median_chi": float(s.chi.median()),
            "median_D": float(s.D.median()), "n_coarse": int(s.n_coarse.iloc[0]),
        }
        LOGGER.info(
            f"  {name:<16}{s.width.iloc[0]:>6}{s['rank'].iloc[0]:>7}"
            f"{s.C.median():>11.4f}{s.chi.median():>12.4f}{s.D.median():>11.4f}"
            f"{s.n_coarse.iloc[0]:>10,}"
        )
    LOGGER.info("\n  per-gang detail (gangs with >= 10 nodes):")
    big = df[df["size"] >= 10].pivot_table(
        index=["gang", "size"], columns="target", values=["C", "chi", "D"]
    )
    for line in big.round(4).to_string().splitlines():
        LOGGER.info("    " + line)

    (args.out / "margin_vs_width.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info(f"\nCSV + JSON -> {args.out}")


if __name__ == "__main__":
    main()
