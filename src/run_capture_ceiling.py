"""Per-gang capture *ceiling* analysis on Elliptic++ (Theorem 6.5 reachability).

For each gang S the trained bank's capture is bounded by the M_tau-projection of
the normalized indicator v_hat_S onto the FULL Chebyshev dictionary
``col T = span{T_k(A_hat) x_a : k <= K, a <= d}`` -- by Theorem 6.2 this is the
best capture ANY filter Theta of degree K can attain for S with these features
(the reachability ceiling of Theorem 6.5).  If the ceiling itself is tiny, no
objective/optimizer/epoch budget can raise capture: the features simply do not
reach the gang's energetic bands, and the fix is feature width, not training.

The script computes, per gang on the training-day graph:

* ceiling for the wallet features at several degrees K (does propagation help?);
* ceiling for random structural range-finders of several widths (how much does
  isotropic width buy?), plus wallet+random concatenation;
* a tau sensitivity row (screening moves the energy mean m_tilde_1 toward Phi);
* the label-free eigenspace span(U_K) capture for reference;
* the chance floor  q/N  (expected capture of a *random* q-dim subspace), so a
  ceiling is judged against its own dimension.

Optionally merges the *achieved* trained-bank captures + detection outcomes from
an existing run's ``per_gang_diagnostics.csv`` for ceiling-vs-achieved and
detected-vs-missed contrasts.

Run::

    conda activate FedStruct
    python -m src.run_capture_ceiling --day-start 25 --day-end 25 \
        --diagnostics results/elliptic_modular/<stamp>/per_gang_diagnostics.csv
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

from src.loukas_sgc_detection import graph_operators
from src.run_collective_bank_detection import (
    _basis_stack,
    _m_apply,
    degree_weighted_indicators,
    _l_apply,
    random_structural_features,
)
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import (
    build_torch_graph,
    load_node_features,
    make_patterns,
)
from src.utils.utils import LOGGER, now


# --------------------------------------------------------------------------- #
# core: capture of v_hat_S onto span(B) in the M_tau inner product
# --------------------------------------------------------------------------- #
def subspace_captures(
    a_hat: torch.Tensor, B: torch.Tensor, V: torch.Tensor, phi: torch.Tensor,
    tau: float, rel_cutoff: float = 1e-10,
) -> tuple[np.ndarray, int]:
    """``C_j = ||Pi^{M_tau}_span(B) v_hat_j||^2_{M_tau}`` for every gang column of V.

    Rank-revealing: the Gram is eigendecomposed and eigenvalues below
    ``rel_cutoff * lambda_max`` are dropped, so linearly dependent dictionary
    columns (saturated Krylov spaces) do not corrupt the projection.  Returns the
    per-gang captures and the effective dimension q = rank(span(B)).
    """

    MB = _m_apply(a_hat, B, tau)  # (N, q)
    G = (B.T @ MB).cpu()  # Gram in M_tau
    G = 0.5 * (G + G.T)
    m_vhat = (_l_apply(a_hat, V) + tau * V) / (phi + tau).sqrt().unsqueeze(0)
    b = (B.T @ m_vhat).cpu()  # (q, m)
    evals, evecs = torch.linalg.eigh(G)
    keep = evals > float(evals.max()) * rel_cutoff
    q_eff = int(keep.sum())
    coeff = evecs[:, keep].T @ b  # (q_eff, m)
    cap = (coeff**2 / evals[keep].unsqueeze(1)).sum(0)
    return cap.clamp(0.0, 1.0).numpy(), q_eff


def eigenspace_captures(
    a_hat: torch.Tensor, V: torch.Tensor, phi: torch.Tensor, tau: float, K: int
) -> tuple[np.ndarray, int] | None:
    """Capture of span(U_K) (smallest-K Laplacian eigenvectors), for reference."""

    try:
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh

        A = a_hat.coalesce()
        idx, val = A.indices().numpy(), A.values().numpy()
        N = A.shape[0]
        A_sp = sp.csr_matrix((val, (idx[0], idx[1])), shape=(N, N))
        L = sp.eye(N, format="csr") - A_sp
        evals, U = eigsh(L, k=K, sigma=-1e-2, which="LM")  # smallest of L
        UK = torch.as_tensor(U, dtype=V.dtype)
        return subspace_captures(a_hat, UK, V, phi, tau)
    except Exception as e:  # eigensolver failures are non-fatal (reference only)
        LOGGER.info(f"  span(U_K) reference skipped: {e}")
        return None


def summarize(name: str, cap: np.ndarray, q: int, N: int, detected=None) -> dict:
    row = {
        "config": name,
        "q_dim": q,
        "chance_q_over_N": q / N,
        "median": float(np.median(cap)),
        "mean": float(np.mean(cap)),
        "min": float(np.min(cap)),
        "max": float(np.max(cap)),
        "median_over_chance": float(np.median(cap) / (q / N)),
    }
    if detected is not None and len(detected) == len(cap):
        det = np.asarray(detected, dtype=bool)
        if det.any():
            row["median_detected"] = float(np.median(cap[det]))
        if (~det).any():
            row["median_missed"] = float(np.median(cap[~det]))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--day-start", type=int, default=25)
    ap.add_argument("--day-end", type=int, default=25)
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--diagnostics", type=Path, default=None,
        help="per_gang_diagnostics.csv of a pipeline run (achieved capture + detected)",
    )
    ap.add_argument("--out", default=Path(f"results/capture_ceiling/{now}/"), type=Path)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # --- graph, gangs, indicators -------------------------------------------
    LOGGER.info(f"=== capture ceiling | days {args.day_start}-{args.day_end} "
                f"tau={args.tau} ===")
    A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, args.day_start, args.day_end)
    X_wallet = load_node_features(args.data_dir, nodes_df, args.day_start, args.day_end)
    graph = build_torch_graph(A_w, A_unw, cls, X_wallet, weighted=False)
    a_hat, adjacency = graph_operators(graph)
    N = int(a_hat.shape[0])

    illicit_idx = np.where(cls == 1)[0]
    gang_sets = connected_components_sets(A_unw, illicit_idx, args.min_gang_size)
    gangs = make_patterns(gang_sets, "alert", "gang", "g")
    LOGGER.info(f"  N={N:,}  gangs={len(gangs)}")

    V = degree_weighted_indicators(adjacency, gangs).to(a_hat.dtype)  # (N, m)
    phi = (V * _l_apply(a_hat, V)).sum(0).clamp_min(torch.finfo(V.dtype).eps)

    # achieved captures + detection outcomes from an existing pipeline run
    detected = achieved = None
    if args.diagnostics is not None and args.diagnostics.exists():
        diag = pd.read_csv(args.diagnostics)
        diag = diag[diag.day == args.day_start]
        if len(diag) == len(gangs):
            gid = {p.id: i for i, p in enumerate(gangs)}
            diag = diag.set_index("gang").reindex([p.id for p in gangs])
            detected = diag["detected"].to_numpy()
            achieved = diag["capture"].to_numpy()
            LOGGER.info(f"  merged achieved capture + outcomes from {args.diagnostics}")
        else:
            LOGGER.info(
                f"  diagnostics row count {len(diag)} != {len(gangs)} gangs -> skipped"
            )

    d = X_wallet.shape[1]
    X_wallet = X_wallet.to(a_hat.dtype)
    rows, per_gang = [], {}

    def run_config(name: str, X: torch.Tensor, K: int, tau: float) -> None:
        stack = _basis_stack(a_hat, X, K, "chebyshev", tau)
        B = torch.cat(stack, dim=1)
        cap, q = subspace_captures(a_hat, B, V, phi, tau)
        del stack, B
        row = summarize(name, cap, q, N, detected)
        rows.append(row)
        per_gang[name] = cap
        LOGGER.info(
            f"  {name:<34} q={q:>5}  chance={row['chance_q_over_N']:.4f}  "
            f"median={row['median']:.4f} ({row['median_over_chance']:.1f}x chance)  "
            f"mean={row['mean']:.4f}  min={row['min']:.4f}"
        )

    LOGGER.info("\n-- wallet features: degree sweep (does propagation help?) --")
    for K in (0, 2, 4, 8, 16, 32):
        run_config(f"wallet_d{d}_K{K}", X_wallet, K, args.tau)

    LOGGER.info("\n-- random structural range-finder: width sweep (K=32) --")
    for w in (16, 64, 128):
        Xr = random_structural_features(N, w, args.seed).to(a_hat.dtype)
        run_config(f"random_w{w}_K32", Xr, 32, args.tau)
    Xr = random_structural_features(N, 256, args.seed).to(a_hat.dtype)
    run_config("random_w256_K8", Xr, 8, args.tau)

    LOGGER.info("\n-- wallet + random concat (K=32) --")
    Xr = random_structural_features(N, 64, args.seed).to(a_hat.dtype)
    run_config(f"wallet+rand64_K32", torch.cat([X_wallet, Xr], dim=1), 32, args.tau)

    LOGGER.info("\n-- tau sensitivity (wallet, K=32) --")
    for tau in (0.1, 2.0):
        run_config(f"wallet_d{d}_K32_tau{tau}", X_wallet, 32, tau)

    LOGGER.info("\n-- label-free eigenspace span(U_K) reference --")
    for K in (d, 4 * d):
        ref = eigenspace_captures(a_hat, V, phi, args.tau, K)
        if ref is not None:
            cap, q = ref
            row = summarize(f"eigenspace_U{K}", cap, q, N, detected)
            rows.append(row)
            per_gang[f"eigenspace_U{K}"] = cap
            LOGGER.info(
                f"  {'eigenspace_U' + str(K):<34} q={q:>5}  "
                f"chance={row['chance_q_over_N']:.4f}  median={row['median']:.4f} "
                f"({row['median_over_chance']:.1f}x chance)"
            )

    # --- tables --------------------------------------------------------------
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "capture_ceiling_summary.csv", index=False)
    pg = pd.DataFrame(per_gang)
    pg.insert(0, "gang", [p.id for p in gangs])
    pg.insert(1, "size", [p.num_nodes for p in gangs])
    pg.insert(2, "Phi", phi.numpy())  # ||v_S||_L^2 = conductance Phi(S)
    if achieved is not None:
        pg["achieved_trained_bank"] = achieved
        pg["detected"] = detected
    pg.to_csv(args.out / "capture_ceiling_per_gang.csv", index=False)

    LOGGER.info("\n" + "=" * 96)
    LOGGER.info("CAPTURE CEILING SUMMARY (per-gang M_tau capture of the FULL dictionary span)")
    LOGGER.info("=" * 96)
    h = (f"  {'config':<34}{'q':>6}{'q/N':>8}{'median':>9}{'x chance':>9}"
         f"{'mean':>8}{'min':>8}" + ("{:>10}{:>9}".format("med det", "med miss") if detected is not None else ""))
    LOGGER.info(h)
    LOGGER.info("  " + "-" * (len(h) - 2))
    for r in rows:
        line = (f"  {r['config']:<34}{r['q_dim']:>6}{r['chance_q_over_N']:>8.4f}"
                f"{r['median']:>9.4f}{r['median_over_chance']:>9.1f}"
                f"{r['mean']:>8.4f}{r['min']:>8.4f}")
        if detected is not None:
            line += f"{r.get('median_detected', float('nan')):>10.4f}{r.get('median_missed', float('nan')):>9.4f}"
        LOGGER.info(line)
    if achieved is not None:
        LOGGER.info(
            f"\n  achieved (trained bank, d={d} cols): median={np.median(achieved):.4f} "
            f"mean={np.mean(achieved):.4f}  -- compare to wallet_d{d}_K32 ceiling"
        )
    (args.out / "capture_ceiling.json").write_text(
        json.dumps({"rows": rows, "tau": args.tau, "N": N,
                    "n_gangs": len(gangs)}, indent=2) + "\n"
    )
    LOGGER.info(f"\nCSV + JSON -> {args.out}")


if __name__ == "__main__":
    main()
