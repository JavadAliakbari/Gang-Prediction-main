"""Multi-head Chebyshev filter bank: inductive high-capture target learning.

The single shared bank has ONE hop-profile per feature channel
(``Theta in R^{(K+1) x d}``); the capture-gap diagnosis showed this
parameterization -- not the objective -- is what pins per-gang capture ~6x below
the reachability ceiling: different gangs want different filters on the same
channel and must compromise.  Indicator/dictionary targets fix that but are NOT
inductive (they need the gang's node set, unavailable on new days).

The multi-head bank keeps everything inductive.  ``Theta in R^{H x (K+1) x d}``
holds ``H`` independent filter banks; the coarsening target is the concatenated
span ``Z = [Z^(1) | ... | Z^(H)]`` (``H*d`` columns).  Each motif type can align
with its own head -- no explicit attention is needed, because the capture of a
span is itself a maximum over linear combinations: projection "attends" for
free.  All parameters are Chebyshev coefficients (graph-independent), so the
frozen ``Theta`` transfers to unseen days exactly like the single-head bank.

Key construction: the closed-form per-gang optimum ``z_hat_j`` (Theorem 6.2) IS
a filter bank output with coefficients ``W_j in R^{(K+1) x d}``.  A bank with
``H = #train-gangs`` heads **warm-started at the per-gang closed-form solves**
therefore contains every train gang's optimum in its span at initialization --
ceiling capture from epoch 0, with purely inductive parameters; training then
only has to preserve/generalize it.

Compares (day 25 -> frozen transfer to days 26-35):
  H1-trace   single shared bank, mean-capture objective (the fair baseline)
  H4-trace   4 heads, cold init
  H8-trace   8 heads, cold init
  H8-softmin 8 heads, worst-gang capture objective
  H8-warm    8 heads, warm-started at the per-gang closed-form solves

Per-gang capture (rank-revealing, vs the wallet ceiling) on the training day,
capture of *unseen* gangs on transfer days, and label-free Ward detection
(epsilon-budget stop) for the flagged configs.

Run::

    conda activate FedStruct
    python -m src.run_multihead_bank \
        --ceiling-csv results/capture_ceiling/<stamp>/capture_ceiling_per_gang.csv
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
from torch import nn

from src.loukas_sgc_detection import evaluate_loukas_patterns, graph_operators
from src.run_capture_ceiling import subspace_captures
from src.run_collective_bank_detection import (
    _basis_stack,
    _collective_gamma_mz,
    _filtered_bank,
    _l_apply,
    _m_apply,
    _soft_min_values,
    _train_gang_m_vhat,
    degree_weighted_indicators,
    ward_tree_coarsen,
)
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import (
    build_torch_graph,
    load_node_features,
    make_patterns,
    split_train_test,
)
from src.utils.utils import LOGGER, now


# --------------------------------------------------------------------------- #
# multi-head bank
# --------------------------------------------------------------------------- #
def unit_channels(theta: torch.Tensor) -> torch.Tensor:
    """Normalize each head's per-channel filter to the unit sphere (H, K+1, d)."""

    eps = torch.finfo(theta.dtype).eps
    return theta / theta.norm(dim=1, keepdim=True).clamp_min(eps)


def multihead_bank(propagated: list, theta: torch.Tensor) -> torch.Tensor:
    """``Z = [Z^(1) | ... | Z^(H)]`` with ``Z^(h) = g_{theta_h}(A_hat) X``  (N, H*d)."""

    return torch.cat(
        [_filtered_bank(propagated, theta[h]) for h in range(theta.shape[0])], dim=1
    )


def closed_form_gang_filters(
    a_hat, propagated: list, m_vhat: torch.Tensor
) -> torch.Tensor:
    """Per-gang Theorem 6.2 coefficients ``W_j`` (m, K+1, d): z_hat_j = sum_a bank cols.

    Rank-revealing solve of the full-dictionary Gram (identical to the
    ``coarsen_target="dictionary"`` branch of :func:`build_bank_subspace`).
    """

    B = torch.cat(propagated, dim=1)  # (N, (K+1)d), col index = k*d + a
    m_b = _m_apply(a_hat, B, _m_apply_cached["tau"])
    gram = B.T @ m_b
    gram = 0.5 * (gram + gram.T)
    rhs = B.T @ m_vhat  # ((K+1)d, m)
    evals, evecs = torch.linalg.eigh(gram)
    keep = evals > evals.max() * 1e-10
    coeffs = evecs[:, keep] @ ((evecs[:, keep].T @ rhs) / evals[keep].unsqueeze(1))
    K1 = len(propagated)
    d = propagated[0].shape[1]
    return coeffs.T.reshape(-1, K1, d)  # (m, K+1, d)


_m_apply_cached: dict = {}


def train_multihead(
    propagated: list,
    m_propagated: list,
    m_vhat_train: torch.Tensor,
    *,
    heads: int,
    objective: str,
    epochs: int,
    lr: float,
    ridge: float,
    seed: int,
    init: "torch.Tensor | None" = None,
    softmin_temperature: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Ascend per-gang capture of the concatenated multi-head span (train gangs)."""

    torch.manual_seed(seed)
    K1, d = len(propagated), propagated[0].shape[1]
    dtype = propagated[0].dtype
    if init is None:
        raw0 = torch.ones(heads, K1, d, dtype=dtype)
        raw0 += 0.1 * torch.randn(heads, K1, d, dtype=dtype)  # break head symmetry
    else:
        raw0 = init.clone().to(dtype)
        if raw0.shape[0] < heads:  # pad extra heads with noisy low-pass
            pad = torch.ones(heads - raw0.shape[0], K1, d, dtype=dtype)
            pad += 0.1 * torch.randn_like(pad)
            raw0 = torch.cat([raw0, pad], dim=0)
    raw = nn.Parameter(raw0)
    opt = torch.optim.Adam([raw], lr=lr, weight_decay=1e-4, eps=1e-8)

    hist = {}
    for ep in range(epochs + 1):
        th = unit_channels(raw)
        Z = multihead_bank(propagated, th)
        m_z = multihead_bank(m_propagated, th)
        gamma = _collective_gamma_mz(Z, m_z, m_vhat_train, ridge)
        diag = torch.diagonal(gamma)
        obj = (
            diag.mean()
            if objective == "trace"
            else _soft_min_values(diag, softmin_temperature)
        )
        if ep == 0:
            hist["init_mean"], hist["init_min"] = float(diag.mean()), float(diag.min())
        if ep == epochs:
            break
        opt.zero_grad(set_to_none=True)
        (-obj).backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            LOGGER.info(
                f"      epoch {ep + 1:>4}: mean C={float(diag.mean()):.4f} "
                f"min C={float(diag.min()):.4f}"
            )
    hist["final_mean"], hist["final_min"] = float(diag.mean()), float(diag.min())
    return unit_channels(raw).detach(), hist


# --------------------------------------------------------------------------- #
# evaluation helpers
# --------------------------------------------------------------------------- #
def gang_indicator_stats(a_hat, adjacency, gangs):
    eps_v = 1e-30
    V = degree_weighted_indicators(adjacency, gangs).to(a_hat.dtype)
    phi = (V * _l_apply(a_hat, V)).sum(0).clamp_min(eps_v)
    return V, phi


def per_gang_capture_of_span(a_hat, adjacency, gangs, Z, tau) -> np.ndarray:
    """Rank-revealing per-gang capture of span(Z) (same math as the ceiling)."""

    V, phi = gang_indicator_stats(a_hat, adjacency, gangs)
    cap, _ = subspace_captures(a_hat, Z, V, phi, tau)
    return cap


def ward_detect(adjacency, Z, gangs, y, *, tau, epsilon, threshold, num_cuts):
    coarsening, _ = ward_tree_coarsen(
        adjacency,
        Z,
        gangs,
        y,
        tau=tau,
        laplacian="symmetric",
        threshold=threshold,
        stop="epsilon",
        epsilon_budget=epsilon,
        num_cuts=num_cuts,
    )
    res, by_label = evaluate_loukas_patterns(
        gangs, coarsening.node_to_supernode, y, threshold=threshold
    )
    a = by_label.get("alert", {})
    return {
        "detection_rate": a.get("detection_rate", 0.0),
        "mean_recall": a.get("mean_recall", 0.0),
        "mean_precision": a.get("mean_precision", 0.0),
        "mean_f1": float(np.mean([r.f1 for r in res])) if res else 0.0,
        "detected": int(a.get("detected", 0)),
        "total": int(a.get("total", 0)),
        "n_coarse": int(coarsening.n_coarse),
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--day-start", type=int, default=25)
    ap.add_argument("--day-end", type=int, default=25)
    ap.add_argument("--transfer-days", type=int, default=10)
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--train-ratio", type=float, default=0.4)
    ap.add_argument("--degree", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=0.02)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--epsilon", type=float, default=0.85, help="label-free Ward stop")
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--ward-num-cuts", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--detect-configs",
        default="H1-trace,H8-trace,H8-warm",
        help="comma list of configs that also get the (slower) Ward detection eval",
    )
    ap.add_argument("--ceiling-csv", type=Path, default=None)
    ap.add_argument("--out", default=Path(f"results/multihead_bank/{now}/"), type=Path)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    # --- training day --------------------------------------------------------
    LOGGER.info(f"=== multi-head bank | day {args.day_start}-{args.day_end} "
                f"K={args.degree} tau={args.tau} ===")
    A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, args.day_start, args.day_end)
    Xfeat, feature_columns = load_node_features(
        args.data_dir, nodes_df, args.day_start, args.day_end, return_columns=True
    )
    graph = build_torch_graph(A_w, A_unw, cls, Xfeat, weighted=False)
    a_hat, adjacency = graph_operators(graph)
    y = graph.y
    gang_sets = connected_components_sets(
        A_unw, np.where(cls == 1)[0], args.min_gang_size
    )
    gangs = make_patterns(gang_sets, "alert", "gang", "g")
    rng = np.random.default_rng(args.seed)
    gang_train, gang_test = split_train_test(gangs, args.train_ratio, rng)
    train_ids = {p.id for p in gang_train}
    LOGGER.info(f"  gangs={len(gangs)} train={len(gang_train)} test={len(gang_test)}")

    X = Xfeat.to(a_hat.dtype)
    propagated = _basis_stack(a_hat, X, args.degree, "chebyshev", args.tau)
    m_propagated = [_m_apply(a_hat, P, args.tau) for P in propagated]
    m_vhat_train = _train_gang_m_vhat(a_hat, adjacency, gang_train, args.tau)

    # warm start: per-gang closed-form filters (Theorem 6.2)
    _m_apply_cached.clear()
    _m_apply_cached["tau"] = args.tau
    W = closed_form_gang_filters(a_hat, propagated, m_vhat_train)  # (m, K+1, d)
    LOGGER.info(f"  closed-form warm start: {W.shape[0]} gang filters "
                f"(K+1={W.shape[1]}, d={W.shape[2]})")

    configs = [
        ("H1-trace", dict(heads=1, objective="trace", init=None)),
        ("H4-trace", dict(heads=4, objective="trace", init=None)),
        ("H8-trace", dict(heads=8, objective="trace", init=None)),
        ("H8-softmin", dict(heads=8, objective="softmin", init=None)),
        ("H8-warm", dict(heads=8, objective="trace", init=W)),
    ]
    detect_set = set(args.detect_configs.split(","))

    thetas: dict[str, torch.Tensor] = {}
    fit_hist: dict[str, dict] = {}
    day25_caps: dict[str, np.ndarray] = {}
    day25_det: dict[str, dict] = {}
    for name, kw in configs:
        LOGGER.info(f"\n  [{name}] training ({kw['heads']} heads, {kw['objective']}"
                    f"{', warm' if kw['init'] is not None else ''}) ...")
        theta, hist = train_multihead(
            propagated, m_propagated, m_vhat_train,
            heads=kw["heads"], objective=kw["objective"], epochs=args.epochs,
            lr=args.learning_rate, ridge=args.ridge, seed=args.seed,
            init=kw["init"],
        )
        thetas[name], fit_hist[name] = theta, hist
        Z = multihead_bank(propagated, theta)
        day25_caps[name] = per_gang_capture_of_span(
            a_hat, adjacency, gangs, Z, args.tau
        )
        LOGGER.info(
            f"    train-gang C: init mean {hist['init_mean']:.4f} -> final mean "
            f"{hist['final_mean']:.4f} (min {hist['final_min']:.4f})"
        )
        if name in detect_set:
            day25_det[name] = ward_detect(
                adjacency, Z, gangs, y, tau=args.tau, epsilon=args.epsilon,
                threshold=args.threshold, num_cuts=args.ward_num_cuts,
            )
            r = day25_det[name]
            LOGGER.info(
                f"    day-25 detection (eps<={args.epsilon}): "
                f"R={r['mean_recall']:.3f} P={r['mean_precision']:.3f} "
                f"F1={r['mean_f1']:.3f} det={r['detection_rate']:.1%} "
                f"({r['detected']}/{r['total']}, n_coarse={r['n_coarse']})"
            )

    # --- day-25 per-gang table ----------------------------------------------
    df = pd.DataFrame({
        "gang": [p.id for p in gangs],
        "size": [p.num_nodes for p in gangs],
        "train": [p.id in train_ids for p in gangs],
        **{name: day25_caps[name] for name, _ in configs},
    })
    if args.ceiling_csv is not None and args.ceiling_csv.exists():
        ceil = pd.read_csv(args.ceiling_csv)[["gang", "wallet_d55_K32"]]
        df = df.merge(ceil.rename(columns={"wallet_d55_K32": "ceiling"}), on="gang",
                      how="left")
    df = df.sort_values("size", ascending=False)
    df.to_csv(args.out / "day25_per_gang_capture.csv", index=False)

    names = [n for n, _ in configs]
    cols = (["ceiling"] if "ceiling" in df else []) + names
    LOGGER.info("\n" + "=" * (18 + 12 * len(cols)))
    LOGGER.info("DAY-25 PER-GANG CAPTURE (rank-revealing, degree-weighted v_hat)")
    LOGGER.info("=" * (18 + 12 * len(cols)))
    hdr = f"  {'gang':<5}{'size':>5}{'train':>6}" + "".join(f"{c:>12}" for c in cols)
    LOGGER.info(hdr)
    LOGGER.info("  " + "-" * (len(hdr) - 2))
    for _, r in df.iterrows():
        LOGGER.info(
            f"  {r.gang:<5}{r['size']:>5}{str(bool(r.train)):>6}"
            + "".join(f"{r[c]:>12.4f}" for c in cols)
        )
    LOGGER.info("  " + "-" * (len(hdr) - 2))
    for label, sub in (("train", df[df.train]), ("test", df[~df.train])):
        LOGGER.info(
            f"  median {label:<11}" + "".join(f"{sub[c].median():>12.4f}" for c in cols)
        )

    # --- frozen transfer -----------------------------------------------------
    transfer_rows, transfer_det = [], []
    for k in range(1, args.transfer_days + 1):
        day = args.day_end + k
        LOGGER.info(f"\n--- transfer day {day} (frozen Theta) ---")
        A_unw_d, A_w_d, cls_d, nodes_d = build_graph(args.data_dir, day, day)
        X_d = load_node_features(
            args.data_dir, nodes_d, day, day, keep_columns=feature_columns
        )
        graph_d = build_torch_graph(A_w_d, A_unw_d, cls_d, X_d, weighted=False)
        a_hat_d, adj_d = graph_operators(graph_d)
        gsets_d = connected_components_sets(
            A_unw_d, np.where(cls_d == 1)[0], args.min_gang_size
        )
        gangs_d = make_patterns(gsets_d, "alert", "gang", "g")
        if not gangs_d:
            LOGGER.info("  no gangs -> skip")
            continue
        prop_d = _basis_stack(a_hat_d, X_d.to(a_hat_d.dtype), args.degree,
                              "chebyshev", args.tau)
        for name in names:
            Z_d = multihead_bank(prop_d, thetas[name])
            cap = per_gang_capture_of_span(a_hat_d, adj_d, gangs_d, Z_d, args.tau)
            transfer_rows += [
                {"day": day, "config": name, "gang": p.id, "size": p.num_nodes,
                 "capture": float(c)}
                for p, c in zip(gangs_d, cap)
            ]
            if name in detect_set:
                det = ward_detect(
                    adj_d, Z_d, gangs_d, graph_d.y, tau=args.tau,
                    epsilon=args.epsilon, threshold=args.threshold,
                    num_cuts=args.ward_num_cuts,
                )
                transfer_det.append({"day": day, "config": name, **det})
        caps_day = {name: np.median([r["capture"] for r in transfer_rows
                                     if r["day"] == day and r["config"] == name])
                    for name in names}
        LOGGER.info("  median capture: " +
                    "  ".join(f"{n}={caps_day[n]:.4f}" for n in names))

    tdf = pd.DataFrame(transfer_rows)
    tdf.to_csv(args.out / "transfer_per_gang_capture.csv", index=False)
    ddf = pd.DataFrame(transfer_det)
    if len(ddf):
        ddf.to_csv(args.out / "transfer_detection.csv", index=False)

    # --- summaries -----------------------------------------------------------
    LOGGER.info("\n" + "=" * 84)
    LOGGER.info("SUMMARY: capture of UNSEEN gangs (median over the 10 transfer days' gangs)")
    LOGGER.info("=" * 84)
    for name in names:
        sub = tdf[tdf.config == name]
        big = sub[sub["size"] >= 10]
        LOGGER.info(
            f"  {name:<12} all-gangs median={sub.capture.median():.4f} "
            f"mean={sub.capture.mean():.4f}   large(>=10) median="
            f"{big.capture.median() if len(big) else float('nan'):.4f}"
        )
    if len(ddf):
        LOGGER.info("\nSUMMARY: label-free Ward detection "
                    f"(eps<={args.epsilon}) averaged over transfer days")
        for name in sorted(ddf.config.unique()):
            sub = ddf[ddf.config == name]
            LOGGER.info(
                f"  {name:<12} R={sub.mean_recall.mean():.3f} "
                f"P={sub.mean_precision.mean():.3f} F1={sub.mean_f1.mean():.3f} "
                f"det={sub.detection_rate.mean():.1%} "
                f"({int(sub.detected.sum())}/{int(sub.total.sum())})"
            )

    (args.out / "summary.json").write_text(json.dumps({
        "configs": {n: fit_hist[n] for n in names},
        "day25_train_median": {n: float(df[df.train][n].median()) for n in names},
        "day25_test_median": {n: float(df[~df.train][n].median()) for n in names},
        "transfer_median": {n: float(tdf[tdf.config == n].capture.median())
                            for n in names},
    }, indent=2) + "\n")
    LOGGER.info(f"\nCSV + JSON -> {args.out}")


if __name__ == "__main__":
    main()
