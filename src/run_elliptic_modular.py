"""Apply the modular :class:`CollectiveBankDetector` to the Elliptic++ Actors graph.

This is a *thin data adapter*: it loads the Elliptic++ wallet-address transaction
graph, forms the illicit gangs (connected components of the illicit subgraph) as
:class:`Pattern` objects, wraps everything in a dataset-agnostic
:class:`~src.collective_detector.GraphData`, and hands it to the same
:class:`~src.collective_detector.CollectiveBankDetector` that runs on the
synthetic benchmark -- no algorithm code is duplicated.

Run::

    conda activate FedStruct
    python -m src.run_elliptic_modular --day-start 24 --day-end 26 \
        --feature-mode wallet --coarsening-method ward-tree --ward-stop f1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import torch

from src.analyze_elliptic_coarsening import analyze_coarsening
from src.collective_detector import CollectiveBankDetector, DetectorConfig, GraphData
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import (
    build_torch_graph,
    load_node_features,
    make_patterns,
    split_train_test,
)


def _random_structural_features(num_nodes: int, width: int, seed: int) -> torch.Tensor:
    """Isotropic random range-finder ``Omega`` (the structural feature channel)."""

    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(num_nodes, width, dtype=torch.float64, generator=gen)
    return (X - X.mean(0, keepdim=True)) / X.std(0, keepdim=True).clamp_min(1e-8)


def _evaluate_transfer_day(
    det: CollectiveBankDetector,
    args: argparse.Namespace,
    day: int,
    feature_columns: "list[str] | None",
) -> dict:
    """Apply the *already-fit* detector to a single held-out day ``day``.

    The learned filter bank ``det.theta_`` is frozen; only the target subspace
    ``R = span(g_Theta(A_hat) X)`` and the RSA/Ward coarsening are recomputed on
    this day's graph.  The Ward cut is chosen by the label-free epsilon budget
    (``det.config.ward_stop == "epsilon"``), so the day's own gang labels never
    influence the coarsening -- they are used only to *score* the resulting
    supernodes.  Returns a per-day record (report + graph sizes).
    """

    print(f"\n--- transfer day {day} ---")
    A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, day, day)
    if args.feature_mode == "wallet":
        Xfeat = load_node_features(
            args.data_dir, nodes_df, day, day, keep_columns=feature_columns
        )
    else:
        Xfeat = _random_structural_features(
            int(A_unw.shape[0]), args.random_width, args.seed + day
        )
    graph = build_torch_graph(A_w, A_unw, cls, Xfeat, weighted=args.weighted)

    illicit_idx = np.where(cls == 1)[0]
    gang_sets = connected_components_sets(A_unw, illicit_idx, args.min_gang_size)
    day_gangs = make_patterns(gang_sets, "alert", "gang", "g")
    print(f"  gangs (illicit CC>={args.min_gang_size}): {len(day_gangs)}")

    data = GraphData.from_graph(graph)
    if data.feature_dim != int(det.theta_.shape[1]):
        raise ValueError(
            f"day {day} feature-dim {data.feature_dim} != trained filter "
            f"feature-dim {int(det.theta_.shape[1])}; cannot apply frozen filter."
        )

    record: dict = {
        "day": day,
        "n_nodes": data.num_nodes,
        "n_gangs": len(day_gangs),
        "report": None,
    }
    if not day_gangs:
        print("  no gangs on this day -> skipping evaluation")
        return record

    basis = det.target_subspace(data, day_gangs)  # R = span(Z); labels unused
    coarsening, _ = det.coarsen(data, basis, day_gangs)
    report = det.evaluate(data, coarsening, {"all": day_gangs})
    record["report"] = report.get("all")
    record["coarsening"] = {
        "n_original": int(coarsening.n_original),
        "n_coarse": int(coarsening.n_coarse),
        "epsilon": float(getattr(coarsening, "epsilon", float("nan"))),
    }
    r = record["report"]
    if r is not None:
        print(
            f"  recall={r['mean_recall']:.3f} precision={r['mean_precision']:.3f} "
            f"f1={r['mean_f1']:.3f} detection={r['detection_rate']:.1%} "
            f"({r['detected']}/{r['total']})"
        )
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # --- dataset ---
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--day-start", type=int, default=24)
    ap.add_argument("--day-end", type=int, default=24)
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--weighted", action="store_true", default=False)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument(
        "--transfer-days",
        type=int,
        default=10,
        help="apply the trained (frozen) filter to this many single days *after* "
        "--day-end, evaluating each day's graph individually (0 = disable). Each "
        "day builds its own graph g_k (day-end+k) and is scored with the label-free "
        "epsilon-budget Ward cut so the transfer is honest.",
    )
    ap.add_argument(
        "--feature-mode",
        choices=["wallet", "random"],
        default="wallet",
        help="'wallet' uses the real z-scored wallet features as the bank input X; "
        "'random' uses an isotropic structural range-finder of --random-width columns "
        "(more capacity when there are many training gangs).",
    )
    ap.add_argument("--random-width", type=int, default=64)
    ap.add_argument(
        "--max-train-gangs",
        type=int,
        default=0,
        help="cap the number of training gangs (0 = all). The collective objective "
        "saturates when #train-gangs > feature-dim (capacity threshold); capping or "
        "raising the feature width avoids lambda_min collapsing to 0.",
    )
    # --- detector hyperparameters (mirror DetectorConfig) ---
    ap.add_argument("--degree", type=int, default=32, help="polynomial degree K")
    ap.add_argument("--basis", choices=["chebyshev", "monomial"], default="chebyshev")
    ap.add_argument("--tau", type=float, default=0.5, help="screening (0 = Cor 4.7)")
    ap.add_argument("--epochs", type=int, default=500, help="training epochs")
    ap.add_argument("--learning-rate", type=float, default=0.02)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument(
        "--optimizer", choices=["projected", "riemannian"], default="riemannian"
    )
    ap.add_argument("--softmin-temperature", type=float, default=0.2)
    ap.add_argument("--conf-weight", type=float, default=10.0)
    ap.add_argument("--conf-reduce", choices=["max", "mean"], default="mean")
    ap.add_argument("--conf-delta", type=float, default=0.0)
    ap.add_argument("--structural-width", type=int, default=0)
    ap.add_argument("--coarsen-target", choices=["bank", "indicators"], default="bank")
    # --- coarsening ---
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
        "full Ward tree (heavier) and stops per --ward-stop.",
    )
    ap.add_argument(
        "--coarsening-laplacian",
        choices=["symmetric", "combinatorial"],
        default="symmetric",
    )
    ap.add_argument("--reduction", type=float, default=0.3)
    ap.add_argument("--epsilon", type=float, default=0.5)
    ap.add_argument("--max-levels", type=int, default=10)
    ap.add_argument("--ward-stop", choices=["epsilon", "f1"], default="epsilon")
    ap.add_argument("--ward-num-cuts", type=int, default=500)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--max-normal-patterns", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/elliptic_modular", type=Path)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    # --- 1. load the Elliptic++ graph + illicit gangs -----------------------
    print(f"=== Elliptic++ (modular) | days {args.day_start}-{args.day_end} ===")
    A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, args.day_start, args.day_end)
    feature_columns: "list[str] | None" = None
    if args.feature_mode == "wallet":
        Xfeat, feature_columns = load_node_features(
            args.data_dir, nodes_df, args.day_start, args.day_end, return_columns=True
        )
    else:
        Xfeat = _random_structural_features(
            int(A_unw.shape[0]), args.random_width, args.seed
        )
        print(
            f"  Feature matrix X: {Xfeat.shape[0]:,} x {Xfeat.shape[1]} (random structural)"
        )
    graph = build_torch_graph(A_w, A_unw, cls, Xfeat, weighted=args.weighted)

    illicit_idx = np.where(cls == 1)[0]
    gang_sets = connected_components_sets(A_unw, illicit_idx, args.min_gang_size)
    gangs = make_patterns(gang_sets, "alert", "gang", "g")
    print(
        f"  Gangs (illicit CC>={args.min_gang_size}): {len(gangs)}  "
        f"| sizes: {sorted((p.num_nodes for p in gangs), reverse=True)[:12]}..."
    )
    licit_sets = connected_components_sets(
        A_unw, np.where(cls == 2)[0], args.min_gang_size
    )
    licit_sets = sorted(licit_sets, key=len, reverse=True)[: args.max_normal_patterns]
    normals = make_patterns(licit_sets, "normal", "normal", "n")
    print(
        f"  gangs={len(gangs)} (sizes {sorted((p.num_nodes for p in gangs), reverse=True)[:8]}...)  normals={len(normals)}"
    )

    rng = np.random.default_rng(args.seed)
    gang_train, gang_test = split_train_test(gangs, args.train_ratio, rng)
    if args.max_train_gangs and len(gang_train) > args.max_train_gangs:
        gang_train = gang_train[: args.max_train_gangs]
    print(
        f"  train gangs: {len(gang_train)}  test gangs: {len(gang_test)}  "
        f"feature-dim: {graph.x.shape[1]}"
    )
    if len(gang_train) > graph.x.shape[1]:
        print(
            f"  WARNING: #train-gangs ({len(gang_train)}) > feature-dim "
            f"({graph.x.shape[1]}): capacity threshold -> lambda_min may be ~0. "
            "Use --feature-mode random --random-width, or --max-train-gangs."
        )

    # --- 2. dataset-agnostic wrapper + detector -----------------------------
    data = GraphData.from_graph(graph)  # features already set on graph.x
    cfg = DetectorConfig(
        degree=args.degree,
        basis=args.basis,
        tau=args.tau,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        ridge=args.ridge,
        optimizer=args.optimizer,
        softmin_temperature=args.softmin_temperature,
        conf_weight=args.conf_weight,
        conf_reduce=args.conf_reduce,
        conf_delta=args.conf_delta,
        structural_width=args.structural_width,
        coarsen_target=args.coarsen_target,
        coarsening_method=args.coarsening_method,
        coarsening_laplacian=args.coarsening_laplacian,
        reduction=args.reduction,
        epsilon=args.epsilon,
        max_levels=args.max_levels,
        ward_stop=args.ward_stop,
        ward_num_cuts=args.ward_num_cuts,
        threshold=args.threshold,
        seed=args.seed,
    )
    det = CollectiveBankDetector(cfg)

    print(
        f"\n  Fitting collective bank (basis={cfg.basis}, tau={cfg.tau}, "
        f"K={cfg.degree}, opt={cfg.optimizer}) …"
    )
    result = det.run(data, gang_train, gang_test, all_patterns=gangs)

    fit = result["fit"]
    print(
        f"    lambda_min(Gamma): {fit['init_objective']:.4g} -> {fit['objective']:.4g}"
    )
    if cfg.conf_weight > 0:
        print(
            f"    confusability chi: {fit['confusability_init']:.4g} -> {fit['confusability']:.4g}"
        )
    co = result["coarsening"]
    print(
        f"  coarsening ({cfg.coarsening_method}): N={co.n_original:,} -> "
        f"n_coarse={co.n_coarse:,}  epsilon={co.epsilon:.4g}"
    )

    # --- 3. report ----------------------------------------------------------
    print("\n" + "=" * 74)
    print(
        f"ELLIPTIC++ GANG DETECTION (modular)  days {args.day_start}-{args.day_end}  "
        f"N={data.num_nodes:,}  {len(gangs)} gangs"
    )
    print("=" * 74)
    hdr = f"  {'split':<6} {'recall':>8} {'precision':>10} {'f1':>7} {'detection':>10} {'det/tot':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name in ("train", "test", "all"):
        r = result["report"].get(name)
        if r is None:
            continue
        print(
            f"  {name:<6} {r['mean_recall']:>8.3f} {r['mean_precision']:>10.3f} "
            f"{r['mean_f1']:>7.3f} {r['detection_rate']:>10.1%} "
            f"{r['detected']:>4}/{r['total']:<5}"
        )

    # --- 3b. transfer: apply the frozen filter to the next N single days ----
    transfer_records: list[dict] = []
    transfer_summary: dict | None = None
    if args.transfer_days > 0:
        # Freeze the learned filter; recompute only the subspace + coarsening per
        # day. Force the label-free epsilon Ward stop so no day's own gangs steer
        # its coarsening (honest transfer).
        det_transfer = CollectiveBankDetector(replace(cfg, ward_stop="epsilon"))
        det_transfer.theta_ = det.theta_
        det_transfer.fit_info_ = det.fit_info_

        print("\n" + "=" * 74)
        print(
            f"TRANSFER: frozen filter (trained on days {args.day_start}-{args.day_end}) "
            f"applied to days {args.day_end + 1}-{args.day_end + args.transfer_days}"
        )
        print("=" * 74)
        for k in range(1, args.transfer_days + 1):
            transfer_records.append(
                _evaluate_transfer_day(
                    det_transfer, args, args.day_end + k, feature_columns
                )
            )

        # per-day table + average over the days that had gangs
        print("\n" + "=" * 74)
        print("PER-DAY TRANSFER PERFORMANCE (frozen filter, honest epsilon cut)")
        print("=" * 74)
        thdr = (
            f"  {'day':<5} {'nodes':>8} {'gangs':>6} {'recall':>8} {'precision':>10} "
            f"{'f1':>7} {'detection':>10} {'det/tot':>10}"
        )
        print(thdr)
        print("  " + "-" * (len(thdr) - 2))
        scored = [rec for rec in transfer_records if rec["report"] is not None]
        for rec in transfer_records:
            r = rec["report"]
            if r is None:
                print(
                    f"  {rec['day']:<5} {rec['n_nodes']:>8,} {rec['n_gangs']:>6} "
                    f"{'--':>8} {'--':>10} {'--':>7} {'--':>10} {'--':>10}"
                )
                continue
            print(
                f"  {rec['day']:<5} {rec['n_nodes']:>8,} {rec['n_gangs']:>6} "
                f"{r['mean_recall']:>8.3f} {r['mean_precision']:>10.3f} "
                f"{r['mean_f1']:>7.3f} {r['detection_rate']:>10.1%} "
                f"{r['detected']:>4}/{r['total']:<5}"
            )
        if scored:
            keys = ("mean_recall", "mean_precision", "mean_f1", "detection_rate")
            avg = {
                k: float(np.mean([rec["report"][k] for rec in scored])) for k in keys
            }
            tot_det = int(sum(rec["report"]["detected"] for rec in scored))
            tot_all = int(sum(rec["report"]["total"] for rec in scored))
            transfer_summary = {
                "n_days_scored": len(scored),
                **avg,
                "detected": tot_det,
                "total": tot_all,
            }
            print("  " + "-" * (len(thdr) - 2))
            print(
                f"  {'avg':<5} {'':>8} {'':>6} "
                f"{avg['mean_recall']:>8.3f} {avg['mean_precision']:>10.3f} "
                f"{avg['mean_f1']:>7.3f} {avg['detection_rate']:>10.1%} "
                f"{tot_det:>4}/{tot_all:<5}"
            )

        # echo the training-day test-pattern performance for side-by-side reading
        test_r = result["report"].get("test")
        if test_r is not None:
            print(
                f"\n  training-day (d{args.day_start}-{args.day_end}) TEST patterns: "
                f"recall={test_r['mean_recall']:.3f} "
                f"precision={test_r['mean_precision']:.3f} "
                f"f1={test_r['mean_f1']:.3f} "
                f"detection={test_r['detection_rate']:.1%} "
                f"({test_r['detected']}/{test_r['total']})"
            )

    out_json = args.out / f"elliptic_modular_d{args.day_start}-{args.day_end}.json"
    payload = {
        "dataset": "elliptic++",
        "day_start": args.day_start,
        "day_end": args.day_end,
        "n_nodes": data.num_nodes,
        "feature_mode": args.feature_mode,
        "feature_dim": data.feature_dim,
        "n_gangs": len(gangs),
        "n_train_gangs": len(gang_train),
        "config": cfg.to_dict(),
        "lambda_min_init": fit["init_objective"],
        "lambda_min_final": fit["objective"],
        "coarsening": {
            "n_original": co.n_original,
            "n_coarse": co.n_coarse,
            "epsilon": co.epsilon,
        },
        "report": result["report"],
        "transfer": {
            "days": transfer_records,
            "average": transfer_summary,
        },
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\nJSON report: {out_json}")

    basis = det.target_subspace(data, gang_train)
    coarsening, _ = det.coarsen(data, basis, gang_train)
    n2s = coarsening.node_to_supernode

    analyze_coarsening(
        data,
        basis,
        gangs,
        det,
        gang_train,
        args,
        cfg,
        coarsening,
        n2s,
        normals,
    )


if __name__ == "__main__":
    main()
