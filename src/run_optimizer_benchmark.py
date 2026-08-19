"""Which optimizer actually maximizes ``lambda_min(Gamma)`` fastest?

Adam on the soft-min capture objective converges slowly, and the reason is
structural rather than a tuning accident:

* with the N-independent Gram kernel the objective is **deterministic and
  full-batch** (no sampling noise unless negatives are on) and **low-dimensional**
  (``(K+1) d`` unknowns) -- the regime where quasi-Newton methods dominate;
* the Gram ``Z^T M_tau Z`` is ill-conditioned, and Adam's *diagonal* preconditioner
  carries no information about that conditioning, so it takes tiny steps along the
  stiff directions that matter;
* ``lambda_min`` is a **max-min eigenvalue** problem: at temperature 0 the gradient
  is supported on a single eigenvector (Danskin), so each step only helps the
  current argmin gang and the others drift back -- the zig-zag that makes the
  curve creep.

This benchmark holds the objective, data and seed fixed and varies only the
optimizer / initialization / temperature schedule, reporting the reached
``lambda_min`` versus both epochs and wall-clock seconds.

Variants:
  riem-adam         Riemannian Adam, flat init                (current default)
  proj-adam         projected Adam, flat init
  riem-adam-warm    Riemannian Adam from the closed-form init
  lbfgs             L-BFGS, flat init
  lbfgs-warm        L-BFGS from the closed-form init
  lbfgs-warm-anneal L-BFGS, closed-form init, annealed soft-min temperature

Run::

    conda activate FedStruct
    python -m src.run_optimizer_benchmark --day-start 25 --day-end 25
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.loukas_sgc_detection import graph_operators
from src.run_collective_bank_detection import fit_collective_bank
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import (
    build_torch_graph,
    load_node_features,
    make_patterns,
    random_structural_features,
    split_train_test,
)
from src.utils.utils import LOGGER, now


VARIANTS = {
    "riem-adam": dict(optimizer_kind="riemannian", warm_start="ones", anneal=1.0),
    "proj-adam": dict(optimizer_kind="projected", warm_start="ones", anneal=1.0),
    "riem-adam-warm": dict(
        optimizer_kind="riemannian", warm_start="closed_form", anneal=1.0
    ),
    "lbfgs": dict(optimizer_kind="lbfgs", warm_start="ones", anneal=1.0),
    "lbfgs-warm": dict(optimizer_kind="lbfgs", warm_start="closed_form", anneal=1.0),
    "lbfgs-warm-anneal": dict(
        optimizer_kind="lbfgs", warm_start="closed_form", anneal=20.0
    ),
}


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
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--conf-weight", type=float, default=10.0)
    ap.add_argument("--softmin-temperature", type=float, default=0.2)
    ap.add_argument("--capture-objective", default="lambda_min")
    ap.add_argument("--adam-epochs", type=int, default=1500)
    ap.add_argument("--lbfgs-epochs", type=int, default=40,
                    help="each 'epoch' is up to 20 inner quasi-Newton steps")
    ap.add_argument("--learning-rate", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=Path(f"results/optimizer_bench/{now}/"), type=Path)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    LOGGER.info(f"=== optimizer benchmark | day {args.day_start}-{args.day_end} "
                f"K={args.degree} tau={args.tau} obj={args.capture_objective} ===")
    A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, args.day_start, args.day_end)
    if args.feature_mode in ("wallet", "wallet+random"):
        X = load_node_features(args.data_dir, nodes_df, args.day_start, args.day_end)
        if args.feature_mode == "wallet+random":
            X = torch.cat(
                [X, random_structural_features(
                    int(A_unw.shape[0]), args.random_width, args.seed)], dim=1
            )
    else:
        X = random_structural_features(int(A_unw.shape[0]), args.random_width, args.seed)
    graph = build_torch_graph(A_w, A_unw, cls, X, weighted=False)
    a_hat, adjacency = graph_operators(graph)
    X = graph.x.to(a_hat.dtype)
    gang_sets = connected_components_sets(
        A_unw, np.where(cls == 1)[0], args.min_gang_size
    )
    gangs = make_patterns(gang_sets, "alert", "gang", "g")
    rng = np.random.default_rng(args.seed)
    gang_train, _ = split_train_test(gangs, args.train_ratio, rng)
    LOGGER.info(f"  {len(gangs)} gangs ({len(gang_train)} train), "
                f"d={X.shape[1]}, params=(K+1)d={(args.degree + 1) * X.shape[1]:,}")

    rows, curves = [], {}
    for name, cfg in VARIANTS.items():
        epochs = args.lbfgs_epochs if cfg["optimizer_kind"] == "lbfgs" else args.adam_epochs
        LOGGER.info(f"\n  [{name}] {cfg['optimizer_kind']}, init={cfg['warm_start']}, "
                    f"anneal={cfg['anneal']}, epochs={epochs}")
        t0 = time.perf_counter()
        fit = fit_collective_bank(
            [("train", a_hat, adjacency, gang_train, X)],
            degree=args.degree, epochs=epochs, learning_rate=args.learning_rate,
            ridge=args.ridge, fit_seed=args.seed, tau=args.tau,
            softmin_temperature=args.softmin_temperature,
            capture_objective=args.capture_objective,
            conf_weight=args.conf_weight, conf_reduce="mean",
            optimizer_kind=cfg["optimizer_kind"], warm_start=cfg["warm_start"],
            softmin_anneal=cfg["anneal"], snapshot_interval=0,
        )
        elapsed = time.perf_counter() - t0
        hist = np.asarray(fit["history"], dtype=float)
        energy = np.asarray(fit["energy_history"], dtype=float)
        curves[name] = {"lambda_min": hist, "capture_mean": energy,
                        "seconds": elapsed, "epochs": epochs}
        rows.append({
            "variant": name,
            "optimizer": cfg["optimizer_kind"],
            "init": cfg["warm_start"],
            "anneal": cfg["anneal"],
            "epochs": epochs,
            "seconds": elapsed,
            "lambda_min_init": float(fit["init_objective"]),
            "lambda_min_best": float(fit["objective"]),
            "lambda_min_final": float(hist[-1]) if len(hist) else float("nan"),
            "capture_mean_final": float(energy[-1]) if len(energy) else float("nan"),
            "confusability": float(fit["confusability"]),
            "margin": float(fit["margin"]),
        })
        LOGGER.info(
            f"    lambda_min {fit['init_objective']:.5f} -> {fit['objective']:.5f}"
            f"   capture_mean {energy[-1] if len(energy) else float('nan'):.4f}"
            f"   chi {fit['confusability']:.5f}   [{elapsed:.1f}s]"
        )

    df = pd.DataFrame(rows).sort_values("lambda_min_best", ascending=False)
    df.to_csv(args.out / "optimizer_benchmark.csv", index=False)

    base = df[df.variant == "riem-adam"].lambda_min_best.iloc[0]
    LOGGER.info("\n" + "=" * 104)
    LOGGER.info("OPTIMIZER BENCHMARK (identical objective, data and seed)")
    LOGGER.info("=" * 104)
    hdr = (f"  {'variant':<20}{'epochs':>8}{'sec':>8}{'lambda_min':>13}"
           f"{'vs riem-adam':>14}{'capture':>10}{'chi':>10}")
    LOGGER.info(hdr)
    LOGGER.info("  " + "-" * (len(hdr) - 2))
    for _, r in df.iterrows():
        LOGGER.info(
            f"  {r.variant:<20}{int(r.epochs):>8}{r.seconds:>8.1f}"
            f"{r.lambda_min_best:>13.5f}{r.lambda_min_best / base:>13.1f}x"
            f"{r.capture_mean_final:>10.4f}{r.confusability:>10.5f}"
        )

    # convergence figure: value vs epoch and vs wall-clock
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    cmap = plt.get_cmap("tab10")
    for i, (name, c) in enumerate(curves.items()):
        y = c["lambda_min"]
        if not len(y):
            continue
        col = cmap(i % 10)
        axes[0].plot(np.arange(len(y)), y, color=col, lw=1.5, label=name)
        axes[1].plot(np.linspace(0, c["seconds"], len(y)), y, color=col, lw=1.5,
                     label=name)
    axes[0].set_xlabel("epoch (L-BFGS: one epoch = up to 20 inner steps)")
    axes[1].set_xlabel("wall-clock seconds")
    axes[1].set_xscale("log")
    for a in axes:
        a.set_ylabel("$\\lambda_{\\min}(\\Gamma)$")
        a.grid(alpha=0.25, lw=0.5)
        a.spines[["top", "right"]].set_visible(False)
        a.legend(fontsize=7, frameon=False)
    axes[0].set_title("convergence per epoch")
    axes[1].set_title("convergence per second")
    fig.suptitle(
        f"capture optimizer benchmark -- elliptic++ d{args.day_start}-{args.day_end}, "
        f"{args.capture_objective}, K={args.degree}, {len(gang_train)} train gangs",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out / "optimizer_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    (args.out / "optimizer_benchmark.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    LOGGER.info(f"\nCSV + plot + JSON -> {args.out}")


if __name__ == "__main__":
    main()
