"""Why is the trained bank's capture ~10x below the reachability ceiling? Fix it.

Decomposes the gap between the achieved per-gang capture (median ~0.03 on
Elliptic++ day 25) and the full-dictionary ceiling (~0.27) into its candidate
causes, then verifies the fix:

1. **objective / regularization** -- refit the same shared bank with
   ``capture_objective="trace"`` and ``conf_weight=0`` (pure mean-capture
   ascent, no confusability penalty, no lambda_min coupling).  If capture stays
   low, training harder was never the answer.
2. **parameterization** -- the bank is one hop-profile per feature channel
   (``Theta in R^{(K+1) x d}``, rank one per channel); gangs wanting different
   filters on the same channel must compromise.  The full dictionary has no such
   constraint.
3. **target construction** -- ``coarsen_target="indicators"`` projects the gang
   indicators onto span(Z), so it *inherits* the bank's capture instead of the
   Theorem 6.2 closed form.  The fix (``coarsen_target="dictionary"``) projects
   onto the full dictionary col T and attains each TRAIN gang's ceiling by
   construction; ``"bank+dictionary"`` appends the bank span so held-out gangs
   keep a generalizing target.

Prints one per-gang table (ceiling, both trained banks, both indicator targets,
detected flag) and a factor-attribution summary.

Run::

    conda activate FedStruct
    python -m src.run_capture_gap_diagnosis \
        --ceiling-csv results/capture_ceiling/<stamp>/capture_ceiling_per_gang.csv
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
import pandas as pd
import torch

from src.collective_detector import CollectiveBankDetector, DetectorConfig, GraphData
from src.run_collective_bank_detection import (
    _collective_gamma,
    _train_gang_m_vhat,
)
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import (
    build_torch_graph,
    load_node_features,
    make_patterns,
    split_train_test,
)
from src.utils.utils import LOGGER, now


def per_gang_capture(a_hat, adjacency, gangs, basis, tau, ridge=1e-9) -> np.ndarray:
    """``Gamma_jj`` of every gang under ``span(basis)`` (degree-weighted v_hat)."""

    m_vhat = _train_gang_m_vhat(a_hat, adjacency, gangs, tau)
    gamma = _collective_gamma(a_hat, basis, m_vhat, ridge, tau)
    return torch.diagonal(gamma).clamp(0.0, 1.0).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--day-start", type=int, default=25)
    ap.add_argument("--day-end", type=int, default=25)
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--train-ratio", type=float, default=0.4)
    ap.add_argument("--degree", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ceiling-csv", type=Path, default=None)
    ap.add_argument("--out", default=Path(f"results/capture_gap/{now}/"), type=Path)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    # --- data + identical gang split ----------------------------------------
    A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, args.day_start, args.day_end)
    Xfeat = load_node_features(args.data_dir, nodes_df, args.day_start, args.day_end)
    graph = build_torch_graph(A_w, A_unw, cls, Xfeat, weighted=False)
    data = GraphData.from_graph(graph)
    gang_sets = connected_components_sets(
        A_unw, np.where(cls == 1)[0], args.min_gang_size
    )
    gangs = make_patterns(gang_sets, "alert", "gang", "g")
    rng = np.random.default_rng(args.seed)
    gang_train, gang_test = split_train_test(gangs, args.train_ratio, rng)
    train_ids = {p.id for p in gang_train}
    LOGGER.info(f"gangs={len(gangs)} train={len(gang_train)} test={len(gang_test)}")

    # pipeline-default config (mirrors run_elliptic_modular defaults)
    base = DetectorConfig(
        degree=args.degree, basis="chebyshev", tau=args.tau, epochs=args.epochs,
        learning_rate=0.02, ridge=1e-3, optimizer="riemannian",
        softmin_temperature=0.2, capture_objective="lambda_min",
        conf_weight=10.0, conf_reduce="mean", coarsen_target="bank",
        seed=args.seed,
    )

    # --- 1. bank as trained by the pipeline (lambda_min + confusability) ----
    LOGGER.info("\n[1/4] fitting pipeline bank (lambda_min, conf_weight=10) ...")
    det1 = CollectiveBankDetector(base).fit(data, gang_train)
    Z1 = det1.target_subspace(data, gang_train)  # span(Z), coarsen_target="bank"
    cap_bank = per_gang_capture(data.a_hat, data.adjacency, gangs, Z1, args.tau)

    # --- 2. same bank, pure capture objective (no conf, no lambda_min) ------
    LOGGER.info("[2/4] fitting ablation bank (trace objective, conf_weight=0) ...")
    det2 = CollectiveBankDetector(
        replace(base, capture_objective="trace", conf_weight=0.0)
    ).fit(data, gang_train)
    Z2 = det2.target_subspace(data, gang_train)
    cap_trace = per_gang_capture(data.a_hat, data.adjacency, gangs, Z2, args.tau)

    # --- 3. old indicators target (projection onto span Z) ------------------
    LOGGER.info("[3/4] building span(Z)-projected indicators (old 'indicators') ...")
    det1.config = replace(base, coarsen_target="indicators")
    T_ind = det1.target_subspace(data, gang_train)
    cap_ind = per_gang_capture(data.a_hat, data.adjacency, gangs, T_ind, args.tau)

    # --- 4. FIX: full-dictionary projected indicators (Theorem 6.2) ---------
    LOGGER.info("[4/4] building full-dictionary indicators ('dictionary' fix) ...")
    det1.config = replace(base, coarsen_target="dictionary")
    T_dict = det1.target_subspace(data, gang_train)
    cap_dict = per_gang_capture(data.a_hat, data.adjacency, gangs, T_dict, args.tau)
    det1.config = replace(base, coarsen_target="bank+dictionary")
    T_bd = det1.target_subspace(data, gang_train)
    cap_bd = per_gang_capture(data.a_hat, data.adjacency, gangs, T_bd, args.tau)
    det1.config = base

    # --- assemble ------------------------------------------------------------
    df = pd.DataFrame({
        "gang": [p.id for p in gangs],
        "size": [p.num_nodes for p in gangs],
        "train": [p.id in train_ids for p in gangs],
        "bank_lammin_conf": cap_bank,
        "bank_trace_noconf": cap_trace,
        "indicators_spanZ": cap_ind,
        "dictionary_fix": cap_dict,
        "bank+dictionary": cap_bd,
    })
    if args.ceiling_csv is not None and args.ceiling_csv.exists():
        ceil = pd.read_csv(args.ceiling_csv)[["gang", "wallet_d55_K32", "detected"]]
        ceil = ceil.rename(columns={"wallet_d55_K32": "ceiling_K32"})
        df = df.merge(ceil, on="gang", how="left")
    df = df.sort_values("size", ascending=False)
    df.to_csv(args.out / "capture_gap_per_gang.csv", index=False)

    LOGGER.info("\n" + "=" * 110)
    LOGGER.info("PER-GANG CAPTURE: trained banks vs indicator targets vs ceiling "
                f"(day {args.day_start}, tau={args.tau}, K={args.degree})")
    LOGGER.info("=" * 110)
    cols = [c for c in ("ceiling_K32", "bank_lammin_conf", "bank_trace_noconf",
                        "indicators_spanZ", "dictionary_fix", "bank+dictionary") if c in df]
    hdr = f"  {'gang':<5}{'size':>5}{'train':>6}" + "".join(f"{c:>19}" for c in cols)
    LOGGER.info(hdr)
    LOGGER.info("  " + "-" * (len(hdr) - 2))
    for _, r in df.iterrows():
        LOGGER.info(
            f"  {r.gang:<5}{r['size']:>5}{str(bool(r.train)):>6}"
            + "".join(f"{r[c]:>19.4f}" for c in cols)
        )
    tr = df[df.train]
    LOGGER.info("  " + "-" * (len(hdr) - 2))
    LOGGER.info(
        f"  {'median (train gangs)':<16}" + "".join(f"{tr[c].median():>19.4f}" for c in cols)
    )
    te = df[~df.train]
    LOGGER.info(
        f"  {'median (test gangs)':<16}" + "".join(f"{te[c].median():>19.4f}" for c in cols)
    )

    # factor attribution on the train gangs
    summary = {
        "train_median": {c: float(tr[c].median()) for c in cols},
        "test_median": {c: float(te[c].median()) for c in cols},
        "fit1_objective": det1.fit_info_.get("objective"),
        "fit2_objective": det2.fit_info_.get("objective"),
    }
    (args.out / "capture_gap_summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n"
    )
    LOGGER.info("\nFactor attribution (train-gang medians):")
    if "ceiling_K32" in df:
        LOGGER.info(
            f"  ceiling {tr['ceiling_K32'].median():.3f}"
            f"  | shared bank (pipeline obj) {tr['bank_lammin_conf'].median():.3f}"
            f"  | shared bank (pure capture) {tr['bank_trace_noconf'].median():.3f}"
            f"  -> objective explains {tr['bank_trace_noconf'].median() - tr['bank_lammin_conf'].median():+.3f}"
        )
        LOGGER.info(
            f"  span(Z) indicators {tr['indicators_spanZ'].median():.3f} (capped by bank)"
            f"  | dictionary fix {tr['dictionary_fix'].median():.3f}"
            f" (ceiling attained: {tr['dictionary_fix'].median() / max(tr['ceiling_K32'].median(), 1e-12):.1%})"
        )
    LOGGER.info(f"\nCSV + JSON -> {args.out}")


if __name__ == "__main__":
    main()
