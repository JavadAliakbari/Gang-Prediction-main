"""Does the closed-form target generalize?  Capture on train vs held-out gangs.

The closed-form collective solve (Theorem A) maximizes ``lambda_min(Gamma)`` over
the *training* gangs exactly, yet detects worse than the gradient-trained bank
even on the epsilon-free PR frontier.  Two hypotheses explain that:

* **overfitting** -- ``Theta_beta = G_beta^{-1} Bhat`` has ``m (K+1) d`` free
  parameters fitted to ``m`` indicators, so its columns capture the training
  gangs and nothing else;
* **localization** -- each column is essentially ``Pi v_hat_{S_j}``, concentrated
  on gang ``j`` and ~0 elsewhere, so the target gives Ward no geometry on the
  rest of the graph (merging zero-rows costs no RSA distortion, which is why its
  epsilon stays tiny while it coarsens deeply).

Both are measured here, for the closed-form target and the bank side by side:

1. per-gang capture on the TRAIN gangs, the HELD-OUT gangs of the same day, and
   every gang of an unseen transfer day (frozen coefficients, no refit);
2. the row-norm profile of the target -- what fraction of the graph's nodes the
   target actually "sees" (norm above a fraction of the max), and how much of
   the total row mass sits on the training gangs' nodes.

A large train/test capture ratio implicates overfitting; a tiny seen-fraction
implicates localization.  They are not exclusive.

Run::

    conda activate FedStruct
    python -m src.run_target_generalization --day-start 26 --day-end 26
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
from src.margin_pencil import apply_dictionary_theta, collective_pencil_theta
from src.run_capture_ceiling import subspace_captures
from src.run_collective_bank_detection import _l_apply, degree_weighted_indicators
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import (
    build_torch_graph,
    load_node_features,
    make_patterns,
    random_structural_features,
    split_train_test,
)
from src.utils.utils import LOGGER, now


def per_gang_capture(a_hat, adjacency, gangs, R, tau) -> np.ndarray:
    """Rank-revealing per-gang capture of ``span(R)``."""

    V = degree_weighted_indicators(adjacency, gangs).to(a_hat.dtype)
    phi = (V * _l_apply(a_hat, V)).sum(0).clamp_min(1e-300)
    cap, _ = subspace_captures(a_hat, R, V, phi, tau)
    return cap


def row_profile(R: torch.Tensor, gang_nodes: set, top_frac: float = 0.01) -> dict:
    """How much of the graph does this target actually see?

    ``seen_frac`` is the fraction of nodes whose row norm exceeds ``top_frac`` of
    the largest row norm; ``mass_on_train`` is the share of total squared row
    mass sitting on the training gangs' nodes (their share of nodes is given for
    reference -- a target that merely tracks the gangs would match it).
    """

    nrm = R.norm(dim=1)
    n = int(nrm.numel())
    thresh = float(nrm.max()) * top_frac
    idx = torch.as_tensor(sorted(gang_nodes), dtype=torch.long)
    sq = nrm.pow(2)
    return {
        "seen_frac": float((nrm > thresh).to(torch.float64).mean()),
        "mass_on_train_gangs": float(sq[idx].sum() / sq.sum().clamp_min(1e-300)),
        "train_gang_node_frac": len(gang_nodes) / n,
        "median_over_max_rownorm": float(nrm.median() / nrm.max().clamp_min(1e-300)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--day-start", type=int, default=26)
    ap.add_argument("--day-end", type=int, default=26)
    ap.add_argument("--transfer-day", type=int, default=27)
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--train-ratio", type=float, default=0.6)
    ap.add_argument("--feature-mode", default="wallet+random",
                    choices=["wallet", "random", "wallet+random"])
    ap.add_argument("--random-width", type=int, default=64)
    ap.add_argument("--degree", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--pencil-beta", type=float, default=10.0)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--conf-weight", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=Path(f"results/target_generalization/{now}/"),
                    type=Path)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    def load_day(d0, d1, keep_cols=None):
        A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, d0, d1)
        if args.feature_mode in ("wallet", "wallet+random"):
            out = load_node_features(args.data_dir, nodes_df, d0, d1,
                                     keep_columns=keep_cols,
                                     return_columns=keep_cols is None)
            X, cols = out if keep_cols is None else (out, keep_cols)
            if args.feature_mode == "wallet+random":
                X = torch.cat([X, random_structural_features(
                    int(A_unw.shape[0]), args.random_width, args.seed)], dim=1)
        else:
            X = random_structural_features(int(A_unw.shape[0]), args.random_width,
                                           args.seed)
            cols = None
        g = build_torch_graph(A_w, A_unw, cls, X, weighted=False)
        gangs = make_patterns(
            connected_components_sets(A_unw, np.where(cls == 1)[0],
                                      args.min_gang_size),
            "alert", "gang", "g")
        return GraphData.from_graph(g), gangs, cols

    LOGGER.info(f"=== target generalization | train day {args.day_start}-"
                f"{args.day_end}, transfer day {args.transfer_day} ===")
    data, gangs, cols = load_day(args.day_start, args.day_end)
    rng = np.random.default_rng(args.seed)
    gang_train, gang_test = split_train_test(gangs, args.train_ratio, rng)
    train_nodes = {int(v) for p in gang_train for v in p.node_indices}
    LOGGER.info(f"  {len(gangs)} gangs: {len(gang_train)} train / "
                f"{len(gang_test)} held-out, d={data.feature_dim}")

    # ---- the two targets ---------------------------------------------------
    LOGGER.info(f"\n  [closed-form] Theta_beta, beta={args.pencil_beta} ...")
    theta_p, rep = collective_pencil_theta(
        data.a_hat, data.adjacency, data.X, gang_train,
        degree=args.degree, tau=args.tau, beta=args.pencil_beta,
    )
    R_pencil = data.X.new_tensor([]) if False else apply_dictionary_theta(
        data.a_hat, data.X, theta_p, degree=args.degree, tau=args.tau)
    LOGGER.info(f"    lambda_min(Gamma)={rep['lambda_min_Gamma']:.4f} "
                f"(N_0={rep['lambda_min_N0']:.4f}), width={R_pencil.shape[1]}")

    LOGGER.info(f"  [bank] gradient fit, {args.epochs} epochs, "
                f"{args.heads} heads ...")
    cfg = DetectorConfig(
        degree=args.degree, basis="chebyshev", tau=args.tau, epochs=args.epochs,
        learning_rate=0.02, ridge=1e-3, optimizer="projected",
        capture_objective="lambda_min", conf_weight=args.conf_weight,
        conf_reduce="mean", heads=args.heads, coarsen_target="bank",
        seed=args.seed,
    )
    det = CollectiveBankDetector(cfg).fit(data, gang_train)
    R_bank = det.target_subspace(data, gang_train)
    LOGGER.info(f"    lambda_min(Gamma)={det.fit_info_['objective']:.4f}, "
                f"width={R_bank.shape[1]}")

    targets = {"closed-form": R_pencil, "bank": R_bank}

    # ---- 1. capture: train vs held-out vs unseen day -----------------------
    rows = []
    for name, R in targets.items():
        cap = per_gang_capture(data.a_hat, data.adjacency, gangs, R, args.tau)
        tr_ids = {p.id for p in gang_train}
        for p, c in zip(gangs, cap):
            rows.append({"target": name, "day": args.day_start, "gang": p.id,
                         "size": p.num_nodes,
                         "split": "train" if p.id in tr_ids else "held-out",
                         "capture": float(c)})

    # unseen transfer day: frozen coefficients, no refit
    data_t, gangs_t, _ = load_day(args.transfer_day, args.transfer_day, cols)
    R_pencil_t = apply_dictionary_theta(data_t.a_hat, data_t.X, theta_p,
                                        degree=args.degree, tau=args.tau)
    R_bank_t = det.target_subspace(data_t, gang_train)
    for name, R in (("closed-form", R_pencil_t), ("bank", R_bank_t)):
        cap = per_gang_capture(data_t.a_hat, data_t.adjacency, gangs_t, R, args.tau)
        for p, c in zip(gangs_t, cap):
            rows.append({"target": name, "day": args.transfer_day, "gang": p.id,
                         "size": p.num_nodes, "split": "transfer",
                         "capture": float(c)})
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "capture_by_split.csv", index=False)

    LOGGER.info("\n" + "=" * 86)
    LOGGER.info("CAPTURE BY SPLIT (median per-gang retained M_tau-energy)")
    LOGGER.info("=" * 86)
    hdr = (f"  {'target':<14}{'train':>12}{'held-out':>12}{'transfer':>12}"
           f"{'held-out/train':>17}{'transfer/train':>16}")
    LOGGER.info(hdr)
    LOGGER.info("  " + "-" * (len(hdr) - 2))
    summary = {}
    for name in targets:
        s = df[df.target == name]
        tr = float(s[s.split == "train"].capture.median())
        te = float(s[s.split == "held-out"].capture.median())
        tf = float(s[s.split == "transfer"].capture.median())
        summary[name] = {"train": tr, "held_out": te, "transfer": tf,
                         "held_out_over_train": te / max(tr, 1e-300),
                         "transfer_over_train": tf / max(tr, 1e-300)}
        LOGGER.info(f"  {name:<14}{tr:>12.4f}{te:>12.4f}{tf:>12.4f}"
                    f"{te / max(tr, 1e-300):>17.3f}{tf / max(tr, 1e-300):>16.3f}")

    # ---- 2. localization ---------------------------------------------------
    LOGGER.info("\n" + "=" * 86)
    LOGGER.info("TARGET LOCALIZATION (does the target see the rest of the graph?)")
    LOGGER.info("=" * 86)
    hdr2 = (f"  {'target':<14}{'width':>7}{'seen frac':>12}{'median/max':>13}"
            f"{'mass on train gangs':>22}{'their node frac':>17}")
    LOGGER.info(hdr2)
    LOGGER.info("  " + "-" * (len(hdr2) - 2))
    for name, R in targets.items():
        pr = row_profile(R, train_nodes)
        summary[name].update(pr)
        LOGGER.info(
            f"  {name:<14}{R.shape[1]:>7}{pr['seen_frac']:>12.4f}"
            f"{pr['median_over_max_rownorm']:>13.2e}"
            f"{pr['mass_on_train_gangs']:>22.4f}"
            f"{pr['train_gang_node_frac']:>17.5f}"
        )
    LOGGER.info(
        "\n  'seen frac' = nodes with row norm > 1% of the max.  A target that is "
        "\n  near-zero off the training gangs gives Ward no metric there: merging "
        "\n  those rows costs ~no RSA distortion, which is why epsilon stays tiny."
    )

    (args.out / "target_generalization.json").write_text(
        json.dumps({"beta": args.pencil_beta, "epochs": args.epochs,
                    "lambda_min_pencil": rep["lambda_min_Gamma"],
                    "lambda_min_N0": rep["lambda_min_N0"],
                    "lambda_min_bank": det.fit_info_["objective"],
                    "summary": summary}, indent=2, default=float) + "\n")
    LOGGER.info(f"\nCSV + JSON -> {args.out}")


if __name__ == "__main__":
    main()
