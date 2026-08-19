"""Supervised GNN node-classifier baseline for Elliptic++ gang detection.

The "node classification + post-hoc grouping" paradigm the coarsening method is
argued against: a class-weighted 2-layer GCN predicts illicit probability per
node; predicted-illicit nodes are grouped into connected components (>= the same
min gang size); each component plays the role of a supernode and is scored with
the *identical* gang criterion as the coarsening pipeline
(:func:`evaluate_loukas_patterns`, recall & precision > threshold), so the
tables are row-for-row comparable with :mod:`src.run_elliptic_modular`.

Fairness protocol (mirrors the pipeline run):
* same day window, same gang definition (illicit CC >= min size), same seed and
  ``split_train_test`` call order  -> identical train/test gang split;
* supervision = the TRAIN gangs' nodes as positives + the licit (class 2) nodes
  as negatives; test-gang / other-illicit / unknown nodes are never trained on;
* class imbalance handled by inverse-frequency class weights in the CE loss
  (all licit negatives are used rather than subsampling);
* the probability threshold for "illicit" is chosen by TRAIN-gang mean F1 on the
  training day only, then frozen (like the filter) for test gangs and the
  transfer days, whose labels are used only for scoring.

Run::

    conda activate FedStruct
    python -m src.run_elliptic_gnn_baseline --day-start 25 --day-end 25
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
import torch.nn.functional as F
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import average_precision_score, roc_auc_score

from src.loukas_sgc_detection import evaluate_loukas_patterns, graph_operators
from src.propagation_encoders import GCN2Encoder
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import (
    build_torch_graph,
    load_node_features,
    make_patterns,
    split_train_test,
)
from src.utils.utils import LOGGER, now


# --------------------------------------------------------------------------- #
# prediction -> components -> gang scoring
# --------------------------------------------------------------------------- #
def components_to_supernodes(
    A_unw, pred_mask: np.ndarray, min_size: int
) -> torch.Tensor:
    """Node->supernode map: each predicted-illicit component (>= min_size) is one
    supernode, every remaining node stays a singleton -- the exact analogue of a
    coarsening that collapsed only the predicted gangs."""

    N = int(A_unw.shape[0])
    n2s = np.arange(N, dtype=np.int64)  # singletons by default
    idx = np.nonzero(pred_mask)[0]
    if len(idx):
        sub = A_unw[idx][:, idx]
        n_comp, lab = connected_components(sub, directed=False)
        next_id = N
        for c in range(n_comp):
            members = idx[lab == c]
            if len(members) >= min_size:
                n2s[members] = next_id
                next_id += 1
    # compress ids to 0..n-1 (evaluate_loukas_patterns only groups by id)
    _, n2s = np.unique(n2s, return_inverse=True)
    return torch.as_tensor(n2s, dtype=torch.long)


def score_gangs(gangs, n2s: torch.Tensor, y: torch.Tensor, threshold: float) -> dict:
    res, by_label = evaluate_loukas_patterns(gangs, n2s, y, threshold=threshold)
    a = by_label.get("alert", {})
    f1 = float(np.mean([r.f1 for r in res])) if res else 0.0
    return {
        "detection_rate": a.get("detection_rate", 0.0),
        "mean_recall": a.get("mean_recall", 0.0),
        "mean_precision": a.get("mean_precision", 0.0),
        "mean_f1": f1,
        "detected": int(a.get("detected", 0)),
        "total": int(a.get("total", 0)),
    }


def evaluate_at_threshold(
    prob: np.ndarray, A_unw, gangs, y, prob_thr: float, min_size: int, det_thr: float
) -> dict:
    n2s = components_to_supernodes(A_unw, prob >= prob_thr, min_size)
    return score_gangs(gangs, n2s, y, det_thr)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train_gcn(
    a_hat, X, train_idx: torch.Tensor, y_train: torch.Tensor, *,
    embed_dim: int, epochs: int, lr: float, seed: int,
) -> GCN2Encoder:
    """Class-weighted 2-layer GCN (all licit negatives; inverse-freq weights)."""

    torch.manual_seed(seed)
    enc = GCN2Encoder(in_dim=X.shape[1], embed_dim=embed_dim, num_classes=2).to(
        dtype=X.dtype
    )
    opt = torch.optim.Adam(enc.parameters(), lr=lr, weight_decay=5e-4)
    counts = torch.bincount(y_train, minlength=2).to(dtype=X.dtype)
    w = (counts.sum() / counts.clamp_min(1.0)) / 2.0  # inverse-frequency weights
    LOGGER.info(
        f"  train nodes: {len(train_idx):,} "
        f"({int((y_train == 1).sum()):,} gang, {int((y_train == 0).sum()):,} licit) "
        f"class weights: licit {float(w[0]):.3f} / gang {float(w[1]):.3f}"
    )
    for ep in range(epochs):
        enc.train()
        opt.zero_grad(set_to_none=True)
        _, logits = enc(a_hat, X)
        loss = F.cross_entropy(logits[train_idx], y_train, weight=w)
        loss.backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            LOGGER.info(f"    epoch {ep + 1:>4}: CE = {float(loss):.4f}")
    return enc


@torch.no_grad()
def predict_prob(enc: GCN2Encoder, a_hat, X) -> np.ndarray:
    enc.eval()
    _, logits = enc(a_hat, X)
    return torch.softmax(logits, dim=1)[:, 1].cpu().numpy()


def node_level_metrics(prob: np.ndarray, cls: np.ndarray) -> dict:
    """Node AUC / PR-AUC of illicit vs licit (labelled nodes only) and vs rest."""

    out = {}
    lab = np.isin(cls, (1, 2))
    y_ill = (cls == 1).astype(int)
    try:
        out["auc_vs_licit"] = float(roc_auc_score(y_ill[lab], prob[lab]))
        out["prauc_vs_licit"] = float(average_precision_score(y_ill[lab], prob[lab]))
        out["auc_vs_all"] = float(roc_auc_score(y_ill, prob))
        out["prauc_vs_all"] = float(average_precision_score(y_ill, prob))
    except ValueError:
        pass
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--day-start", type=int, default=25)
    ap.add_argument("--day-end", type=int, default=25)
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--train-ratio", type=float, default=0.4)
    ap.add_argument("--transfer-days", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.51, help="gang det. thr")
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--compare-run",
        type=Path,
        default=None,
        help="an elliptic_modular results dir; prints its numbers side by side",
    )
    ap.add_argument("--out", default=Path(f"results/elliptic_gnn_baseline/{now}/"), type=Path)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    # --- 1. training-day graph, gangs, identical split ----------------------
    LOGGER.info(f"=== GNN baseline | days {args.day_start}-{args.day_end} ===")
    A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, args.day_start, args.day_end)
    Xfeat, feature_columns = load_node_features(
        args.data_dir, nodes_df, args.day_start, args.day_end, return_columns=True
    )
    graph = build_torch_graph(A_w, A_unw, cls, Xfeat, weighted=False)
    a_hat, _ = graph_operators(graph)
    y = graph.y

    illicit_idx = np.where(cls == 1)[0]
    gang_sets = connected_components_sets(A_unw, illicit_idx, args.min_gang_size)
    gangs = make_patterns(gang_sets, "alert", "gang", "g")
    rng = np.random.default_rng(args.seed)  # same call order as the pipeline
    gang_train, gang_test = split_train_test(gangs, args.train_ratio, rng)
    LOGGER.info(
        f"  gangs: {len(gangs)} (train {len(gang_train)} / test {len(gang_test)})"
    )

    # --- 2. supervision: train-gang nodes vs ALL licit nodes ----------------
    pos = sorted({int(v) for p in gang_train for v in p.node_indices})
    neg = np.nonzero(cls == 2)[0]  # licit; unknown & non-train illicit excluded
    train_idx = torch.as_tensor(np.r_[pos, neg], dtype=torch.long)
    y_train = torch.zeros(len(train_idx), dtype=torch.long)
    y_train[: len(pos)] = 1

    enc = train_gcn(
        a_hat, graph.x, train_idx, y_train,
        embed_dim=args.embed_dim, epochs=args.epochs,
        lr=args.learning_rate, seed=args.seed,
    )
    prob = predict_prob(enc, a_hat, graph.x)
    nm = node_level_metrics(prob, cls)
    LOGGER.info(
        f"  node-level (day {args.day_start}-{args.day_end}): "
        f"AUC vs licit {nm.get('auc_vs_licit', float('nan')):.3f}  "
        f"PR-AUC vs licit {nm.get('prauc_vs_licit', float('nan')):.3f}  "
        f"AUC vs all {nm.get('auc_vs_all', float('nan')):.3f}"
    )

    # --- 3. probability threshold: TRAIN-gang F1 only, then frozen ----------
    # coarse grid + a fine tail near 1: with 34x class weights the calibrated
    # probabilities pile up near 1, so the useful cuts live in [0.95, 1).
    thr_grid = np.r_[np.round(np.arange(0.05, 0.95, 0.05), 2),
                     0.95, 0.97, 0.99, 0.995, 0.999]
    sweep = []
    for t in thr_grid:
        r_tr = evaluate_at_threshold(
            prob, A_unw, gang_train, y, t, args.min_gang_size, args.threshold
        )
        sweep.append({"prob_thr": float(t), **{f"train_{k}": v for k, v in r_tr.items()}})
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(args.out / "threshold_sweep_train.csv", index=False)
    best = sweep_df.loc[sweep_df.train_mean_f1.idxmax()]
    prob_thr = float(best.prob_thr)
    LOGGER.info(
        f"  prob threshold (best TRAIN-gang F1): {prob_thr:.2f} "
        f"(train F1 {best.train_mean_f1:.3f}, det {best.train_detection_rate:.1%})"
    )

    report = {
        name: evaluate_at_threshold(
            prob, A_unw, pats, y, prob_thr, args.min_gang_size, args.threshold
        )
        for name, pats in (("train", gang_train), ("test", gang_test), ("all", gangs))
    }

    LOGGER.info("\n" + "=" * 74)
    LOGGER.info(
        f"GNN BASELINE  days {args.day_start}-{args.day_end}  "
        f"N={graph.num_nodes:,}  {len(gangs)} gangs  prob_thr={prob_thr:.2f}"
    )
    LOGGER.info("=" * 74)
    hdr = f"  {'split':<6} {'recall':>8} {'precision':>10} {'f1':>7} {'detection':>10} {'det/tot':>10}"
    LOGGER.info(hdr)
    LOGGER.info("  " + "-" * (len(hdr) - 2))
    for name in ("train", "test", "all"):
        r = report[name]
        LOGGER.info(
            f"  {name:<6} {r['mean_recall']:>8.3f} {r['mean_precision']:>10.3f} "
            f"{r['mean_f1']:>7.3f} {r['detection_rate']:>10.1%} "
            f"{r['detected']:>4}/{r['total']:<5}"
        )

    # --- 4. transfer: frozen GCN + frozen threshold on the next N days ------
    transfer_records = []
    for k in range(1, args.transfer_days + 1):
        day = args.day_end + k
        LOGGER.info(f"\n--- transfer day {day} ---")
        A_unw_d, A_w_d, cls_d, nodes_d = build_graph(args.data_dir, day, day)
        X_d = load_node_features(
            args.data_dir, nodes_d, day, day, keep_columns=feature_columns
        )
        graph_d = build_torch_graph(A_w_d, A_unw_d, cls_d, X_d, weighted=False)
        a_hat_d, _ = graph_operators(graph_d)
        ill_d = np.where(cls_d == 1)[0]
        gsets_d = connected_components_sets(A_unw_d, ill_d, args.min_gang_size)
        gangs_d = make_patterns(gsets_d, "alert", "gang", "g")
        rec = {"day": day, "n_nodes": graph_d.num_nodes, "n_gangs": len(gangs_d),
               "report": None, "node_metrics": None}
        if gangs_d:
            prob_d = predict_prob(enc, a_hat_d, graph_d.x)
            rec["node_metrics"] = node_level_metrics(prob_d, cls_d)
            rec["report"] = evaluate_at_threshold(
                prob_d, A_unw_d, gangs_d, graph_d.y, prob_thr,
                args.min_gang_size, args.threshold,
            )
            r = rec["report"]
            LOGGER.info(
                f"  gangs={len(gangs_d)}  recall={r['mean_recall']:.3f} "
                f"precision={r['mean_precision']:.3f} f1={r['mean_f1']:.3f} "
                f"detection={r['detection_rate']:.1%} ({r['detected']}/{r['total']})  "
                f"node-AUC(vs licit)={rec['node_metrics'].get('auc_vs_licit', float('nan')):.3f}"
            )
        else:
            LOGGER.info("  no gangs on this day -> skipping")
        transfer_records.append(rec)

    scored = [r for r in transfer_records if r["report"] is not None]
    transfer_summary = None
    if scored:
        keys = ("mean_recall", "mean_precision", "mean_f1", "detection_rate")
        avg = {k: float(np.mean([r["report"][k] for r in scored])) for k in keys}
        transfer_summary = {
            "n_days_scored": len(scored), **avg,
            "detected": int(sum(r["report"]["detected"] for r in scored)),
            "total": int(sum(r["report"]["total"] for r in scored)),
        }

    LOGGER.info("\n" + "=" * 74)
    LOGGER.info("PER-DAY TRANSFER (frozen GCN, frozen prob threshold)")
    LOGGER.info("=" * 74)
    thdr = (
        f"  {'day':<5} {'nodes':>8} {'gangs':>6} {'recall':>8} {'precision':>10} "
        f"{'f1':>7} {'detection':>10} {'det/tot':>10}"
    )
    LOGGER.info(thdr)
    LOGGER.info("  " + "-" * (len(thdr) - 2))
    for rec in transfer_records:
        r = rec["report"]
        if r is None:
            LOGGER.info(f"  {rec['day']:<5} {rec['n_nodes']:>8,} {rec['n_gangs']:>6} " + " ".join(["--"] * 5))
            continue
        LOGGER.info(
            f"  {rec['day']:<5} {rec['n_nodes']:>8,} {rec['n_gangs']:>6} "
            f"{r['mean_recall']:>8.3f} {r['mean_precision']:>10.3f} "
            f"{r['mean_f1']:>7.3f} {r['detection_rate']:>10.1%} "
            f"{r['detected']:>4}/{r['total']:<5}"
        )
    if transfer_summary:
        LOGGER.info("  " + "-" * (len(thdr) - 2))
        LOGGER.info(
            f"  {'avg':<5} {'':>8} {'':>6} "
            f"{transfer_summary['mean_recall']:>8.3f} "
            f"{transfer_summary['mean_precision']:>10.3f} "
            f"{transfer_summary['mean_f1']:>7.3f} "
            f"{transfer_summary['detection_rate']:>10.1%} "
            f"{transfer_summary['detected']:>4}/{transfer_summary['total']:<5}"
        )

    # --- 5. side-by-side with a coarsening pipeline run ---------------------
    comparison = None
    if args.compare_run is not None:
        cand = sorted(args.compare_run.glob("elliptic_modular_d*.json"))
        if cand:
            pipe = json.loads(cand[0].read_text())
            comparison = {"pipeline_run": str(args.compare_run)}
            LOGGER.info("\n" + "=" * 74)
            LOGGER.info(f"COMPARISON  (pipeline run: {args.compare_run})")
            LOGGER.info("=" * 74)
            chdr = (
                f"  {'method':<22} {'split':<10} {'recall':>8} {'precision':>10} "
                f"{'f1':>7} {'detection':>10} {'det/tot':>9}"
            )
            LOGGER.info(chdr)
            LOGGER.info("  " + "-" * (len(chdr) - 2))

            def _row(method, split, r):
                if r is None:
                    return
                LOGGER.info(
                    f"  {method:<22} {split:<10} {r['mean_recall']:>8.3f} "
                    f"{r['mean_precision']:>10.3f} {r['mean_f1']:>7.3f} "
                    f"{r['detection_rate']:>10.1%} "
                    f"{r['detected']:>4}/{r['total']:<4}"
                )

            for split in ("train", "test", "all"):
                _row("coarsening (ward)", split, pipe["report"].get(split))
                _row("GNN baseline", split, report.get(split))
                LOGGER.info("")
            _row("coarsening (ward)", "transfer", pipe.get("transfer", {}).get("average"))
            _row("GNN baseline", "transfer", transfer_summary)
            comparison["pipeline_report"] = pipe["report"]
            comparison["pipeline_transfer_average"] = pipe.get("transfer", {}).get("average")

    payload = {
        "dataset": "elliptic++",
        "method": "gnn_node_classifier_components",
        "day_start": args.day_start,
        "day_end": args.day_end,
        "n_nodes": graph.num_nodes,
        "feature_dim": int(graph.x.shape[1]),
        "n_gangs": len(gangs),
        "n_train_gangs": len(gang_train),
        "embed_dim": args.embed_dim,
        "epochs": args.epochs,
        "prob_threshold": prob_thr,
        "detection_threshold": args.threshold,
        "node_metrics_train_day": nm,
        "report": report,
        "transfer": {
            "days": transfer_records,
            "average": transfer_summary,
        },
        "comparison": comparison,
    }
    out_json = args.out / f"gnn_baseline_d{args.day_start}-{args.day_end}.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    LOGGER.info(f"\nJSON report: {out_json}")


if __name__ == "__main__":
    main()
