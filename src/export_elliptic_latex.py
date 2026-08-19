"""Turn a ``run_elliptic_modular`` output directory into pgfplots-ready CSVs.

The shared artifacts (training trace, Ward sweep, per-gang diagnostics) are
exported by :mod:`src.export_paper_csv`; this driver adds what only the
Elliptic\\texttt{++} run produces --- the per-day transfer table for the frozen
filter, and the licit-group selectivity control.

Usage::

    python -m src.export_elliptic_latex <run_dir> [--out <tex_dir>] [--every N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.export_paper_csv import (
    Writer,
    binned_rate,
    write_capture,
    write_diagnostics,
    write_pr_sweep,
    write_sweep_summary,
    write_training,
)


def export(run: Path, out: Path, every: int) -> None:
    w = Writer(out, "ell")

    hits = sorted(run.glob("elliptic_modular_d*.json"))
    if not hits:
        raise SystemExit(f"no elliptic_modular_d*.json in {run}")
    report = json.loads(hits[0].read_text())

    # ---- shared artifacts ---------------------------------------------------
    write_training(run, w, every)
    write_pr_sweep(run, w)
    write_sweep_summary(run, w, label="graph")
    diag = write_diagnostics(run, w, unit="gang", stem="gangs")
    write_capture(report.get("captures") or {}, w)

    # ---- headline splits on the reported (training) day range --------------
    rows = []
    for split in ("train", "test", "all"):
        r = (report.get("report") or {}).get(split)
        if r:
            rows.append({"split": split, "gangs": r["total"], "recall": r["mean_recall"],
                         "precision": r["mean_precision"], "f1": r["mean_f1"],
                         "det": r["detection_rate"]})
    w("splits", pd.DataFrame(rows), "detection per split on the training day range")

    # ---- the shared filter on each TRAINING day's own graph -----------------
    per_group = report.get("per_group_report") or {}
    if per_group:
        w("train_days", pd.DataFrame([
            {"day": lbl, "gangs": r["total"], "recall": r["mean_recall"],
             "precision": r["mean_precision"], "f1": r["mean_f1"],
             "det": r["detection_rate"]}
            for lbl, r in per_group.items()
        ]), "shared filter on each training day's own graph")

    # ---- per-day transfer of the FROZEN filter ------------------------------
    days = (report.get("transfer") or {}).get("days") or []
    trows = []
    for rec in days:
        r = rec.get("report")
        if not r:
            continue
        row = {"day": rec["day"], "n_nodes": rec["n_nodes"], "gangs": r["total"],
               "recall": r["mean_recall"], "precision": r["mean_precision"],
               "f1": r["mean_f1"], "det": r["detection_rate"]}
        co = rec.get("coarsening") or {}
        row["n_coarse"] = co.get("n_coarse")
        row["epsilon"] = co.get("epsilon")
        ps = rec.get("pr_sweep") or {}
        row["pr_auc"] = ps.get("pr_auc")
        trows.append(row)
    if trows:
        w("transfer", pd.DataFrame(trows),
          "per-day transfer of the frozen filter (day, size, metrics, PR-AUC)")

    # ---- detection rate by gang size ----------------------------------------
    if diag is not None and "size" in diag:
        binned_rate(diag, "size",
                    edges=[0, 3, 5, 10, 25, 10 ** 9],
                    labels=["2-3", "4-5", "6-10", "11-25", "26+"],
                    w=w, stem="by_size",
                    what="detection rate by gang size band")
        binned_rate(diag, "Phi",
                    edges=[-0.001, 0.2, 0.35, 0.5, 0.65, 1.001],
                    labels=["0-.20", ".20-.35", ".35-.50", ".50-.65", ".65-1"],
                    w=w, stem="by_phi",
                    what="detection rate by gang conductance band")

    # ---- selectivity: illicit gangs vs licit control groups -----------------
    apath = run / "analysis.json"
    if apath.exists():
        a = json.loads(apath.read_text())
        ng = a.get("nongang") or {}
        cond = a.get("conductance") or {}
        dvm = a.get("gang_conductance_detected_vs_missed") or {}
        gang = (report.get("report") or {}).get("all") or {}
        w("selectivity", pd.DataFrame([
            {"group": "illicit gangs", "n": gang.get("total"),
             "recall": gang.get("mean_recall"), "precision": gang.get("mean_precision"),
             "collapse_rate": gang.get("detection_rate")},
            {"group": "licit control", "n": ng.get("n_normals"),
             "recall": ng.get("normal_mean_recall"),
             "precision": ng.get("normal_mean_precision"),
             "collapse_rate": ng.get("normal_false_collapse_rate")},
        ]), "selectivity: illicit gangs vs licit control groups")
        w("conductance", pd.DataFrame([
            {"quantity": "gang supernodes", "median_phi": cond.get("gang_median_conductance"),
             "n": cond.get("n_gang_supernodes")},
            {"quantity": "background supernodes",
             "median_phi": cond.get("background_median_conductance"),
             "n": cond.get("n_background_supernodes")},
            {"quantity": "detected gangs", "median_phi": dvm.get("detected_median"), "n": None},
            {"quantity": "missed gangs", "median_phi": dvm.get("missed_median"), "n": None},
        ]), "supernode / gang conductance medians")

    w.report()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path, help="a run_elliptic_modular output dir")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--every", type=int, default=10,
                    help="subsample training curves to every N-th epoch")
    a = ap.parse_args()
    export(a.run, a.out or (a.run / "latex"), a.every)


if __name__ == "__main__":
    main()
