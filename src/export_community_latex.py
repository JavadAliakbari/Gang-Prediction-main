"""Turn ``run_graph_fraud_gang_detection`` runs into pgfplots-ready CSVs.

Handles several datasets at once (com-Amazon / com-DBLP / com-Youtube) and emits
both per-dataset series (training trace, Ward sweep, per-community scatter) and
*combined* tables with a ``dataset`` column, so one pgfplots axis can overlay the
three curves.  The shared writers live in :mod:`src.export_paper_csv`.

Usage::

    python -m src.export_community_latex <parent_dir> --datasets comamazon,comdblp,comyoutube
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.export_paper_csv import Writer, subsample, tex_safe, write_diagnostics

# what the community diagnostics record (no capture/star-ness; overlap instead)
COMMUNITY_STATS = ("size", "Phi", "mbar1", "density", "overlap_mean", "overlap_max")

PRETTY = {"comamazon": "com-Amazon", "comdblp": "com-DBLP",
          "comyoutube": "com-Youtube", "comlj": "com-LiveJournal",
          "comorkut": "com-Orkut"}


def _encoder_row(report: dict) -> dict:
    """The (single) encoder block of a run's JSON report."""
    encs = report.get("encoders") or []
    return encs[0] if encs else {}


def export(parent: Path, datasets: list[str], out: Path, every: int) -> None:
    w = Writer(out, "com")
    summary, by_size, selectivity, aucs, describe = [], [], [], [], []

    for ds in datasets:
        run = parent / f"com_{ds}"
        if not run.exists():
            run = parent / ds
        jpath = run / f"gang_detection_{ds}.json"
        if not jpath.exists():
            print(f"  ! skipping {ds}: no {jpath.name}")
            continue
        report = json.loads(jpath.read_text())
        enc = _encoder_row(report)
        name = PRETTY.get(ds, ds)

        # ---- per-dataset training trace ------------------------------------
        hist_path = run / f"training_history_{ds}.csv"
        if hist_path.exists():
            hist = pd.read_csv(hist_path)
            cols = [c for c in ("epoch", "lambda_min", "capture_mean", "capture_min",
                                "confusability", "objective", "loss") if c in hist]
            h = subsample(hist[cols], every).copy()
            # normalised loss so the three datasets share one axis despite very
            # different absolute scales
            lo, hi = h["loss"].min(), h["loss"].max()
            h["loss_norm"] = (h["loss"] - lo) / (hi - lo) if hi > lo else 0.0
            w(f"{ds}_training", h, f"{name}: training trace")

        # ---- per-dataset Ward sweep ----------------------------------------
        sweeps = sorted(run.glob(f"pr_sweep_{ds}_*.csv"))
        if sweeps:
            sw = pd.read_csv(sweeps[0])
            ek = "epsilon_exact" if "epsilon_exact" in sw else "epsilon"
            keep = [c for c in (ek, "n_coarse", "mean_recall", "mean_precision",
                                "mean_f1", "mean_jaccard", "det_rate") if c in sw]
            sw = sw[keep].sort_values(ek)
            if ek != "epsilon_exact":  # one column name for the .tex either way
                sw = sw.rename(columns={ek: "epsilon_exact"})
            w(f"{ds}_pr_sweep", subsample(sw, max(1, len(sw) // 200)),
              f"{name}: full Ward sweep")

        # ---- per-community diagnostics + AUC --------------------------------
        diag_files = sorted(run.glob(f"community_diag_{ds}_*.csv"))
        diag = None
        if diag_files:
            wd = Writer(out, f"com_{ds}")
            diag = write_diagnostics(
                diag_files[0].parent, wd, unit="community", stem="communities",
                filename=diag_files[0].name, stats=COMMUNITY_STATS,
                id_cols=("community",),
            )
            w.written.extend(wd.written)
            if diag is not None:
                a = pd.read_csv(out / f"com_{ds}_auc.csv")
                a.insert(0, "dataset", name)
                aucs.append(a)

        # ---- combined: detection by size band -------------------------------
        bs = sorted(run.glob(f"by_size_{ds}_*.csv"))
        if bs:
            b = pd.read_csv(bs[0])
            # per-dataset file too: filtering a combined table by a string column
            # inside pgfplots is fragile, so each curve gets its own table
            b.insert(0, "idx", range(len(b)))
            w(f"{ds}_by_size", b, f"{name}: detection rate by size band")
            b2 = b.drop(columns="idx").copy()
            b2.insert(0, "dataset", name)
            by_size.append(b2)

        # ---- combined: selectivity -----------------------------------------
        if "control_collapse_rate" in enc:
            selectivity.append({
                "dataset": name,
                "communities": enc.get("n_gangs"),
                "community_rate": enc.get("det_rate"),
                "controls": enc.get("n_controls"),
                "control_rate": enc.get("control_collapse_rate"),
                "gap": enc.get("selectivity_gap"),
            })

        # ---- combined: headline summary -------------------------------------
        sizes = diag["size"] if diag is not None else pd.Series(dtype=float)
        summary.append({
            "dataset": name,
            "n_nodes": report.get("n_nodes"),
            "n_edges": report.get("n_edges"),
            "communities": enc.get("n_gangs"),
            "n_coarse": enc.get("n_coarse"),
            "det_rate": enc.get("det_rate"),
            "f1": enc.get("meanF1_all"),
            "jaccard": enc.get("meanJac_all"),
            "recall": enc.get("gang_mean_recall"),
            "precision": enc.get("gang_mean_precision"),
            "f1_top5": enc.get("meanF1_top5"),
            "control_rate": enc.get("control_collapse_rate"),
            "gap": enc.get("selectivity_gap"),
            "phi_community": enc.get("supernode_phi_community_median"),
            "phi_background": enc.get("supernode_phi_background_median"),
        })

        # ---- dataset description -------------------------------------------
        describe.append({
            "dataset": name,
            "n_nodes": report.get("n_nodes"),
            "n_edges": report.get("n_edges"),
            "communities_total": report.get("n_communities_total"),
            "evaluated": enc.get("n_gangs"),
            "size_median": float(sizes.median()) if len(sizes) else np.nan,
            "size_mean": float(sizes.mean()) if len(sizes) else np.nan,
            "size_max": int(sizes.max()) if len(sizes) else 0,
            "avg_degree": (2.0 * report["n_edges"] / report["n_nodes"]
                           if report.get("n_nodes") else np.nan),
        })

    if summary:
        w("summary", pd.DataFrame(summary), "headline metrics per dataset")
    if describe:
        w("datasets", pd.DataFrame(describe), "dataset descriptions")
    if selectivity:
        w("selectivity", pd.DataFrame(selectivity),
          "communities vs size-matched random sets, per dataset")
    if by_size:
        w("by_size", pd.concat(by_size, ignore_index=True),
          "detection rate by community-size band (all datasets)")
    if aucs:
        a = pd.concat(aucs, ignore_index=True)
        w("auc", a, "AUC of each structural statistic, per dataset")
        # wide form: one row per statistic, one column per dataset
        wide = a.pivot(index="statistic", columns="dataset", values="auc").reset_index()
        wide.columns = [tex_safe(c) for c in wide.columns]
        w("auc_wide", wide, "AUC table, statistics x datasets")

    w.report()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parent", type=Path, help="directory holding the per-dataset runs")
    ap.add_argument("--datasets", default="comamazon,comdblp,comyoutube")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--every", type=int, default=10)
    a = ap.parse_args()
    export(a.parent, [d.strip() for d in a.datasets.split(",") if d.strip()],
           a.out or (a.parent / "latex_community"), a.every)


if __name__ == "__main__":
    main()
