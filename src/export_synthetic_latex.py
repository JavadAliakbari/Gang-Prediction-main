"""Turn a ``run_synthetic_modular`` output directory into pgfplots-ready CSVs.

The shared artifacts (training trace, Ward sweep, per-unit diagnostics) are
exported by :mod:`src.export_paper_csv`; this driver adds what only the synthetic
run produces --- the planted motif's *type*, recovered from its pattern id, and
the size banding that the variable-size planting makes meaningful.

Usage::

    python -m src.export_synthetic_latex <run_dir> [--out <tex_dir>] [--every N]
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


def _add_motif_type(d: pd.DataFrame) -> pd.DataFrame:
    """The motif type is the prefix of the pattern id (``random_7`` -> ``random``)."""
    d = d.copy()
    d["type"] = d["gang"].astype(str).str.replace(r"_\d+$", "", regex=True)
    return d


def export(run: Path, out: Path, every: int) -> None:
    w = Writer(out, "synth")
    report = json.loads((run / "synthetic_modular_report.json").read_text())

    # ---- shared artifacts ---------------------------------------------------
    write_training(run, w, every)
    write_pr_sweep(run, w)
    write_sweep_summary(run, w, label="graph", unit_col="motifs")
    diag = write_diagnostics(run, w, unit="motif", stem="motifs",
                             derive=_add_motif_type, extra_cols=("type",))
    write_capture(report.get("captures") or {}, w)

    # ---- detection rate by motif size band ---------------------------------
    if diag is not None and "size" in diag:
        binned_rate(diag, "size",
                    edges=[0, 8, 11, 14, 17, 10 ** 9],
                    labels=["7-8", "9-11", "12-14", "15-17", "18-20"],
                    w=w, stem="by_size", what="detection rate by motif size band")

    # ---- headline splits + per transfer graph -------------------------------
    rows = []
    for split in ("train", "test", "all"):
        r = (report.get("report") or {}).get(split)
        if r:
            rows.append({"split": split, "motifs": r["total"], "recall": r["mean_recall"],
                         "precision": r["mean_precision"], "f1": r["mean_f1"],
                         "det": r["detection_rate"]})
    for i, r in enumerate(report.get("transfer") or []):
        rows.append({"split": f"transfer T{i}", "motifs": r["total"],
                     "recall": r["mean_recall"], "precision": r["mean_precision"],
                     "f1": r["mean_f1"], "det": r["detection_rate"]})
    w("splits", pd.DataFrame(rows), "headline detection per split + per transfer graph")

    w.report()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path, help="a run_synthetic_modular output dir")
    ap.add_argument("--out", type=Path, default=None, help="where to write the CSVs")
    ap.add_argument("--every", type=int, default=25,
                    help="subsample training curves to every N-th epoch")
    a = ap.parse_args()
    export(a.run, a.out or (a.run / "latex"), a.every)


if __name__ == "__main__":
    main()
