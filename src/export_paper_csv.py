"""Shared CSV writers turning a modular-run output directory into pgfplots tables.

Both :mod:`src.run_synthetic_modular` and :mod:`src.run_elliptic_modular` write the
same artifacts (``training_history.csv``, ``pr_sweep_*.csv``, ``pr_sweep_summary.csv``,
``per_gang_diagnostics.csv``), so the export logic lives here once and the two
dataset drivers (:mod:`src.export_synthetic_latex`, :mod:`src.export_elliptic_latex`)
only add what is genuinely dataset-specific -- motif type/size bands on one side,
transfer *days* and the licit-group control on the other.

Every emitted CSV is comma-separated with a single header row, which is what
``\\pgfplotstableread[col sep=comma]`` expects.  String cells are escaped for
LaTeX text mode, because ``pgfplotstabletypeset`` typesets them verbatim.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Paper-ready labels for the structural statistics; ``pgfplotstabletypeset``
# writes string cells verbatim, so a bare ``_`` would be a math-mode error.
STAT_LABEL = {
    "overlap_mean": "overlap (mean)",
    "overlap_max": "overlap (max)",
    "size": r"$|S|$",
    "Phi": r"$\Phi$",
    "mbar1": r"$\tilde m_1$",
    "capture": r"$C_S^\tau$",
    "density": "density",
    "starness": "star-ness",
    "deg_ratio": "deg.\\ ratio",
}

DIAG_STATS = ("size", "Phi", "mbar1", "capture", "density", "starness", "deg_ratio")


def tex_safe(s) -> str:
    """Escape a bare identifier for LaTeX text mode."""
    return str(s).replace("_", r"\_")


def subsample(df: pd.DataFrame, every: int) -> pd.DataFrame:
    """Keep every ``every``-th row plus the last (keeps files Overleaf-sized)."""
    if every <= 1 or len(df) <= every:
        return df
    keep = list(range(0, len(df), every))
    if keep[-1] != len(df) - 1:
        keep.append(len(df) - 1)
    return df.iloc[keep]


class Writer:
    """Collects CSVs into ``out`` under a common ``prefix``, logging what it wrote."""

    def __init__(self, out: Path, prefix: str):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.written: list[tuple[str, str]] = []

    def __call__(self, stem: str, df: pd.DataFrame, what: str) -> None:
        name = f"{self.prefix}_{stem}.csv"
        df.to_csv(self.out / name, index=False, float_format="%.6g")
        self.written.append((name, what))

    def report(self) -> None:
        print(f"wrote {len(self.written)} CSVs to {self.out}")
        for n, w in self.written:
            print(f"  {n:34} {w}")


# --------------------------------------------------------------------------- #
# artifacts shared by both drivers
# --------------------------------------------------------------------------- #
def write_training(run: Path, w: Writer, every: int = 25,
                   filename: str = "training_history.csv",
                   stem: str = "training") -> None:
    """Per-epoch training trace: capture floor, per-gang capture, chi, loss.

    ``filename`` lets callers point at a tagged history (the fraud/community
    driver writes ``training_history_<dataset>.csv``).
    """

    path = run / filename
    if not path.exists():
        return
    hist = pd.read_csv(path)
    cols = [c for c in ("epoch", "lambda_min", "capture_mean", "capture_min",
                        "confusability", "objective", "loss", "lambda_min_test",
                        "capture_mean_test") if c in hist]
    w(stem, subsample(hist[cols], every),
      "training curves: lambda_min / capture / chi / loss vs epoch")


def write_pr_sweep(run: Path, w: Writer, pattern: str = "pr_sweep_train_*.csv",
                   max_rows: int = 200) -> str:
    """The full Ward sweep on the reported graph; returns the epsilon column used."""

    hits = sorted(run.glob(pattern))
    if not hits:
        return "epsilon"
    sw = pd.read_csv(hits[0])
    eps_key = "epsilon_exact" if "epsilon_exact" in sw else "epsilon"
    keep = [c for c in (eps_key, "n_coarse", "mean_recall", "mean_precision",
                        "mean_f1", "mean_jaccard", "det_rate") if c in sw]
    sw = sw[keep].sort_values(eps_key)
    w("pr_sweep", subsample(sw, max(1, len(sw) // max_rows)),
      "full Ward sweep: metrics vs epsilon (PR curve + eps* panel)")
    return eps_key


def write_sweep_summary(run: Path, w: Writer, label: str = "graph",
                        unit_col: str = "gangs") -> None:
    """Per-graph PR-AUC and the best-F1 operating point ``eps*``.

    ``unit_col`` names the count column ("gangs" on Elliptic, "motifs" on the
    synthetic benchmark) -- the .tex files select columns by name.
    """

    path = run / "pr_sweep_summary.csv"
    if not path.exists():
        return
    s = pd.read_csv(path)
    ek = s["eps_key"].iloc[0] if "eps_key" in s else "epsilon"
    keep = {"tag": label, "n_gangs": unit_col, "levels": "levels", "pr_auc": "pr_auc",
            f"best_{ek}": "eps_star", "best_n_coarse": "n_coarse",
            "best_mean_recall": "recall", "best_mean_precision": "precision",
            "best_mean_f1": "f1", "best_det_rate": "det"}
    avail = {k: v for k, v in keep.items() if k in s}
    s = s[list(avail)].rename(columns=avail)
    s[label] = s[label].map(tex_safe)
    w("sweep_summary", s, "per-graph PR-AUC, eps*, and best-F1 operating point")


def write_diagnostics(run: Path, w: Writer, unit: str = "gang",
                      stem: str = "units",
                      derive: "callable | None" = None,
                      extra_cols: "tuple[str, ...]" = (),
                      filename: str = "per_gang_diagnostics.csv",
                      stats: "tuple[str, ...]" = DIAG_STATS,
                      id_cols: "tuple[str, ...]" = ("day", "gang"),
                      ) -> "pd.DataFrame | None":
    """Per-unit structural stats vs detection outcome; also the AUC table.

    Writes the pooled scatter table, the detected/missed split (so pgfplots can
    draw two colours without filtering in TeX), and the AUC of each statistic as
    a detection score.  Returns the pooled frame for dataset-specific follow-ups.

    ``derive`` may add dataset-specific columns to the frame before writing (the
    synthetic driver uses it to recover each planted motif's *type* from its id);
    ``extra_cols`` names those columns so they survive the column selection.
    ``filename``/``stats``/``id_cols`` adapt to a different diagnostics schema --
    the community driver records overlap instead of star-ness and degree ratio.
    """

    path = run / filename
    if not path.exists():
        return None
    d = pd.read_csv(path)
    if derive is not None:
        d = derive(d)
    keep = [c for c in (*id_cols, *extra_cols, *stats, "f1", "recall", "precision",
                        "detected")
            if c in d]
    d_out = d[keep]
    w(stem, d_out, f"per-{unit} scatter: Phi / capture / size vs detected (0/1)")
    w(f"{stem}_detected", d_out[d_out.detected == 1], f"detected {unit}s only")
    w(f"{stem}_missed", d_out[d_out.detected == 0], f"missed {unit}s only")

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return d
    rows = []
    for f in stats:
        if f not in d:
            continue
        rows.append({
            "statistic": STAT_LABEL.get(f, f),
            "detected_median": d[d.detected == 1][f].median(),
            "missed_median": d[d.detected == 0][f].median(),
            "auc": (roc_auc_score(d.detected, d[f])
                    if d.detected.nunique() > 1 else float("nan")),
        })
    w("auc", pd.DataFrame(rows),
      "AUC of each structural statistic as a detection predictor")
    return d


def write_capture(captures: dict, w: Writer) -> None:
    """Retained ``M_tau``-energy per split."""

    if not captures:
        return
    w("capture", pd.DataFrame([
        {"split": k, "mean_capture": v["mean_capture"], "min_capture": v["min_capture"],
         "lambda_min": v["lambda_min_gamma"]}
        for k, v in captures.items()
    ]), "capture (retained M_tau energy) per split")


def binned_rate(d: pd.DataFrame, column: str, edges: list, labels: list,
                w: Writer, stem: str, what: str) -> None:
    """Detection rate bucketed by a numeric column (size bands, conductance bands)."""

    b = d.copy()
    b["band"] = pd.cut(b[column], bins=edges, labels=labels, right=True)
    g = b.groupby("band", observed=True).agg(
        detected=("detected", "sum"), total=("detected", "size"),
        median_phi=("Phi", "median"), median_capture=("capture", "median"),
    ).reset_index()
    g["rate"] = g.detected / g.total
    w(stem, g, what)
