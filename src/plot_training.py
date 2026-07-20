"""Training-curve plots for the collective filter bank fit.

Turns the per-epoch history recorded by
:func:`~src.run_collective_bank_detection.fit_collective_bank` into one figure
showing what the optimizer actually did:

* **loss** -- the negated ascended objective, ``-(capture - beta*chi -
  label_weight*CE)``.  This is the only curve that is guaranteed to trend down;
  the others are its components and may trade against each other.
* **capture** -- mean and worst-gang retained energy ``Gamma_jj``, plus the
  collective floor ``lambda_min(Gamma)`` (which is what ``capture_objective=
  "lambda_min"`` ascends; it sits below the mean by the cross-gang alignment).
* **confusability** ``chi`` -- the necessity-side penalty (eq. 40); it should
  fall while capture rises, and the two panels together show whether the margin
  ``D = C - chi`` was bought or merely traded.
* **per-gang capture** -- each training gang's own ``Gamma_jj`` over the fit,
  taken from the snapshots, so a gang that never lifts is visible immediately.

``plot_training_curves(fit_info, path)`` writes the PNG and returns the CSV-able
history frame; callers usually also dump that frame next to the plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def history_frame(fit: dict) -> pd.DataFrame:
    """Per-epoch history as a tidy frame (one row per epoch)."""

    cols = {
        "lambda_min": fit.get("history") or [],
        "capture_mean": fit.get("energy_history") or [],
        "capture_min": fit.get("capture_min_history") or [],
        "confusability": fit.get("conf_history") or [],
        "objective": fit.get("margin_history") or [],
        "label_ce": fit.get("ce_history") or [],
        "neg_energy": fit.get("neg_history") or [],
    }
    n = max((len(v) for v in cols.values()), default=0)
    if n == 0:
        return pd.DataFrame()
    data = {"epoch": np.arange(n)}
    for k, v in cols.items():
        if len(v) == n and np.any(np.asarray(v, dtype=float) != 0.0):
            data[k] = np.asarray(v, dtype=float)
    df = pd.DataFrame(data)
    if "objective" in df:
        df["loss"] = -df["objective"]
    return df


def _per_gang_frame(fit: dict) -> "pd.DataFrame | None":
    """Per-gang capture over the snapshot epochs (wide: one column per gang)."""

    snaps = fit.get("snapshots") or []
    rows = [s for s in snaps if s.get("gamma_diag")]
    if len(rows) < 2:
        return None
    m = len(rows[0]["gamma_diag"])
    return pd.DataFrame(
        {"epoch": [s["epoch"] for s in rows],
         **{f"gang_{j}": [s["gamma_diag"][j] for s in rows] for j in range(m)}}
    )


def plot_training_curves(
    fit: dict, path: "str | Path", *, title: str = "collective bank training"
) -> pd.DataFrame:
    """Write the training-curve figure for one ``fit_collective_bank`` result."""

    df = history_frame(fit)
    if df.empty:
        return df
    pg = _per_gang_frame(fit)
    has_conf = "confusability" in df
    panels = 2 + int(has_conf) + int(pg is not None)
    fig, axes = plt.subplots(1, panels, figsize=(5.0 * panels, 4.0))
    axes = np.atleast_1d(axes)
    ax = iter(axes)

    # 1. loss (what was actually minimized)
    a = next(ax)
    if "loss" in df:
        a.plot(df.epoch, df.loss, color="#B4442E", lw=1.6)
        a.set_ylabel("loss  $-(C - \\beta\\chi)$")
    else:
        a.plot(df.epoch, -df.lambda_min, color="#B4442E", lw=1.6)
        a.set_ylabel("loss")
    a.set_title("training loss")
    a.set_xlabel("epoch")

    # 2. capture: floor, mean, worst gang
    a = next(ax)
    a.plot(df.epoch, df.lambda_min, color="#1F4E79", lw=1.8,
           label="$\\lambda_{\\min}(\\Gamma)$  (collective floor)")
    if "capture_mean" in df:
        a.plot(df.epoch, df.capture_mean, color="#2E7D5B", lw=1.4,
               label="mean $C_j$")
    if "capture_min" in df:
        a.plot(df.epoch, df.capture_min, color="#2E7D5B", lw=1.2, ls="--",
               label="worst $C_j$")
    a.set_title("capture (retained $M_\\tau$-energy)")
    a.set_xlabel("epoch")
    a.set_ylabel("capture")
    a.legend(fontsize=7, frameon=False)

    # 3. confusability
    if has_conf:
        a = next(ax)
        a.plot(df.epoch, df.confusability, color="#7A4EA8", lw=1.6, label="$\\chi$")
        if "capture_mean" in df:
            a.plot(df.epoch, df.capture_mean - df.confusability, color="#B8860B",
                   lw=1.2, ls="--", label="margin $D = C - \\chi$")
        a.set_title("confusability and margin")
        a.set_xlabel("epoch")
        a.set_ylabel("$\\chi$")
        a.legend(fontsize=7, frameon=False)

    # 4. per-gang capture trajectories
    if pg is not None:
        a = next(ax)
        gcols = [c for c in pg.columns if c.startswith("gang_")]
        cmap = plt.get_cmap("viridis")
        for i, c in enumerate(gcols):
            a.plot(pg.epoch, pg[c], lw=1.1,
                   color=cmap(i / max(len(gcols) - 1, 1)), alpha=0.9)
        a.set_title(f"per-gang capture ({len(gcols)} train gangs)")
        a.set_xlabel("epoch")
        a.set_ylabel("$C_j = \\Gamma_{jj}$")

    for a in axes:
        a.grid(alpha=0.25, lw=0.5)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df


def write_training_report(fit: dict, out_dir: "str | Path", *, tag: str = "",
                          title: str = "collective bank training") -> "Path | None":
    """Plot + CSVs for one fit; returns the figure path (``None`` if no history)."""

    out_dir = Path(out_dir)
    suffix = f"_{tag}" if tag else ""
    fig_path = out_dir / f"training_curves{suffix}.png"
    df = plot_training_curves(fit, fig_path, title=title)
    if df.empty:
        return None
    df.to_csv(out_dir / f"training_history{suffix}.csv", index=False)
    pg = _per_gang_frame(fit)
    if pg is not None:
        pg.to_csv(out_dir / f"training_per_gang_capture{suffix}.csv", index=False)
    return fig_path
