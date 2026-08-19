"""Re-define gangs to the population the method actually models, then re-measure.

Every sweep so far (tau, epsilon, family, features) says the same thing: the large
sparse *fans* -- illicit connected components at density ~0.02 with ``m_bar1 ~ 0.7``
-- are not single-supernode objects, and the day-to-day detection swing tracks the
day's median ``m_bar1`` at ``r = -0.84``.  The model assumes low-conductance *dense*
groups.  So: split the giant CCs into dense sub-communities and re-measure.

Three gang definitions, all evaluated through the identical frozen-filter pipeline:

* ``cc``       -- the original: illicit connected components (size >= 2).
* ``kcore2``   -- connected components of the **2-core** of each illicit CC.  A tree
                  or pure star has an *empty* 2-core, so this strips exactly the
                  fan fringes and keeps cycles/dense cores.  Principled: "a gang is
                  a mutually-reinforcing core, not a payout tree."
* ``louvain``  -- Louvain sub-communities of each illicit CC, kept only if their
                  internal density >= ``--min-density``.

For each definition we re-train on ``--day-start`` and transfer to the same days,
reporting the population stats, detection, and -- the actual claim under test --
the **per-day variance** and the ``r(detection_rate, median m_bar1)`` correlation.
If the fan/dense split explains the swing, that ``-0.84`` should collapse.

NOTE: this changes the *task*, not just the method: a definition that removes the
hard cases will score higher for free.  The population stats are reported alongside
so the comparison stays honest.

Run::

    python -m src.run_gang_redefinition --day-start 24 --transfer-days 10 \
        --out results/gang_redef
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))
sys.path.insert(0, str(Path.cwd()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch

from src.analyze_missed_gangs import cached_wallet_features, gang_moments, gang_structure
from src.collective_detector import CollectiveBankDetector, DetectorConfig, GraphData
from src.loukas_sgc_detection import evaluate_loukas_patterns
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import build_torch_graph, make_patterns, split_train_test


def _subgraph(A_unw, nodes):
    """networkx view of the induced subgraph on ``nodes`` (0..s-1 <-> nodes[i])."""

    sub = A_unw[nodes][:, nodes]
    return nx.from_scipy_sparse_array(sub)


def split_kcore(A_unw, cc_nodes, k=2, min_size=2):
    """Connected components of the k-core: strips tree/star fringes entirely."""

    G = _subgraph(A_unw, cc_nodes)
    try:
        core = nx.k_core(G, k=k)
    except Exception:
        return []
    out = []
    for comp in nx.connected_components(core):
        if len(comp) >= min_size:
            out.append([int(cc_nodes[i]) for i in comp])
    return out


def split_louvain(A_unw, cc_nodes, min_density=0.15, min_size=2, seed=0):
    """Louvain sub-communities of the CC, kept only if internally dense enough."""

    G = _subgraph(A_unw, cc_nodes)
    if G.number_of_nodes() < 2:
        return []
    try:
        comms = nx.community.louvain_communities(G, seed=seed)
    except Exception:
        comms = [set(G.nodes())]
    out = []
    for c in comms:
        if len(c) < min_size:
            continue
        H = G.subgraph(c)
        s, e = len(c), H.number_of_edges()
        dens = 2 * e / (s * (s - 1)) if s > 1 else 0.0
        if dens >= min_density:
            out.append([int(cc_nodes[i]) for i in c])
    return out


def build_gangs(A_unw, cls, definition, args):
    """Gang node-sets for one day under a given definition."""

    ccs = connected_components_sets(A_unw, np.where(cls == 1)[0], args.min_gang_size)
    if definition == "cc":
        return [list(map(int, S)) for S in ccs]
    out = []
    for S in ccs:
        S = list(map(int, S))
        if definition == "kcore2":
            out.extend(split_kcore(A_unw, S, k=args.kcore, min_size=args.min_gang_size))
        elif definition == "louvain":
            out.extend(split_louvain(A_unw, S, min_density=args.min_density,
                                     min_size=args.min_gang_size, seed=args.seed))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--day-start", type=int, default=24)
    ap.add_argument("--transfer-days", type=int, default=10)
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--kcore", type=int, default=2)
    ap.add_argument("--min-density", type=float, default=0.15)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--degree", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--conf-weight", type=float, default=10.0)
    ap.add_argument("--epsilon", type=float, default=0.5)
    ap.add_argument("--ward-num-cuts", type=int, default=150)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/gang_redef", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    DEFS = ["cc", "kcore2", "louvain"]

    # --- cache each day's graph once; build all three gang populations ------
    print("=== caching days ===")
    days, cols = {}, None
    for k in range(args.transfer_days + 1):
        day = args.day_start + k
        A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, day, day)
        if cols is None:
            X, cols = cached_wallet_features(args.data_dir, nodes_df, day, return_columns=True)
        else:
            X = cached_wallet_features(args.data_dir, nodes_df, day, keep_columns=cols)
        g = build_torch_graph(A_w, A_unw, cls, X, weighted=False)
        base = GraphData.from_graph(g)
        pops = {}
        for d in DEFS:
            sets = build_gangs(A_unw, cls, d, args)
            if not sets:
                continue
            gangs = make_patterns(sets, "alert", "gang", "g")
            gof = torch.full((base.num_nodes,), -1, dtype=torch.long)
            for gi, S in enumerate(sets):
                gof[torch.as_tensor(S, dtype=torch.long)] = gi
            phi, mbar1 = gang_moments(base.a_hat, base.adjacency, gangs)
            struct = gang_structure(g.edge_index, gof, len(gangs), base.num_nodes)
            pops[d] = {"gangs": gangs, "struct": struct, "phi": phi.numpy(),
                       "mbar1": mbar1.numpy()}
        days[day] = {"base": base, "pops": pops}
        counts = "  ".join(f"{d}={len(pops[d]['gangs'])}" for d in DEFS if d in pops)
        print(f"  day {day}: N={base.num_nodes:,}  {counts}")

    train_day = args.day_start
    rows = []
    for d in DEFS:
        if d not in days[train_day]["pops"]:
            continue
        P0 = days[train_day]["pops"][d]
        tr, _ = split_train_test(P0["gangs"], args.train_ratio,
                                 np.random.default_rng(args.seed))
        cfg = DetectorConfig(degree=args.degree, tau=args.tau, epochs=args.epochs,
                             conf_weight=args.conf_weight, conf_reduce="mean",
                             optimizer="riemannian", coarsen_target="bank",
                             coarsening_method="ward-tree", ward_stop="epsilon",
                             epsilon=args.epsilon, ward_num_cuts=args.ward_num_cuts,
                             threshold=args.threshold, seed=args.seed)
        det = CollectiveBankDetector(cfg)
        print(f"\n=== definition '{d}': training on day {train_day} "
              f"({len(tr)} train gangs) ===")
        det.fit(days[train_day]["base"], tr)
        print(f"  lambda_min {det.fit_info_['init_objective']:.4g} -> "
              f"{det.fit_info_['objective']:.4g}")
        for day, D in days.items():
            if d not in D["pops"]:
                continue
            P, data = D["pops"][d], D["base"]
            gangs = P["gangs"]
            basis = det.target_subspace(data, gangs)
            co, _ = det.coarsen(data, basis, gangs)
            res, _ = evaluate_loukas_patterns(gangs, co.node_to_supernode, data.y,
                                              threshold=args.threshold)
            for gi in range(len(gangs)):
                rows.append({"definition": d, "day": day,
                             "size": P["struct"][gi]["size"],
                             "density": P["struct"][gi]["density"],
                             "mbar1": float(P["mbar1"][gi]), "Phi": float(P["phi"][gi]),
                             "detected": int(res[gi].detected),
                             "recall": res[gi].recall, "precision": res[gi].precision,
                             "f1": res[gi].f1})
        sub = pd.DataFrame([r for r in rows if r["definition"] == d])
        print(f"  -> {len(sub)} gangs, detection {sub.detected.mean():.1%}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "gang_redef_per_gang.csv", index=False)

    # --- report -------------------------------------------------------------
    print("\n" + "=" * 108)
    print("GANG RE-DEFINITION  (does splitting giant CCs into dense cores explain the swing?)")
    print("=" * 108)
    print(f"{'definition':<10}{'n_gangs':>9}{'med_size':>10}{'med_dens':>10}{'med_mbar1':>11}"
          f"{'fan%':>7}{'detect':>9}{'precision':>11}{'f1':>8}{'per-day std':>13}{'r(det,mbar1)':>14}")
    print("-" * 108)
    S = []
    for d in DEFS:
        sub = df[df.definition == d]
        if not len(sub):
            continue
        per = sub.groupby("day").agg(det=("detected", "mean"),
                                     mb=("mbar1", "median"))
        r = per["det"].corr(per["mb"]) if len(per) > 2 else float("nan")
        rec = {"definition": d, "n_gangs": len(sub),
               "med_size": sub["size"].median(), "med_density": sub.density.median(),
               "med_mbar1": sub.mbar1.median(),
               "fan_frac": (sub["size"] >= 30).mean(),
               "detect": sub.detected.mean(), "precision": sub.precision.mean(),
               "f1": sub.f1.mean(), "per_day_std": per["det"].std(), "r_det_mbar1": r}
        S.append(rec)
        print(f"{d:<10}{rec['n_gangs']:>9}{rec['med_size']:>10.0f}{rec['med_density']:>10.3f}"
              f"{rec['med_mbar1']:>11.3f}{rec['fan_frac']:>7.1%}{rec['detect']:>9.1%}"
              f"{rec['precision']:>11.3f}{rec['f1']:>8.3f}{rec['per_day_std']:>13.3f}"
              f"{rec['r_det_mbar1']:>14.2f}")
    pd.DataFrame(S).to_csv(args.out / "gang_redef_summary.csv", index=False)

    print("\nper-day detection rate by definition:")
    piv = df.pivot_table(index="day", columns="definition", values="detected", aggfunc="mean")
    print(piv.round(3).to_string())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for d in DEFS:
        if d in piv.columns:
            a1.plot(piv.index, piv[d], "o-", label=d)
    a1.set_xlabel("day"); a1.set_ylabel("detection rate"); a1.set_ylim(0, 1)
    a1.set_title("Per-day detection by gang definition"); a1.legend(); a1.grid(alpha=0.3)
    Sd = pd.DataFrame(S)
    x = np.arange(len(Sd))
    a2.bar(x - 0.2, Sd.detect, 0.4, label="detection", color="tab:blue")
    a2.bar(x + 0.2, Sd.per_day_std, 0.4, label="per-day std", color="tab:red")
    a2.set_xticks(x); a2.set_xticklabels(Sd.definition)
    a2.set_title("Detection and day-to-day variance"); a2.legend(); a2.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out / "gang_redef.png", dpi=150); plt.close(fig)
    print(f"\nCSV + plot -> {args.out}")


if __name__ == "__main__":
    main()
