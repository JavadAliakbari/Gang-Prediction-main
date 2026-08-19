"""Key-query attention over a FIXED small bank of filter heads (inductive routing).

Fixes both downsides of the one-head-per-gang bank:

* **Scalability / overfitting** -- the number of heads ``H`` is a small constant,
  independent of the number of training gangs; many gangs share and shape the
  same few heads, and only the tiny routing net grows nothing at all.
* **New motifs on new days** -- every parameter (head filters ``Theta_h``, keys
  ``k_h``, query map ``W_q``) is frozen after training.  A new candidate motif
  ``S`` (e.g. a test gang, or an alert on an unseen day) gets its filter by
  routing, not by training:

      descriptor  b_S[k, a] = <T_k(A_hat) x_a, v_hat_S>_{M_tau}   (label-free,
                  normalized to unit Frobenius norm -> size/scale invariant)
      query       q_S = W_q vec(b_S)
      attention   alpha = softmax(K q_S / sqrt(p))          (K = stacked keys)
      filter      Theta_S = sum_h alpha_h Theta_h           (convex combination)
      column      z_S = sum_{k,a} Theta_S[k,a] T_k(A_hat) x_a

  The coarsening target on a day is the stack of routed columns of that day's
  candidate motifs -- ONE column per candidate (the Remark 6.15 economy: no
  within-gang-varying directions wasting RSA budget), versus ``H*d`` columns for
  the concatenated multi-head span.

Trained end-to-end on the training gangs by ascending the single-column capture
Rayleigh quotient ``C_j = <z_j, v_hat_j>_M^2 / ||z_j||_M^2 / (Phi_j + tau)``.
Heads can be cold-started or warm-started at k-means centroids of the per-gang
closed-form filters (Theorem 6.2), so the head basis begins spanning the motif
archetypes seen in training.

Candidate-set caveat (stated plainly): routing needs the candidate motif's node
set at evaluation time -- the deployment regime is "candidate alerts are given,
the pipeline tailors a filter and certifies them", not blind discovery.  The
single-head baseline's span target needs no candidates; both are reported under
the same Ward stop so the trade is visible.

Run::

    conda activate FedStruct
    python -m src.run_attention_bank \
        --ceiling-csv results/capture_ceiling/<stamp>/capture_ceiling_per_gang.csv
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
from torch import nn

from src.loukas_sgc_detection import graph_operators
from src.run_capture_ceiling import subspace_captures
from src.run_collective_bank_detection import (
    _basis_stack,
    _l_apply,
    _m_apply,
    _train_gang_m_vhat,
    degree_weighted_indicators,
)
from src.run_multihead_bank import (
    _m_apply_cached,
    closed_form_gang_filters,
    gang_indicator_stats,
    multihead_bank,
    per_gang_capture_of_span,
    train_multihead,
    unit_channels,
    ward_detect,
)
from src.run_elliptic_gang_conductance import build_graph, connected_components_sets
from src.run_elliptic_gang_detection import (
    build_torch_graph,
    load_node_features,
    make_patterns,
    split_train_test,
)
from src.utils.utils import LOGGER, now


# --------------------------------------------------------------------------- #
# descriptors and the attention module
# --------------------------------------------------------------------------- #
def motif_descriptors(propagated: list, m_vhat: torch.Tensor) -> torch.Tensor:
    """``b_S[k, a] = <T_k x_a, v_hat_S>_M`` per motif, unit-normalized  (m, K+1, d)."""

    b = torch.stack([P.T @ m_vhat for P in propagated], dim=0)  # (K+1, d, m)
    b = b.permute(2, 0, 1)  # (m, K+1, d)
    nrm = b.flatten(1).norm(dim=1).clamp_min(torch.finfo(b.dtype).eps)
    return b / nrm.view(-1, 1, 1)


class AttentionBank(nn.Module):
    """H fixed filter heads + cosine key-query routing -> one filter per motif.

    The query IS the motif's unit-normalized descriptor (no learned projection:
    with only a handful of training gangs a learned ``W_q`` both starves the
    softmax of logit scale -- uniform attention -- and invites overfitting).
    Keys live in the same descriptor space and a learnable temperature sharpens
    the cosine logits; warm init places ``key_h`` at the mean descriptor of the
    gangs whose closed-form filters formed head ``h``, so routing is sensible
    from epoch 0.
    """

    def __init__(self, heads: int, K1: int, d: int, dtype, init=None,
                 key_init=None, temp0: float = 0.1):
        super().__init__()
        if init is None:
            raw = torch.ones(heads, K1, d, dtype=dtype)
            raw += 0.1 * torch.randn(heads, K1, d, dtype=dtype)
        else:
            raw = init.clone().to(dtype)
        self.theta = nn.Parameter(raw)  # (H, K+1, d)
        if key_init is None:
            keys = torch.randn(heads, K1 * d, dtype=dtype)
        else:
            keys = key_init.clone().to(dtype)
        self.keys = nn.Parameter(keys)
        self.log_temp = nn.Parameter(torch.tensor(float(np.log(temp0)), dtype=dtype))

    def attention(self, desc: torch.Tensor) -> torch.Tensor:
        """(m, K+1, d) descriptors -> (m, H) softmax attention weights."""

        eps = torch.finfo(desc.dtype).eps
        q = desc.flatten(1)  # unit-normalized by motif_descriptors
        k = self.keys / self.keys.norm(dim=1, keepdim=True).clamp_min(eps)
        return torch.softmax(q @ k.T / self.log_temp.exp().clamp_min(1e-3), dim=1)

    def motif_filters(self, desc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Routed per-motif filters ``Theta_S`` (m, K+1, d) and the weights.

        Heads are normalized as WHOLE filters (unit Frobenius norm), not per
        channel: the routed column sums channels with the filter's own weights,
        so per-channel normalization would destroy the cross-channel structure
        (e.g. of a warm-started closed-form filter).  Span targets are invariant
        to column scaling; single routed columns are not.
        """

        eps = torch.finfo(self.theta.dtype).eps
        nrm = self.theta.flatten(1).norm(dim=1).clamp_min(eps).view(-1, 1, 1)
        alpha = self.attention(desc)  # (m, H)
        return torch.einsum("mh,hkd->mkd", alpha, self.theta / nrm), alpha


def routed_columns(P: torch.Tensor, theta_m: torch.Tensor) -> torch.Tensor:
    """``z_S = sum_{k,a} Theta_S[k,a] T_k x_a`` for every motif  (N, m).

    ``P`` is the stacked dictionary ``(K+1, N, d)``.
    """

    return torch.einsum("knd,mkd->nm", P, theta_m)


def single_column_captures(
    z: torch.Tensor, m_z: torch.Tensor, m_vhat: torch.Tensor
) -> torch.Tensor:
    """Capture of each motif by ITS OWN routed column (Rayleigh quotient), (m,)."""

    eps = torch.finfo(z.dtype).eps
    align = (z * m_vhat).sum(0)  # <z_S, M v_hat_S>
    energy = (z * m_z).sum(0).clamp_min(eps)  # ||z_S||_M^2
    return (align**2 / energy).clamp(0.0, 1.0)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train_attention_bank(
    P: torch.Tensor, m_P: torch.Tensor, desc: torch.Tensor, m_vhat: torch.Tensor,
    *, heads: int, epochs: int, lr: float, weight_decay: float,
    seed: int, init=None, key_init=None,
) -> tuple[AttentionBank, dict]:
    torch.manual_seed(seed)
    K1, _, d = P.shape
    model = AttentionBank(heads, K1, d, P.dtype, init=init, key_init=key_init)
    opt = torch.optim.Adam(
        [
            {"params": [model.theta, model.log_temp], "weight_decay": 0.0},
            {"params": [model.keys], "weight_decay": weight_decay},
        ],
        lr=lr,
    )
    hist = {}
    for ep in range(epochs + 1):
        theta_m, alpha = model.motif_filters(desc)
        z = routed_columns(P, theta_m)
        m_z = routed_columns(m_P, theta_m)
        cap = single_column_captures(z, m_z, m_vhat)
        if ep == 0:
            hist["init_mean"], hist["init_min"] = float(cap.mean()), float(cap.min())
        if ep == epochs:
            break
        opt.zero_grad(set_to_none=True)
        (-cap.mean()).backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            LOGGER.info(
                f"      epoch {ep + 1:>4}: mean C={float(cap.mean()):.4f} "
                f"min C={float(cap.min()):.4f}  attn-max={float(alpha.max(1).values.mean()):.2f}"
            )
    hist["final_mean"], hist["final_min"] = float(cap.mean()), float(cap.min())
    return model, hist


def kmeans_filters(
    W: torch.Tensor, heads: int, seed: int
) -> tuple[torch.Tensor, np.ndarray]:
    """k-means centroids of the (unit-normalized) closed-form gang filters.

    Returns ``(centroids (H, K+1, d), labels (m,))`` -- the labels let the caller
    warm-start the routing keys at the matching cluster-mean descriptors.
    """

    flat = W.flatten(1)
    flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-30)
    X = flat.numpy()
    rng = np.random.default_rng(seed)
    cent = X[rng.choice(len(X), size=min(heads, len(X)), replace=False)].copy()
    lab = np.zeros(len(X), dtype=int)
    for _ in range(50):
        dist = ((X[:, None, :] - cent[None]) ** 2).sum(-1)
        lab = dist.argmin(1)
        for h in range(len(cent)):
            if (lab == h).any():
                cent[h] = X[lab == h].mean(0)
    if len(cent) < heads:  # more heads than gangs: pad with noise
        pad = np.ones((heads - len(cent), X.shape[1])) / X.shape[1] ** 0.5
        cent = np.r_[cent, pad]
    return (
        torch.as_tensor(cent, dtype=W.dtype).reshape(heads, W.shape[1], W.shape[2]),
        lab,
    )


def cluster_mean_keys(
    desc: torch.Tensor, labels: np.ndarray, heads: int
) -> torch.Tensor:
    """Warm routing keys: unit mean descriptor of each head's gang cluster."""

    flat = desc.flatten(1)
    keys = []
    for h in range(heads):
        members = np.nonzero(labels == h)[0]
        k = flat[members].mean(0) if len(members) else torch.randn_like(flat[0])
        keys.append(k / k.norm().clamp_min(1e-30))
    return torch.stack(keys, dim=0)


# --------------------------------------------------------------------------- #
# per-day evaluation
# --------------------------------------------------------------------------- #
def eval_day(model, P, m_P, a_hat, adjacency, gangs, y, tau, args, detect: bool):
    """Routed-column captures + (optionally) detection with two targets:

    ``routed``  -- one tailored column per candidate motif (maximal economy);
    ``+heads``  -- the routed columns plus the H head banks' span (H*d + m
                   columns): the label-free heads protect the background and
                   held-out structure while each candidate keeps its column.
    """

    m_vhat = _train_gang_m_vhat(a_hat, adjacency, gangs, tau)
    prop_list = [P[k] for k in range(P.shape[0])]
    desc = motif_descriptors(prop_list, m_vhat)
    with torch.no_grad():
        theta_m, alpha = model.motif_filters(desc)
        z = routed_columns(P, theta_m)
        m_z = routed_columns(m_P, theta_m)
        cap = single_column_captures(z, m_z, m_vhat)
    dets = {}
    if detect:
        dets["routed"] = ward_detect(
            adjacency, z, gangs, y, tau=tau, epsilon=args.epsilon,
            threshold=args.threshold, num_cuts=args.ward_num_cuts,
        )
        with torch.no_grad():
            Zh = multihead_bank(prop_list, unit_channels(model.theta.detach()))
        dets["+heads"] = ward_detect(
            adjacency, torch.cat([Zh, z], dim=1), gangs, y, tau=tau,
            epsilon=args.epsilon, threshold=args.threshold,
            num_cuts=args.ward_num_cuts,
        )
    return cap.numpy(), alpha.numpy(), dets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/elliptic_actors", type=Path)
    ap.add_argument("--day-start", type=int, default=25)
    ap.add_argument("--day-end", type=int, default=25)
    ap.add_argument("--transfer-days", type=int, default=10)
    ap.add_argument("--min-gang-size", type=int, default=2)
    ap.add_argument("--train-ratio", type=float, default=0.4)
    ap.add_argument("--degree", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--epsilon", type=float, default=0.85)
    ap.add_argument("--threshold", type=float, default=0.51)
    ap.add_argument("--ward-num-cuts", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ceiling-csv", type=Path, default=None)
    ap.add_argument("--out", default=Path(f"results/attention_bank/{now}/"), type=Path)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    # --- training day --------------------------------------------------------
    LOGGER.info(f"=== attention bank | day {args.day_start}-{args.day_end} "
                f"H={args.heads} K={args.degree} tau={args.tau} ===")
    A_unw, A_w, cls, nodes_df = build_graph(args.data_dir, args.day_start, args.day_end)
    Xfeat, feature_columns = load_node_features(
        args.data_dir, nodes_df, args.day_start, args.day_end, return_columns=True
    )
    graph = build_torch_graph(A_w, A_unw, cls, Xfeat, weighted=False)
    a_hat, adjacency = graph_operators(graph)
    y = graph.y
    gang_sets = connected_components_sets(
        A_unw, np.where(cls == 1)[0], args.min_gang_size
    )
    gangs = make_patterns(gang_sets, "alert", "gang", "g")
    rng = np.random.default_rng(args.seed)
    gang_train, gang_test = split_train_test(gangs, args.train_ratio, rng)
    train_ids = {p.id for p in gang_train}
    LOGGER.info(f"  gangs={len(gangs)} train={len(gang_train)} test={len(gang_test)}")

    X = Xfeat.to(a_hat.dtype)
    prop = _basis_stack(a_hat, X, args.degree, "chebyshev", args.tau)
    P = torch.stack(prop, dim=0)  # (K+1, N, d)
    m_P = torch.stack([_m_apply(a_hat, Pk, args.tau) for Pk in prop], dim=0)
    m_vhat_tr = _train_gang_m_vhat(a_hat, adjacency, gang_train, args.tau)
    desc_tr = motif_descriptors(prop, m_vhat_tr)

    _m_apply_cached.clear()
    _m_apply_cached["tau"] = args.tau
    W = closed_form_gang_filters(a_hat, prop, m_vhat_tr)  # (m, K+1, d)
    W_cent, W_lab = kmeans_filters(W, args.heads, args.seed)
    warm_keys = cluster_mean_keys(desc_tr, W_lab, args.heads)

    configs = [
        # cold: everything from gradient training (expected weak -- shown honestly)
        (f"attn-H{args.heads}-cold",
         dict(init=None, key_init=None, epochs=args.epochs, lr=args.learning_rate)),
        # frozen: closed-form heads + cluster-mean keys, ZERO gradient training --
        # nothing to overfit; the whole method is two closed-form solves + k-means
        (f"attn-H{args.heads}-frozen",
         dict(init=W_cent, key_init=warm_keys, epochs=0, lr=args.learning_rate)),
        # warm: gentle fine-tune of the frozen solution (10x smaller lr)
        (f"attn-H{args.heads}-warm",
         dict(init=W_cent, key_init=warm_keys, epochs=args.epochs,
              lr=args.learning_rate / 10.0)),
    ]

    # --- single-head span baseline (label-free target, same protocol) --------
    LOGGER.info("\n  [H1-span baseline] training single shared bank (trace) ...")
    theta1, _ = train_multihead(
        prop, [m_P[k] for k in range(m_P.shape[0])], m_vhat_tr,
        heads=1, objective="trace", epochs=args.epochs, lr=args.learning_rate,
        ridge=1e-3, seed=args.seed,
    )

    models: dict[str, AttentionBank] = {}
    day25 = {}
    for name, kw in configs:
        LOGGER.info(f"\n  [{name}] training ...")
        model, hist = train_attention_bank(
            P, m_P, desc_tr, m_vhat_tr,
            heads=args.heads, epochs=kw["epochs"],
            lr=kw["lr"], weight_decay=args.weight_decay,
            seed=args.seed, init=kw["init"], key_init=kw["key_init"],
        )
        models[name] = model
        LOGGER.info(f"    train C: {hist['init_mean']:.4f} -> {hist['final_mean']:.4f} "
                    f"(min {hist['final_min']:.4f})")
        cap, alpha, dets = eval_day(
            model, P, m_P, a_hat, adjacency, gangs, y, args.tau, args, detect=True
        )
        day25[name] = {"cap": cap, "alpha": alpha, "det": dets}
        for tgt, r in dets.items():
            LOGGER.info(
                f"    day-25 detection [{tgt}] (eps<={args.epsilon}): "
                f"R={r['mean_recall']:.3f} P={r['mean_precision']:.3f} "
                f"F1={r['mean_f1']:.3f} det={r['detection_rate']:.1%} "
                f"({r['detected']}/{r['total']})"
            )

    # single-head span: per-gang capture + detection
    Z1 = multihead_bank(prop, theta1)
    cap1 = per_gang_capture_of_span(a_hat, adjacency, gangs, Z1, args.tau)
    det1 = ward_detect(adjacency, Z1, gangs, y, tau=args.tau, epsilon=args.epsilon,
                       threshold=args.threshold, num_cuts=args.ward_num_cuts)
    LOGGER.info(
        f"\n  [H1-span] day-25: R={det1['mean_recall']:.3f} "
        f"P={det1['mean_precision']:.3f} F1={det1['mean_f1']:.3f} "
        f"det={det1['detection_rate']:.1%}"
    )

    # --- day-25 table --------------------------------------------------------
    df = pd.DataFrame({
        "gang": [p.id for p in gangs],
        "size": [p.num_nodes for p in gangs],
        "train": [p.id in train_ids for p in gangs],
        "H1_span": cap1,
        **{name: day25[name]["cap"] for name, _ in configs},
    })
    if args.ceiling_csv is not None and args.ceiling_csv.exists():
        ceil = pd.read_csv(args.ceiling_csv)[["gang", "wallet_d55_K32"]]
        df = df.merge(ceil.rename(columns={"wallet_d55_K32": "ceiling"}), on="gang",
                      how="left")
    df = df.sort_values("size", ascending=False)
    df.to_csv(args.out / "day25_per_gang.csv", index=False)
    names = ["H1_span"] + [n for n, _ in configs]
    cols = (["ceiling"] if "ceiling" in df else []) + names
    LOGGER.info("\nDAY-25 PER-GANG CAPTURE (H1 = span of d cols; attn = own routed column)")
    hdr = f"  {'gang':<5}{'size':>5}{'train':>6}" + "".join(f"{c:>16}" for c in cols)
    LOGGER.info(hdr)
    for _, r in df.iterrows():
        LOGGER.info(f"  {r.gang:<5}{r['size']:>5}{str(bool(r.train)):>6}"
                    + "".join(f"{r[c]:>16.4f}" for c in cols))
    for label, sub in (("train", df[df.train]), ("test", df[~df.train])):
        LOGGER.info(f"  median {label:<9}" + "".join(f"{sub[c].median():>16.4f}"
                                                     for c in cols))
    # attention assignment table (which head each day-25 gang uses)
    for name, _ in configs:
        A = day25[name]["alpha"]
        LOGGER.info(f"\n  {name} attention (rows=gangs, cols=heads):")
        for p_, a_ in zip(gangs, A):
            LOGGER.info(f"    {p_.id:<5} size={p_.num_nodes:<5} "
                        + " ".join(f"{v:.2f}" for v in a_)
                        + ("   [train]" if p_.id in train_ids else ""))

    # --- frozen transfer -----------------------------------------------------
    rows, det_rows = [], []
    for k in range(1, args.transfer_days + 1):
        day = args.day_end + k
        LOGGER.info(f"\n--- transfer day {day} (frozen heads + routing) ---")
        A_unw_d, A_w_d, cls_d, nodes_d = build_graph(args.data_dir, day, day)
        X_d = load_node_features(args.data_dir, nodes_d, day, day,
                                 keep_columns=feature_columns)
        graph_d = build_torch_graph(A_w_d, A_unw_d, cls_d, X_d, weighted=False)
        a_hat_d, adj_d = graph_operators(graph_d)
        gsets_d = connected_components_sets(
            A_unw_d, np.where(cls_d == 1)[0], args.min_gang_size
        )
        gangs_d = make_patterns(gsets_d, "alert", "gang", "g")
        if not gangs_d:
            LOGGER.info("  no gangs -> skip")
            continue
        prop_d = _basis_stack(a_hat_d, X_d.to(a_hat_d.dtype), args.degree,
                              "chebyshev", args.tau)
        P_d = torch.stack(prop_d, dim=0)
        m_P_d = torch.stack([_m_apply(a_hat_d, Pk, args.tau) for Pk in prop_d], dim=0)

        for name, _ in configs:
            cap, alpha, dets = eval_day(
                models[name], P_d, m_P_d, a_hat_d, adj_d, gangs_d, graph_d.y,
                args.tau, args, detect=True,
            )
            rows += [{"day": day, "config": name, "gang": p_.id,
                      "size": p_.num_nodes, "capture": float(c)}
                     for p_, c in zip(gangs_d, cap)]
            for tgt, det in dets.items():
                det_rows.append({"day": day, "config": f"{name}[{tgt}]", **det})
        Z1_d = multihead_bank(prop_d, theta1)
        cap1_d = per_gang_capture_of_span(a_hat_d, adj_d, gangs_d, Z1_d, args.tau)
        rows += [{"day": day, "config": "H1_span", "gang": p_.id,
                  "size": p_.num_nodes, "capture": float(c)}
                 for p_, c in zip(gangs_d, cap1_d)]
        det_rows.append({"day": day, "config": "H1_span",
                         **ward_detect(adj_d, Z1_d, gangs_d, graph_d.y,
                                       tau=args.tau, epsilon=args.epsilon,
                                       threshold=args.threshold,
                                       num_cuts=args.ward_num_cuts)})

    tdf = pd.DataFrame(rows)
    tdf.to_csv(args.out / "transfer_per_gang.csv", index=False)
    ddf = pd.DataFrame(det_rows)
    ddf.to_csv(args.out / "transfer_detection.csv", index=False)

    LOGGER.info("\n" + "=" * 84)
    LOGGER.info("SUMMARY: transfer (unseen days) -- capture and label-free-stop detection")
    LOGGER.info("=" * 84)
    for name in ["H1_span"] + [n for n, _ in configs]:
        sub = tdf[tdf.config == name]
        big = sub[sub["size"] >= 10]
        LOGGER.info(
            f"  {name:<20} capture median={sub.capture.median():.4f} "
            f"large(>=10)={big.capture.median() if len(big) else float('nan'):.4f}"
        )
    LOGGER.info("")
    for name in sorted(ddf.config.unique()):
        d_ = ddf[ddf.config == name]
        LOGGER.info(
            f"  {name:<24} det: R={d_.mean_recall.mean():.3f} "
            f"P={d_.mean_precision.mean():.3f} F1={d_.mean_f1.mean():.3f} "
            f"rate={d_.detection_rate.mean():.1%} "
            f"({int(d_.detected.sum())}/{int(d_.total.sum())})"
        )

    (args.out / "summary.json").write_text(json.dumps({
        "heads": args.heads,
        "day25_train_median": {n: float(df[df.train][n].median()) for n in names},
        "day25_test_median": {n: float(df[~df.train][n].median()) for n in names},
        "transfer_median": {n: float(tdf[tdf.config == n].capture.median())
                            for n in ["H1_span"] + [n_ for n_, _ in configs]},
    }, indent=2) + "\n")
    LOGGER.info(f"\nCSV + JSON -> {args.out}")


if __name__ == "__main__":
    main()
