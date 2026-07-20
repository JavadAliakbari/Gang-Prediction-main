"""Dataset-agnostic collective filter-bank gang detector.

This is the *algorithm*, separated from any particular graph.  The learnable
Chebyshev filter bank, the collective ``lambda_min(Gamma)`` capture objective,
the confusability margin (eq. 40), the Riemannian/projected optimizers and the
RSA / Ward-tree coarsening all live (and stay tested) in
:mod:`src.run_collective_bank_detection`; this module wraps those *primitives*
into a small reusable API with the graph data factored out:

* :class:`DetectorConfig` -- every hyperparameter (no argparse, no globals).
* :class:`GraphData`      -- the only dataset-specific object: the normalized
  adjacency ``a_hat``, raw adjacency, node features ``X`` and node labels ``y``.
  Build it from any ``torch_geometric``-style graph via :meth:`GraphData.from_graph`.
* :class:`CollectiveBankDetector` -- ``fit`` -> ``target_subspace`` -> ``coarsen``
  -> ``evaluate``, or the one-shot :meth:`run`.  Nothing here knows or cares
  whether the gangs came from a synthetic planter or from Elliptic++.

Apply to a new dataset in three lines::

    data = GraphData.from_graph(my_graph)            # any graph
    det = CollectiveBankDetector(DetectorConfig(tau=0.3, conf_weight=5.0))
    result = det.run(data, train_patterns, test_patterns, all_patterns=gangs)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from src.loukas_sgc_detection import (
    evaluate_loukas_patterns,
    graph_operators,
    loukas_coarsen_pytorch,
)
from src.run_collective_bank_detection import (
    _basis_stack,
    _m_apply,
    _train_gang_m_vhat,
    build_bank_subspace,
    channel_gram_cond,
    fit_collective_bank,
    make_negative_sampler,
    retained_energy,
    ward_tree_coarsen,
)

#: coarsen targets built from a frozen bank of filter HEADS (see
#: :meth:`CollectiveBankDetector._multihead_target`) rather than the single
#: shared filter -- all fully inductive except the routed columns of "attn".
MULTIHEAD_TARGETS = ("multihead", "multihead-warm", "attn")


# --------------------------------------------------------------------------- #
# configuration (all algorithm knobs; no argparse, no module globals)
# --------------------------------------------------------------------------- #
@dataclass
class DetectorConfig:
    """Every hyperparameter of the collective-bank detector."""

    # --- learnable filter bank ------------------------------------------------
    degree: int = 10  # Chebyshev/monomial degree K
    basis: str = "chebyshev"  # "chebyshev" | "monomial" | "lanczos"
    tau: float = 0.3  # screened metric M_tau = L + tau I
    epochs: int = 800
    learning_rate: float = 0.02
    ridge: float = 1e-4
    optimizer: str = "projected"  # "projected" | "riemannian"
    softmin_temperature: float = 0.2  # 0 = hard lambda_min
    # what the bank ascends: "lambda_min" (capture + cross-gang separation, carries
    # the m>d capacity wall) | "trace" (mean per-gang capture, no separation term,
    # no capacity wall) | "softmin_diag" (worst gang's capture, no separation).
    # trace/softmin_diag rest on the connectivity constraint: a local-variation
    # coarsener never merges non-adjacent gangs, so cross-gang separation is free
    # (Prop 8.5) and only neighbour separation (the confusability chi) is needed.
    capture_objective: str = "lambda_min"
    # optional supervised head on the same embedding, trained jointly with theta
    label_weight: float = 0.0

    # --- confusability margin (eq. 40) ---------------------------------------
    conf_weight: float = 0.0  # beta; 0 = pure detect-all objective
    conf_reduce: str = "max"  # "max" (eq. 40) | "mean"
    conf_delta: float = 0.0  # 0 = hard chi^tau; >0 = delta-leaky cone
    conf_halo_hops: int = 1

    # --- negative "repeller" sets --------------------------------------------
    num_neg: int = 0
    neg_weight: float = 0.0
    neg_temperature: float = 0.1
    neg_size_min: int = 3
    neg_size_max: int = 10

    # --- target subspace handed to the coarsener -----------------------------
    # "bank"            span(Z) of the single shared filter (d cols, inductive)
    # "indicators"      v_hat projected onto span(Z) (capped at bank capture)
    # "dictionary"      Theorem 6.2 closed-form projection onto the FULL
    #                   dictionary (per-gang capture ceiling; needs node sets)
    # "bank+dictionary" both concatenated
    # "multihead"       span of `heads` filter banks trained cold by capture
    #                   ascent (heads*d cols, inductive)
    # "multihead-warm"  span of k-means-compressed closed-form gang filters,
    #                   zero training by default (heads*d cols, inductive)
    # "attn"            multihead-warm span + KQV-routed per-candidate columns
    #                   (routing frozen; candidate node sets needed at eval)
    coarsen_target: str = "bank"
    structural_width: int = 0
    indicator: str = "degree_weighted"  # capture/energy indicator
    # --- multi-head / attention head bank (multihead* / attn targets) --------
    heads: int = 4  # number of filter heads (fixed, independent of #gangs)
    head_epochs: int = 300  # cold training epochs for "multihead"
    head_finetune: int = 0  # gentle (lr/10) fine-tune epochs for warm heads

    # --- coarsening + detection ----------------------------------------------
    coarsening_method: str = "ward-tree"
    coarsening_laplacian: str = "symmetric"  # "symmetric" | "combinatorial"
    reduction: float = 0.7
    epsilon: float | None = 0.5  # RSA distortion budget (None -> reduction only)
    epsilon_ramp_levels: int = 5
    max_levels: int = 1000
    ward_stop: str = "epsilon"  # "epsilon" | "f1"
    ward_num_cuts: int = 200
    threshold: float = 0.51

    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# graph data (the only dataset-specific object)
# --------------------------------------------------------------------------- #
@dataclass
class GraphData:
    """Everything the algorithm needs from a graph, and nothing else.

    * ``a_hat``     -- symmetric-normalized adjacency (self-loops), the operator
      the filter bank propagates on.
    * ``adjacency`` -- raw sparse weight matrix ``W`` (for degrees / coarsening).
    * ``X``         -- node feature matrix ``(N, d)`` (the filter's input signal).
    * ``y``         -- node class labels ``(N,)`` (gang nodes marked ``1``); used
      only for evaluation and, optionally, negative sampling.
    """
    edge_index: torch.Tensor
    a_hat: torch.Tensor
    adjacency: torch.Tensor
    X: torch.Tensor
    y: torch.Tensor

    @classmethod
    def from_graph(cls, graph, *, features: "torch.Tensor | None" = None) -> "GraphData":
        """Build from any ``torch_geometric``-style graph (``edge_index``, ``x``, ``y``).

        ``features`` overrides ``graph.x`` (e.g. to swap real node features for a
        random structural range-finder).  The features are cast to the operator's
        dtype/device so the bank stays numerically consistent.
        """
        edge_index = graph.edge_index
        a_hat, adjacency = graph_operators(graph)
        X = graph.x if features is None else features
        X = X.to(device=a_hat.device, dtype=a_hat.dtype)
        y = graph.y.to(a_hat.device)
        return cls(edge_index=edge_index, a_hat=a_hat, adjacency=adjacency, X=X, y=y)

    @property
    def num_nodes(self) -> int:
        return int(self.a_hat.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.X.shape[1])


# --------------------------------------------------------------------------- #
# the algorithm
# --------------------------------------------------------------------------- #
class CollectiveBankDetector:
    """Collective learnable filter-bank gang detector (dataset-agnostic).

    Usage::

        det = CollectiveBankDetector(DetectorConfig(...))
        det.fit(data, train_patterns)               # learn Theta*
        basis = det.target_subspace(data, train_patterns)   # R = span(Z)
        coarsening, traj = det.coarsen(data, basis, train_patterns)
        report = det.evaluate(data, coarsening, {"train": ..., "test": ...})

    or the one-shot :meth:`run`.  After :meth:`fit`, ``theta_`` and ``fit_info_``
    hold the learned filter and its training diagnostics.
    """

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self.theta_: torch.Tensor | None = None
        self.fit_info_: dict | None = None
        # frozen multi-head / attention state (multihead* / attn targets): built
        # once from the training day's patterns, then reused verbatim -- copy it
        # onto a transfer detector alongside ``theta_`` to keep transfer honest.
        self.heads_state_: dict | None = None

    # -- internals ---------------------------------------------------------- #
    def _negative_sampler(self, data: GraphData, train_patterns: list):
        c = self.config
        if c.num_neg <= 0 or c.neg_weight <= 0.0:
            return None
        # avoid every known gang node so negatives are genuine background sets
        avoid = torch.nonzero(data.y == 1, as_tuple=False).flatten().tolist()
        for p in train_patterns:
            avoid.extend(int(v) for v in p.node_indices)
        edge_index = data.adjacency.coalesce().indices()
        return make_negative_sampler(
            edge_index,
            data.num_nodes,
            num_sets=c.num_neg,
            size_min=c.neg_size_min,
            size_max=c.neg_size_max,
            avoid=sorted(set(avoid)),
            rng=np.random.default_rng(c.seed + 1),
        )

    # -- steps -------------------------------------------------------------- #
    def fit(self, data: GraphData, train_patterns: list, *, label_y=None,
            label_idx=None) -> "CollectiveBankDetector":
        """Learn the filter bank ``Theta*`` on the *training* gangs.

        ``label_y`` / ``label_idx`` optionally attach a supervised node head
        (config ``label_weight`` > 0) trained jointly with the filter.
        """

        c = self.config
        self.fit_info_ = fit_collective_bank(
            data.a_hat,
            data.adjacency,
            train_patterns,
            data.X,
            degree=c.degree,
            epochs=c.epochs,
            learning_rate=c.learning_rate,
            ridge=c.ridge,
            fit_seed=c.seed,
            tau=c.tau,
            neg_sampler=self._negative_sampler(data, train_patterns),
            neg_weight=c.neg_weight,
            neg_temperature=c.neg_temperature,
            softmin_temperature=c.softmin_temperature,
            basis=c.basis,
            conf_weight=c.conf_weight,
            conf_reduce=c.conf_reduce,
            conf_delta=c.conf_delta,
            conf_halo_hops=c.conf_halo_hops,
            optimizer_kind=c.optimizer,
            capture_objective=c.capture_objective,
            label_weight=c.label_weight,
            label_y=label_y,
            label_idx=label_idx,
        )
        self.theta_ = self.fit_info_["theta"]
        return self

    def _multihead_target(self, data: GraphData, patterns: list) -> torch.Tensor:
        """Multi-head / attention coarsening targets (frozen inductive heads).

        On the FIRST call the heads are built from ``patterns`` (the training
        day's train gangs) and stored in ``heads_state_``; every later call --
        including transfer days -- reuses them frozen.

        * ``multihead``       H heads trained cold by mean-capture ascent;
                              target = concatenated span (H*d columns).
        * ``multihead-warm``  heads = k-means centroids of the closed-form
                              per-gang filters (Theorem 6.2); optional gentle
                              (lr/10) fine-tune via ``head_finetune``; target =
                              concatenated span.  Zero training by default.
        * ``attn``            the multihead-warm span PLUS one KQV-routed column
                              per pattern in ``patterns`` (cosine attention of
                              the pattern's spectral descriptor against frozen
                              cluster-mean keys).  The routed columns need the
                              candidate node sets of the evaluation day -- the
                              certification regime, not blind discovery.
        """

        from src.run_attention_bank import (
            cluster_mean_keys,
            kmeans_filters,
            motif_descriptors,
            routed_columns,
        )
        from src.run_multihead_bank import (
            _m_apply_cached,
            closed_form_gang_filters,
            multihead_bank,
            train_multihead,
            unit_channels,
        )

        c = self.config
        eps = torch.finfo(data.a_hat.dtype).eps
        prop = _basis_stack(data.a_hat, data.X, c.degree, c.basis, c.tau)

        if self.heads_state_ is None:  # training day: build + freeze the heads
            m_vhat = _train_gang_m_vhat(data.a_hat, data.adjacency, patterns, c.tau)
            if c.coarsen_target == "multihead":
                m_prop = [_m_apply(data.a_hat, P, c.tau) for P in prop]
                theta, _ = train_multihead(
                    prop, m_prop, m_vhat, heads=c.heads, objective="trace",
                    epochs=c.head_epochs, lr=c.learning_rate, ridge=c.ridge,
                    seed=c.seed,
                )
                self.heads_state_ = {"theta": theta}
            else:  # warm heads from the closed-form per-gang filters
                _m_apply_cached.clear()
                _m_apply_cached["tau"] = c.tau
                W = closed_form_gang_filters(data.a_hat, prop, m_vhat)
                labels = np.arange(W.shape[0])
                if W.shape[0] > c.heads:
                    W, labels = kmeans_filters(W, c.heads, c.seed)
                if c.head_finetune > 0:
                    m_prop = [_m_apply(data.a_hat, P, c.tau) for P in prop]
                    W, _ = train_multihead(
                        prop, m_prop, m_vhat, heads=W.shape[0], objective="trace",
                        epochs=c.head_finetune, lr=c.learning_rate / 10.0,
                        ridge=c.ridge, seed=c.seed, init=W,
                    )
                self.heads_state_ = {"theta": W}
                if c.coarsen_target == "attn":
                    desc = motif_descriptors(prop, m_vhat)
                    self.heads_state_["keys"] = cluster_mean_keys(
                        desc, labels, W.shape[0]
                    )
                    self.heads_state_["temp"] = 0.1

        st = self.heads_state_
        # span target: per-channel normalization is harmless for a span
        target = multihead_bank(prop, unit_channels(st["theta"]))
        if c.coarsen_target == "attn":
            # routed column per candidate pattern (whole-head normalization --
            # single columns are NOT invariant to per-channel rescaling)
            m_vhat = _train_gang_m_vhat(data.a_hat, data.adjacency, patterns, c.tau)
            desc = motif_descriptors(prop, m_vhat).flatten(1)
            keys = st["keys"] / st["keys"].norm(dim=1, keepdim=True).clamp_min(eps)
            alpha = torch.softmax(desc @ keys.T / st["temp"], dim=1)
            th = st["theta"]
            th = th / th.flatten(1).norm(dim=1).clamp_min(eps).view(-1, 1, 1)
            theta_m = torch.einsum("mh,hkd->mkd", alpha, th)
            z = routed_columns(torch.stack(prop, dim=0), theta_m)
            target = torch.cat([target, z], dim=1)
        return target

    def target_subspace(self, data: GraphData, train_patterns: list) -> torch.Tensor:
        """Coarsening target ``R = span(Z)`` from the learned filter (needs :meth:`fit`)."""

        self._require_fit()
        c = self.config
        if c.coarsen_target in MULTIHEAD_TARGETS:
            return self._multihead_target(data, train_patterns)
        return build_bank_subspace(
            data.a_hat,
            data.adjacency,
            data.X,
            self.theta_,
            c.ridge,
            train_patterns,
            c.tau,
            structural_width=c.structural_width,
            seed=c.seed,
            coarsen_target=c.coarsen_target,
            basis=c.basis,
        )

    def capture(self, data: GraphData, patterns: list) -> dict:
        """Per-gang retained ``M_tau``-energy (capture) of ``patterns``."""

        self._require_fit()
        c = self.config
        return retained_energy(
            data.a_hat,
            data.adjacency,
            patterns,
            data.X,
            self.theta_,
            c.ridge,
            c.tau,
            indicator=c.indicator,
            basis=c.basis,
        )

    def gram_condition(self, data: GraphData) -> dict:
        """Channel-Gram conditioning in each basis at the learned filter (Prop 6.3)."""

        self._require_fit()
        c = self.config
        return {
            "chebyshev": channel_gram_cond(data.a_hat, data.X, self.theta_, c.tau, "chebyshev"),
            "monomial": channel_gram_cond(data.a_hat, data.X, self.theta_, c.tau, "monomial"),
        }

    def coarsen(self, data: GraphData, basis: torch.Tensor, train_patterns: list):
        """Coarsen with target ``R = span(basis)``; returns ``(coarsening, trajectory)``.

        ``trajectory`` is the fine->coarse Ward sweep (``None`` for the greedy
        methods).  ``coarsening_method="ward-tree"`` uses the tree cut chosen by
        ``ward_stop`` (RSA-epsilon budget or best training F1); the other methods
        run the Loukas local-variation greedy under ``reduction``/``epsilon``.
        """

        c = self.config
        if c.coarsening_method == "ward-tree":
            return ward_tree_coarsen(
                data.adjacency,
                basis,
                train_patterns,
                data.y,
                tau=c.tau,
                laplacian=c.coarsening_laplacian,
                threshold=c.threshold,
                stop=c.ward_stop,
                epsilon_budget=(c.epsilon if c.epsilon is not None else math.inf),
                num_cuts=c.ward_num_cuts,
            )
        if c.epsilon is not None:
            budget = dict(
                reduction=c.reduction,
                epsilon=c.epsilon,
                epsilon_ramp_levels=c.epsilon_ramp_levels,
            )
        else:
            budget = dict(reduction=c.reduction)
        coarsening = loukas_coarsen_pytorch(
            data.adjacency,
            basis,
            method=c.coarsening_method,
            laplacian=c.coarsening_laplacian,
            max_levels=c.max_levels,
            tau=c.tau,
            **budget,
        )
        return coarsening, None

    def evaluate(self, data: GraphData, coarsening, splits: dict) -> dict:
        """Alert recall / precision / F1 / detection rate per named split of patterns."""

        report = {}
        for name, patterns in splits.items():
            if not patterns:
                continue
            results, by_label = evaluate_loukas_patterns(
                patterns,
                coarsening.node_to_supernode,
                data.y,
                threshold=self.config.threshold,
            )
            alert = by_label.get("alert", {})
            f1 = float(np.mean([r.f1 for r in results])) if results else 0.0
            report[name] = {
                "detection_rate": alert.get("detection_rate", 0.0),
                "mean_recall": alert.get("mean_recall", 0.0),
                "mean_precision": alert.get("mean_precision", 0.0),
                "mean_f1": f1,
                "detected": int(alert.get("detected", 0)),
                "total": int(alert.get("total", 0)),
            }
        return report

    def run(
        self,
        data: GraphData,
        train_patterns: list,
        test_patterns: list,
        *,
        all_patterns: "list | None" = None,
        label_y=None,
        label_idx=None,
    ) -> dict:
        """Full pipeline: fit -> target -> coarsen -> evaluate, returning a results dict.

        ``label_y`` / ``label_idx`` optionally attach the joint supervised node head
        (active when ``config.label_weight > 0``) during the fit.
        """

        self.fit(data, train_patterns, label_y=label_y, label_idx=label_idx)
        basis = self.target_subspace(data, train_patterns)
        coarsening, trajectory = self.coarsen(data, basis, train_patterns)
        splits = {
            "train": train_patterns,
            "test": test_patterns,
            "all": all_patterns if all_patterns is not None else train_patterns + test_patterns,
        }
        report = self.evaluate(data, coarsening, splits)
        captures = {
            name: self.capture(data, pats) for name, pats in splits.items() if pats
        }
        return {
            "config": self.config.to_dict(),
            "theta": self.theta_,
            "fit": self.fit_info_,
            "basis": basis,
            "coarsening": coarsening,
            "trajectory": trajectory,
            "report": report,
            "captures": captures,
        }

    # -- misc --------------------------------------------------------------- #
    def _require_fit(self) -> None:
        if self.theta_ is None:
            raise RuntimeError("call fit(data, train_patterns) before this step")
