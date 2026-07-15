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
    build_bank_subspace,
    channel_gram_cond,
    fit_collective_bank,
    make_negative_sampler,
    retained_energy,
    ward_tree_coarsen,
)


# --------------------------------------------------------------------------- #
# configuration (all algorithm knobs; no argparse, no module globals)
# --------------------------------------------------------------------------- #
@dataclass
class DetectorConfig:
    """Every hyperparameter of the collective-bank detector."""

    # --- learnable filter bank ------------------------------------------------
    degree: int = 10  # Chebyshev/monomial degree K
    basis: str = "chebyshev"  # "chebyshev" | "monomial"
    tau: float = 0.3  # screened metric M_tau = L + tau I
    epochs: int = 800
    learning_rate: float = 0.02
    ridge: float = 1e-4
    optimizer: str = "projected"  # "projected" | "riemannian"
    softmin_temperature: float = 0.2  # 0 = hard lambda_min

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
    coarsen_target: str = "bank"  # "bank" | "indicators"
    structural_width: int = 0
    indicator: str = "degree_weighted"  # capture/energy indicator

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

        a_hat, adjacency = graph_operators(graph)
        X = graph.x if features is None else features
        X = X.to(device=a_hat.device, dtype=a_hat.dtype)
        y = graph.y.to(a_hat.device)
        return cls(a_hat=a_hat, adjacency=adjacency, X=X, y=y)

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
    def fit(self, data: GraphData, train_patterns: list) -> "CollectiveBankDetector":
        """Learn the filter bank ``Theta*`` on the *training* gangs."""

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
        )
        self.theta_ = self.fit_info_["theta"]
        return self

    def target_subspace(self, data: GraphData, train_patterns: list) -> torch.Tensor:
        """Coarsening target ``R = span(Z)`` from the learned filter (needs :meth:`fit`)."""

        self._require_fit()
        c = self.config
        return build_bank_subspace(
            data.a_hat,
            data.adjacency,
            train_patterns,
            data.X,
            self.theta_,
            c.ridge,
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
    ) -> dict:
        """Full pipeline: fit -> target -> coarsen -> evaluate, returning a results dict."""

        self.fit(data, train_patterns)
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
