"""SGC, APPNP, and 2-layer GCN node encoders for the coarsening comparison.

Each encoder maps node features ``X`` to a ``d``-dim node embedding ``H`` by a
different propagation scheme and is trained by *supervised node classification*
(predict the SAR/alert label on the pattern-train nodes).  ``H`` then serves the
same two jobs as the linear SGC encoders: its span ``R = span(H)`` is the Loukas
coarsening target and a ridge-LDA head reads the label.  Keeping the training
objective and the evaluation identical isolates the only thing that changes --
the propagation architecture:

    sgc    :  H = (A_hat^K X) W            (linear, fixed K-hop diffusion)
    appnp  :  H = PPR_alpha(A_hat) (X W)   (linear, personalized-PageRank teleport)
    gcn2   :  H = A_hat relu(A_hat X W0)   (2-layer, nonlinear)

``A_hat`` is the symmetric normalized adjacency with self-loops returned by
:func:`graph_operators` (the same operator the SGC encoders propagate on).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _spmm(adjacency: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(adjacency, x)


class SGCEncoder(nn.Module):
    """``H = g_theta(A_hat) X W`` with a learnable polynomial filter ``g_theta``.

    ``g_theta(A_hat) = sum_k theta_k A_hat^k`` -- the same SGC filter the joint
    encoder learns, here as a module so the *same* retention objective can train
    ``theta`` (init = ``A_hat^K``, vanilla SGC).
    """

    def __init__(self, in_dim: int, embed_dim: int, num_classes: int, degree: int):
        super().__init__()
        self.degree = degree
        self.theta = nn.Parameter(torch.zeros(degree + 1))
        with torch.no_grad():
            self.theta[-1] = 1.0
        self.embed = nn.Linear(in_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, adjacency: torch.Tensor, x: torch.Tensor):
        filtered = self.theta[0] * x
        propagated = x
        for k in range(1, self.degree + 1):
            propagated = _spmm(adjacency, propagated)
            filtered = filtered + self.theta[k] * propagated
        h = self.embed(filtered)
        return h, self.classifier(h)


class APPNPEncoder(nn.Module):
    """``H0 = X W``; ``H = PPR_alpha(A_hat) H0`` -- transform then teleport-propagate.

    The personalized-PageRank propagation ``Z <- (1-alpha) A_hat Z + alpha H0``
    keeps an ``alpha`` fraction anchored to the un-propagated transform every hop,
    so deep propagation does not over-smooth (the APPNP fix for SGC/GCN depth).
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        num_classes: int,
        degree: int,
        alpha: float,
        learn_alpha: bool = False,
    ):
        super().__init__()
        self.degree = degree
        self.learn_alpha = learn_alpha
        alpha = min(max(float(alpha), 1e-4), 1.0 - 1e-4)
        if learn_alpha:
            # train the teleport by the retention objective (its "theta" analog),
            # kept in (0, 1) via a sigmoid on an unconstrained logit.
            self._alpha_logit = nn.Parameter(torch.logit(torch.tensor(alpha)))
        else:
            self.register_buffer("_alpha_const", torch.tensor(alpha))
        self.embed = nn.Linear(in_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self._alpha_logit) if self.learn_alpha else self._alpha_const

    def forward(self, adjacency: torch.Tensor, x: torch.Tensor):
        a = self.alpha
        h0 = self.embed(x)
        z = h0
        for _ in range(self.degree):
            z = (1.0 - a) * _spmm(adjacency, z) + a * h0
        return z, self.classifier(z)


class GCN2Encoder(nn.Module):
    """``H = A_hat relu(A_hat X W0)`` then ``A_hat H W1`` -- a 2-layer nonlinear GCN.

    The embedding is the first-layer hidden representation ``H`` (``d``-dim); the
    second propagated linear layer produces the class logits.
    """

    def __init__(
        self, in_dim: int, embed_dim: int, num_classes: int, dropout: float = 0.5
    ):
        super().__init__()
        self.lin0 = nn.Linear(in_dim, embed_dim)
        self.lin1 = nn.Linear(embed_dim, num_classes)
        self.dropout = dropout

    def forward(self, adjacency: torch.Tensor, x: torch.Tensor):
        h = F.relu(_spmm(adjacency, self.lin0(x)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = _spmm(adjacency, self.lin1(h))
        return h, logits


def fit_encoder(
    encoder: nn.Module,
    adjacency: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    *,
    epochs: int = 200,
    learning_rate: float = 1e-2,
    weight_decay: float = 5e-4,
    seed: int = 0,
) -> torch.Tensor:
    """Train ``encoder`` by class-weighted cross-entropy on ``train_idx`` nodes.

    Returns the detached ``(N, d)`` node embedding ``H``.  The loss is inverse-
    frequency weighted because SAR/alert nodes are rare; without it the encoder
    collapses to the majority (normal) class and ``H`` carries no label signal.
    """

    torch.manual_seed(seed)
    encoder = encoder.to(dtype=x.dtype)
    optimizer = torch.optim.Adam(
        encoder.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    y = y.to(torch.long)
    train_idx = train_idx.to(torch.long)

    num_classes = int(y.max().item()) + 1
    counts = torch.bincount(y[train_idx], minlength=num_classes).to(dtype=x.dtype)
    class_weight = (counts.sum() / counts.clamp_min(1.0)) / num_classes

    for _ in range(epochs):
        encoder.train()
        optimizer.zero_grad(set_to_none=True)
        _, logits = encoder(adjacency, x)
        loss = F.cross_entropy(
            logits[train_idx], y[train_idx], weight=class_weight.to(logits.dtype)
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss while training encoder")
        loss.backward()
        optimizer.step()

    encoder.eval()
    with torch.no_grad():
        embedding, _ = encoder(adjacency, x)
    return embedding.detach()


def fit_retention_encoder(
    encoder: nn.Module,
    adjacency: torch.Tensor,
    x: torch.Tensor,
    retain_patterns,
    *,
    ridge: float = 1e-3,
    epochs: int = 200,
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    retention_mode: str = "auto",
    retention_reduce: str = "softmin",
    retention_temp: float = 0.1,
    seed: int = 0,
) -> tuple[torch.Tensor, float]:
    """Train an encoder by the **unsupervised retention objective** (no labels).

    This is the SGC joint-encoder objective with the linear filter ``g_theta``
    swapped for an arbitrary encoder ``H = Encoder(A_hat, X)``.  The encoder's
    parameters (``theta`` for SGC, the teleport ``alpha`` and ``W`` for APPNP,
    ``W0, W1`` for the GCN) are trained to maximize ``reduce(lambda(G))`` -- the
    same soft-min / channel-side retention machinery as
    :func:`sgc_detection._retention_lambda_min` -- on the retain-pattern Gram

        G = Y Y^T ,   Y = V^T H        (Y_j = energy-pooled embedding of pattern j).

    The signature matrix ``Y`` is renormalized to unit Frobenius each step so the
    objective is scale-invariant (the encoder cannot cheat by inflating ``H``),
    playing the role of the unit-norm constraints on ``theta``/``W``.  Returns the
    best embedding ``H`` and the best objective value.
    """

    from src.sgc_detection import _retention_lambda_min, pattern_indicator_matrix

    torch.manual_seed(seed)
    encoder = encoder.to(dtype=x.dtype)
    V = pattern_indicator_matrix(
        retain_patterns, adjacency.shape[0], dtype=x.dtype, device=x.device
    )  # (N, m)
    eps = torch.finfo(x.dtype).eps
    optimizer = torch.optim.Adam(
        encoder.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_objective = float("-inf")
    best_state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    for _ in range(epochs):
        encoder.train()
        optimizer.zero_grad(set_to_none=True)
        embedding, _ = encoder(adjacency, x)  # (N, d)
        signatures = V.t() @ embedding  # (m, d) pattern signatures
        signatures = signatures / signatures.norm().clamp_min(eps)
        objective = _retention_lambda_min(
            signatures.t(),  # (d, m): r=d channel dim, m patterns
            mode=retention_mode,
            ridge=ridge,
            reduce=retention_reduce,
            temp=retention_temp,
        )
        if not torch.isfinite(objective):
            raise FloatingPointError("non-finite retention objective while training")
        (-objective).backward()
        optimizer.step()

        value = float(objective.detach().cpu())
        if value > best_objective:
            best_objective = value
            best_state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}

    encoder.load_state_dict(best_state)
    encoder.eval()
    with torch.no_grad():
        embedding, _ = encoder(adjacency, x)
    return embedding.detach(), best_objective
