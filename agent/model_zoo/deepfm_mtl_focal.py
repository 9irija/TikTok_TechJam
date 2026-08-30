"""Multi-Task DeepFM + focal loss on the main task (Lin et al. 2017, "Focal
Loss for Dense Object Detection", arXiv:1708.02002) -- the top pick from an
explicit brainstorm for a genuinely different lever, not another variant of
something already tried.

Why this, and why it's actually different: every model in this project so
far trains with plain BCE (or BPR/listwise/PDAOM's own pairwise/listwise
variants) -- every row contributes to the gradient with equal weight
regardless of how easy or hard it currently is for the model. Focal loss
changes WHICH EXAMPLES MATTER during training, not the model architecture,
not the auxiliary signal set, not the ranking-loss family -- a different
axis entirely from everything else tried in this Model Zoo:

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

where `p_t` is the model's predicted probability for the TRUE class (p if
y=1, 1-p if y=0). As a prediction gets more confidently correct, p_t -> 1
and (1-p_t)^gamma -> 0 -- the modulating factor -- so well-classified
examples contribute almost nothing to the gradient, and the loss
concentrates on examples the model still gets wrong.

Directly motivated by an existing finding, not a guess: the per-segment
diagnosis (P2_FEATURES_AND_RESULTS.md Sec.15) showed deepfm_mtl_v1's
ranking quality is measurably worse on mid-popularity items specifically,
while plain BCE trains on every row uniformly regardless of that
per-segment difficulty. Focal loss should automatically reallocate
gradient toward exactly the harder examples across the whole training set
(mid-popularity items and any other systematically-harder slice included)
instead of being dominated by the easy majority -- a real, checkable
hypothesis, not a hope.

Isolated to ONE variable against the current best: identical architecture
and auxiliary setup to deepfm_mtl.py (same aux_heads, same aux_weight,
same aux_loss_fn), only the MAIN task's loss function changes from
BCEWithLogitsLoss to focal loss. `alpha=0.5` by default (a neutral
class-balance weight) deliberately isolates the hard-example-focusing
effect (`gamma`) from class-imbalance correction (`alpha`'s other
purpose in the original paper) -- this dataset's ~31-34% positive rate
isn't the extreme 1:1000-style imbalance focal loss was originally
designed for, so conflating the two effects would muddy which one, if
either, is doing the work.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Standard binary focal loss, computed from LOGITS via
    `binary_cross_entropy_with_logits(reduction="none")` for the same
    numerically-stable log-sigmoid every other model's BCEWithLogitsLoss
    already relies on -- never takes log(p_t) directly, which could
    under/overflow for very confident, very wrong predictions."""
    def __init__(self, gamma: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        modulating = (1 - p_t).clamp(min=0.0) ** self.gamma
        return (alpha_t * modulating * bce).mean()


class _Net(nn.Module):
    """Identical architecture to deepfm_mtl.py's _Net -- the loss function
    is the only variable this model tests."""
    def __init__(self, dim: int, k: int, hidden: list[int], n_fields: int):
        super().__init__()
        self.V = nn.Embedding(dim, k)
        nn.init.normal_(self.V.weight, 0.0, 0.01)
        self.W = nn.Embedding(dim, 1)
        nn.init.zeros_(self.W.weight)
        self.b = nn.Parameter(torch.zeros(1))

        sizes = [n_fields * k] + list(hidden)
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        self.deep = nn.Sequential(*layers)
        self.deep_out = nn.Linear(sizes[-1], 1)
        self.aux_heads = nn.Linear(sizes[-1], 4)  # is_like/is_follow/is_comment/is_forward, same as deepfm_mtl.py

    def forward(self, X: torch.Tensor):
        E = self.V(X)
        S = E.sum(1)
        fm_inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        fm_linear = self.W(X).sum((1, 2))
        B, Fd, K = E.shape
        h = self.deep(E.reshape(B, Fd * K))
        main_logit = self.b + fm_linear + fm_inter + self.deep_out(h).squeeze(-1)
        aux_logits = self.aux_heads(h)
        return main_logit, aux_logits


class DeepFM_MTL_Focal:
    def __init__(self, dim: int, k: int = 16, hidden: list[int] | None = None,
                 n_fields: int = 5, lr: float = 0.001, l2: float = 1e-6,
                 aux_weight: float = 0.2, focal_gamma: float = 2.0, focal_alpha: float = 0.5,
                 seed: int = 0):
        hidden = list(hidden) if hidden else [128, 64]
        torch.manual_seed(seed)
        self.dim, self.k, self.hidden, self.n_fields = dim, k, hidden, n_fields
        self.aux_weight = aux_weight
        self.net = _Net(dim, k, hidden, n_fields)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=l2)
        self.main_loss_fn = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.aux_loss_fn = nn.BCEWithLogitsLoss()  # unchanged -- isolates the main-task loss as the only variable

    def mtl_step(self, X: np.ndarray, y: np.ndarray, aux: np.ndarray) -> float:
        self.net.train()
        Xt = torch.from_numpy(X).long()
        yt = torch.from_numpy(y).float()
        auxt = torch.from_numpy(aux).float()

        self.opt.zero_grad()
        main_logit, aux_logits = self.net(Xt)
        main_loss = self.main_loss_fn(main_logit, yt)
        aux_loss = self.aux_loss_fn(aux_logits, auxt)
        loss = main_loss + self.aux_weight * aux_loss
        loss.backward()
        self.opt.step()
        return float(main_loss.item())  # loss curve tracks the main task only, same convention as deepfm_mtl.py

    def predict(self, X: np.ndarray, bs: int = 200_000) -> np.ndarray:
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                Xt = torch.from_numpy(X[i:i + bs]).long()
                main_logit, _ = self.net(Xt)
                out.append(main_logit.numpy())
        return np.concatenate(out)

    def get_state(self):
        return {k: v.clone() for k, v in self.net.state_dict().items()}

    def set_state(self, state) -> None:
        self.net.load_state_dict(state)

    def num_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))


def build(dim: int, n_fields: int = 5, **hp) -> DeepFM_MTL_Focal:
    keys = {"k", "hidden", "lr", "l2", "aux_weight", "focal_gamma", "focal_alpha", "seed"}
    return DeepFM_MTL_Focal(dim, n_fields=n_fields, **{k: v for k, v in hp.items() if k in keys})
