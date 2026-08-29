"""DeepFM Multi-Task with LEARNED, per-task uncertainty weighting -- P2,
a different lever than re-tuning a fixed scalar (which agent/hpo.py's
Optuna search over `deepfm_mtl_v1`'s `aux_weight` already tried and found
nothing better than 0.2, tagged `regression`).

`deepfm_mtl_v1` combines its 4 auxiliary BCE losses into one number
(mean over tasks) and multiplies the result by one fixed, hand-picked
`aux_weight` shared across all 4 -- is_like, is_follow, is_comment, and
is_forward all get the same weight despite having very different base
rates (is_like is common; is_comment/is_forward are rare) and plausibly
different genuine relevance to `long_view`. Kendall, Gal & Cipolla 2018
("Multi-Task Learning Using Uncertainty to Weigh Losses for Scene
Geometry and Semantics") replaces every fixed task weight with a learned
per-task log-variance parameter, optimized jointly with the network:
`loss = sum_i exp(-log_var_i) * task_loss_i + log_var_i` -- a task whose
loss the model can't easily reduce automatically gets down-weighted
(pushing log_var_i up lowers its precision term's influence) rather than
fighting the shared embedding table for gradient share.

5 learned scalars total: 1 for the main task, 4 for the individual aux
tasks (not a single combined aux weight -- the whole point is letting
is_comment/is_forward find their own natural weight, not inherit
is_like's). Kept in their own optimizer param group with weight_decay=0
(L2-regularizing an uncertainty parameter would bias it toward log_var=0,
defeating the point of letting it move freely).

Same architecture as agent/model_zoo/deepfm_mtl.py otherwise (identical
_Net structure) -- the only change is how the 5 task losses are combined,
so any effect is attributable to the weighting scheme alone, isolating it
as the only variable versus deepfm_mtl_v1.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_AUX_TASKS = 4  # is_like, is_follow, is_comment, is_forward -- same as deepfm_mtl.py


class _Net(nn.Module):
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
        self.aux_heads = nn.Linear(sizes[-1], N_AUX_TASKS)

    def forward(self, X: torch.Tensor):
        E = self.V(X)
        S = E.sum(1)
        fm_inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        fm_linear = self.W(X).sum((1, 2))
        B, F_, K = E.shape
        h = self.deep(E.reshape(B, F_ * K))
        main_logit = self.b + fm_linear + fm_inter + self.deep_out(h).squeeze(-1)
        aux_logits = self.aux_heads(h)
        return main_logit, aux_logits


class DeepFM_MTL_Uncertainty:
    def __init__(self, dim: int, k: int = 16, hidden: list[int] | None = None,
                 n_fields: int = 5, lr: float = 0.001, l2: float = 1e-6, seed: int = 0):
        hidden = list(hidden) if hidden else [128, 64]
        torch.manual_seed(seed)
        self.dim, self.k, self.hidden, self.n_fields = dim, k, hidden, n_fields
        self.net = _Net(dim, k, hidden, n_fields)
        # log_vars[0] = main task; log_vars[1:5] = the 4 aux tasks individually.
        # Initialized to 0 -> exp(-0) = 1, an unweighted start (matches aux_weight=1
        # before any adaptation, not deepfm_mtl_v1's hand-picked 0.2 -- the model
        # has to learn its own balance from scratch, that's the whole point).
        self.log_vars = nn.Parameter(torch.zeros(1 + N_AUX_TASKS))
        self.opt = torch.optim.Adam([
            {"params": self.net.parameters(), "weight_decay": l2},
            {"params": [self.log_vars], "weight_decay": 0.0},
        ], lr=lr)

    def mtl_step(self, X: np.ndarray, y: np.ndarray, aux: np.ndarray) -> float:
        self.net.train()
        Xt = torch.from_numpy(X).long()
        yt = torch.from_numpy(y).float()
        auxt = torch.from_numpy(aux).float()

        self.opt.zero_grad()
        main_logit, aux_logits = self.net(Xt)
        main_loss = F.binary_cross_entropy_with_logits(main_logit, yt)
        aux_losses = [F.binary_cross_entropy_with_logits(aux_logits[:, i], auxt[:, i])
                      for i in range(N_AUX_TASKS)]
        task_losses = [main_loss] + aux_losses
        total = sum(torch.exp(-self.log_vars[i]) * task_losses[i] + self.log_vars[i]
                    for i in range(len(task_losses)))
        total.backward()
        self.opt.step()
        return float(main_loss.item())  # matches deepfm_mtl.py's convention -- logged loss
        # curve tracks the main task only, not the uncertainty-weighted total.

    def predict(self, X: np.ndarray, bs: int = 200_000) -> np.ndarray:
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                Xt = torch.from_numpy(X[i:i + bs]).long()
                main_logit, _ = self.net(Xt)
                out.append(main_logit.numpy())
        return np.concatenate(out)

    def learned_weights(self) -> dict[str, float]:
        """exp(-log_var) for each task -- the effective weight the model
        converged to, for post-hoc inspection (not used by training)."""
        names = ["main", "is_like", "is_follow", "is_comment", "is_forward"]
        with torch.no_grad():
            w = torch.exp(-self.log_vars).tolist()
        return dict(zip(names, w))

    def get_state(self):
        return {"net": {k: v.clone() for k, v in self.net.state_dict().items()},
                "log_vars": self.log_vars.detach().clone()}

    def set_state(self, state) -> None:
        self.net.load_state_dict(state["net"])
        with torch.no_grad():
            self.log_vars.copy_(state["log_vars"])

    def num_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters())) + self.log_vars.numel()


def build(dim: int, n_fields: int = 5, **hp) -> DeepFM_MTL_Uncertainty:
    keys = {"k", "hidden", "lr", "l2", "seed"}
    return DeepFM_MTL_Uncertainty(dim, n_fields=n_fields, **{k: v for k, v in hp.items() if k in keys})
