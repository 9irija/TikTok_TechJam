"""Deep(er) auxiliary heads for Multi-Task DeepFM -- a refinement angle on
`deepfm_mtl_v1` never actually tried. Every prior MTL refinement changed
*how much* the auxiliary losses count (uncertainty-weighting, P2 Sec.12)
or *how their gradients combine* with the main task's (PCGrad, P2 Sec.17)
-- neither ever questioned whether the aux heads have enough capacity to
extract a clean signal in the first place. `deepfm_mtl.py`'s `_Net` maps
all 4 auxiliary tasks (`is_like`/`is_follow`/`is_comment`/`is_forward`)
through ONE shared `nn.Linear(hidden[-1], 4)` -- literally the same weight
matrix for every task, no task-specific nonlinearity at all. That's a
real, previously-untested capacity bottleneck: if `is_like` and
`is_forward` need to read different sub-signals out of the same pooled
trunk representation, one shared linear projection can't express that,
only a single blended compromise direction.

This gives each of the 4 auxiliary tasks its own small private MLP
(`Linear(hidden[-1], head_hidden) -> ReLU -> Linear(head_hidden, 1)`,
`head_hidden=32` by default) instead of the single shared linear layer --
same shared trunk/embeddings as `deepfm_mtl_v1` (main task unchanged), only
the auxiliary heads' own capacity is the variable being isolated.

Same object-shaped interface as `deepfm_mtl.py` (`mtl_step`, `predict`,
`get_state`/`set_state`, `num_params`) -- runs through the exact same
`is_mtl` branch in `agent/experiment.py`, no new training-loop code needed.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

N_AUX_TASKS = 4  # is_like, is_follow, is_comment, is_forward -- agent/features.py's AUX_LABEL_FIELDS


class _Net(nn.Module):
    def __init__(self, dim: int, k: int, hidden: list[int], n_fields: int, head_hidden: int):
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

        # One small private MLP per auxiliary task, instead of deepfm_mtl.py's
        # single nn.Linear(sizes[-1], N_AUX_TASKS) shared across all 4.
        self.aux_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(sizes[-1], head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))
            for _ in range(N_AUX_TASKS)
        ])

    def forward(self, X: torch.Tensor):
        E = self.V(X)                              # (B, F, k)
        S = E.sum(1)                                # (B, k)
        fm_inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        fm_linear = self.W(X).sum((1, 2))
        B, F, K = E.shape
        h = self.deep(E.reshape(B, F * K))
        main_logit = self.b + fm_linear + fm_inter + self.deep_out(h).squeeze(-1)
        aux_logits = torch.cat([head(h) for head in self.aux_heads], dim=1)  # (B, 4)
        return main_logit, aux_logits


class DeepFM_MTL_DeepHeads:
    def __init__(self, dim: int, k: int = 16, hidden: list[int] | None = None,
                 n_fields: int = 5, lr: float = 0.001, l2: float = 1e-6,
                 aux_weight: float = 0.2, head_hidden: int = 32, seed: int = 0):
        hidden = list(hidden) if hidden else [128, 64]
        torch.manual_seed(seed)
        self.dim, self.k, self.hidden, self.n_fields = dim, k, hidden, n_fields
        self.aux_weight, self.head_hidden = aux_weight, head_hidden
        self.net = _Net(dim, k, hidden, n_fields, head_hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=l2)
        self.main_loss_fn = nn.BCEWithLogitsLoss()
        self.aux_loss_fn = nn.BCEWithLogitsLoss()

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
        return float(main_loss.item())

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


def build(dim: int, n_fields: int = 5, **hp) -> DeepFM_MTL_DeepHeads:
    keys = {"k", "hidden", "lr", "l2", "aux_weight", "head_hidden", "seed"}
    return DeepFM_MTL_DeepHeads(dim, n_fields=n_fields, **{k: v for k, v in hp.items() if k in keys})
