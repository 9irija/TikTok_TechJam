"""DeepFM + Multi-Task, plus a Watch-Time auxiliary head -- P2, standalone
check first (same verify-before-integrating discipline already used for
LightGBM and DIN: get a real, honest number before investing in full
Model Zoo registry / agent/experiment.py integration).

Extends `deepfm_mtl_v1`'s proven shared-bottom setup
(agent/model_zoo/deepfm_mtl.py: 4 binary auxiliary sigmoid heads --
is_like/is_follow/is_comment/is_forward -- reading the same pooled deep
trunk as the main long_view logit) with a 5th auxiliary head: a
CONTINUOUS regression target, `agent/features.py`'s `load_watch_ratio`
(a clipped, normalized play_time_ms/duration_ms completion ratio). This is
CLAUDE.md's "Unexplored headroom" item #4, "Watch-time modeling: censored
regression on play_time... Still open" -- the one item on that list this
project hadn't tried yet.

Not a leakage concern despite play_time_ms being what `long_view` is
derived from -- see `load_watch_ratio`'s own docstring for why using it as
an auxiliary TRAINING TARGET is a structurally different, already-
established pattern here (same role is_like/is_follow/etc. already play),
not the INPUT-feature leak `agent/features.py`'s module docstring warns
about. The main long_view logit -- the only thing GAUC/nDCG@5 ever score
-- has no access to it at train or inference time either way.

The open empirical question isn't "is this safe" (structurally it is,
same as the other 4 heads) but "does a denser, continuous training signal
regularize the shared embedding table any better than 4 binary ones
alone." Parent in the Research Map (if/when promoted): `deepfm_mtl_v1`,
isolating the watch-time head as the only variable.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

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
        self.aux_heads = nn.Linear(sizes[-1], N_AUX_TASKS)   # shares the deep trunk's last hidden layer
        self.watch_head = nn.Linear(sizes[-1], 1)             # same trunk, one more head

    def forward(self, X: torch.Tensor):
        E = self.V(X)                              # (B, F, k)
        S = E.sum(1)                                # (B, k)
        fm_inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        fm_linear = self.W(X).sum((1, 2))
        B, F, K = E.shape
        h = self.deep(E.reshape(B, F * K))
        main_logit = self.b + fm_linear + fm_inter + self.deep_out(h).squeeze(-1)
        aux_logits = self.aux_heads(h)              # (B, 4)
        watch_pred = self.watch_head(h).squeeze(-1)  # (B,) -- regression, no sigmoid
        return main_logit, aux_logits, watch_pred


class DeepFM_MTL_Watch:
    def __init__(self, dim: int, k: int = 16, hidden: list[int] | None = None,
                 n_fields: int = 5, lr: float = 0.001, l2: float = 1e-6,
                 aux_weight: float = 0.2, watch_weight: float = 0.2, seed: int = 0):
        hidden = list(hidden) if hidden else [128, 64]
        torch.manual_seed(seed)
        self.dim, self.k, self.hidden, self.n_fields = dim, k, hidden, n_fields
        self.aux_weight, self.watch_weight = aux_weight, watch_weight
        self.net = _Net(dim, k, hidden, n_fields)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=l2)
        self.main_loss_fn = nn.BCEWithLogitsLoss()
        self.aux_loss_fn = nn.BCEWithLogitsLoss()
        self.watch_loss_fn = nn.MSELoss()

    def mtl_watch_step(self, X: np.ndarray, y: np.ndarray, aux: np.ndarray, watch: np.ndarray) -> float:
        self.net.train()
        Xt = torch.from_numpy(X).long()
        yt = torch.from_numpy(y).float()
        auxt = torch.from_numpy(aux).float()
        watcht = torch.from_numpy(watch).float()

        self.opt.zero_grad()
        main_logit, aux_logits, watch_pred = self.net(Xt)
        main_loss = self.main_loss_fn(main_logit, yt)
        aux_loss = self.aux_loss_fn(aux_logits, auxt)
        watch_loss = self.watch_loss_fn(watch_pred, watcht)
        loss = main_loss + self.aux_weight * aux_loss + self.watch_weight * watch_loss
        loss.backward()
        self.opt.step()
        return float(main_loss.item())  # logged/plotted loss curve tracks the *main* task only,
        # matching deepfm_mtl.py's own convention -- aux/watch losses are training-time
        # regularizers, not part of what "loss went down" means to a human reading it.

    def predict(self, X: np.ndarray, bs: int = 200_000) -> np.ndarray:
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                Xt = torch.from_numpy(X[i:i + bs]).long()
                main_logit, _, _ = self.net(Xt)
                out.append(main_logit.numpy())
        return np.concatenate(out)

    def get_state(self):
        return {k: v.clone() for k, v in self.net.state_dict().items()}

    def set_state(self, state) -> None:
        self.net.load_state_dict(state)

    def num_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))


def build(dim: int, n_fields: int = 5, **hp) -> DeepFM_MTL_Watch:
    keys = {"k", "hidden", "lr", "l2", "aux_weight", "watch_weight", "seed"}
    return DeepFM_MTL_Watch(dim, n_fields=n_fields, **{k: v for k, v in hp.items() if k in keys})
