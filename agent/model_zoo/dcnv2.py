"""DCNv2 (Deep & Cross Network v2, Wang et al. 2020 "DCN V2: Improved Deep
& Cross Network and Practical Lessons for Web-scale Learning to Rank
Systems") -- the one architecture-level lever from the roadmap's Model Zoo
section that was still genuinely untried (`docs/POLISH_PASS_RESULTS.md` /
README's former Limitations section listed it alongside Wide&Deep).

Explicitly the LOWEST-priority lever in this project's own ranking (see
CLAUDE.md's "Unexplored headroom" #5) -- every capacity/architecture
change actually tried came back flat or negative (embedding width k=8/16/32
flat; CWM's extra feature domains no gain; DIN's attention block a
ranking_tradeoff, not a clean win; LightGBM's tree-based splits a real
loss vs. the embedding-based models). Built anyway for completeness once
DCNv2 was the one specific architecture named but never checked, with
expectations set accordingly -- not expected to beat `deepfm_mtl_v1`,
worth confirming rather than assuming.

Replaces DeepFM's 2nd-order FM interaction term with a Cross Network:
explicit, bounded-degree feature crosses learned via `n_cross_layers`
stacked layers, each of the form (the paper's own "vector" cross,
Eq. 2 with a full-rank weight matrix, not the more parameter-frugal
low-rank variant -- this benchmark's embedding table is small enough,
~80-dim flattened x0 at k=16/5 fields, that a full-rank cross matrix is
cheap):

    x_{l+1} = x_0 * (W_l @ x_l + b_l) + x_l

run in PARALLEL with a plain deep MLP tower (both read the same flattened
embedding x0), concatenated before one final linear output layer -- the
paper's own "parallel" structure (vs. "stacked", where the deep tower
would read the cross network's output instead of x0 directly).

Torch, not hand-rolled numpy, for the same reason as the whole
`deepfm_mtl.py` family: this is a genuinely new architecture built from
scratch (not an extension of `deepfm.py`'s already hand-derived FM+MLP
backward pass the way `deepfm_bpr.py`/`deepfm_din.py` are), and a
multi-layer cross network's backward pass is exactly the kind of thing
autograd is for rather than a shortcut around.

Plain pointwise BCE (`step`, no special branch needed) -- architecture is
the only variable being isolated here, matching `deepfm_regularized`'s
training objective exactly, so it's directly comparable and can run
through the normal `agent/experiment.py` path (registered in
`registry.py`, no standalone-check-first workaround needed, unlike the
per-user-batched loss variants).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _Net(nn.Module):
    def __init__(self, dim: int, k: int, hidden: list[int], n_fields: int, n_cross_layers: int):
        super().__init__()
        self.V = nn.Embedding(dim, k)
        nn.init.normal_(self.V.weight, 0.0, 0.01)

        x0_dim = n_fields * k
        self.cross_W = nn.ParameterList([nn.Parameter(torch.empty(x0_dim, x0_dim)) for _ in range(n_cross_layers)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(x0_dim)) for _ in range(n_cross_layers)])
        for W in self.cross_W:
            nn.init.xavier_normal_(W)

        sizes = [x0_dim] + list(hidden)
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        self.deep = nn.Sequential(*layers)

        self.out = nn.Linear(x0_dim + sizes[-1], 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        E = self.V(X)                     # (B, F, k)
        B, F, K = E.shape
        x0 = E.reshape(B, F * K)          # (B, F*k) -- the cross network's fixed anchor, every layer

        xl = x0
        for W, b in zip(self.cross_W, self.cross_b):
            xl = x0 * (xl @ W.T + b) + xl  # DCNv2 Eq. 2: x0 * (W_l x_l + b_l) + x_l

        deep_out = self.deep(x0)
        combined = torch.cat([xl, deep_out], dim=1)
        return self.out(combined).squeeze(-1)


class DCNv2:
    def __init__(self, dim: int, k: int = 16, hidden: list[int] | None = None,
                 n_fields: int = 5, n_cross_layers: int = 2,
                 lr: float = 0.001, l2: float = 1e-4, seed: int = 0):
        hidden = list(hidden) if hidden else [128, 64]
        torch.manual_seed(seed)
        self.dim, self.k, self.hidden, self.n_fields, self.n_cross_layers = dim, k, hidden, n_fields, n_cross_layers
        self.net = _Net(dim, k, hidden, n_fields, n_cross_layers)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=l2)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def step(self, X: np.ndarray, y: np.ndarray) -> float:
        self.net.train()
        Xt = torch.from_numpy(X).long()
        yt = torch.from_numpy(y).float()
        self.opt.zero_grad()
        logit = self.net(Xt)
        loss = self.loss_fn(logit, yt)
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def predict(self, X: np.ndarray, bs: int = 200_000) -> np.ndarray:
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                Xt = torch.from_numpy(X[i:i + bs]).long()
                out.append(self.net(Xt).numpy())
        return np.concatenate(out)

    def get_state(self):
        return {k: v.clone() for k, v in self.net.state_dict().items()}

    def set_state(self, state) -> None:
        self.net.load_state_dict(state)

    def num_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))


def build(dim: int, n_fields: int = 5, **hp) -> DCNv2:
    keys = {"k", "hidden", "n_cross_layers", "lr", "l2", "seed"}
    return DCNv2(dim, n_fields=n_fields, **{k: v for k, v in hp.items() if k in keys})
