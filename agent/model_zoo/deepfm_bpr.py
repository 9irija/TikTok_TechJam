"""DeepFM trained with BPR pairwise ranking loss (P2) -- combining two
independently-partial results from earlier in this project instead of
trying a third, unrelated idea: `fm_bpr` (P1) showed the loss/metric-
alignment direction is real but plateaued ~0.002 below the FM baseline in
plain FM form (3 diagnosis-driven rounds, P1_FEATURES_AND_RESULTS.md); the
deep component (`deepfm`/`deepfm_regularized`) independently, separately
proved to help. Neither was ever combined with the other. Flagged as "a
natural, cheap extension of what's already built" in this project's own
roadmap notes (CLAUDE.md) since before either FM_BPR round concluded, and
never actually attempted until now.

Architecture: identical DeepFM backbone (shared field-embedding table ->
FM linear + 2nd-order interaction, deep MLP over concatenated embeddings)
as `agent/model_zoo/deepfm.py`/`deepfm_mtl.py`, but trained on the same
BPR pairwise objective as `agent/model_zoo/fm_bpr.py` instead of pointwise
logloss: `agent/experiment.py`'s existing `is_bpr = hasattr(model,
"bpr_step")` branch (added for `fm_bpr`, framework-agnostic -- it just
calls `model.bpr_step(X_pos, X_neg)` and `model.predict(X)` on whatever
plain numpy arrays `agent.model_zoo.fm_bpr.sample_bpr_batches` yields)
picks this up with ZERO changes to the training loop. This is the first
Model Zoo entry that's both torch AND wired into the real pipeline
(registry.py, Research Map, Multi-Fidelity Runner) from day one, rather
than standalone-checked first -- justified because the training-loop
integration risk here is genuinely near zero (the exact same `bpr_step`/
`predict` contract `fm_bpr` already validated in P1, just backed by a
deeper network), unlike deepfm_mtl/deepfm_din which needed new branches.

Why torch for the pairwise gradient through a deep MLP, not hand-derived
like fm_bpr.py's FM version: `fm_bpr.py`'s `bpr_step` already needed a
non-trivial derivation (running the positive-row and negative-row
gradients through the SAME accumulator with opposite signs before one
combined Adam update -- see that file's own docstring). Doing the
equivalent correctly through a multi-layer MLP by hand is exactly the kind
of derivation this project would rather not risk getting subtly wrong;
autograd removes that risk entirely for a training signal it wasn't
originally built to handle.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        E = self.V(X)                               # (B, F, k)
        S = E.sum(1)
        fm_inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        fm_linear = self.W(X).sum((1, 2))
        B, Fd, K = E.shape
        h = self.deep(E.reshape(B, Fd * K))
        return self.b + fm_linear + fm_inter + self.deep_out(h).squeeze(-1)


class DeepFM_BPR:
    def __init__(self, dim: int, k: int = 16, hidden: list[int] | None = None,
                 n_fields: int = 5, lr: float = 0.001, l2: float = 1e-6, seed: int = 0):
        hidden = list(hidden) if hidden else [128, 64]
        torch.manual_seed(seed)
        self.dim, self.k, self.hidden, self.n_fields = dim, k, hidden, n_fields
        self.net = _Net(dim, k, hidden, n_fields)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=l2)

    def bpr_step(self, X_pos: np.ndarray, X_neg: np.ndarray) -> float:
        """Same pairwise objective as agent/model_zoo/fm_bpr.py's bpr_step:
        loss = -log(sigmoid(z_pos - z_neg)) -- `F.logsigmoid` is the
        numerically-stable form of that, so this is exactly the same math,
        not an approximation."""
        self.net.train()
        Xp = torch.from_numpy(X_pos).long()
        Xn = torch.from_numpy(X_neg).long()

        self.opt.zero_grad()
        d = self.net(Xp) - self.net(Xn)
        loss = -F.logsigmoid(d).mean()
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


def build(dim: int, n_fields: int = 5, **hp) -> DeepFM_BPR:
    keys = {"k", "hidden", "lr", "l2", "seed"}
    return DeepFM_BPR(dim, n_fields=n_fields, **{k: v for k, v in hp.items() if k in keys})
