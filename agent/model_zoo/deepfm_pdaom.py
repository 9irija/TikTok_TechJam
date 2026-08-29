"""DeepFM trained with a pairwise exponential AUC-optimization loss using
per-user HARD-PAIR MINING ("PDAOM" -- Personalized Differentiable AUC
Optimization with Maximum violation, arXiv:2304.09176) -- a genuinely
different bet than fm_bpr.py/deepfm_bpr.py's pairwise BPR loss (tested
twice already, both plateauing below the FM baseline), not a repeat of it.

NOTE ON SOURCE FIDELITY: the paper's PDF text could not be machine-
extracted (image-embedded PDF, not searchable text), so this is a
faithful reconstruction from its abstract-level description --
"pairwise exponential loss with difficult pairs of positive and negative
samples within sub-batches grouped by user ID... guide the classifier to
pay attention to the relation between hard-distinguished pairs" -- built
from two well-established, named techniques rather than the paper's own
exact equations: an exponential pairwise AUC-surrogate loss (the classic
AUC-optimization literature, e.g. Herschtal & Raskutti 2004) and
batch-hard positive/negative mining (metric learning, e.g. Hermans,
Beyer & Leibe 2017, "In Defense of the Triplet Loss for Person
Re-Identification"). The exact scaling/margin constants from the
original paper are unknown; this file's defaults are standard, reasonable
choices for this loss family, not a citation of the paper's own tuned
values.

Two concrete differences from BPR, the thing that makes this worth
trying as a genuinely separate bet rather than a third BPR run:
1. Loss shape: BPR uses log(1+exp(-(s_pos-s_neg))) (bounded, sigmoid-
   based). This uses exp(-(s_pos-s_neg)) (steeper for large violations --
   pays disproportionately more attention to badly-mis-ranked pairs, less
   to already-easy ones).
2. Pair selection: BPR samples one UNIFORMLY-RANDOM positive + one
   uniformly-random negative per user per batch. This selects, per user,
   the SINGLE HARDEST pair available that batch (lowest-scoring positive,
   highest-scoring negative among a capped candidate pool) -- concentrates
   gradient on the pairs the model currently gets most wrong.

Same DeepFM backbone as deepfm_regularized (shared field-embedding table
-> FM linear + 2nd-order interaction, deep MLP). Standalone check
(tools/check_pdaom.py) -- hard-pair mining needs a forward pass over a
user's full candidate pool before selecting which single pair to backprop
through, a different shape than agent/experiment.py's existing `is_bpr`
dispatch (which drives fm_bpr/deepfm_bpr's uniform-sampling
`sample_bpr_batches`), so not wired into that path.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


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
        """X: (..., n_fields) -- any leading shape, flattened/reshaped
        (same trick as deepfm_listwise.py's _Net) so this scores both
        plain (B, n_fields) rows (predict) and (B, K, n_fields) padded
        candidate pools (hard-mining) with one code path."""
        lead_shape = X.shape[:-1]
        Xf = X.reshape(-1, X.shape[-1])
        E = self.V(Xf)
        S = E.sum(1)
        fm_inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        fm_linear = self.W(Xf).sum((1, 2))
        N, Fd, K = E.shape
        h = self.deep(E.reshape(N, Fd * K))
        logit = self.b + fm_linear + fm_inter + self.deep_out(h).squeeze(-1)
        return logit.reshape(*lead_shape)


class DeepFM_PDAOM:
    def __init__(self, dim: int, k: int = 16, hidden: list[int] | None = None,
                 n_fields: int = 5, lr: float = 0.001, l2: float = 1e-6, seed: int = 0):
        hidden = list(hidden) if hidden else [128, 64]
        torch.manual_seed(seed)
        self.dim, self.k, self.hidden, self.n_fields = dim, k, hidden, n_fields
        self.net = _Net(dim, k, hidden, n_fields)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=l2)

    def pdaom_step(self, X_pos: np.ndarray, y_pos_mask: np.ndarray,
                   X_neg: np.ndarray, y_neg_mask: np.ndarray) -> float:
        """X_pos: (B, Kp, n_fields) a user's positive candidates, padded;
        X_neg: (B, Kn, n_fields) their negative candidates, padded;
        y_pos_mask/y_neg_mask: (B, Kp)/(B, Kn) bool, True = real (not pad).

        Picks the hardest (lowest-scoring) positive and hardest (highest-
        scoring) negative per user, then applies the exponential AUC loss
        to just that one pair -- gradient flows through the selected
        min/max score exactly like torch.min/max's own backward pass, the
        standard mechanism batch-hard mining relies on."""
        self.net.train()
        Xp = torch.from_numpy(X_pos).long()
        Xn = torch.from_numpy(X_neg).long()
        mp = torch.from_numpy(y_pos_mask).bool()
        mn = torch.from_numpy(y_neg_mask).bool()

        self.opt.zero_grad()
        s_pos = self.net(Xp).masked_fill(~mp, 1e9)     # (B, Kp) -- padding can never be "hardest positive"
        s_neg = self.net(Xn).masked_fill(~mn, -1e9)     # (B, Kn) -- padding can never be "hardest negative"
        hardest_pos = s_pos.min(dim=1).values           # (B,) -- lowest-scoring real positive
        hardest_neg = s_neg.max(dim=1).values           # (B,) -- highest-scoring real negative
        margin = hardest_pos - hardest_neg
        loss = torch.exp(torch.clamp(-margin, max=30.0)).mean()  # clamp: numerical safety only,
        # not a behavioral cap -- prevents exp() overflowing to inf/nan on a badly-mis-ranked
        # pair early in training, while still growing very large (correctly) for such pairs.
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


def build(dim: int, n_fields: int = 5, **hp) -> DeepFM_PDAOM:
    keys = {"k", "hidden", "lr", "l2", "seed"}
    return DeepFM_PDAOM(dim, n_fields=n_fields, **{k: v for k, v in hp.items() if k in keys})
