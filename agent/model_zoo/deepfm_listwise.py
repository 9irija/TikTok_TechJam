"""DeepFM trained with a per-user LISTWISE softmax ranking loss (P2) --
the starter kit's own documented "top guess" for the loss-function lever
("pairwise BPR or listwise per-user softmax"), the half of that guess
never actually tested: `fm_bpr` (P1) and `deepfm_bpr` (P2) both tried
pairwise BPR and both plateaued below the plain FM baseline. This is the
first attempt at the listwise half.

Same DeepFM backbone as agent/model_zoo/deepfm.py (shared field-embedding
table -> FM linear + 2nd-order interaction, deep MLP) -- only the training
signal changes. Instead of "predict P(long_view) for this row" (pointwise)
or "rank this positive above one sampled negative" (pairwise/BPR),
listwise asks "rank this user's ENTIRE shown impression set correctly,
all at once": a per-user softmax cross-entropy (ListNet/Plackett-Luce
style) -- treat a user's row scores as logits over their own impression
set, and push probability mass toward their actual positive item(s). This
is a closer theoretical match to GAUC (within-user rank consistency across
ALL impressions, not just one pair) and nDCG@5 (top-k ranking quality)
than either pointwise or pairwise training.

Training loop shape is structurally different from every other model in
this Model Zoo (plain row-shuffled minibatches): each batch is a set of
USERS, not rows -- every impression belonging to a sampled user is
gathered, padded to a fixed max length (masked out of the softmax, same
masking pattern agent/model_zoo/deepfm_din.py already established for
padding), so the softmax is computed over each user's own impression set,
not the whole batch. Users with fewer than 1 positive AND 1 negative
impression are excluded (same filtering agent/model_zoo/fm_bpr.py already
uses -- a user with no negative to rank below a positive, or no positive
to rank up, gives zero listwise signal either way).

Standalone check only (tools/check_listwise.py) -- this batching scheme
doesn't fit agent/experiment.py's existing row-shuffled or BPR-triple
training loops, so it isn't wired into the Model Zoo registry.
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
        """X: (..., n_fields) -- any leading shape (plain (B, n_fields) for
        row-wise predict, or (B, L, n_fields) for a padded per-user batch
        during listwise training). Flattened to 2D, scored, reshaped back."""
        lead_shape = X.shape[:-1]
        Xf = X.reshape(-1, X.shape[-1])
        E = self.V(Xf)                              # (N, F, k)
        S = E.sum(1)
        fm_inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        fm_linear = self.W(Xf).sum((1, 2))
        N, Fd, K = E.shape
        h = self.deep(E.reshape(N, Fd * K))
        logit = self.b + fm_linear + fm_inter + self.deep_out(h).squeeze(-1)
        return logit.reshape(*lead_shape)


class DeepFM_Listwise:
    def __init__(self, dim: int, k: int = 16, hidden: list[int] | None = None,
                 n_fields: int = 5, lr: float = 0.001, l2: float = 1e-6, seed: int = 0):
        hidden = list(hidden) if hidden else [128, 64]
        torch.manual_seed(seed)
        self.dim, self.k, self.hidden, self.n_fields = dim, k, hidden, n_fields
        self.net = _Net(dim, k, hidden, n_fields)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=l2)

    def listwise_step(self, X_padded: np.ndarray, y_padded: np.ndarray, mask: np.ndarray) -> float:
        """X_padded: (B, L, n_fields) int, y_padded: (B, L) float 0/1,
        mask: (B, L) bool (True = a real row, False = padding). Padded
        positions are masked to -1e9 before the softmax (same convention
        as deepfm_din.py's attention mask) so they never receive any
        probability mass regardless of what garbage field-ids sit in the
        padding slots."""
        self.net.train()
        Xt = torch.from_numpy(X_padded).long()
        yt = torch.from_numpy(y_padded).float()
        maskt = torch.from_numpy(mask).bool()

        self.opt.zero_grad()
        logits = self.net(Xt)                        # (B, L)
        logits = logits.masked_fill(~maskt, -1e9)
        log_probs = F.log_softmax(logits, dim=1)      # (B, L), per-user softmax over their own impressions
        pos_mask = (yt > 0.5) & maskt
        n_pos = pos_mask.sum().clamp(min=1)
        loss = -(log_probs * pos_mask.float()).sum() / n_pos
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


def build(dim: int, n_fields: int = 5, **hp) -> DeepFM_Listwise:
    keys = {"k", "hidden", "lr", "l2", "seed"}
    return DeepFM_Listwise(dim, n_fields=n_fields, **{k: v for k, v in hp.items() if k in keys})
