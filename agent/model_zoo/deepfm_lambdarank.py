"""DeepFM trained with a LambdaRank-style, nDCG@5-weighted pairwise loss
(Burges et al. 2006, "Learning to Rank with Nonsmooth Cost Functions") --
the most structurally different lever left untried in this project, built
after explicit direction to look beyond incrementally retraining what
already exists.

Why this is different from every pairwise/listwise loss already tried,
not just another variant: `fm_bpr`/`deepfm_bpr` (pairwise BPR) weight
every violated pair EQUALLY -- "the positive should outrank the negative"
matters the same whether that pair sits at rank 2 vs. 3 or rank 40 vs. 41.
`deepfm_listwise` (per-user softmax) considers the whole impression set at
once but has no explicit connection to nDCG@5's actual top-k, discounted
shape. LambdaRank is the one loss family here that's a DIRECT function of
the metric being scored: each pair's gradient is scaled by |delta nDCG@5|
-- the actual change in nDCG@5 that swapping the pair's current ranks
would cause. A pair entirely outside the top 5 (both ranks > 5) gets
essentially zero weight; a pair straddling rank 5 gets a large one. This
naturally concentrates training exactly where nDCG@5 is sensitive,
something none of BPR/listwise/PDAOM's loss shapes do explicitly.

Batching reuses `deepfm_listwise.py`'s per-user padded-batch scheme (same
architecture, same masking convention) -- ranks and IDCG are computed PER
USER, so grouping by user is required regardless of pairwise vs. listwise
framing. Within each user's real (non-padded) impressions, up to
`max_pairs_per_user` (positive, negative) pairs are sampled per step (all
pairs would be O(n_pos * n_neg), unbounded for very active users -- capped
the same way `fm_bpr.py`'s BPR sampling already bounds cost per user).

Ranks are computed from the model's CURRENT scores, `.detach()`-ed before
use -- they are gradient WEIGHTS (scalars), never differentiated through
themselves; only the underlying pairwise score difference
`(s_pos - s_neg)` carries gradient, exactly like BPR's own loss term.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Net(nn.Module):
    """Identical to deepfm_listwise.py's _Net -- same architecture, same
    flexible leading-shape forward pass (plain (B, n_fields) for predict,
    (B, L, n_fields) for a padded per-user training batch)."""
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


def dcg_discount(rank_1_indexed: int) -> float:
    """1/log2(rank+1), the standard DCG position discount -- rank is
    1-indexed (the top-ranked item has rank=1)."""
    return 1.0 / math.log2(rank_1_indexed + 1)


def idcg_at_5(n_positives: int) -> float:
    """Ideal DCG@5 for a user with `n_positives` relevant (binary-relevance)
    items -- the best possible arrangement puts all of them (up to 5) in
    the top 5 positions."""
    k = min(5, n_positives)
    return sum(dcg_discount(p) for p in range(1, k + 1)) if k > 0 else 0.0


def delta_ndcg5_for_pair(rank_pos: int, rank_neg: int, idcg5: float) -> float:
    """|delta nDCG@5| from swapping a positive item currently at
    `rank_pos` with a negative item currently at `rank_neg` (both
    1-indexed, both computed over the SAME user's full real impression
    list). Binary relevance (rel in {0,1}), so the usual
    `abs(2^rel_i - 2^rel_j)` gain term collapses to exactly 1 for any
    positive/negative pair. Only the DISCOUNT at each rank matters, and
    only ranks <= 5 contribute anything (nDCG@5's own definition) -- a
    pair entirely below rank 5 contributes exactly 0, naturally
    concentrating gradient near the top of the ranking without an
    explicit cutoff rule.
    """
    if idcg5 <= 0:
        return 0.0
    d_pos = dcg_discount(rank_pos) if rank_pos <= 5 else 0.0
    d_neg = dcg_discount(rank_neg) if rank_neg <= 5 else 0.0
    return abs(d_pos - d_neg) / idcg5


class DeepFM_LambdaRank:
    def __init__(self, dim: int, k: int = 16, hidden: list[int] | None = None,
                 n_fields: int = 5, lr: float = 0.001, l2: float = 1e-6,
                 max_pairs_per_user: int = 10, seed: int = 0):
        hidden = list(hidden) if hidden else [128, 64]
        torch.manual_seed(seed)
        self.dim, self.k, self.hidden, self.n_fields = dim, k, hidden, n_fields
        self.max_pairs_per_user = max_pairs_per_user
        self.net = _Net(dim, k, hidden, n_fields)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=l2)
        self._rng = np.random.default_rng(seed)

    def lambdarank_step(self, X_padded: np.ndarray, y_padded: np.ndarray, mask: np.ndarray) -> float:
        """X_padded: (B, L, n_fields), y_padded: (B, L) float 0/1,
        mask: (B, L) bool. Returns the mean weighted pairwise loss across
        all sampled pairs in the batch (for logging -- the loss curve)."""
        self.net.train()
        Xt = torch.from_numpy(X_padded).long()
        yt = torch.from_numpy(y_padded).float()
        maskt = torch.from_numpy(mask).bool()

        self.opt.zero_grad()
        logits = self.net(Xt)  # (B, L), grad-tracked

        pair_losses = []
        pair_weights = []
        with torch.no_grad():
            scores_np = logits.detach().numpy()
        B = X_padded.shape[0]
        for b in range(B):
            real = np.where(mask[b])[0]
            if len(real) < 2:
                continue
            y_real = y_padded[b, real]
            pos_local = np.where(y_real > 0.5)[0]
            neg_local = np.where(y_real <= 0.5)[0]
            if len(pos_local) == 0 or len(neg_local) == 0:
                continue

            # Ranks (1-indexed, descending by current score) over this user's REAL impressions only.
            order = np.argsort(-scores_np[b, real])  # indices into `real`, best score first
            rank_of = np.empty(len(real), dtype=np.int64)
            rank_of[order] = np.arange(1, len(real) + 1)

            idcg5 = idcg_at_5(len(pos_local))
            n_pairs = min(self.max_pairs_per_user, len(pos_local) * len(neg_local))
            for _ in range(n_pairs):
                pi = pos_local[self._rng.integers(len(pos_local))]
                ni = neg_local[self._rng.integers(len(neg_local))]
                weight = delta_ndcg5_for_pair(int(rank_of[pi]), int(rank_of[ni]), idcg5)
                if weight <= 0:
                    continue
                s_pos = logits[b, real[pi]]
                s_neg = logits[b, real[ni]]
                pair_losses.append(F.softplus(-(s_pos - s_neg)))  # -log(sigmoid(s_pos - s_neg)), same shape as BPR
                pair_weights.append(weight)

        if not pair_losses:
            return 0.0  # no eligible pairs in this batch (e.g. every user single-class) -- a real, valid outcome
        losses_t = torch.stack(pair_losses)
        weights_t = torch.tensor(pair_weights, dtype=torch.float32)
        loss = (losses_t * weights_t).sum() / weights_t.sum().clamp(min=1e-9)
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


def build(dim: int, n_fields: int = 5, **hp) -> DeepFM_LambdaRank:
    keys = {"k", "hidden", "lr", "l2", "max_pairs_per_user", "seed"}
    return DeepFM_LambdaRank(dim, n_fields=n_fields, **{k: v for k, v in hp.items() if k in keys})
