"""FM trained with BPR pairwise ranking loss (P1) -- the new candidate
direction this pass adds to the Model Zoo.

Same architecture as agent/model_zoo/fm.py's FM (identical embeddings V,
linear term W, bias b, same Adam optimizer) -- the only thing that changes
is the training signal. Pointwise FM asks "predict P(long_view) for this
row"; BPR asks "for this user, rank their positive item above one of their
negative items" -- which is directly what GAUC and nDCG@5 (both *ranking*
metrics, both computed within-user) actually reward.

This is the starter kit README's own #1-ranked untested direction:
"现在是 pointwise logloss，但指标（GAUC / nDCG）是排序指标。换成 pairwise
(BPR)... 目标函数和评测口径对齐，这是我们认为最可能有效的一条。" (loss/metric
mismatch is their top guess for where headroom actually is -- ranked ahead
of sequence modeling, multi-task learning, watch-time modeling, and model
architecture, in that order).

Training loop shape: `agent/experiment.py`'s `run_experiment` special-cases
any model exposing `bpr_step` (vs. the pointwise `step`) to sample
(user, positive_item, negative_item) triples instead of plain row batches --
see `sample_bpr_batches` below. Evaluation is completely unaffected: BPR
models still implement `predict`/`get_state`/`set_state`/`num_params`
identically to every other RankingModel, so the organizer's `evaluate.py`
never needs to know how a model was trained.
"""
from __future__ import annotations

import copy

import numpy as np

from .fm import sigmoid


class FM_BPR:
    def __init__(self, dim: int, k: int = 16, lr: float = 0.001, l2: float = 1e-6, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.dim, self.k = dim, k
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X: np.ndarray):
        E = self.V[X]                                     # (B,F,k)
        S = E.sum(1)                                        # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def _accumulate(self, X: np.ndarray, E: np.ndarray, S: np.ndarray, g: np.ndarray,
                     gW: np.ndarray, gV: np.ndarray) -> float:
        """Adds this batch's contribution (weighted by per-row gradient `g`)
        into the running gW/gV accumulators -- identical scatter-add math to
        FM.step, factored out so bpr_step can call it once for the positive
        rows and once (negated) for the negative rows before a single
        shared Adam update."""
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        return float(g.sum())

    def bpr_step(self, X_pos: np.ndarray, X_neg: np.ndarray) -> float:
        """One gradient step from paired (positive-item-row, negative-item-row)
        batches for the same set of users. Returns the mean BPR loss.

        Derivation: loss = -log(sigmoid(z_pos - z_neg)); with d = z_pos - z_neg,
        dLoss/dd = sigmoid(d) - 1, so dLoss/dz_pos = sigmoid(d)-1 and
        dLoss/dz_neg = 1-sigmoid(d) -- i.e. the negative-row batch gets
        exactly the negated gradient of the positive-row batch, which is
        why _accumulate can be called twice (once with +g, once with -g)
        into the same gW/gV before one combined update.
        """
        B = len(X_pos)
        z_pos, E_pos, S_pos = self.logits(X_pos)
        z_neg, E_neg, S_neg = self.logits(X_neg)
        d = z_pos - z_neg
        g = ((sigmoid(d) - 1.0) / B).astype(np.float32)

        gW = np.zeros_like(self.W); gV = np.zeros_like(self.V)
        b_grad = self._accumulate(X_pos, E_pos, S_pos, g, gW, gV)
        b_grad += self._accumulate(X_neg, E_neg, S_neg, -g, gW, gV)

        gV += self.l2 * self.V
        gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * b_grad

        return float(-np.mean(np.log(sigmoid(d) + 1e-9)))

    def predict(self, X: np.ndarray, bs: int = 200_000) -> np.ndarray:
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])

    def get_state(self):
        return (self.V.copy(), self.W.copy(), np.float32(self.b))

    def set_state(self, state) -> None:
        self.V, self.W, self.b = copy.deepcopy(state)

    def num_params(self) -> int:
        return int(self.V.size + self.W.size + 1)


def build(dim: int, **hp) -> FM_BPR:
    keys = {"k", "lr", "l2", "seed"}
    return FM_BPR(dim, **{k: v for k, v in hp.items() if k in keys})


def build_user_pos_neg_index(users: list, y: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Groups row indices by user into (positive_row_indices, negative_row_indices),
    keeping only users who have at least one of each -- a user with all-positive
    or all-negative impressions gives no pairwise ranking signal at all (the
    exact same "0 < positives < impressions" condition GAUC itself uses in
    evaluate.py, not a new convention invented here)."""
    import collections
    pos: dict[str, list[int]] = collections.defaultdict(list)
    neg: dict[str, list[int]] = collections.defaultdict(list)
    for i, (u, label) in enumerate(zip(users, y)):
        (pos if label else neg)[u].append(i)
    # sorted(), not the raw set -- Python randomizes string hash seeds per
    # process by default (PYTHONHASHSEED), so `pos.keys() & neg.keys()`'s
    # iteration order (and therefore which user a given rng-sampled index
    # lands on in sample_bpr_batches) would otherwise silently differ
    # between two runs of the exact same seed, breaking reproducibility.
    return {u: (np.array(pos[u], dtype=np.int64), np.array(neg[u], dtype=np.int64))
            for u in sorted(pos.keys() & neg.keys())}


def sample_bpr_batches(rng: np.random.Generator, X: np.ndarray,
                        user_index: dict[str, tuple[np.ndarray, np.ndarray]],
                        n_batches: int, batch_size: int):
    """Yields `n_batches` (X_pos, X_neg) pairs of shape (batch_size, F) each,
    sampling `batch_size` eligible users (uniformly, with replacement) per
    batch and one random positive + one random negative row index per
    sampled user."""
    eligible_users = list(user_index.keys())
    if not eligible_users:
        raise ValueError("No users have both a positive and a negative row -- cannot train BPR on this split.")
    for _ in range(n_batches):
        sampled = rng.choice(len(eligible_users), size=batch_size, replace=True)
        pos_idx = np.empty(batch_size, dtype=np.int64)
        neg_idx = np.empty(batch_size, dtype=np.int64)
        for j, ui in enumerate(sampled):
            p, n = user_index[eligible_users[ui]]
            pos_idx[j] = p[rng.integers(len(p))]
            neg_idx[j] = n[rng.integers(len(n))]
        yield X[pos_idx], X[neg_idx]
