"""DeepFM -- second Model Zoo entry (P0), numpy-only, no torch dependency.

Standard DeepFM design (Guo et al. 2017): a shared field-embedding table `V`
feeds two parallel components whose logits are summed before the sigmoid --

    * FM component  : identical linear + pairwise-interaction term as
                      agent/model_zoo/fm.py (captures 2nd-order crosses
                      cheaply, exactly what the FM baseline already does).
    * Deep component: the same embeddings, concatenated per row and passed
                      through a small ReLU MLP (captures higher-order,
                      nonlinear feature interactions the FM term can't).

This is our answer to the starter kit README's headroom item #5 ("换模型
DeepFM / DCN / xDeepFM") and gives the Orchestrator a genuinely different
hypothesis to test against the baseline (P0 Key Feature: "Model Zoo (base):
FM (baseline) + DeepFM to start, so the agent selects/configures instead of
writing models from scratch").

Everything is hand-rolled (forward + manual backprop + Adam) in the same
style as the starter kit's FM, so no torch/autograd dependency is introduced
at the P0 layer -- this keeps `python run.py` working in the exact
numpy-only environment the starter kit targets.
"""
from __future__ import annotations

import copy

import numpy as np

from .fm import sigmoid


class DeepFM:
    def __init__(self, dim: int, k: int = 16, hidden: list[int] | None = None,
                 n_fields: int = 5, lr: float = 0.001, l2: float = 1e-6, seed: int = 0):
        hidden = list(hidden) if hidden else [64, 32]
        rng = np.random.default_rng(seed)
        self.dim, self.k, self.hidden, self.n_fields = dim, k, hidden, n_fields
        self.lr, self.l2 = lr, l2

        # Shared embedding table (FM order-2 term + deep component input)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)   # FM linear term
        self.b = np.float32(0.0)

        # Deep component: MLP over concat(embeddings), sizes [F*k, h1, h2, ..., 1]
        sizes = [n_fields * k] + hidden + [1]
        self.Ws = [rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])).astype(np.float32)
                   for i in range(len(sizes) - 1)]
        self.bs = [np.zeros(sizes[i + 1], dtype=np.float32) for i in range(len(sizes) - 1)]

        # Adam moment buffers -- one (m, v) pair per trainable array
        self._params = [self.V, self.W] + self.Ws + self.bs
        self._m = [np.zeros_like(p) for p in self._params]
        self._v = [np.zeros_like(p) for p in self._params]
        self.t = 0

    # ---------------------------------------------------------------- forward
    def _deep_forward(self, concat: np.ndarray):
        a = concat
        acts, pre_acts = [a], []
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            z = a @ W + b
            pre_acts.append(z)
            a = np.maximum(z, 0.0)  # ReLU
            acts.append(a)
        z_out = a @ self.Ws[-1] + self.bs[-1]      # (B,1), linear output
        return z_out[:, 0], acts, pre_acts

    def logits(self, X: np.ndarray):
        E = self.V[X]                               # (B,F,k)
        S = E.sum(1)                                 # (B,k)
        fm_inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        fm_linear = self.W[X].sum(1)
        B, F, K = E.shape
        concat = E.reshape(B, F * K)
        deep_out, acts, pre_acts = self._deep_forward(concat)
        z = self.b + fm_linear + fm_inter + deep_out
        return z, E, S, acts, pre_acts

    # --------------------------------------------------------------- backward
    def step(self, X: np.ndarray, y: np.ndarray) -> float:
        B = len(y)
        z, E, S, acts, pre_acts = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)   # dL/dz, shape (B,)

        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)

        # --- FM component gradient (identical to fm.py) ---
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))

        # --- Deep component gradient (manual backprop through the MLP) ---
        gWs = [None] * len(self.Ws)
        gbs = [None] * len(self.bs)
        d_out = g[:, None]                              # dL/dz_out, (B,1), linear output layer
        gWs[-1] = acts[-1].T @ d_out
        gbs[-1] = d_out.sum(0)
        da = d_out @ self.Ws[-1].T                       # (B, h_last)
        for i in range(len(self.Ws) - 2, -1, -1):
            dz = da * (pre_acts[i] > 0)                   # ReLU backward
            gWs[i] = acts[i].T @ dz
            gbs[i] = dz.sum(0)
            da = dz @ self.Ws[i].T

        # da is now dL/d(concat), shape (B, F*k) -> route back into shared V
        B_, F, K = E.shape
        d_concat = da.reshape(B_, F, K)
        np.add.at(gV, X, d_concat)

        gV += self.l2 * self.V
        gW += self.l2 * self.W
        gWs = [g_ + self.l2 * w for g_, w in zip(gWs, self.Ws)]

        self._adam_update([self.V, self.W] + self.Ws + self.bs,
                           [gV, gW] + gWs + gbs)
        self.b -= self.lr * g.sum()

        loss = -np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9))
        return float(loss)

    def _adam_update(self, params, grads):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in zip(params, grads, self._m, self._v):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)

    def predict(self, X: np.ndarray, bs: int = 200_000) -> np.ndarray:
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])

    def get_state(self):
        return (self.V.copy(), self.W.copy(), np.float32(self.b),
                [w.copy() for w in self.Ws], [b.copy() for b in self.bs])

    def set_state(self, state) -> None:
        V, W, b, Ws, bs = copy.deepcopy(state)
        self.V, self.W, self.b, self.Ws, self.bs = V, W, b, Ws, bs
        self._params = [self.V, self.W] + self.Ws + self.bs

    def num_params(self) -> int:
        return int(self.V.size + self.W.size + 1
                    + sum(w.size for w in self.Ws) + sum(b.size for b in self.bs))


def build(dim: int, n_fields: int = 5, **hp) -> DeepFM:
    keys = {"k", "hidden", "lr", "l2", "seed"}
    return DeepFM(dim, n_fields=n_fields, **{k: v for k, v in hp.items() if k in keys})
