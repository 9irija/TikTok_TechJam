"""Factorization Machine -- the official baseline model (P0 Model Zoo).

Numerically identical to kuairand-starter-kit/baseline.py's FM class (same
architecture, same Adam hyperparameters, same update equations) so that
reproducing the official baseline's published scores is guaranteed by
construction. The only difference is this version implements the shared
`RankingModel` interface (get_state/set_state/num_params) so the Orchestrator
can checkpoint it the same way it checkpoints every other model in the zoo.

Do not "improve" this file to chase a better baseline reproduction score --
if this file's baseline run doesn't match kuairand-starter-kit/baseline.py's,
that's a bug, not an experiment.
"""
from __future__ import annotations

import copy

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
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
        E = self.V[X]                                    # (B,F,k)
        S = E.sum(1)                                      # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X: np.ndarray, y: np.ndarray) -> float:
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)      # (B,)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def predict(self, X: np.ndarray, bs: int = 200_000) -> np.ndarray:
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])

    def get_state(self):
        return (self.V.copy(), self.W.copy(), np.float32(self.b))

    def set_state(self, state) -> None:
        self.V, self.W, self.b = copy.deepcopy(state)

    def num_params(self) -> int:
        return int(self.V.size + self.W.size + 1)


def build(dim: int, **hp) -> FM:
    """Factory used by agent.model_zoo.registry -- keeps hyperparameter
    filtering (e.g. dropping unknown keys another model might accept) in one
    place instead of scattered across the orchestrator."""
    keys = {"k", "lr", "l2", "seed"}
    return FM(dim, **{k: v for k, v in hp.items() if k in keys})
