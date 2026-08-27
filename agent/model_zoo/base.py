"""Model Zoo (base) -- common interface (P0).

The Orchestrator and Experiment Runner only ever talk to this interface; they
never need to know whether the concrete model is FM, DeepFM, or something
added later in P2 (DCNv2, Wide&Deep, LightGBM). That's what lets a config's
`model` field pick a model with a single string instead of writing training
code per model -- the point of the Structured Experiment Interface.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RankingModel(Protocol):
    """A pointwise scoring model for the KuaiRand within-user ranking task.

    Every model consumes the same encoded input `X` (int32, shape (N, F))
    produced by kuairand-starter-kit's `data.encode()`, and a binary label
    vector `y` (float32, shape (N,)). Only relative score order within a user
    matters -- nothing here needs to be a calibrated probability.
    """

    def step(self, X: np.ndarray, y: np.ndarray) -> float:
        """One mini-batch gradient step. Returns the batch loss (float)."""
        ...

    def predict(self, X: np.ndarray, bs: int = 200_000) -> np.ndarray:
        """One real-valued score per row."""
        ...

    def get_state(self) -> Any:
        """Deep copy of trainable parameters -- used to checkpoint the
        validation-best model the Convergence Detector flags."""
        ...

    def set_state(self, state: Any) -> None:
        """Restore a previously captured state (e.g. roll back to
        validation-best after early stopping)."""
        ...

    def num_params(self) -> int:
        """Total trainable parameter count -- logged for the Feasibility &
        Practicality resource-usage deliverable."""
        ...
