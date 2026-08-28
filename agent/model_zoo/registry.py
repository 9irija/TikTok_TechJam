"""Model Zoo registry -- the one place an ExperimentConfig.model string
resolves to a concrete builder. Adding a model later (DCNv2, Wide&Deep,
LightGBM -- P2 Extended Model Zoo) means adding one line here, not touching
the orchestrator or experiment runner.
"""
from __future__ import annotations

from typing import Callable

from . import deepfm, fm, fm_bpr

MODELS: dict[str, Callable[..., object]] = {
    "fm": fm.build,
    "deepfm": deepfm.build,
    "fm_bpr": fm_bpr.build,  # P1: pairwise BPR loss instead of pointwise logloss
}


def build(model_name: str, dim: int, n_fields: int, **hyperparams):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(MODELS)}")
    if model_name == "deepfm":
        return MODELS[model_name](dim, n_fields=n_fields, **hyperparams)
    return MODELS[model_name](dim, **hyperparams)
