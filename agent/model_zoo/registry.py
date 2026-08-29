"""Model Zoo registry -- the one place an ExperimentConfig.model string
resolves to a concrete builder. Adding a model later (DCNv2, Wide&Deep,
LightGBM -- P2 Extended Model Zoo) means adding one line here, not touching
the orchestrator or experiment runner.
"""
from __future__ import annotations

from typing import Callable

from . import deepfm, fm, fm_bpr

try:
    from . import deepfm_bpr, deepfm_mtl
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False  # torch not installed -- every other model in this file still works

MODELS: dict[str, Callable[..., object]] = {
    "fm": fm.build,
    "deepfm": deepfm.build,
    "fm_bpr": fm_bpr.build,  # P1: pairwise BPR loss instead of pointwise logloss
}
if _HAS_TORCH:
    MODELS["deepfm_mtl"] = deepfm_mtl.build  # P2: multi-task DeepFM (torch) -- see deepfm_mtl.py
    MODELS["deepfm_bpr"] = deepfm_bpr.build  # P2: DeepFM + pairwise BPR loss -- see deepfm_bpr.py

_NEEDS_N_FIELDS = {"deepfm", "deepfm_mtl", "deepfm_bpr"}  # models whose forward pass depends on field count


def build(model_name: str, dim: int, n_fields: int, **hyperparams):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(MODELS)}")
    if model_name in _NEEDS_N_FIELDS:
        return MODELS[model_name](dim, n_fields=n_fields, **hyperparams)
    return MODELS[model_name](dim, **hyperparams)
