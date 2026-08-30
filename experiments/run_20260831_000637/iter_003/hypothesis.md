# iter_003 -- deepfm_mtl_capacity_v1

## Hypothesis
deepfm_mtl_v1 currently holds the highest validation score (0.6046). Previous attempts to modify deepfm_mtl focused on loss weighting (aux_weight_tuning, focal_soft) or auxiliary task structures (deep_heads, click), which either regressed or hit noise floors. Expanding the deep MLP component capacity from [64, 32] to [128, 64] while retaining the proven auxiliary weight (0.2) and embedding dimension (k=16) will allow the shared feature interactions and high-level representations to better capture complex user-item signals across tasks without over-regularizing the main long_view predictor.

## Notes
LLM-proposed (Phase 4 Research Strategist). priority=0.85

## Diff vs previous iteration
```json
{
  "hyperparams": {
    "before": {
      "k": 16,
      "lr": 0.001,
      "l2": 1e-05,
      "batch": 2048,
      "epochs": 20,
      "patience": 3,
      "hidden": [
        64,
        32
      ],
      "aux_weight": 0.1,
      "focal_gamma": 1.0,
      "focal_alpha": 0.5
    },
    "after": {
      "k": 16,
      "lr": 0.001,
      "l2": 1e-05,
      "batch": 2048,
      "epochs": 15,
      "patience": 3,
      "hidden": [
        128,
        64
      ],
      "aux_weight": 0.2
    }
  },
  "model": {
    "before": "deepfm_mtl_focal",
    "after": "deepfm_mtl"
  }
}
```
