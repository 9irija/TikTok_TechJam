# iter_002 -- deepfm_mtl_focal_soft_v1

## Hypothesis
deepfm_mtl_focal_v1 regressed because focal_gamma=2.0 was overly aggressive in down-weighting well-classified negatives on highly imbalanced recommendation signals. Relaxing focal_gamma to 1.0 with aux_weight=0.1 will softly focus gradient updates on hard impression pairs without destabilizing the multi-task embedding representations established in deepfm_mtl_v1.

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
      "aux_weight": 0.05
    },
    "after": {
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
    }
  },
  "model": {
    "before": "deepfm_mtl",
    "after": "deepfm_mtl_focal"
  }
}
```
