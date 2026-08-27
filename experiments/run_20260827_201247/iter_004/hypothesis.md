# iter_004 -- deepfm_wider

## Hypothesis
If deepfm_default improves on the FM baseline, test whether more MLP capacity (64,32 -> 128,64) buys further gain. If deepfm_default does NOT improve, this experiment is a diagnostic: capacity is unlikely to be the bottleneck (the starter kit's own ablation already showed embedding-dim 8/16/32 barely moves the score for FM), so a wider net failing too would point at the deep pathway itself, not under-parameterization -- a concrete negative result for the log.

## Notes
Same 3-seed rigor as deepfm_default -- this is the wider variant of whichever model may end up being validation-best, so it gets the same scrutiny.

## Diff vs previous iteration
```json
{
  "hyperparams": {
    "before": {
      "k": 16,
      "hidden": [
        64,
        32
      ],
      "lr": 0.001,
      "batch": 8192,
      "epochs": 40,
      "patience": 4
    },
    "after": {
      "k": 16,
      "hidden": [
        128,
        64
      ],
      "lr": 0.001,
      "batch": 8192,
      "epochs": 40,
      "patience": 4
    }
  },
  "parent_id": {
    "before": "fm_baseline_repro",
    "after": "deepfm_default"
  }
}
```
