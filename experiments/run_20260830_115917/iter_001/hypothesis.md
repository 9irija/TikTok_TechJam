# iter_001 -- deepfm_mtl_focal_v1

## Hypothesis
Replaces 'deepfm_mtl_v1''s main-task BCE with focal loss (Lin et al. 2017) -- a genuinely different lever than any architecture, auxiliary-signal, or MTL-mechanism change already tried: it changes which examples matter during training, not the model or the signal set. Directly motivated by the per-segment diagnosis (P2_FEATURES_AND_RESULTS.md Sec.15), which found ranking quality measurably worse on mid-popularity items specifically -- plain BCE trains every row uniformly regardless of that per-segment difficulty; focal loss's (1-p_t)^gamma modulating factor should automatically reallocate gradient toward harder examples across the dataset. gamma=2.0 (paper default), alpha=0.5 (neutral -- isolates the hard-example effect from class-balance correction, a separate question this dataset's mild ~31-34% imbalance doesn't obviously need). Same fields/k/hidden/l2/aux_weight/aux_heads as the parent, so the loss function is the only variable.

## Notes
agent/model_zoo/deepfm_mtl_focal.py -- reuses agent/experiment.py's existing is_mtl branch (same mtl_step(X,y,aux) contract, default aux_fields), wired directly into the real pipeline, not standalone-checked first.

## Diff vs previous iteration
```json
{
  "__root__": true
}
```
