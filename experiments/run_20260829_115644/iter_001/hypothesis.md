# iter_001 -- deepfm_bpr_v1

## Hypothesis
Combines two independently-partial results rather than a new, unrelated idea: fm_bpr's pairwise loss (P1, real loss/metric-alignment signal, plateaued ~0.002 below the FM baseline in plain FM form) with 'deepfm_regularized's deep architecture (valid primary 0.6034507203196051), which independently proved to help. Same fields/k/hidden/l2 as the parent, so any effect is attributable to the pairwise-vs-pointwise objective change alone -- exactly the same isolation discipline fm_bpr_default used against fm_baseline_repro.

## Notes
agent/model_zoo/deepfm_bpr.py -- first torch model wired directly into the real pipeline (registry.py + agent/experiment.py's existing is_bpr branch) rather than standalone-checked first, since the bpr_step/predict contract fm_bpr already validated in P1 carries over with zero training-loop changes needed.

## Diff vs previous iteration
```json
{
  "__root__": true
}
```
