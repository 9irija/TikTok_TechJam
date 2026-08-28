# iter_001 -- fm_bpr_default

## Hypothesis
Starter kit README ranks a pairwise/listwise loss change as its #1 guess for untested headroom -- GAUC/nDCG are ranking metrics, but every model trained so far (FM, DeepFM) uses pointwise logloss. BPR directly optimizes P(positive item ranked above a negative item for the same user), which is exactly what GAUC measures. Same k=16 embedding size as the FM baseline, so any effect is attributable to the loss/objective change alone, not a confounded architecture change.

## Notes
agent/model_zoo/fm_bpr.py -- P1's first new algorithmic direction (a training-objective change, not just a model/hyperparameter tweak).

## Diff vs previous iteration
```json
{
  "__root__": true
}
```
