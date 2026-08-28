# iter_001 -- deepfm_mtl_v1

## Hypothesis
Multi-task learning, not a hyperparameter or input-feature change: adds auxiliary is_like/is_follow/is_comment/is_forward sigmoid heads sharing 'deepfm_regularized's embedding table (valid primary 0.6034507203196051), a shared-bottom setup in the public ESMM/MMoE recsys line. Hypothesis: long_view and these other logged engagement signals share structure (all downstream of genuine enjoyment of the video), so their gradients can regularize the shared embeddings even though only long_view is ever scored. Same fields/k/hidden/lr/l2 as the parent, so any effect is attributable to the multi-task objective alone.

## Notes
agent/model_zoo/deepfm_mtl.py -- first torch model in the Model Zoo; agent/experiment.py's is_mtl branch feeds it agent/features.py's load_aux_labels() alongside (X, y).

## Diff vs previous iteration
```json
{
  "__root__": true
}
```
