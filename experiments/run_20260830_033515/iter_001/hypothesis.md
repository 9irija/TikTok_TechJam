# iter_001 -- deepfm_mtl_pcgrad_v1

## Hypothesis
Refines 'deepfm_mtl_v1''s own multi-task mechanism (valid primary 0.6046394259487893) rather than trying a new architecture: PCGrad (Yu et al. 2020, 'Gradient Surgery for Multi-Task Learning') resolves conflicting gradient DIRECTIONS between the main task and the auxiliary tasks on their shared parameters, before combining them -- a different pathology than deepfm_mtl_uncertainty_v1's loss-magnitude weighting (already tried, tied), which cannot address direction conflict even in principle. Simplified to a 2-task formulation (main vs. combined aux loss, not full 5-way pairwise) since aux_heads' 4 outputs share one weight matrix, not cleanly separable parameters. Same fields/k/hidden/l2/aux_weight as the parent, so gradient surgery is the only variable.

## Notes
agent/model_zoo/deepfm_mtl_pcgrad.py -- reuses agent/experiment.py's existing is_mtl branch (same mtl_step(X,y,aux) contract as deepfm_mtl.py), wired directly into the real pipeline.

## Diff vs previous iteration
```json
{
  "__root__": true
}
```
