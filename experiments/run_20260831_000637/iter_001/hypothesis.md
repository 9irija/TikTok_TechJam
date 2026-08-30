# iter_001 -- deepfm_mtl_aux_weight_tuning

## Hypothesis
Auxiliary engagement signals (is_like, is_follow, is_comment, is_forward) are extremely sparse (0.1% to 1.8% positive rate). A standard auxiliary weight of 0.2 may exert excessive gradient force on the shared embedding space for these rare events. Reducing aux_weight to 0.05 will preserve the multi-task regularization benefit while keeping shared representations primarily optimized for the main long_view target.

## Notes
LLM-proposed (Phase 4 Research Strategist). priority=0.85

## Diff vs previous iteration
```json
{
  "__root__": true
}
```
