# iter_001 -- fm_baseline_repro

## Hypothesis
Reproduce the official FM baseline exactly (k=16, lr=0.001, batch=8192, 5 fields, seed 0) before writing anything original. If our harness can't hit the published validation primary (~0.6016), nothing downstream -- including every later experiment's delta-over-baseline -- can be trusted.

## Notes
Config matches kuairand-starter-kit/baseline_scores.json exactly.

## Diff vs previous iteration
```json
{
  "__root__": true
}
```
