# iter_001 -- fm_bpr_slow_and_steady

## Hypothesis
Manual curve-shape analysis, not yet something the Diagnosis Engine checks automatically (a real gap -- it compares best_epoch/epochs_run ratios, not full trajectory shape against a healthy reference): fm_baseline_repro's own valid-primary curve rises steadily for 7 epochs (0.5869 -> 0.6015) before declining -- gradual, sustained convergence. Both prior BPR attempts (default AND regularized) instead plateau by epoch 2-4 and never show that gradual climb -- fast convergence to a shallow optimum, a different failure mode than 'overfitting', which the L2/patience fix in fm_bpr_regularized correctly stabilized (flatter curve) without correctly diagnosing (the real ceiling didn't move). This candidate targets convergence speed directly: lr cut ~3x (0.001 -> 0.0003) for a more gradual trajectory, patience and epoch budget both raised to give that slower process room to actually get there.

## Notes
Hand-authored from direct comparison against fm_baseline_repro's own curve -- not (yet) an automated Diagnosis Engine trigger; see P1_FEATURES_AND_RESULTS.md.

## Diff vs previous iteration
```json
{
  "__root__": true
}
```
