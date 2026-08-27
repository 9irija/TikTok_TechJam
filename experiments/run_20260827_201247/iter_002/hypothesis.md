# iter_002 -- fm_seed_variance

## Hypothesis
Run the identical baseline config on 3 seeds to measure our own harness's noise floor before accepting any future result as a real improvement. The organizer's own 5-seed std is 0.0008 against a convergence epsilon of 0.002 (only ~2.5 sigma of margin) -- a single noisy run can look like a false improvement or a false convergence, so this is cheap insurance, not busywork.

## Notes
Noise-Aware Convergence Check, P1 table -- pulled forward because it's cheap and protects every later decision from being poisoned by seed noise.

## Diff vs previous iteration
```json
{
  "parent_id": {
    "before": null,
    "after": "fm_baseline_repro"
  },
  "seeds": {
    "before": [
      0
    ],
    "after": [
      0,
      1,
      2
    ]
  }
}
```
