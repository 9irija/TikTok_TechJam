# iter_002 -- fm_wider_k32

## Hypothesis
Direct replication check of the starter kit's own ablation (README: k=8/16/32 -> 0.5895/0.5902/0.5887 test primary; conclusion: 'capacity is not the bottleneck') against OUR harness, at k=32. If it also shows no gain here, that's independent confirmation of a documented dead end using our own pipeline -- worth recording in the Research Map so the Best-First Selector learns to deprioritize pure-capacity changes on future rounds, not just trusting the starter kit's separate ablation script.

## Notes
(none)

## Diff vs previous iteration
```json
{
  "hyperparams": {
    "before": {
      "k": 16,
      "lr": 0.001,
      "batch": 8192,
      "epochs": 40,
      "patience": 4
    },
    "after": {
      "k": 32,
      "lr": 0.001,
      "batch": 8192,
      "epochs": 40,
      "patience": 4
    }
  },
  "model": {
    "before": "fm_bpr",
    "after": "fm"
  }
}
```
