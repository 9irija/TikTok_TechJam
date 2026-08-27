# iter_003 -- deepfm_default

## Hypothesis
Test whether a deep MLP component over the same shared embeddings captures higher-order, nonlinear feature interactions the FM's pairwise term structurally can't. Starter kit README headroom item #5 ranks 'change model' behind loss function / sequence / multi-task changes, but it is the cheapest of the P0 Model Zoo entries to stand up first and gives later experiments a second model family to branch from, not just tuning the same FM. Run on 3 seeds (matching fm_baseline_repro's own noise-floor rigor from fm_seed_variance) rather than 1 -- a single-seed 'win' over a baseline whose own 5-seed std is 0.0008 is exactly the false-improvement risk fm_seed_variance exists to guard against, so the model this run may end up selecting as validation-best deserves the same scrutiny.

## Notes
Guo et al. 2017 DeepFM architecture; shared field embeddings between the FM and deep components (see agent/model_zoo/deepfm.py docstring).

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
      "k": 16,
      "hidden": [
        64,
        32
      ],
      "lr": 0.001,
      "batch": 8192,
      "epochs": 40,
      "patience": 4
    }
  },
  "model": {
    "before": "fm",
    "after": "deepfm"
  }
}
```
