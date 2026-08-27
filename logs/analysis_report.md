# Run & Iteration Log -- Analysis Report

_Generated 2026-08-27 20:35:08 from `logs\run_log.jsonl`_

## Run overview

- **Run ID:** `run_20260827_201247`
- **Iterations:** 4 total -- 4 succeeded, 0 failed after recovery was exhausted
- **Converged:** True (epsilon=0.002, N=3, per organizer's baseline_scores.json)
- **Validation-best:** `iter_004` (valid primary = 0.6028)
- **Manual interventions:** 0 (lower is better -- this is what judges use to score Autonomy)
- **Resource usage:** wall-clock 1083.3s total | LLM tokens 0 | GPU-hours 0.0
- **Final submission (deepfm_wider), hidden-test-style split:** GAUC 0.6643 (+0.0033), nDCG@5 0.5306 (+0.0024), primary-metric delta +0.0029

## Validation-primary trajectory

```
  ▆█
min=0.6014  max=0.6028  n=4
```

## Per-iteration log

| # | Config | Model | Status | Valid primary | Δ vs baseline (mean) | Wall time | Recovery events | Hypothesis |
|---|---|---|---|---|---|---|---|---|
| iter_001 | `fm_baseline_repro` | fm | ok | 0.6015 | +0.0007 | 92.8s | 0 | Reproduce the official FM baseline exactly (k=16, lr=0.001, batch=8192, 5 fields, seed 0) before wri... |
| iter_002 | `fm_seed_variance` | fm | ok | 0.6014 | +0.0004 | 260.2s | 0 | Run the identical baseline config on 3 seeds to measure our own harness's noise floor before accepti... |
| iter_003 | `deepfm_default` | deepfm | ok | 0.6026 | +0.0027 | 277.4s | 0 | Test whether a deep MLP component over the same shared embeddings captures higher-order, nonlinear f... |
| iter_004 | `deepfm_wider` | deepfm | ok | 0.6028 | +0.0030 | 452.9s | 0 | If deepfm_default improves on the FM baseline, test whether more MLP capacity (64,32 -> 128,64) buys... |

_Δ vs baseline is computed on the locally-held test split for tracking parity with the organizer's own baseline.py; it is never used to pick a config -- only the Valid primary column drives Convergence Detector / config-selection decisions, per Task Requirement 2 (train+validation only)._

## Code diffs (structured-config)

- **iter_001** (`fm_baseline_repro`): root config, no prior iteration to diff against.
- **iter_002** (`fm_seed_variance`): `parent_id`: None → fm_baseline_repro, `seeds`: [0] → [0, 1, 2]
- **iter_003** (`deepfm_default`): `hyperparams`: {'k': 16, 'lr': 0.001, 'batch': 8192, 'epochs': 40, 'patience': 4} → {'k': 16, 'hidden': [64, 32], 'lr': 0.001, 'batch': 8192, 'epochs': 40, 'patience': 4}, `model`: fm → deepfm
- **iter_004** (`deepfm_wider`): `hyperparams`: {'k': 16, 'hidden': [64, 32], 'lr': 0.001, 'batch': 8192, 'epochs': 40, 'patience': 4} → {'k': 16, 'hidden': [128, 64], 'lr': 0.001, 'batch': 8192, 'epochs': 40, 'patience': 4}, `parent_id`: fm_baseline_repro → deepfm_default

## Error / recovery events

None -- every iteration completed on its first attempt.

## Manual interventions

None recorded -- fully autonomous run.
