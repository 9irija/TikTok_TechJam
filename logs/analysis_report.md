# Run & Iteration Log -- Analysis Report

_Generated 2026-08-31 00:35:16 from `logs\run_log.jsonl`_

## Project-best (across all phases -- the actual current result)

- **Node:** `deepfm_mtl_v1` (deepfm_mtl) -- diagnosis `clear_improvement`, 3-seed
- **Valid primary:** 0.6046 +/- 0.0003
- **Test primary:** 0.5974 +/- 0.0006 (+0.0028 vs. official baseline)
- Research Map total: 33 nodes explored across every phase (Foundation/P1/Phase 4/P2) -- see `docs/dashboard.html` for the full tree.

## Phase 0's own convergence run

_The literal single continuous run the Problem Statement's convergence rule describes (read baseline -> iterate -> converge automatically, no LLM yet) -- superseded as "the project result" by the section above, kept here as its own correctly-scoped record, not the whole project's outcome._

- **Run ID:** `run_20260827_201247`
- **Iterations:** 34 total -- 33 succeeded, 1 failed after recovery was exhausted
- **Converged:** True (epsilon=0.002, N=3, per organizer's baseline_scores.json)
- **Validation-best (this run only):** `iter_004` (valid primary = 0.6028)
- **Manual interventions (this run only):** 0 (lower is better -- this is what judges use to score Autonomy; see docs/RESULTS_SUMMARY.md's own Autonomy breakdown for the whole-project picture)
- **Resource usage (this run only):** wall-clock 1083.3s total | LLM tokens 0 | GPU-hours 0.0
- **This run's own best submission (deepfm_wider), hidden-test-style split:** GAUC 0.6643 (+0.0033), nDCG@5 0.5306 (+0.0024), primary-metric delta +0.0029

## Validation-primary trajectory

```
▇▇▇▇▇▇▇▇▇▂▇▇▄▇▇█▇▇▇▇▇▇▇▇ ▅▇▇▇▇▇▇▇
min=0.5483  max=0.6050  n=33
```

Official FM baseline (validation): **0.6016** -- reference line for the trajectory above.

## Per-iteration log

| # | Config | Model | Status | Valid primary | Δ vs baseline (mean) | Wall time | Recovery events | Hypothesis |
|---|---|---|---|---|---|---|---|---|
| iter_001 | `fm_baseline_repro` | fm | ok | 0.6015 | +0.0007 | 92.8s | 0 | Reproduce the official FM baseline exactly (k=16, lr=0.001, batch=8192, 5 fields, seed 0) before wri... |
| iter_002 | `fm_seed_variance` | fm | ok | 0.6014 | +0.0004 | 260.2s | 0 | Run the identical baseline config on 3 seeds to measure our own harness's noise floor before accepti... |
| iter_003 | `deepfm_default` | deepfm | ok | 0.6026 | +0.0027 | 277.4s | 0 | Test whether a deep MLP component over the same shared embeddings captures higher-order, nonlinear f... |
| iter_004 | `deepfm_wider` | deepfm | ok | 0.6028 | +0.0030 | 452.9s | 0 | If deepfm_default improves on the FM baseline, test whether more MLP capacity (64,32 -> 128,64) buys... |
| iter_001 | `fm_bpr_default` | fm_bpr | ok | 0.5981 | -0.0020 | 136.8s | 0 | Starter kit README ranks a pairwise/listwise loss change as its #1 guess for untested headroom -- GA... |
| iter_002 | `fm_wider_k32` | fm | ok | 0.6009 | -0.0002 | 122.9s | 0 | Direct replication check of the starter kit's own ablation (README: k=8/16/32 -> 0.5895/0.5902/0.588... |
| iter_001 | `fm_bpr_regularized` | fm_bpr | ok | 0.5989 | -0.0028 | 165.6s | 0 | Directly acts on fm_bpr_default's own diagnosis, rather than treating BPR as a dead end: its overfit... |
| iter_001 | `fm_bpr_slow_and_steady` | fm_bpr | ok | 0.5994 | -0.0019 | 300.5s | 0 | Manual curve-shape analysis, not yet something the Diagnosis Engine checks automatically (a real gap... |
| iter_001 | `deepfm_regularized` | deepfm | ok | 0.6035 | +0.0036 | 70.1s | 0 | Both deepfm_default and deepfm_wider achieved top validation scores (~0.6028) but were flagged for o... |
| iter_002 | `deepfm_higher_l2` | deepfm | ok | 0.5683 | -0.0345 | 181.3s | 3 | deepfm_regularized achieved our current best primary score (0.6035), but its diagnosis engine flagge... |
| iter_001 | `features_v1` | deepfm | ok | 0.6030 | +0.0026 | 203.3s | 0 | Tests TikTok's own disclosed strong signals as train-only aggregate features on top of the current b... |
| iter_001 | `deepfm_mtl_v1` | deepfm_mtl | ok | 0.6049 | +0.0033 | 435.5s | 0 | Multi-task learning, not a hyperparameter or input-feature change: adds auxiliary is_like/is_follow/... |
| iter_001 | `deepfm_bpr_v1` | deepfm_bpr | ok | 0.5819 | -0.0233 | 186.5s | 0 | Combines two independently-partial results rather than a new, unrelated idea: fm_bpr's pairwise loss... |
| iter_001 | `deepfm_bpr_v1_regularized` | deepfm_bpr | ok | 0.5980 | -0.0019 | 247.7s | 0 | Directly acts on deepfm_bpr_v1's own diagnosis, same pattern that worked for fm_bpr_default->fm_bpr_... |
| iter_001 | `deepfm_mtl_pcgrad_v1` | deepfm_mtl_pcgrad | ok | 0.6027 | +0.0025 | 163.7s | 3 | Refines 'deepfm_mtl_v1''s own multi-task mechanism (valid primary 0.6046394259487893) rather than tr... |
| iter_001 | `deepfm_mtl_click_v1` | deepfm_mtl_click | ok | 0.6050 | +0.0030 | 404.4s | 0 | Adds is_click as a 5th auxiliary head to 'deepfm_mtl_v1''s proven multi-task recipe (valid primary 0... |
| iter_001 | `deepfm_mtl_focal_v1` | deepfm_mtl_focal | ok | 0.6030 | +0.0014 | 322.2s | 0 | Replaces 'deepfm_mtl_v1''s main-task BCE with focal loss (Lin et al. 2017) -- a genuinely different ... |
| lgbm_baseline | `lgbm_baseline` | lgbm | ok | 0.5995 | -0.0000 | 141.9s | 0 | Genuinely different model family (gradient-boosted trees, not learned embeddings), per the user expl... |
| deepfm_mtl_v1_hpo | `deepfm_mtl_v1_hpo` | deepfm_mtl | ok | 0.6015 | -0.0009 | 378.8s | 0 | Optuna search (agent/hpo.py) around 'deepfm_mtl_v1''s hyperparameters, 15 trials at reduced fidelity... |
| deepfm_din_v1 | `deepfm_din_v1` | deepfm_din | ok | 0.6036 | +0.0027 | 79.9s | 0 | DIN-style attention (Zhou et al. 2018) over each user's recent watch history should let the model we... |
| deepfm_mtl_watch_v1 | `deepfm_mtl_watch_v1` | deepfm_mtl_watch | ok | 0.6043 | +0.0036 | 28.1s | 0 | Extends deepfm_mtl_v1 (4 binary auxiliary heads: is_like/is_follow/is_comment/is_forward) with a 5th... |
| deepfm_din_mtl_v1 | `deepfm_din_mtl_v1` | deepfm_din_mtl | ok | 0.6033 | +0.0025 | 60.2s | 0 | Combines two independently-partial results into one model, same reasoning deepfm_bpr_v1 already used... |
| deepfm_mtl_uncertainty_v1 | `deepfm_mtl_uncertainty_v1` | deepfm_mtl_uncertainty | ok | 0.6048 | +0.0038 | 106.0s | 0 | Replaces deepfm_mtl_v1's single fixed aux_weight=0.2 (hand-picked, and already ruled out as tunable-... |
| deepfm_listwise_v1 | `deepfm_listwise_v1` | deepfm_listwise | ok | 0.6033 | +0.0027 | 23.8s | 0 | The starter kit README's own top guess for the loss-function lever was "pairwise (BPR) or listwise (... |
| deepfm_pdaom_v1 | `deepfm_pdaom_v1` | deepfm_pdaom | ok | 0.5483 | -0.0590 | 11.2s | 0 | PDAOM (arXiv:2304.09176) reconstruction: a pairwise EXPONENTIAL AUC-optimization loss (steeper than ... |
| deepfm_lambdarank_v1 | `deepfm_lambdarank_v1` | deepfm_lambdarank | ok | 0.5856 | -0.0144 | 86.5s | 0 | Direct follow-up to the ten-refinements-of-deepfm_mtl_v1 pattern (P2 SS10-19): a structurally differ... |
| dcnv2_v1 | `dcnv2_v1` | dcnv2 | ok | 0.6039 | +0.0027 | 866.3s | 0 | The one specific architecture named in this project roadmap alongside Wide&Deep, never tried -- buil... |
| deepfm_mtl_deep_heads_v1 | `deepfm_mtl_deep_heads_v1` | deepfm_mtl_deep_heads | ok | 0.6038 | +0.0027 | 1774.5s | 0 | Every prior MTL refinement (uncertainty-weighting, PCGrad) changed how much the auxiliary losses cou... |
| deepfm_mtl_gnn_init_v1 | `deepfm_mtl_gnn_init_v1` | deepfm_mtl | ok | 0.6045 | +0.0029 | not recorded | 0 | A genuinely different mechanism from every other lever tried: every prior attempt changed the loss, ... |
| deepfm_mtl_gnn_feature_v1 | `deepfm_mtl_gnn_feature_v1` | deepfm_mtl_gnn_feature | ok | 0.6047 | +0.0033 | not recorded | 0 | Direct diagnosis-driven follow-up to deepfm_mtl_gnn_init_v1 (P2 Sec.27), which found graph-propagate... |
| iter_001 | `deepfm_mtl_aux_weight_tuning` | deepfm_mtl | ok | 0.6036 | +0.0034 | 205.8s | 0 | Auxiliary engagement signals (is_like, is_follow, is_comment, is_forward) are extremely sparse (0.1%... |
| iter_002 | `deepfm_mtl_focal_soft_v1` | deepfm_mtl_focal | ok | 0.6029 | +0.0028 | 203.5s | 0 | deepfm_mtl_focal_v1 regressed because focal_gamma=2.0 was overly aggressive in down-weighting well-c... |
| iter_003 | `deepfm_mtl_capacity_v1` | deepfm_mtl | failed | -- | -- | 51.1s | 4 | deepfm_mtl_v1 currently holds the highest validation score (0.6046). Previous attempts to modify dee... |
| deepfm_mtl_capacity_v1_retry | `deepfm_mtl_capacity_v1` | deepfm_mtl | ok | 0.6036 | +0.0035 | 401.5s | 0 | deepfm_mtl_v1 currently holds the highest validation score (0.6046). Previous attempts to modify dee... |

_Δ vs baseline is computed on the locally-held test split for tracking parity with the organizer's own baseline.py; it is never used to pick a config -- only the Valid primary column drives Convergence Detector / config-selection decisions, per Task Requirement 2 (train+validation only)._

## Code diffs (structured-config)

- **iter_001** (`fm_baseline_repro`): root config, no prior iteration to diff against.
- **iter_002** (`fm_seed_variance`): `parent_id`: None → fm_baseline_repro, `seeds`: [0] → [0, 1, 2]
- **iter_003** (`deepfm_default`): `hyperparams`: {'k': 16, 'lr': 0.001, 'batch': 8192, 'epochs': 40, 'patience': 4} → {'k': 16, 'hidden': [64, 32], 'lr': 0.001, 'batch': 8192, 'epochs': 40, 'patience': 4}, `model`: fm → deepfm
- **iter_004** (`deepfm_wider`): `hyperparams`: {'k': 16, 'hidden': [64, 32], 'lr': 0.001, 'batch': 8192, 'epochs': 40, 'patience': 4} → {'k': 16, 'hidden': [128, 64], 'lr': 0.001, 'batch': 8192, 'epochs': 40, 'patience': 4}, `parent_id`: fm_baseline_repro → deepfm_default
- **iter_001** (`fm_bpr_default`): root config, no prior iteration to diff against.
- **iter_002** (`fm_wider_k32`): `hyperparams`: {'k': 16, 'lr': 0.001, 'batch': 8192, 'epochs': 40, 'patience': 4} → {'k': 32, 'lr': 0.001, 'batch': 8192, 'epochs': 40, 'patience': 4}, `model`: fm_bpr → fm
- **iter_001** (`fm_bpr_regularized`): root config, no prior iteration to diff against.
- **iter_001** (`fm_bpr_slow_and_steady`): root config, no prior iteration to diff against.
- **iter_001** (`deepfm_regularized`): root config, no prior iteration to diff against.
- **iter_002** (`deepfm_higher_l2`): `hyperparams`: {'k': 16, 'lr': 0.001, 'l2': 0.0001, 'batch': 8192, 'epochs': 20, 'patience': 5, 'hidden': [128, 64]} → {'k': 16, 'lr': 0.0005, 'l2': 0.01, 'batch': 2048, 'epochs': 20, 'patience': 5, 'hidden': [64, 32]}, `parent_id`: deepfm_wider → deepfm_regularized
- **iter_001** (`features_v1`): root config, no prior iteration to diff against.
- **iter_001** (`deepfm_mtl_v1`): root config, no prior iteration to diff against.
- **iter_001** (`deepfm_bpr_v1`): root config, no prior iteration to diff against.
- **iter_001** (`deepfm_bpr_v1_regularized`): root config, no prior iteration to diff against.
- **iter_001** (`deepfm_mtl_pcgrad_v1`): root config, no prior iteration to diff against.
- **iter_001** (`deepfm_mtl_click_v1`): root config, no prior iteration to diff against.
- **iter_001** (`deepfm_mtl_focal_v1`): root config, no prior iteration to diff against.
- **lgbm_baseline** (`lgbm_baseline`): `edge_type`: None → draft, `hyperparams`: {'k': 16, 'lr': 0.001, 'batch': 8192, 'epochs': 40, 'patience': 4} → {'objective': 'binary', 'num_leaves': 63, 'learning_rate': 0.05, 'min_data_in_leaf': 50, 'feature_fraction': 0.9, 'bagging_fraction': 0.9, 'num_boost_round': 500, 'early_stopping_rounds': 30}, `model`: fm → lgbm, `parent_id`: None → fm_baseline_repro
- **deepfm_mtl_v1_hpo** (`deepfm_mtl_v1_hpo`): `hyperparams`: {'k': 16, 'lr': 0.001, 'l2': 0.0001, 'batch': 8192, 'epochs': 20, 'patience': 5, 'hidden': [128, 64], 'aux_weight': 0.2} → {'k': 16, 'lr': 0.0022319714240418083, 'l2': 0.0009751103055468681, 'batch': 8192, 'epochs': 20, 'patience': 5, 'hidden': [128, 64], 'aux_weight': 0.3330384922384841}, `parent_id`: deepfm_regularized → deepfm_mtl_v1
- **deepfm_din_v1** (`deepfm_din_v1`): `edge_type`: improve → None, `hyperparams`: {'k': 16, 'lr': 0.001, 'l2': 0.0001, 'batch': 8192, 'epochs': 20, 'patience': 5, 'hidden': [128, 64]} → {'k': 16, 'hidden': [128, 64], 'lr': 0.001, 'l2': 0.0001, 'seq_len': 20, 'epochs': 20, 'patience': 5}, `model`: deepfm → deepfm_din, `parent_id`: deepfm_wider → deepfm_regularized, `seeds`: [0] → [0, 1, 2]
- **deepfm_mtl_watch_v1** (`deepfm_mtl_watch_v1`): `edge_type`: improve → None, `hyperparams`: {'k': 16, 'lr': 0.001, 'l2': 0.0001, 'batch': 8192, 'epochs': 20, 'patience': 5, 'hidden': [128, 64], 'aux_weight': 0.2} → {'k': 16, 'hidden': [128, 64], 'lr': 0.001, 'l2': 1e-06, 'aux_weight': 0.2, 'watch_weight': 0.2, 'epochs': 20, 'patience': 5}, `model`: deepfm_mtl → deepfm_mtl_watch, `parent_id`: deepfm_regularized → deepfm_mtl_v1, `seeds`: [0] → [0, 1, 2]
- **deepfm_din_mtl_v1** (`deepfm_din_mtl_v1`): `edge_type`: improve → None, `hyperparams`: {'k': 16, 'lr': 0.001, 'l2': 0.0001, 'batch': 8192, 'epochs': 20, 'patience': 5, 'hidden': [128, 64]} → {'k': 16, 'hidden': [128, 64], 'lr': 0.001, 'l2': 1e-06, 'aux_weight': 0.2, 'seq_len': 20, 'epochs': 20, 'patience': 5}, `model`: deepfm → deepfm_din_mtl, `parent_id`: deepfm_wider → deepfm_regularized, `seeds`: [0] → [0, 1, 2]
- **deepfm_mtl_uncertainty_v1** (`deepfm_mtl_uncertainty_v1`): `edge_type`: improve → None, `hyperparams`: {'k': 16, 'lr': 0.001, 'l2': 0.0001, 'batch': 8192, 'epochs': 20, 'patience': 5, 'hidden': [128, 64], 'aux_weight': 0.2} → {'k': 16, 'hidden': [128, 64], 'lr': 0.001, 'l2': 1e-06, 'epochs': 20, 'patience': 5}, `model`: deepfm_mtl → deepfm_mtl_uncertainty, `parent_id`: deepfm_regularized → deepfm_mtl_v1, `seeds`: [0] → [0, 1, 2, 3, 4, 5, 6, 7]
- **deepfm_listwise_v1** (`deepfm_listwise_v1`): `edge_type`: improve → None, `hyperparams`: {'k': 16, 'lr': 0.001, 'l2': 0.0001, 'batch': 8192, 'epochs': 20, 'patience': 5, 'hidden': [128, 64]} → {'k': 16, 'hidden': [128, 64], 'lr': 0.001, 'l2': 1e-05, 'max_len': 64, 'batch_users': 256, 'epochs': 20, 'patience': 5}, `model`: deepfm → deepfm_listwise, `parent_id`: deepfm_wider → deepfm_regularized, `seeds`: [0] → [0, 1, 2]
- **deepfm_pdaom_v1** (`deepfm_pdaom_v1`): `edge_type`: improve → None, `hyperparams`: {'k': 16, 'lr': 0.001, 'l2': 0.0001, 'batch': 8192, 'epochs': 20, 'patience': 5, 'hidden': [128, 64]} → {'k': 16, 'hidden': [128, 64], 'lr': 0.001, 'l2': 0.0001, 'max_candidates': 8, 'batch_users': 512, 'epochs': 20, 'patience': 5}, `model`: deepfm → deepfm_pdaom, `parent_id`: deepfm_wider → deepfm_regularized
- **deepfm_lambdarank_v1** (`deepfm_lambdarank_v1`): `edge_type`: improve → None, `hyperparams`: {'k': 16, 'lr': 0.001, 'l2': 0.0001, 'batch': 8192, 'epochs': 20, 'patience': 5, 'hidden': [128, 64]} → {'k': 16, 'hidden': [128, 64], 'lr': 0.001, 'l2': 0.0001, 'max_len': 64, 'batch_users': 256, 'max_pairs_per_user': 10, 'epochs': 20, 'patience': 5}, `model`: deepfm → deepfm_lambdarank, `parent_id`: deepfm_wider → deepfm_regularized
- **dcnv2_v1** (`dcnv2_v1`): `edge_type`: improve → None, `hyperparams`: {'k': 16, 'lr': 0.001, 'l2': 0.0001, 'batch': 8192, 'epochs': 20, 'patience': 5, 'hidden': [128, 64]} → {'k': 16, 'hidden': [128, 64], 'n_cross_layers': 2, 'lr': 0.001, 'l2': 0.0001, 'epochs': 40, 'patience': 4, 'batch': 8192}, `model`: deepfm → dcnv2, `parent_id`: deepfm_wider → deepfm_regularized
- **deepfm_mtl_deep_heads_v1** (`deepfm_mtl_deep_heads_v1`): `edge_type`: improve → None, `hyperparams`: {'k': 16, 'lr': 0.001, 'l2': 0.0001, 'batch': 8192, 'epochs': 20, 'patience': 5, 'hidden': [128, 64], 'aux_weight': 0.2} → {'k': 16, 'hidden': [128, 64], 'lr': 0.001, 'l2': 0.0001, 'aux_weight': 0.2, 'head_hidden': 32, 'epochs': 20, 'patience': 5, 'batch': 8192}, `model`: deepfm_mtl → deepfm_mtl_deep_heads, `parent_id`: deepfm_regularized → deepfm_mtl_v1
- **deepfm_mtl_gnn_init_v1** (`deepfm_mtl_gnn_init_v1`): `edge_type`: improve → None, `hyperparams`: {'k': 16, 'lr': 0.001, 'l2': 0.0001, 'batch': 8192, 'epochs': 20, 'patience': 5, 'hidden': [128, 64], 'aux_weight': 0.2} → {'k': 16, 'hidden': [128, 64], 'lr': 0.001, 'l2': 0.0001, 'aux_weight': 0.2, 'epochs': 20, 'patience': 5, 'batch': 8192, 'gnn_init_n_layers': 2}, `parent_id`: deepfm_regularized → deepfm_mtl_v1
- **deepfm_mtl_gnn_feature_v1** (`deepfm_mtl_gnn_feature_v1`): `hyperparams`: {'k': 16, 'hidden': [128, 64], 'lr': 0.001, 'l2': 0.0001, 'aux_weight': 0.2, 'epochs': 20, 'patience': 5, 'batch': 8192, 'gnn_init_n_layers': 2} → {'k': 16, 'hidden': [128, 64], 'lr': 0.001, 'l2': 0.0001, 'aux_weight': 0.2, 'epochs': 20, 'patience': 5, 'batch': 8192, 'gnn_n_layers': 2}, `model`: deepfm_mtl → deepfm_mtl_gnn_feature, `parent_id`: deepfm_mtl_v1 → deepfm_mtl_gnn_init_v1
- **iter_001** (`deepfm_mtl_aux_weight_tuning`): root config, no prior iteration to diff against.
- **iter_002** (`deepfm_mtl_focal_soft_v1`): `hyperparams`: {'k': 16, 'lr': 0.001, 'l2': 1e-05, 'batch': 2048, 'epochs': 20, 'patience': 3, 'hidden': [64, 32], 'aux_weight': 0.05} → {'k': 16, 'lr': 0.001, 'l2': 1e-05, 'batch': 2048, 'epochs': 20, 'patience': 3, 'hidden': [64, 32], 'aux_weight': 0.1, 'focal_gamma': 1.0, 'focal_alpha': 0.5}, `model`: deepfm_mtl → deepfm_mtl_focal
- **iter_003** (`deepfm_mtl_capacity_v1`): `hyperparams`: {'k': 16, 'lr': 0.001, 'l2': 1e-05, 'batch': 2048, 'epochs': 20, 'patience': 3, 'hidden': [64, 32], 'aux_weight': 0.1, 'focal_gamma': 1.0, 'focal_alpha': 0.5} → {'k': 16, 'lr': 0.001, 'l2': 1e-05, 'batch': 2048, 'epochs': 15, 'patience': 3, 'hidden': [128, 64], 'aux_weight': 0.2}, `model`: deepfm_mtl_focal → deepfm_mtl
- **deepfm_mtl_capacity_v1_retry** (`deepfm_mtl_capacity_v1`): `hyperparams`: {'k': 16, 'lr': 0.001, 'l2': 0.0001, 'batch': 8192, 'epochs': 20, 'patience': 5, 'hidden': [128, 64], 'aux_weight': 0.2} → {'k': 16, 'lr': 0.001, 'l2': 1e-05, 'batch': 2048, 'epochs': 15, 'patience': 3, 'hidden': [128, 64], 'aux_weight': 0.2}, `parent_id`: deepfm_regularized → deepfm_mtl_v1

## Error / recovery events

- **iter_002** [timeout] attempt 1: Experiment 'deepfm_higher_l2' (seed=0) exceeded 240s wall-clock budget; process terminated.
- **iter_002** [fallback] attempt 2: All 1 attempt(s) of 'deepfm_higher_l2' failed; falling back to a 5-epoch degraded run so this iteration still logs a real result.
- **iter_002** [retry] attempt 2: Retry #1 of 'deepfm_higher_l2' succeeded.
- **iter_001** [timeout] attempt 1: Experiment 'deepfm_mtl_pcgrad_v1' (seed=0) exceeded 600s wall-clock budget; process terminated.
- **iter_001** [fallback] attempt 2: All 1 attempt(s) of 'deepfm_mtl_pcgrad_v1' failed; falling back to a 5-epoch degraded run so this iteration still logs a real result.
- **iter_001** [retry] attempt 2: Retry #1 of 'deepfm_mtl_pcgrad_v1' succeeded.
- **iter_003** [timeout] attempt 1: Experiment 'deepfm_mtl_capacity_v1' (seed=0) exceeded 240s wall-clock budget; process terminated.
- **iter_003** [fallback] attempt 2: All 1 attempt(s) of 'deepfm_mtl_capacity_v1' failed; falling back to a 5-epoch degraded run so this iteration still logs a real result.
- **iter_003** [timeout] attempt 2: Experiment 'deepfm_mtl_capacity_v1' (seed=0) exceeded 240s wall-clock budget; process terminated.
- **iter_003** [abandoned] attempt 3: Degraded fallback for 'deepfm_mtl_capacity_v1' also failed -- abandoning this experiment; orchestrator proceeds to the next config.

## Manual interventions

None recorded -- fully autonomous run.
