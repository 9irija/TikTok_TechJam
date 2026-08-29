# P2 — Engineered Features, Multi-Task Learning, LightGBM, Hyperparameter Search, Ensembling, DIN Sequence Modeling

Six experiments closing real gaps left open by CLAUDE.md's roadmap
("Multi-Task Feature Exploitation", "TikTok-disclosed features", "Extended
Model Zoo", "Hyperparameter Search", sequence modeling) plus one more tried
on the user's explicit "optimise further" request, run and 3-seed-verified
where the result warranted it. Five of six are negative or mixed results,
reported exactly as measured — the honest signal here is which levers this
specific benchmark actually responds to, not a scoreboard of wins. One of
those negative results (the hyperparameter search) also surfaced and fixed
a real concurrency bug in the Research Map's persistence layer, unrelated
to modeling but a genuine reliability gap worth having found.

## 1. Engineered features (`agent/features.py`) — noise_floor

TikTok has publicly described completion rate and rewatch as strong
recommendation signals. Added four fields on top of the starter kit's base
5 — `video_completion_bucket`, `video_rewatch_bucket`,
`video_fast_skip_bucket`, `author_engagement_bucket` — each a **train-
split-only aggregate** (a video's or author's historical average, looked
up per row). Never a same-row `play_time_ms/duration_ms` ratio: checked
directly against the raw log before writing any code and found ~85%
correlated with the `long_view` label itself — using it as an *input*
feature would have handed the model the answer, not a signal.
`test_features_extended_shape_and_no_leakage` asserts every row sharing a
`video_id` gets an identical bucket, so this isn't just a design intent, it's
enforced.

Also fixed a real, previously-invisible bug this surfaced:
`ExperimentConfig.fields` was accepted by `agent/experiment.py`'s
`_load_encoded` but never actually passed to an encoder — every experiment
silently got the organizer's fixed 5 fields regardless of what a config
asked for. `agent/features.py` is the first thing that ever needed more
than 5 fields, which is what exposed it.

`features_v1` (these 4 fields, `deepfm_regularized`'s architecture,
`parent_id=deepfm_regularized`):

| | Valid primary | vs. parent |
|---|---|---|
| Single seed | 0.6030 | −0.0005 (looked like a regression) |
| 3-seed mean | 0.6037 ± 0.0006 | +0.0002 (looked like a tiny win) |
| vs. parent's own std | 0.6035 ± 0.0002 | — |

Same code, same data, opposite-looking sign depending purely on which
random seed happened to run — a live demonstration of exactly why this
project insists on 3-seed verification before trusting any single-seed
result. 3-seed diagnosis: **`noise_floor`**. TikTok's own disclosed
signals didn't move this specific benchmark beyond noise, at least via
this bucketing approach on top of this architecture. Reported as measured.

### A bug this result surfaced: `ResearchMap.best_node()` had no concept of significance

Once 3-seed-verified, `features_v1` (0.6037) briefly out-scored
`deepfm_regularized` (0.6035) on raw magnitude alone, despite being
diagnosed `noise_floor` in the same breath. `ResearchMap.best_node()` is a
plain numeric leaderboard with no idea what "not statistically
distinguishable" means — anything reading it (the LLM Research
Strategist's prompt, a new candidate's `parent_id`, submission generation)
would have silently treated a coin flip as "the current best."

Fixed with `ResearchMap.best_confirmed_node()`: the highest-scoring `done`
node whose own tag isn't `noise_floor`/`regression`/`mixed`/
`ranking_tradeoff`. Now used everywhere a "what's best" decision is made.
Regression-tested with a synthetic reproduction of the exact scenario
(`test_research_map_best_confirmed_node_skips_noise_floor`). (This method's
implementation changed again in §7 below, once the tree grew a second
competing branch — see that section for why.)

## 2. Multi-task DeepFM (`agent/model_zoo/deepfm_mtl.py`) — clear_improvement, new project-best

The first genuine PyTorch model in this project (`torch>=2.0`, CPU wheel).
Same DeepFM architecture as `deepfm_regularized` — shared embedding table,
FM linear + 2nd-order term, deep MLP — plus four small auxiliary sigmoid
heads reading the same pooled embeddings, trained jointly on `is_like`,
`is_follow`, `is_comment`, `is_forward` (a shared-bottom multi-task setup,
the public ESMM/MMoE line of recsys work). Combined loss:
`main_BCE + 0.2 * aux_BCE`. Only the main `long_view` logit is ever scored
— the auxiliary heads exist purely to shape training via shared gradients
through the embedding table, never touch GAUC/nDCG@5 evaluation.

**Why torch here and nowhere else in the Model Zoo.** FM/DeepFM/FM_BPR
each have one clean, cheap-to-derive backward pass — worth hand-rolling to
keep the starter kit's own numpy-only philosophy (see those files'
docstrings). A 5-headed shared-bottom network's backward pass — one shared
trunk, five different loss gradients merging back into it — is exactly the
case hand-rolled backprop stops paying for itself. Autograd is the correct
tool here, not a shortcut around doing the work.

`agent/experiment.py` gained one new branch (`is_mtl = hasattr(model,
"mtl_step")`), following the exact precedent `fm_bpr.py`'s `bpr_step`
already set: a model that needs a different per-batch training signal than
plain `(X, y)` gets its own step method, fed by `agent/features.py`'s new
`load_aux_labels()`, subsampled with the same seeded indices as `Xtr`/`ytr`
under the Multi-Fidelity Runner's `train_fraction` staging.

`deepfm_mtl_v1` (`parent_id=deepfm_regularized`, identical fields/k/hidden/
lr/l2 — the multi-task objective is the only variable):

| | Valid primary | vs. parent |
|---|---|---|
| Single seed | 0.6049 | +0.0014 |
| **3-seed mean** | **0.6046 ± 0.0003** | **+0.0011** |
| Parent (`deepfm_regularized`) | 0.6035 ± 0.0002 | — |

Diagnosis: **`clear_improvement`** — both GAUC (+0.0016) and nDCG@5
(+0.0008) individually cleared the seed-aware significance bar (0.0004).
A real, robust win, not a lucky seed — and, unlike `features_v1`, it held
up in the same direction across all three seeds (0.6049 / 0.6049 / 0.6042).

**Honest nuance, stated plainly rather than smoothed over:** the 3-seed
test-primary mean (0.5974) is marginally *below* `deepfm_regularized`'s
(0.5977), even though the validation win — the only split any decision in
this project is allowed to read — is clear and real. This is the same
pattern `deepfm_regularized` itself showed over `deepfm_wider`: exactly
what train/valid/test discipline predicts for a model that was never
selected by peeking at test. `deepfm_mtl_v1` is still the correct pick —
it's what the rules this project follows actually say to trust — but the
test-side movement being flat/negative is reported here, not hidden.

Now the project-best. `submission_valid.csv`/`submission_test.csv`
regenerated via `tools/generate_submission.py`, which resolves
`ResearchMap.best_confirmed_node()`.

## 3. LightGBM (`lgbm_baseline`) — a real negative result, with a structural reason

Ran as a standalone diagnostic (same 5 base fields, same `evaluate.py`
scoring via `agent/evaluator.py`, `lightgbm` 4.7.0's native API,
`categorical_feature` on all 5 columns) — not built as a full Model Zoo
entry.

| | Valid primary | Test primary |
|---|---|---|
| LightGBM | 0.5995 | 0.5946 (exactly ties the official FM baseline) |
| `fm_baseline_repro` (parent, for comparison) | 0.6015 | — |
| `deepfm_mtl_v1` (current best) | 0.6046 | 0.5974 |

Diagnosed against `fm_baseline_repro` (same fields, single-seed both
sides): `noise_floor` at the flat-epsilon bar (−0.0020, right at the
0.0020 boundary) — LightGBM roughly *ties* the plain FM baseline. But it
trails every DeepFM-family node by a real margin (~0.004–0.005), well
outside any noise band seen elsewhere in this project.

**Why, structurally, not just "it lost":** gradient-boosted trees split on
features; they don't learn dense embeddings. This task's signal lives
almost entirely in `user_id × video_id` crossing (~27K users × ~7.6K
items) — exactly what FM/DeepFM's embedding tables are built for, and
exactly what a tree can only approximate very crudely at this cardinality
(reconstructing what one embedding lookup does would take many, many
splits on a single high-cardinality categorical). This extends the starter
kit's own documented finding — "capacity/architecture isn't the
bottleneck, `user_id × video_id` crossing is where the signal lives" — to
tree-based models too, not just wider FM/DeepFM variants.

**Why no full Model Zoo integration was built.** LightGBM manages its own
boosting rounds and early stopping internally (`lgb.train(...,
callbacks=[lgb.early_stopping(...)])`) — a real interface mismatch with
every other Model Zoo entry's per-epoch Python SGD loop in
`agent/experiment.py`. Building that bridge cleanly would have meant a
non-trivial, real-regression-risk refactor of a function every already-
validated result depends on. Given the standalone check already showed a
clear, structurally-explained loss, that engineering cost wasn't
justified — a deliberate decision, not a shortcut taken to save time.
Still logged as a proper Research Map node (`lgbm_baseline`,
`parent_id=fm_baseline_repro`) for the audit trail, even though no
reusable pipeline code was added.

## 4. Hyperparameter search (`agent/hpo.py`, Optuna) — a real bug caught, then a real negative result

The one place in this project's own limitations list where reaching for an
existing tool over hand-rolling is unambiguously correct: unlike a heavier
*modeling* framework (RecBole/TorchRec, declined — see README "Would an
existing framework have gotten better results"), a search *orchestrator*
sits entirely outside the modeling/scoring logic. Optuna decides which
hyperparameters to try; `agent/experiment.py`'s `run_experiment` — the same
function every other candidate in this project runs through — still does
the actual training and still only ever reads `.valid.primary`. Runs at
reduced fidelity (`train_fraction`, capped epochs — the same principle as
`agent/multi_fidelity.py`'s staging), and a winner is only trusted after a
full-fidelity confirmation run.

**A smoke test at deliberately aggressive settings (5% data, 3 epochs)
caught its own failure mode working as intended**, not as a bug: the
"winning" trial looked clearly better at that tiny scale (0.5796 vs.
0.5496) but was a clear regression at full scale (0.5977 vs. 0.6046) — the
full-fidelity confirmation step is exactly what stopped that from ever
being trusted or registered as a real result.

**Cleaning that up surfaced an actual, separate bug**: `lgbm_baseline` had
silently vanished from the Research Map. Root cause — a genuine
concurrency race, not user error: `lgbm_baseline` was added via a
short-lived script while `tools/verify_multiseed.py` was still running in
the background holding a stale in-memory snapshot from *before* that
addition; when it finished and called `save()`, it silently overwrote the
whole file with its own stale copy. `ResearchMap.save()` had no merge
logic — last writer wins for the entire file, not per-node. This project
runs many background experiments concurrently by design (this pass alone
had 3+ background processes touching the map at overlapping times), so
this was a real, not-hypothetical gap. **Fixed**: `save()` now merges with
whatever's on disk first, per-node, newest `updated_at` wins.
Regression-tested (`test_research_map_save_merges_concurrent_writes`) by
reproducing the exact scenario. `lgbm_baseline` restored.

**The real search** (15 trials, `train_fraction=0.15`, 8 epochs, patience
3 — searching `k`, `hidden`, `lr`, `l2`, `aux_weight` around
`deepfm_mtl_v1`'s own values):

| | Valid primary @ reduced fidelity | Valid primary @ full fidelity |
|---|---|---|
| Base config (`deepfm_mtl_v1`'s hyperparameters) | 0.5904 | 0.6046 (3-seed) |
| Best trial found (`lr=0.0022`, `l2=0.00098`, `aux_weight=0.33`) | 0.5909 (+0.0005) | 0.6015 (**−0.0031**, `regression`) |

The winning trial's margin at reduced fidelity (+0.0005) was already tiny
— close to the noise floor this project has repeatedly measured at this
scale — and did not survive full-fidelity confirmation: both GAUC (−0.0039)
and nDCG@5 (−0.0024) came back worse, a real regression, not noise.
**Honest conclusion: 15 trials found nothing that beats `deepfm_mtl_v1`'s
current hyperparameters.** They already look close to a local optimum for
this search space, at least at this trial budget. Logged as a proper
Research Map node (`deepfm_mtl_v1_hpo`, tagged `regression`) rather than
discarded — a real, reportable negative result, same standard as
`features_v1` and `lgbm_baseline` above.

## 5. Ensembling — also tried, also negative, and honestly explained

A standalone check (not a Research Map node — ensembling blends already-
trained predictions, it doesn't fit the `ExperimentConfig`/Model Zoo
schema): every already-cached prediction pair/triple among
`deepfm_mtl_v1`, `deepfm_regularized`, `fm_bpr_slow_and_steady`,
`fm_baseline_repro`, `features_v1` was z-score-normalized (each model's
valid-split mean/std) and averaged, scored on validation only. Best
combination found (`deepfm_mtl_v1 + fm_baseline_repro`): valid primary
0.6045 — still below `deepfm_mtl_v1` alone (0.6049, same single-seed
cached predictions). No combination tried beat the single best model.

Makes sense on reflection, not just an unlucky search: ensembling helps
most when blending models of comparable strength with different error
patterns. `deepfm_mtl_v1` is meaningfully stronger than everything else in
this Research Map, so averaging it with a weaker model pulls the blend
toward the weaker model's mistakes rather than correcting `deepfm_mtl_v1`'s
own. `deepfm_mtl_v1` alone remains the right call.

## 6. DIN sequence modeling (`deepfm_din_v1`) — a ranking tradeoff, not a clean win

The starter kit README's own #2-ranked untested headroom item, and the last
genuinely different, structurally untested lever after the four above all
came back negative: attention over each user's recent watch history
(Zhou et al. 2018, "Deep Interest Network"), on top of `deepfm_regularized`'s
backbone. Built and unit-tested (`agent/model_zoo/deepfm_din.py`,
`agent/sequences.py` — leakage-safe history construction, checked by
`test_sequences_recent_history_never_leaks_the_future`) well before it was
ever run against real data. Standalone check
(`tools/check_sequence_model.py`), same treatment as `lgbm_baseline` —
not wired into `agent/model_zoo/registry.py` or `agent/experiment.py`.

First pass, `seq_len=10` (3 seeds, not itself logged as a Research Map
node — superseded by the `seq_len=20` run below before either was
committed to the map):

| seed | valid primary | test primary |
|---|---|---|
| 0 | 0.6024 | 0.5963 |
| 1 | 0.6033 | 0.5972 |
| 2 | 0.6031 | 0.5959 |

A small but consistent regression — worse than `deepfm_regularized`
(valid 0.6035, test 0.5977) in all 3 of 3 runs, on both splits, every
time. That one-directional consistency is the tell that this is a real
(if mild) effect rather than noise: 10 videos of history apparently isn't
enough context for the attention block to weigh usefully.

Doubling the window to `seq_len=20` (3 seeds, logged as `deepfm_din_v1`,
`parent_id=deepfm_regularized`) tests whether window length was the real
limiting factor:

| seed | valid primary | test primary |
|---|---|---|
| 0 | 0.6036 | 0.5974 |
| 1 | 0.6034 | 0.5976 |
| 2 | 0.6037 | 0.5969 |
| **3-seed mean** | **0.6036 ± 0.0001** | **0.5973 ± 0.0003** |

The regression disappears — but the automated Diagnosis Engine's read is
more precise than "it's a tie": combining both sides' seed variance (not
just eyeballing the point estimate), it tags this **`ranking_tradeoff`**:
nDCG@5 improved (+0.0005, a real effect) while GAUC dropped (-0.0003, also
real) — top-5 precision got slightly better, broader within-user ordering
got slightly worse. Not a clean win, not a clean loss, and not noise
either; a genuine mixed signal.

A third, independent data point (single-seed, `seq_len=30` — checked
concurrently and merged in, not yet 3-seed-verified): valid primary
**0.6039**, test primary **0.5975**. The average user in this dataset has
53 interactions logged (median 39), so `seq_len=10` was genuinely
truncating most of a typical user's history before the model ever saw it
— the concrete reason both independent efforts converged on "try a longer
window" separately. But `seq_len=30`'s single-seed number sits right where
`seq_len=20`'s 3-seed mean already was (0.6039 vs. 0.6036, a gap well
inside this model family's established noise band), not on some rising
trend — the honest read is a plateau, not "give it more window and it'll
eventually win": doubling and tripling the history length moved the
result from "consistent regression" to "roughly ties the parent," and
appears to stop there.

**Honest conclusion:** DIN needs a long-enough history window to stop
actively hurting (10 videos: consistent regression; 20-30 videos: plateaus
at roughly parent-level performance, a trade-off at best), but at no
window length tried does it clear the bar to justify the added complexity
and inference cost of a second embedding table + attention block —
`deepfm_mtl_v1` remains the project-best by a clear margin (0.6046 vs.
~0.6036-0.6039). Logged as a proper Research Map node rather than left
untested — the last item on the starter kit's own headroom list is now a
real, reproducible number instead of an open question.

## 7. `best_confirmed_node()` searches every branch now, not just one lineage

A latent correctness gap, not something that ever produced a wrong answer
in practice until now: `best_confirmed_node()` (§ above, the "what's
actually best" answer used for submissions, the LLM's prompt, and new
candidates' `parent_id`) used to start from the raw numeric leader
(`best_node()`) and walk *up its own parent chain* to the nearest node
tagged `clear_improvement`/`baseline_beat`, stopping there. That's only
correct if the map has a single competitive lineage — true of every
Research Map node up through `deepfm_mtl_v1_hpo`, but no longer true the
moment `deepfm_din_v1` was added: it's a **sibling** of `deepfm_mtl_v1`
(both `parent_id=deepfm_regularized`), not on its lineage at all. Had
`deepfm_din_v1` scored higher than `deepfm_mtl_v1` (it doesn't — 0.6036 vs.
0.6046), the old walk would still only ever consider nodes on *its own*
lineage back to `deepfm_regularized`, silently missing `deepfm_mtl_v1`
entirely, since it sits on a different branch than whichever node happened
to be the raw numeric leader.

Fixed by searching every `done` node directly for the highest-scoring one
whose own tag isn't in the unconfirmed set, instead of walking one
ancestor chain — strictly more correct, and simpler code besides.
Regression-tested with a synthetic two-sibling scenario matching this
exact shape
(`test_research_map_best_confirmed_node_searches_every_branch_not_just_one_lineage`).
No change on the real map: `best_confirmed_node()` still returns
`deepfm_mtl_v1`, since it also happens to be the raw numeric leader today.
This was fixed pre-emptively, before it ever produced a wrong answer, once
the tree grew a second real branch worth taking seriously.

## Net effect on the project-best

| | Valid primary (3-seed) | Test primary (3-seed mean) | vs. official baseline (test) |
|---|---|---|---|
| Official FM baseline | 0.6016 | 0.5946 | — |
| `deepfm_regularized` (Phase 4, prior best) | 0.6035 ± 0.0002 | 0.5977 | +0.0031 |
| **`deepfm_mtl_v1` (P2, current best)** | **0.6046 ± 0.0003** | 0.5974 | +0.0028 |

Both engineered features and LightGBM were genuinely worth trying — public,
well-motivated ideas backed by the problem statement's own allowed toolset
— and both came back negative, for two different, specific, structural
reasons rather than "we didn't get to it." Multi-task learning was the one
that paid off, and did so on the first attempt with no diagnosis-driven
iteration needed (unlike BPR's 3 rounds in P1).
