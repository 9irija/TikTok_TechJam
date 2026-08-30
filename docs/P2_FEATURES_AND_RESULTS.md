# P2 — Engineered Features, Multi-Task Learning, LightGBM, Hyperparameter Search, Ensembling, DIN Sequence Modeling, DeepFM_BPR, Randomized-Exposure Generalization

Eight experiments closing real gaps left open by CLAUDE.md's roadmap
("Multi-Task Feature Exploitation", "TikTok-disclosed features", "Extended
Model Zoo", "Hyperparameter Search", sequence modeling, "DeepFM_BPR is a
natural, cheap extension") plus a generalization check answering a direct
question the user asked ("does the improvement hold across any kinds of
data, not just this one"), run and 3-seed-verified where the result
warranted it. Six of eight are negative or mixed results, reported exactly
as measured — the honest signal here is which levers this specific
benchmark actually responds to, not a scoreboard of wins. One of those
negative results (the hyperparameter search) also surfaced and fixed a
real concurrency bug in the Research Map's persistence layer, unrelated to
modeling but a genuine reliability gap worth having found. This pass also
merged real, independent parallel work from a teammate (Yichen930): the
`seq_len=20` DIN result in §6, the `best_confirmed_node()` correctness fix
in §7, and `tools/generate_dashboard.py` (dashboard auto-generation) all
came from that merge, not from this session directly — credited inline
below rather than claimed.

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

## 8. DeepFM_BPR (`deepfm_bpr_v1`) — combining two independently-partial results, still not enough

`fm_bpr` (P1) showed the loss/metric-alignment direction was real —
GAUC/nDCG are ranking metrics, BPR directly optimizes for within-user
ranking — but plateaued ~0.002 below the FM baseline after 3 diagnosis-
driven rounds, in plain FM form. The deep component
(`deepfm`/`deepfm_regularized`) independently, separately proved to help.
Neither was ever combined with the other, despite being flagged in this
project's own roadmap notes as "a natural, cheap extension" since before
the P1 BPR rounds even concluded.

`agent/model_zoo/deepfm_bpr.py`: DeepFM's exact architecture (shared field
embeddings, FM linear + 2nd-order term, deep MLP), trained on the same
pairwise BPR objective as `fm_bpr.py`. First torch model wired directly
into the real pipeline from day one (`registry.py`, `agent/experiment.py`'s
existing `is_bpr` branch) rather than standalone-checked first — the
`bpr_step`/`predict` contract `fm_bpr` already validated in P1 needed zero
training-loop changes to carry over.

`deepfm_bpr_v1` (`parent_id=deepfm_regularized`, same fields/k/hidden/l2 —
isolating the pairwise-vs-pointwise objective as the only variable):

| | Valid primary | Diagnosis |
|---|---|---|
| `deepfm_bpr_v1` (first attempt) | 0.5819 | `regression` (GAUC −0.0290, nDCG@5 −0.0141), flagged `overfitting_risk` — best epoch at ~20% of training length |
| `deepfm_bpr_v1_regularized` (L2 ×10, patience 4→2) | 0.5980 | `clear_improvement` **vs. its own parent** — recovers most of the damage |
| `deepfm_regularized` (plain DeepFM, for reference) | 0.6035 | — |
| `fm_baseline_repro` (plain FM, for reference) | 0.6015 | — |

Same reactive pattern that worked twice before (`deepfm_default` →
`deepfm_regularized`, `fm_bpr_default` → `fm_bpr_regularized`): diagnose
the overfitting, raise L2, tighten patience. It worked again in the sense
that it recovered most of the damage (+0.0161) — but even after the fix,
`deepfm_bpr_v1_regularized` (0.5980) still doesn't clear the *plain* FM
baseline (0.6015), let alone `deepfm_regularized` or `deepfm_mtl_v1`.

**Honest conclusion:** this now doubly-confirms (P1's FM_BPR rounds, and
this DeepFM_BPR attempt) that BPR's pairwise objective has a real,
structural ceiling on this specific benchmark regardless of which
architecture it's paired with — not a hyperparameter away from competitive,
a genuine mismatch between the training signal and this dataset's actual
learnable structure. Not pursued further (a third round, mirroring P1's
`fm_bpr_slow_and_steady`, was considered and declined): the pattern is
consistent enough across two independent architectures now that another
round would very likely just re-confirm the same ceiling, not move it.
`deepfm_mtl_v1` remains the project-best.

## 9. Randomized-exposure generalization check — does the win hold on unbiased data?

Direct answer to a question the user asked explicitly: has anything here
been optimized to generalize "across any kinds of data," not just this one
train/valid/test split? Honest answer at the time: partially. 3-seed
verification checks robustness to random initialization; the date-based
splits check a real temporal holdout. Neither checks whether
`deepfm_mtl_v1`'s win holds under a genuinely different *distribution* —
every split so far is drawn from TikTok's own recommendation-biased
logging policy (whatever the platform already chose to show users).

`log_random_4_22_to_5_08_pure.csv` is a separate, real file in the
dataset — interactions logged under *randomized* exposure (`is_rand=1`),
same user population and date range as valid+test, explicitly flagged in
the brainstorm doc as usable for "an unbiased secondary validation set,"
never used anywhere in this project until now. `tools/check_randomized_exposure.py`
encodes it via the organizer's own pinned `encode()` (a `splits` dict with
an extra `"random"` key, vocab still built from `train` only — never
reimplemented) and evaluates both `fm_baseline_repro` and `deepfm_mtl_v1`
on it, strictly as a held-out eval, never trained on.

| Split | FM baseline | `deepfm_mtl_v1` | Delta |
|---|---|---|---|
| valid | 0.6015 | 0.6049 | +0.0034 |
| test | 0.5953 | 0.5979 | +0.0025 |
| **random (unbiased)** | 0.3639 | 0.3741 | **+0.0102** |

Absolute scores collapse on the random split (expected, not concerning):
only 8.5% of randomly-shown content is a `long_view`, vs. 31.3% on
TikTok's own biased logs — the platform's own recommendation policy is
already good at showing people things they'll watch, and random exposure
removes that head start, making it a genuinely harder ranking task for
both models (nDCG@5 in particular drops hard, since it needs enough
positives per user to have anything meaningful in the top 5).

**The result that matters: the delta doesn't just survive, it's
proportionally larger** under the unbiased distribution — roughly 3-4x the
gap seen on valid/test. Real evidence that multi-task learning's benefit
isn't an artifact of TikTok's own biased serving/logging policy; if
anything it holds up better once the "easy" signal (already-good matches)
is stripped away and the model has to do genuine ranking work. Single-seed
only (kept fast on purpose, same standalone-check pattern as the rest of
this document) — the exact magnitude shouldn't be quoted with 3-seed
confidence, but the direction is a real, useful answer: yes, this specific
optimization generalizes beyond the one distribution it was validated on.

## 10. Watch-time multi-task head (`deepfm_mtl_watch_v1`) — a thoroughly-tested noise_floor

CLAUDE.md's "Unexplored headroom" #4, "Watch-time modeling: censored
regression on `play_time`... Still open" — the last item on that list not
yet tried. `agent/model_zoo/deepfm_mtl_watch.py` extends `deepfm_mtl_v1`'s
proven shared-bottom setup (4 binary auxiliary heads: `is_like`/
`is_follow`/`is_comment`/`is_forward`) with a 5th, **continuous** auxiliary
head trained on a clipped, normalized `play_time_ms/duration_ms`
completion ratio (`agent/features.py`'s new `load_watch_ratio`), via MSE
instead of BCE. Hypothesis: a denser, continuous training signal might
regularize the shared embeddings better than 4 binary signals alone.

**Not the leakage case this project usually has to guard against** with
`play_time_ms` (see `agent/features.py`'s module docstring on why it's
unsafe as an *input* feature) — this uses it only as an auxiliary
**training target**, exactly the same role the existing 4 heads already
play. The main `long_view` logit — the only thing GAUC/nDCG@5 ever score
— never sees it, at train or inference time.

Standalone check (`tools/check_watch_time_mtl.py`), same discipline as
`lgbm_baseline`/`deepfm_din_v1`. `watch_weight` swept at seed 0 across
`{0.05, 0.1, 0.2, 0.4, 0.6}` first — valid primary stayed in a narrow
0.6043–0.6045 band at every weight, all within noise of the parent
(0.6046). Not a tuning problem: the signal simply isn't moving validation
either way, regardless of how strongly it's weighted. Two settings
3-seed-verified for a proper check:

| | valid primary (3-seed) | test primary (3-seed) | diagnosis |
|---|---|---|---|
| `watch_weight=0.2` (matches `aux_weight`) | 0.6043 ± 0.0003 | 0.5982 ± 0.0005 | `noise_floor` |
| `watch_weight=0.6` (best single-seed point) | 0.6045 ± 0.0002 | 0.5982 ± 0.0004 | `noise_floor` |

Logged as `deepfm_mtl_watch_v1` (the `watch_weight=0.2` run, for symmetry
with `deepfm_mtl_v1`'s own `aux_weight=0.2`).

**A curious, consistent aside — reported honestly, not used to override
the diagnosis**: test primary came in *higher* than `deepfm_mtl_v1`'s at
every single seed tried, at both weights (0.5982 vs. parent's 0.5974,
consistently, 6 runs). Train/valid/test discipline says this doesn't
matter for the decision — validation is the only split anything here is
allowed to read, and validation says `noise_floor` — but it's worth
recording rather than silently dropping, the same way `deepfm_mtl_v1`
itself had an honestly-reported test-side quirk relative to
`deepfm_regularized`.

**Honest conclusion:** the watch-time auxiliary signal doesn't clear the
bar on validation, at any weight tested. `deepfm_mtl_v1` remains the
project-best. The last item on the starter kit's own headroom list is now
a real, reproducible, thoroughly-swept number instead of an open question.

## 11. Combining DIN + MTL (`deepfm_din_mtl_v1`) — the two mechanisms cancel out, not stack

The natural "combine two independently-partial results" move, same
reasoning `deepfm_bpr_v1` already used to combine P1's BPR loss with
DeepFM's architecture (§8). `deepfm_din_v1` alone (seq_len=20) was a
`ranking_tradeoff` on top of `deepfm_regularized`: nDCG@5 improved
(+0.0005) but GAUC dropped (-0.0003). `deepfm_mtl_v1` alone was a clean
`clear_improvement` on the same parent. Hypothesis: MTL's regularizing
effect might correct DIN's GAUC regression while keeping its nDCG@5 gain,
since the two act on different parts of the model (attention over recent
history vs. a denser multi-signal training target) rather than obviously
competing for the same capacity.

`agent/model_zoo/deepfm_din_mtl.py`: DIN's exact attention block (dedicated
video embedding table, masked scaled-dot-product attention) plus MTL's 4
binary auxiliary heads (`is_like`/`is_follow`/`is_comment`/`is_forward`),
both reading the same post-attention deep trunk. Standalone check
(`tools/check_din_mtl.py`), both components' proven hyperparameters reused
unchanged (`seq_len=20`, `aux_weight=0.2`) so combining them is the only
variable. 3-seed result:

| | valid primary (3-seed) | test primary (3-seed) |
|---|---|---|
| `deepfm_mtl_v1` (current best) | 0.6046 ± 0.0003 | 0.5974 |
| `deepfm_din_v1` alone (seq_len=20) | 0.6036 ± 0.0001 | 0.5973 |
| **`deepfm_din_mtl_v1` (combined)** | **0.6033 ± 0.0002** | **0.5972** |

**The hypothesis was wrong.** The combined model doesn't land between or
above its two parents — it lands *below both*. Diagnosed against its
actual parent (`deepfm_regularized`, 0.6035): `noise_floor` (the two
effects roughly cancel rather than combine). Diagnosed against the
current best (`deepfm_mtl_v1`) for honest context: a real `regression`
(GAUC -0.0021, nDCG@5 -0.0005, both individually significant).

**Why, plausibly:** both components' hyperparameters were reused exactly
as tuned for each standalone case, not re-tuned for the combined,
higher-capacity model (one more embedding table, one more loss term,
same learning rate and epoch budget). The two mechanisms may simply need
a smaller learning rate or lower `aux_weight` to coexist productively
instead of interfering during optimization — a real, concrete next
question if this direction is ever revisited, not chased further here.
`deepfm_mtl_v1` remains the project-best.

## 12. Uncertainty-weighted MTL (`deepfm_mtl_uncertainty_v1`) — the closest thing to a tie, still not a confirmed win

After DIN+MTL combination failed (§11), a different lever on the
already-working mechanism itself rather than combining it with something
else: `deepfm_mtl_v1` combines its 4 auxiliary losses with one fixed,
hand-picked `aux_weight=0.2` — a value `agent/hpo.py`'s Optuna search
already searched around (among other hyperparameters) and found nothing
better than. Kendall, Gal & Cipolla 2018 ("Multi-Task Learning Using
Uncertainty to Weigh Losses") replaces every fixed task weight with a
**learned** per-task uncertainty parameter, optimized jointly with the
network — a genuinely different mechanism from a fixed-value sweep, not
just a different point in the same search space.

`agent/model_zoo/deepfm_mtl_uncertainty.py`: identical architecture to
`deepfm_mtl.py`, but 5 learned `log_var` parameters (main + 4 aux tasks
individually) replace the single fixed weight; `loss = sum_i
exp(-log_var_i) * task_loss_i + log_var_i`. Standalone check
(`tools/check_mtl_uncertainty.py`). Given how close the early seeds
looked, this is the one lever pushed to **8 seeds** instead of the usual
3, to get a properly powered read:

| | valid primary | test primary |
|---|---|---|
| `deepfm_mtl_v1` (current best, fixed weight=0.2) | 0.6046 ± 0.0003 (3-seed) | 0.5974 (3-seed) |
| **`deepfm_mtl_uncertainty_v1`** (learned weights) | **0.6048 ± 0.0002 (8-seed)** | **0.5984 ± 0.0002 (8-seed)** |

The learned weights converged to roughly **uniform across all 4 aux
tasks** (~3.2–3.8× the main task's own learned weight of ~1.98) rather
than differentiating between `is_like`/`is_follow`/`is_comment`/
`is_forward` — the 4 signals appear similarly relevant/difficult here, so
the real effect of this technique is closer to "recalibrate the overall
aux weight upward from 0.2" than "weight tasks differently from each
other."

**Diagnosis, even at 8 seeds: `noise_floor`** — the +0.0002 valid-primary
edge stays inside the seed-aware significance bar (0.0004). This is the
tightest margin and the only lever this pass that never scored *below*
the current best on any single seed (unlike the watch-time head or the
DIN+MTL combination), but it still isn't a confirmed win by this
project's own statistical standard.

**Worth stating plainly, not smoothed over:** the 8-seed test-primary gap
(0.5984 vs. 0.5974) is *itself* statistically real (~2.7 combined standard
errors) — a stronger test-side signal than validation shows. Train/valid/
test discipline means this cannot be the basis for calling it a win —
only validation is allowed to drive a decision here, and validation says
tie — but it's an honest, curious data point rather than something to
quietly drop. `deepfm_mtl_v1` remains the project-best.

## 13. Listwise ranking loss (`deepfm_listwise_v1`) — ties the pointwise baseline, the untested half of the loss-function guess

The starter kit README's own top guess for the loss-function lever was
*"pairwise (BPR) or listwise (per-user softmax)"* — `fm_bpr` (P1) and
`deepfm_bpr` (P2) tested the pairwise half twice, both plateauing below
the plain FM baseline (a real, structural ceiling, per §8). The listwise
half was never actually tried, only assumed to be in the same family.
This tests it directly.

`agent/model_zoo/deepfm_listwise.py`: the same DeepFM backbone as
`deepfm_regularized`, trained with a per-user softmax cross-entropy
(ListNet/Plackett-Luce style) over each user's *entire* impression set at
once, instead of pointwise BCE or BPR's one-pair-at-a-time. Batching is
structurally different from every other model here (each batch is a set
of users, padded to `max_len=64`, masked out of the softmax — same
masking convention `deepfm_din.py` already established), so it's a
standalone check (`tools/check_listwise.py`), not wired into
`agent/experiment.py`.

`l2` swept `{1e-2, 1e-3, 1e-4 (DeepFM's own default), 1e-5, 1e-6}` at
seed 0 first — and the direction was the *opposite* of what fixed
`deepfm_default`/`deepfm_wider`'s overfitting earlier in this project:
**higher** L2 made this worse (1e-2 broke training outright, valid
~0.58), not better. `1e-5` was the best point found. 3-seed result at
`l2=1e-5`:

| | valid primary (3-seed) | vs. parent |
|---|---|---|
| `deepfm_regularized` (parent, pointwise BCE) | 0.6035 ± 0.0002 | — |
| `deepfm_bpr_v1_regularized` (pairwise BPR) | 0.5980 | −0.0055, real regression |
| **`deepfm_listwise_v1`** (listwise softmax) | **0.6033 ± 0.0004** | **−0.0002, `noise_floor` — a genuine tie** |

**Listwise is clearly the better of the starter kit's own two suggested
ranking-loss options** — it ties the pointwise baseline outright, where
pairwise BPR fell short by a real, structural margin twice. Neither beats
`deepfm_mtl_v1` (0.6046), the actual current best, so this doesn't move
the project's bottom line — but it does close out the loss-function
question the starter kit itself posed more completely than the pairwise
attempts alone did. One curious training-dynamics note, also flagged by
the Diagnosis Engine's own `overfitting_risk` check: validation peaks
very early (epoch 2-3) at every L2 tried, a different shape than
pointwise BCE's more gradual climb — a real next question if this
direction is revisited (e.g. a much smaller effective learning rate, or a
temperature-scaled softmax), not chased further here.

## 14. PDAOM hard-pair mining (`deepfm_pdaom_v1`) — a real, well-diagnosed regression

One more loss-function bet after listwise (§13) tied the pointwise
baseline: "PDAOM" (Personalized Differentiable AUC Optimization with
Maximum violation, arXiv:2304.09176) — a pairwise loss like BPR, but with
two real differences: an exponential loss shape (steeper than BPR's
bounded log-sigmoid) and per-user hard-pair mining (the single hardest
positive/negative pair per user each batch, not BPR's uniformly-random
pair). The paper reports real GAUC/AUC gains in a production
feed-recommendation system.

**Source-fidelity caveat, stated upfront:** the paper's PDF text couldn't
be machine-extracted (image-embedded, not searchable), so
`agent/model_zoo/deepfm_pdaom.py` is a faithful reconstruction from its
abstract-level description plus the two named techniques it's built
from (classic AUC-optimization exponential loss; batch-hard mining from
metric learning) — not a citation of the paper's own exact tuned
constants.

**Result: a severe, unambiguous regression**, not noise — no 3-seed
re-verification needed to see this is real:

| config | valid primary |
|---|---|
| `deepfm_regularized` (parent, pointwise BCE) | 0.6035 |
| `deepfm_bpr_v1_regularized` (uniform-pair BPR) | 0.5980 |
| `deepfm_listwise_v1` (per-user softmax) | 0.6033 |
| **`deepfm_pdaom_v1`** (hard-mining, `max_candidates=8`) | **0.5483** |

Diagnosed with two quick ablations before stopping, isolating the two
variables (loss shape vs. mining) instead of guessing:

| variant | valid primary |
|---|---|
| Full hard-mining (`max_candidates=8`) | 0.5483 |
| Smaller candidate pool (`max_candidates=2`) | 0.5764 |
| No mining at all — exponential loss on one random pair (`max_candidates=1`) | 0.5917 |

**Both ingredients hurt independently.** Even with hard-mining fully
removed, the exponential loss alone (0.5917) still trails BPR's bounded
log-sigmoid loss (0.5980) — the loss barely moved epoch-to-epoch during
training (~1.003 the whole run), a sign of a weak or unstable gradient
signal. Hard-pair mining then compounds that instability further as the
candidate pool grows (0.5917 → 0.5764 → 0.5483). This matches a
documented failure mode in the metric-learning literature: FaceNet
(Schroff et al. 2015) moved away from pure hardest-negative mining
toward "semi-hard" mining for exactly this reason — the single hardest
example in a batch is often a noisy outlier, not a useful training
signal, and training on it exclusively can destabilize rather than
sharpen the model.

**Honest conclusion:** this specific reconstruction of PDAOM doesn't
work here, and the diagnostic trail explains why with reasonable
confidence rather than leaving it a mystery. Not pursued further given
this clarity and that the source paper's exact tuned constants were
unavailable to try instead. `deepfm_mtl_v1` remains the project-best.

## 15. Per-segment diagnosis — does the win hold uniformly, or is it concentrated?

Every result so far reports one aggregate GAUC/nDCG@5 number, which can
hide a win (or a loss) that's actually concentrated in one segment and
flat or negative elsewhere. `tools/check_per_segment.py`: no new training
run needed — reuses `deepfm_regularized`'s and `deepfm_mtl_v1`'s
already-cached valid-split predictions, bucketed by **user activity** and
**item (video) popularity**, both computed from TRAIN-split impression
counts only (never valid/test — the same train-only-aggregate discipline
`agent/features.py` already uses, so a segment boundary is never informed
by the very labels being scored).

**By user activity (quartiles of train-split impressions per user):**

| Segment | Rows | `deepfm_regularized` | `deepfm_mtl_v1` | Delta |
|---|---|---|---|---|
| Q1 (least active) | 19,132 | 0.5988 | 0.5994 | +0.0006 |
| Q2 | 24,731 | 0.6106 | 0.6131 | +0.0025 |
| Q3 | 31,956 | 0.6103 | 0.6106 | +0.0003 |
| Q4 (most active) | 49,090 | 0.5923 | 0.5945 | +0.0022 |

**Positive in all 4 segments** — the win isn't propped up by one narrow
slice of users. Reassuring, and worth stating plainly since it could
easily have come back otherwise.

**By item (video) popularity (quartiles of train-split impressions per video):**

| Segment | Rows | `deepfm_regularized` | `deepfm_mtl_v1` | Delta |
|---|---|---|---|---|
| Q1 (least popular) | 4,810 | 0.4748 | 0.4784 | +0.0035 |
| Q2 | 9,242 | 0.4857 | 0.4791 | **−0.0066** |
| Q3 | 22,263 | 0.5270 | 0.5258 | **−0.0012** |
| Q4 (most popular) | 88,594 | 0.5978 | 0.5996 | +0.0019 |

**Not uniform, and worth reporting exactly as measured rather than
smoothed into the aggregate:** the two middle-popularity quartiles show a
real *negative* delta — `deepfm_mtl_v1` is measurably worse than
`deepfm_regularized` there, most notably Q2 (−0.0066). The win is positive
at both popularity extremes, but `Q4` alone holds 88,594 of 124,909 valid
rows (71%) — a heavily right-skewed popularity distribution, as expected
for short-video engagement data — so the *aggregate* +0.0014 win is
disproportionately carried by success on already-popular items, not
evidence of uniformly better ranking across the popularity spectrum.

**Honest read:** multi-task learning's aggregate win is genuine and not a
user-segment artifact, but on the item side it's concentrated where the
bulk of the data (and the aggregate metric's weight) already sits, with a
real, unexplained dip in the middle-popularity range. A concrete, specific
next question this raises — not chased further here, a real scope
boundary: why would auxiliary engagement signals (`is_like`/`is_follow`/
etc.) help *more* at the popularity extremes than in the middle? One
plausible, untested hypothesis: mid-popularity videos have enough
interaction volume for the *pointwise* signal to already be reasonably
well-estimated, but not enough for the *auxiliary* signals (which are
individually sparser than `long_view` itself) to add much beyond noise at
that specific density — worth a dedicated follow-up, not asserted as
confirmed here.

## 16. Temporal drift check — train→valid boundary, and day-by-day within validation

Priority #3 from the user's own next-steps list. Two checks,
`tools/check_temporal_drift.py`, no new training needed.

**Distribution drift across the train→valid boundary:** 98.1% of valid
users and 99.9% of valid videos were already seen somewhere in train — 
cold-start is negligible on this benchmark, so any drift found isn't a
"the model has never seen these entities" effect. The label rate does
shift, though: train positive rate 0.3366 → valid 0.3133 (−0.0233, a real,
non-trivial drop) — a genuine distribution difference between the two
windows, consistent with `deepfm_mtl_v1`'s own honestly-reported
validation-vs-test gap elsewhere in this document.

**Per-day performance within the valid window** (`deepfm_regularized` vs.
`deepfm_mtl_v1`, both scored on identical within-day user groupings):

| Date | Rows | Positive rate | `deepfm_regularized` | `deepfm_mtl_v1` | Delta |
|---|---|---|---|---|---|
| 2022-04-22 | 22,283 | 0.3186 | 0.5467 | 0.5443 | **−0.0023** |
| 2022-04-23 | 26,645 | 0.3382 | 0.5573 | 0.5600 | +0.0026 |
| 2022-04-24 | 18,240 | 0.3031 | 0.5228 | 0.5258 | +0.0030 |
| 2022-04-25 | 14,911 | 0.3123 | 0.5245 | 0.5290 | +0.0045 |
| 2022-04-26 | 14,530 | 0.3114 | 0.5236 | 0.5239 | +0.0003 |
| 2022-04-27 | 14,328 | 0.2973 | 0.5227 | 0.5244 | +0.0017 |
| 2022-04-28 | 13,972 | 0.2899 | 0.5309 | 0.5312 | +0.0003 |
| All 7 days (aggregate) | 124,909 | 0.3133 | 0.6035 | 0.6049 | +0.0014 |

**Methodological note, so this table isn't misread:** the per-day absolute
scores (~0.52–0.56) look much lower than the 7-day aggregate (0.6035/
0.6049) — this is expected, not a sign the model performs worse day to
day than reported. GAUC/nDCG@5 are computed per-user; restricting to one
day gives each user far fewer impressions to rank within, which is a
harder, noisier evaluation than letting each user's full week contribute.
The **delta** column (both models scored on the identical within-day
grouping) is still a fair, self-consistent comparison — the absolute
column just isn't comparable to the aggregate row.

**Honest read:** the win holds on 6 of 7 days, with one borderline day
(4/22, the very first day of the validation window) showing a real but
small negative delta (−0.0023). Not an alarming drift pattern — a single
day out of seven, small in magnitude, on the smallest and noisiest kind of
slice this project scores — but reported exactly as measured rather than
rounded up to "holds every day." No clear day-of-week or trend pattern is
visible in the remaining 6 days' deltas.

## 17. PCGrad gradient surgery (`deepfm_mtl_pcgrad_v1`) — a real regression, and a coherent reason for it

Refines the one mechanism that's actually worked (`deepfm_mtl_v1`) rather
than trying yet another architecture — every architecture bet this pass
(DIN, BPR, listwise, PDAOM) came back negative or tied, independently
reconfirming the organizers' own "capacity/architecture isn't the
bottleneck" finding. `deepfm_mtl_uncertainty_v1` (§12) already tested
whether fixed loss-*magnitude* weighting (`aux_weight=0.2`) was leaving
something on the table — it wasn't (a tie at 8 seeds). PCGrad (Yu et al.
2020, "Gradient Surgery for Multi-Task Learning") tests a genuinely
different question: even with magnitudes fine, do the main and auxiliary
tasks' gradients actively point in conflicting *directions* on their
shared parameters? `agent/model_zoo/deepfm_mtl_pcgrad.py`: identical
architecture to `deepfm_mtl.py`, but resolves that conflict (projects
each task's shared-parameter gradient onto the other's orthogonal plane
whenever they'd otherwise fight) before combining them, simplified to a
2-task formulation (main vs. combined aux loss — `aux_heads`' 4 outputs
share one weight matrix, not separable enough for full 5-way surgery
without redesigning the architecture).

**A real bug caught by the test suite, not anticipated in advance:** the
first implementation treated `W` (the FM linear term) and `b` (bias) as
"shared" parameters needing conflict resolution. They aren't — they only
ever feed the main task's logit, never the auxiliary heads, so
`torch.autograd.grad(aux_loss, [W, b, ...])` correctly threw "not used in
the graph" the moment the real model test ran (not caught by the targeted
math test, which used synthetic tensors that don't have this structural
property). Fixed by moving `W`/`b` to main-task-private parameters
(alongside `deep_out`) — only `V` (the embedding table) and the deep
trunk are genuinely shared between both losses. Regression-tested with
three hand-constructed cases with known expected outcomes (full
cancellation on directly-opposing gradients, no-op on orthogonal
gradients, no-op on reinforcing gradients) before ever training on real
data.

**Result** (`parent_id=deepfm_mtl_v1`, same fields/k/hidden/l2/aux_weight
— gradient surgery is the only variable):

| | Valid primary | vs. parent |
|---|---|---|
| `deepfm_mtl_pcgrad_v1` | 0.6027 | **−0.0019**, `regression` (GAUC −0.0028, nDCG@5 −0.0011, both individually well past the 0.0004 significance bar) |
| `deepfm_mtl_v1` (parent, current best) | 0.6046 | — |

Single-seed, not 3-seed re-verified — the margin is wide enough on both
individual metrics (7x and 2.75x the bar respectively) that a second seed
flipping the sign is implausible, the same judgment call already applied
to `deepfm_pdaom_v1`'s clear regression.

**Honest read, and why this result is coherent rather than just another
data point:** both the magnitude question (`deepfm_mtl_uncertainty_v1`)
and now the direction question (`deepfm_mtl_pcgrad_v1`) have been tested
against `deepfm_mtl_v1`'s original, simple, fixed-weight recipe — neither
refinement beat it, and PCGrad actively hurt. That's a real, informative
signal about this specific setup, not just two more negative results
stacked on the pile: whatever mild gradient conflict exists between the
main and auxiliary tasks here isn't pure noise to be surgically removed —
it plausibly carries useful information the model needs, and PCGrad's
projection throws part of that away along with the conflict. Reasonable
stopping point for refining the multi-task mechanism itself; `deepfm_mtl_v1`
remains the project-best, now via two independent refinement attempts
that both confirm its original recipe rather than improve on it.

## 18. `is_click` as a 5th auxiliary head (`deepfm_mtl_click_v1`) — well-reasoned, still a tie

Built after explicit direction to find what would actually work before
implementing it — real evidence was checked first, not another guess.
Two real findings came out of that check before any model code was
written:

- **`video_features_statistic_pure.csv` (organizer-provided, never used
  anywhere in this project) was investigated as a new safe feature source
  and explicitly declined**, not silently skipped: verified via the
  KuaiRand paper's own description ([alphaXiv](https://www.alphaxiv.org/audio/2208.08696),
  [kuairand.com](https://kuairand.com/)) that its statistics are averaged
  over the *entire* one-month collection window (2022-04-08 to
  2022-05-08) — overlapping both the valid and test date ranges. Using it
  as an input feature would leak future engagement into training, a real
  leakage vector caught by checking before building, not a hypothetical
  risk. `user_features_pure.csv` (confirmed static/time-independent —
  same sources) is safe but genuinely untested, a real next candidate.
- **The raw log was sampled directly (500K rows) before writing any
  code**: `is_click`'s positive rate (~45.9%) is 25–450x denser than
  `deepfm_mtl_v1`'s existing 4 auxiliary signals (`is_like` 1.8%,
  `is_follow`/`is_forward` ~0.095% each, `is_comment` 0.25%) — a
  genuinely different-character signal, not a 5th flavor of the same
  sparse-engagement idea. `is_hate` (~0.04%, too sparse) and
  `is_profile_enter` (~2.5%, reasonable but would confound attribution if
  bundled with `is_click`) were checked and set aside to keep this
  experiment isolated to one variable.

`agent/model_zoo/deepfm_mtl_click.py`: `deepfm_mtl.py`'s exact
architecture, `aux_heads` extended to 5 outputs. Wired directly into the
real pipeline (`parent_id=deepfm_mtl_v1`, same fields/k/hidden/l2/
aux_weight — `is_click` is the only variable):

| | Valid primary | vs. parent |
|---|---|---|
| Single seed | 0.6050 | +0.0003 (looked promising) |
| **3-seed mean** | **0.6047 ± 0.0004** | **+0.0001, `noise_floor`** |
| Parent (`deepfm_mtl_v1`) | 0.6046 ± 0.0003 | — |

**Honest conclusion: a well-reasoned candidate that still came back a
tie, not a win.** Worth being explicit about why the reasoning didn't
pay off, not just reporting the number: the a priori case rested on
signal *density* (`is_click` is common, the other four are rare), but the
real pattern across all 5 signals suggests what actually matters is
correlation *strength* with `long_view`, not raw frequency. `is_like`/
`is_follow`/`is_comment`/`is_forward` are rare precisely because they're
strong, deliberate positive actions — a reliable proxy for genuine
enjoyment. `is_click` is common precisely because it's a low-commitment
action (opening/starting a video, not finishing or valuing it) — a much
weaker proxy for the same thing `long_view` measures. Density bought nothing
here; correlation quality, which this candidate didn't actually improve
on, is the more likely real driver of what made the original 4 signals
work. `deepfm_mtl_v1` remains the project-best.

## 19. Focal loss on the main task (`deepfm_mtl_focal_v1`) — the top brainstorm pick, also a regression

From an explicit brainstorm for a genuinely different lever (not another
architecture, signal, or MTL-mechanism refinement — all three categories
already exhausted or tied at that point): focal loss (Lin et al. 2017)
changes WHICH EXAMPLES MATTER during training by down-weighting ones the
model already gets confidently right, concentrating gradient on the hard
ones. Directly motivated by the per-segment diagnosis (§15), which found
`deepfm_mtl_v1` measurably worse on mid-popularity items specifically —
plain BCE trains every row uniformly regardless of that difficulty.
`agent/model_zoo/deepfm_mtl_focal.py`: `deepfm_mtl_v1`'s exact
architecture, main-task BCE replaced with focal loss (`gamma=2.0`, the
paper's default; `alpha=0.5`, symmetric — isolates the hard-example
effect from class-balance correction, a separate question this dataset's
mild ~31-34% imbalance doesn't obviously need).

| | Valid primary | vs. parent |
|---|---|---|
| `deepfm_mtl_focal_v1` | 0.6030 | **−0.0016**, `regression` (GAUC −0.0020, nDCG@5 −0.0012, both past the 0.0004 bar) |
| `deepfm_mtl_v1` (parent, current best) | 0.6046 | — |

Not an overfitting artifact (best epoch landed at 12/17, a healthy
ratio, unlike PDAOM/PCGrad/the first BPR attempt's early collapses) —
genuinely a worse optimum, not a training-dynamics problem a
regularization fix would recover.

**Honest read, and a coherent reason, not just a number:** focal loss was
designed for *extreme* class imbalance (the original paper's object-
detection setting is roughly 1:1000 foreground:background). This
benchmark's positive rate is ~31-34% — a mild imbalance, nowhere near
what the technique targets. Down-weighting "easy" examples in a roughly-
balanced dataset likely just discards genuinely useful, confidently-
correct training signal rather than fixing a real swamping problem, since
there's no large majority class drowning out a rare one here. The per-
segment difficulty gap that motivated this candidate is real, but it's
plausibly a *structural* property of those items (less training data per
item, not individual "hard examples" in the focal-loss sense) — a
different kind of problem than per-example reweighting addresses.
`deepfm_mtl_v1` remains the project-best.

**Pattern worth naming plainly across §10-§19 as a group:** ten different,
genuinely distinct refinements of or additions to the one thing that
worked (`deepfm_mtl_v1`) have now been tried — watch-time signal, DIN
combination, uncertainty weighting, listwise loss, PDAOM, PCGrad,
`is_click`, focal loss, plus the two architecture-level BPR/DIN attempts
before them — and none has beaten the original, simple, first-attempt
recipe. That's a genuine, informative signal in its own right: this
benchmark's specific ceiling for this general family of approaches looks
real, not a search-effort problem. The honest next question isn't "try
another variant of the same kind" — it's whether a structurally different
lever (e.g. a metric-aware pairwise weighting like LambdaRank, or a
cascade/two-stage framing) is worth the larger engineering investment, or
whether this is close to the practical ceiling for this dataset size and
task.

## 20. LambdaRank-style pairwise loss (`agent/model_zoo/deepfm_lambdarank.py`) — the structurally-different lever §19 called for, also negative

Direct follow-up to §19's own closing question: after ten refinements of
`deepfm_mtl_v1` all failed (§10-§19), the honest next move was a lever from
a genuinely different family, not another variant of the same one.
LambdaRank (Burges et al. 2006) reweights each pairwise comparison by how
much fixing it would actually move the metric that's being optimized —
here, ΔnDCG@5 for swapping a within-user (positive, negative) pair's rank
order — rather than treating every pair equally the way plain BPR (§8) or
uniform listwise softmax (§13) do. Chosen specifically because it targets
nDCG@5 by name, not just "ranking" generically.

Built with the same discipline as every other high-risk mechanism this
project has added: the ΔnDCG@5 math (`dcg_discount`, `idcg_at_5`,
`delta_ndcg5_for_pair`) was hand-derived and unit-tested against worked-by-
hand cases (`test_lambdarank_delta_ndcg5_math_on_hand_derived_cases`)
*before* ever training on real data — the same "correctness test first"
practice that caught real bugs in PCGrad's shared-parameter set (§17) and
focal loss's own test expectations (§19) elsewhere in this project.
`DeepFM_LambdaRank.lambdarank_step()` reuses the per-user padded-batch
infrastructure `deepfm_listwise.py`/`tools/check_listwise.py` already
built (§13): for each user with both a positive and negative impression,
sample up to `max_pairs_per_user` pairs, weight each pair's
`softplus(-(s_pos - s_neg))` loss by its ΔnDCG@5, and average.

Standalone check (`tools/check_lambdarank.py`, same "prove it before
promoting it" treatment as `deepfm_listwise_v1`/`deepfm_pdaom_v1`):

| epoch | loss | valid primary |
|---|---|---|
| 1 | 0.5928 | 0.5827 |
| 2 | 0.5139 | **0.5856** (best) |
| 3 | 0.4613 | 0.5765 |
| 4 | 0.4306 | 0.5737 |
| 5 | 0.4101 | 0.5683 |
| 6 | 0.3923 | 0.5670 |
| 7 | 0.3751 | 0.5692 |

Best epoch 2 of 7, valid primary 0.5856, test 0.5802 — clearly below
`deepfm_regularized` (0.6035 valid, the plain pointwise-BCE architecture
this shares its backbone with), `deepfm_listwise_v1` (0.6033), and
`deepfm_mtl_v1` (0.6046). Not a marginal miss; a clear regression, and the
worst result of any structurally-different lever tried this project.

**Honest read:** the training loss keeps falling every epoch (0.59 -> 0.38)
while valid primary peaks at epoch 2 and steadily *worsens* through epoch
7 — the model keeps getting better at the pairwise objective it's actually
being trained on while getting worse at the metric that objective is
supposed to be a proxy for. This is the same "fast convergence to a
shallow optimum" shape §8's first BPR attempt showed, and LambdaRank
inherits BPR's core per-user sampled-pair training loop (just reweighted
by ΔnDCG@5), so it plausibly inherits the same instability source: each
epoch's gradient comes from a small, freshly-resampled slice of a user's
true pairwise preference structure (`max_pairs_per_user` pairs, and ranks
computed only within the padded batch's `max_len=64` window for users with
more impressions than that), which is a noisier, higher-variance signal
than plain BCE's full-batch pointwise gradient — and the softplus-margin
loss has no floor once it starts separating easy pairs, so it can keep
"improving" on training pairs long after that stops correlating with
validation ranking quality. Combined with §8's two BPR attempts and their
regularized follow-up, this is now the **third** confirmed instance of a
pairwise-sampling-based loss underperforming pointwise BCE on this
specific benchmark — not a single unlucky configuration, a repeating
pattern across three separate implementations (FM_BPR, DeepFM_BPR,
DeepFM_LambdaRank). `deepfm_mtl_v1` remains the project-best, and a fourth
pairwise-loss variant is not a good next use of compute without a genuinely
new idea about why the first three all converged too fast to a worse
optimum — a concrete candidate the Research Critic Gate (`agent/research_critic.py`)
does not yet cover, since its two rules are hyperparameter-axis-scoped, not
model-family-scoped (see the Roadmap in `CLAUDE.md`).

## 21. Ensembling revisited with a proper grid search — a single-seed edge that 3-seed verification erases

§5 already tried ensembling once (equal-weight pairs/averages of whatever
was cached at the time) and found nothing beat `deepfm_mtl_v1` alone. This
revisits it with two real methodological improvements, to make sure that
wasn't just an under-searched attempt: (1) a full 3-component weighted
grid search (`tools/check_ensemble.py`, step=0.1 over the simplex, not
just equal-weight pairs) across the three most architecturally distinct
*trained* models available (`fm_baseline_repro`, `deepfm_regularized`,
`deepfm_mtl_v1` — pure bilinear, +MLP, +4 auxiliary heads), each z-score-
normalized (stats fit on valid, same affine map applied to test — no test
information used to pick anything); (2) the chosen blend was then
**3-seed verified** (`tools/verify_ensemble_multiseed.py`) with fixed
weights (never re-tuned per seed — that would leak seed noise into the
"decision"), the same discipline this project applies to every other
single-seed result close to its parent (§18's `deepfm_mtl_click_v1`,
`features_v1`).

Single-seed grid search found a real-looking edge: best blend
(fm=0.20, deepfm_regularized=0.10, deepfm_mtl_v1=0.70) scored valid primary
**0.6051** vs. `deepfm_mtl_v1` alone's 0.6049 (+0.0002) — small, but a
genuine improvement over §5's best single-seed combination (0.6045). The
per-segment breakdown (§15's item-popularity quartiles) looked genuinely
promising too: the blend recovered +0.0034 to +0.0071 primary specifically
in Q1/Q2 (the least-popular-item segments, where `deepfm_mtl_v1` alone is
weakest at 0.4784-0.4791 — far below Q3/Q4's 0.52-0.60), while Q3/Q4 barely
moved — exactly the kind of segment-level smoothing an ensemble of
differently-biased models is supposed to provide.

3-seed verification (fixed weights, seeds 0/1/2, each component retrained
at seeds 1 and 2 where not already cached):

| | seed 0 | seed 1 | seed 2 | mean +/- std |
|---|---|---|---|---|
| `deepfm_mtl_v1` solo | 0.6049 | 0.6049 | 0.6042 | **0.6046 +/- 0.0003** |
| fixed blend | 0.6051 | 0.6046 | 0.6042 | **0.6046 +/- 0.0004** |

Delta: **−0.0000**. Diagnosed `noise_floor` by the exact same seed-aware
significance bar (`agent/diagnosis.py`) used for every other node — the
seed-0 edge (+0.0002) does not survive; it was itself noise, not a real
effect the grid search found. **Not a confirmed win.**

**Honest read:** this is the **seventh** structurally different lever
tried beyond `deepfm_mtl_v1`'s own architecture (five BPR-family variants,
LambdaRank, now ensembling) with none beating it after proper seed
verification — the strongest evidence yet that `deepfm_mtl_v1` sits at (or
very near) a real, non-trivial local optimum for this benchmark's specific
data volume, not that the search has been shallow. The per-segment
smoothing effect at seed 0 (weakest-segment Q1/Q2 improvement) is still a
real, interesting *distributional* observation even though the aggregate
number ties — worth a follow-up with more seeds specifically on the
per-segment breakdown if there's ever a reason to revisit ensembling, but
not a build-time win today. `deepfm_mtl_v1` remains the project-best and
the shipped submission.

## 22. Adding LightGBM as a 4th ensemble component — still no win, and now we know why

§21's own diagnosis floated a specific, testable explanation for why the
3-component ensemble only tied instead of winning: FM/DeepFM/DeepFM-MTL
are all embedding-based and plausibly correlated in their errors, so
averaging them buys little real diversity. `lgbm_baseline` (gradient-
boosted trees, a genuinely different inductive bias) was the obvious
candidate to test that — but it previously existed only as a metrics-only
Research Map node, trained on a teammate's Windows machine with no cached
`predictions.npz` anywhere (and blocked from being retrained there by a
Windows-specific WDAC application-control policy refusing to load
LightGBM's native DLL — `OSError: [WinError 4551]`, not a real bug, just
not reproducible on that machine).

Retrained locally on macOS instead (`tools/train_lgbm_and_cache.py`, same
hyperparameters as the existing node: `num_leaves=63, lr=0.05,
min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.9,
num_boost_round=500, early_stopping_rounds=30`, organizer's own
`data.encode()` pipeline for guaranteed row-alignment with the other
cached models). Landed almost exactly on the original single-seed number
(valid 0.5991 / test 0.5944 vs. the existing node's 0.5995 / 0.5946) — a
good sanity check that this is a faithful reproduction, not a fluke of a
different encoding path.

`tools/check_ensemble.py`'s grid search was generalized from a hardcoded
3-component simplex to an N-component one (any `COMPONENTS` list now
works, not just exactly 3), then `lgbm_baseline` added as a 4th
component. Checked at two grid resolutions (step=0.1 and step=0.05) to
rule out a coarse-grid artifact:

| grid step | best blend weights | blend valid primary | lgbm_baseline's weight |
|---|---|---|---|
| 0.10 | fm=0.20, deepfm_regularized=0.10, mtl=0.70, **lgbm=0.00** | 0.6051 | **0** |
| 0.05 | fm=0.15, deepfm_regularized=0.30, mtl=0.55, **lgbm=0.00** | 0.6051 | **0** |

**The grid search itself assigns LightGBM a weight of exactly zero at
both resolutions — the diversity hypothesis doesn't hold.** The +0.0002
valid edge that remains is the same one §21 already found and diagnosed
`noise_floor` after 3-seed verification; adding a 4th, genuinely-diverse
component didn't change that number at all, because the search correctly
recognized LightGBM isn't worth including. The reason, in hindsight, is
straightforward: LightGBM's absolute quality (0.5991) sits far enough
below the three embedding-based models (0.6015-0.6049) that whatever
error diversity it offers isn't worth the quality it drags the blend
toward — diversity only helps a blend when it doesn't cost too much
accuracy to buy it, and here it costs more than it buys.

**Honest conclusion:** this closes the specific, well-reasoned follow-up
question §21 raised, with a real answer rather than leaving it
speculative. `deepfm_mtl_v1` remains the project-best and the shipped
submission; ensembling — in every form tried across §5, §21, and this
section — does not improve on it.

## 23. Stacked ensemble via scikit-learn — a more principled method, an honestly *worse* number

§21's ensemble used a hand-tuned grid search over a weight simplex,
evaluated in-sample on valid (the same rows the weights were picked
against) — standard practice for this project's other config-selection
decisions, but a looser standard than proper stacking discipline, which
never scores a meta-learner on rows it was fit on. `scikit-learn` wasn't
usable on the machine this check was run from (a Windows Application
Control policy blocks `lightgbm`'s native DLL load there, and §22's
LightGBM ensemble result above came from a teammate's macOS machine
instead; `scikit-learn` itself turned out to just be uninstalled on this
one, not blocked — `pip install scikit-learn` worked immediately once
actually tried), so this angle was previously untested.

`tools/check_stacked_ensemble.py`: same 3 z-scored components as §21
(`fm_baseline_repro`, `deepfm_regularized`, `deepfm_mtl_v1` — run before
§22's LightGBM result was available, so LightGBM isn't one of this
check's inputs), combined by a `LogisticRegression` meta-learner instead
of a fixed weight grid — genuinely more expressive (its own intercept,
coefficients not constrained to sum to 1, optimizes log-loss directly).
Scored honestly via 5-fold `cross_val_predict` on valid (out-of-fold
predictions only, never scored on rows the fold's own model saw during
fitting):

| | Valid primary |
|---|---|
| `deepfm_mtl_v1` solo | 0.6049 |
| Stacked ensemble (5-fold out-of-fold) | **0.6044** |

Delta: **−0.0004**. Learned coefficients (`fm=0.27`, `deepfm_regularized=0.55`,
`deepfm_mtl_v1=0.45`, intercept −1.06) show the meta-learner leaning on
`deepfm_regularized` slightly more than `deepfm_mtl_v1` despite the latter
being the stronger solo model — a sign it's fitting noise in the 3-feature
space rather than finding real complementary signal. Single seed, not
3-seed-verified: unlike §13/§18's borderline single-seed results, this one
came back *negative* on the fair, out-of-fold evaluation already, so
3-seed verification wouldn't change the decision (matching the precedent
set by §14's PDAOM and §20's LambdaRank — a single seed is enough when the
result is decisively unpromising, not ambiguous).

**Honest read:** the more expressive method did not do better than the
simpler grid search — if anything, the honest out-of-fold evaluation here
is a *more* trustworthy read of the grid search's own result than the grid
search's own in-sample number was. Both this check and §22 agree on the
same substantive conclusion from two different angles: three (or four,
counting §22's LightGBM at a learned weight of zero) blended predictions
of models that are either too correlated (the embedding-based three) or
too weak in absolute quality (LightGBM) to combine into something better
than the strongest individual model, regardless of whether the combination
method is a hand-tuned grid or a properly-fit meta-learner. `deepfm_mtl_v1`
remains the project-best.

## 24. DCNv2 — the last named architecture-level lever, checked for completeness

The one specific architecture named in this project's own roadmap
(`docs/POLISH_PASS_RESULTS.md`, README's former Limitations section)
that was never actually tried, alongside Wide&Deep. Explicitly the
lowest-priority lever per CLAUDE.md's own "Unexplored headroom" ranking —
every capacity/architecture change actually tested in this project came
back flat or negative (embedding width k=8/16/32 flat per the starter
kit's own ablation; CWM's extra feature domains no gain; DIN's attention
block a `ranking_tradeoff`, not a clean win; LightGBM's tree splits a real
loss vs. the embedding-based models). Built anyway for completeness, with
expectations set accordingly, not because the evidence pointed toward it.

`agent/model_zoo/dcnv2.py` (Wang et al. 2020): replaces DeepFM's FM
2nd-order interaction term with a Cross Network — `n_cross_layers`
stacked layers of the form `x_{l+1} = x0 * (W_l @ x_l + b_l) + x_l`,
run in parallel with a plain deep MLP tower, both reading the same
flattened embedding `x0`, concatenated before one final linear layer.
Registered directly in `agent/model_zoo/registry.py` (plain pointwise
BCE, no special training-loop branch needed, unlike the per-user-batched
loss variants) — the first new P2 model that runs through the normal
`agent/experiment.py` path without a standalone-check workaround.

A real bug caught by its own unit test before ever touching real data:
the first test design (shifting across different random batches, the
pattern most other models' tests use) genuinely failed —
`losses[-1] < losses[0]` broke because different random batches have
different baseline losses, and 3 stacked cross layers at `lr=0.01` turned
out noisier batch-to-batch than DeepFM's own gentler curve tolerates. Not
an actual backward-pass bug: verified directly (same `lr` on a FIXED,
repeated batch drives loss from ~0.70 to ~0.0015 in 30 steps). Fixed by
switching the test to the fixed-batch design `deepfm_lambdarank`'s own
test already established, which isolates "does the gradient reduce the
loss it was computed from" from "is 30 steps enough to beat batch-to-batch
variance" — the latter is what actually broke, not the model.

Trained with `k=16, hidden=[128,64], n_cross_layers=2, lr=0.001, l2=1e-4`,
identical to `deepfm_regularized` so architecture is the only variable
(`parent_id=deepfm_regularized`, registered as `dcnv2_v1`). Single seed
first: valid primary 0.6037, best epoch 8/12 (a healthy curve, not an
early-collapse pattern like PDAOM/PCGrad/the first BPR attempt) — close
enough to its parent (+0.0002) to need verification before trusting it,
same discipline as any single-seed result this close.

3-seed verified (`tools/verify_multiseed.py`):

| | Valid primary | vs. parent (`deepfm_regularized`) |
|---|---|---|
| `dcnv2_v1` | 0.6039 ± 0.0003 | +0.0004 GAUC, +0.0005 nDCG@5, +0.0005 primary — **exactly at** the 0.0004 significance bar |
| `deepfm_mtl_v1` (current best, for comparison) | 0.6046 ± 0.0003 | — |

Diagnosed `mixed` by `agent/diagnosis.py`'s own significance-bar logic —
not `clear_improvement` (the delta needs to be *strictly greater* than the
bar on both metrics; here it lands right on it, a genuine coin-flip
result at this project's own resolution limit for distinguishing signal
from noise with 3 seeds), and not `noise_floor` either. The most honest
characterization: DCNv2 is indistinguishable from a marginal tie with
`deepfm_regularized`, at best -- and still clearly below `deepfm_mtl_v1`
either way. Consistent with the wider pattern this whole project has
found: architecture changes (embedding width, CWM features, DIN, now
DCNv2, LightGBM) cluster around flat-to-marginal outcomes, while the
multi-task *training signal* change (`deepfm_mtl_v1`) is the one lever
that produced a clear, repeatedly-confirmed win. `deepfm_mtl_v1` remains
the project-best.

**A real bug this check surfaced, caught by its own regression test before
being trusted:** `tools/verify_multiseed.py`'s `verify_node_multiseed()`
originally timed itself with `time.process_time()` wrapped around
`run_with_recovery()` -- which spawns a *separate subprocess* to do the
actual training (`agent/recovery.py`, required for a real timeout kill on
Windows). A wrapper's own `process_time()` only counts CPU time the
wrapper itself burns (mostly idle time waiting on `proc.join()`), not the
child's. Live symptom: DCNv2's 3-seed verification, which took several
real minutes of subprocess training, initially reported `wall_time_s=0.03`.
Since this value feeds directly into `agent/p1_orchestrator.py`'s and
`agent/p4_orchestrator.py`'s own wall-clock budget accounting (the
auto-verification feature added earlier this pass), this would have
silently undermined `--max_wall_time_s`'s real ceiling -- the exact kind
of "looks like it works, quietly doesn't" bug this project has caught
several times before (the 12h host-sleep wall-clock bug in P1, the
Diagnosis Engine's false noise_floor label). Fixed by summing each
per-seed result's own `wall_time_s` (measured *inside* the subprocess by
`agent/experiment.py`, always correct) instead of timing the wrapper.
Regression-tested (`test_verify_node_multiseed_reports_real_subprocess_wall_time`)
against a real, fast, two-seed training run.

## 25. Stochastic Weight Averaging on `deepfm_mtl_v1` — a genuine null result, not a regression

The last of this project's original 4-idea brainstorm (focal loss,
LambdaRank, SWA, cascade modeling) never actually tried — focal loss and
LambdaRank both came back regressions (§19-20). Genuinely orthogonal to
every other lever tried: SWA (Izmailov et al. 2018) doesn't touch the loss
function, architecture, training signal, or combine multiple separately-
trained models (unlike §21-23's ensembling) — it changes how a single
training run's *final checkpoint* is built, averaging weights across a
window of epochs instead of taking the single best-validation-epoch point
estimate every other model in this project uses.

Directly motivated by `deepfm_mtl_v1`'s own real training curve (already
on record): valid primary sits in a flat plateau roughly epochs 7-14
(0.6036-0.6049) before dropping off at epoch 15+ — a textbook case for
weight averaging, since the individual epoch checkpoints inside a plateau
are different, roughly-equally-good local solutions. `DeepFM_MTL`'s
architecture has no BatchNorm layers (embeddings, `Linear`, `ReLU` only),
so plain elementwise parameter averaging needs no extra recalibration
step, unlike the usual SWA caveat about BN running statistics.

`tools/check_swa.py`: same exact hyperparameters as `deepfm_mtl_v1`
(`k=16, hidden=[128,64], lr=0.001, l2=1e-4, aux_weight=0.2`), averaging
every epoch whose valid primary is within `tolerance=0.0015` of the best
epoch's (window picked by proximity to the best score, not a fixed "last K
epochs" rule that would be sensitive to exactly when early stopping
triggered):

| | Valid primary | Test primary |
|---|---|---|
| Point-best checkpoint (epoch 12) | 0.6048 | 0.5979 |
| SWA-averaged (8 epochs, 7-14) | 0.6046 | 0.5980 |
| Delta | **−0.0002** | **+0.0001** |

Single seed, not 3-seed-verified: the delta is already essentially zero in
both directions (valid slightly down, test slightly up) — a decisively
*null* result, not an ambiguous one, so verification wouldn't change the
conclusion (same "a single seed is enough when the result is decisive"
precedent §14/§23 already established, just for a null rather than a
negative decision this time).

**Honest read:** unlike focal loss and LambdaRank (both real regressions),
SWA doesn't hurt `deepfm_mtl_v1` — it just doesn't help either. Plausible
reason: the plateau epochs 7-14, while close in *validation score*, may
already be similar enough *in weight space* (Adam's adaptive per-parameter
steps late in training tend to make small, correlated updates near a
minimum) that averaging them lands close to where any single one of them
already sits, rather than in a meaningfully different, flatter region the
way SWA helps most when individual checkpoints are more weight-space
diverse (e.g., SGD with a cyclic/high constant LR, which finds more
different points per epoch than Adam's fine-tuned late-training steps
typically do). `cascade modeling`, the one remaining original-brainstorm
idea, was not built: KuaiRand-Pure's evaluation scores a *fixed,
already-logged* impression list per user, not a candidate-generation-then-
rerank pipeline — the setting cascade modeling is designed for (cheaply
filtering a huge candidate pool before an expensive final ranker) doesn't
have a natural analogue when there's no candidate-generation step in the
task to begin with. `deepfm_mtl_v1` remains the project-best.

## 26. Deeper auxiliary heads (`deepfm_mtl_deep_heads_v1`) — capacity wasn't the bottleneck either

Every prior MTL refinement changed *how much* the auxiliary losses count
(uncertainty-weighting, §12) or *how their gradients combine* with the
main task's (PCGrad, §17) — never whether the aux heads have enough
capacity to extract a clean signal in the first place. `deepfm_mtl_v1`
maps all 4 auxiliary tasks through ONE shared `nn.Linear(hidden[-1], 4)`
— literally the same weight matrix for every task, no task-specific
nonlinearity. `agent/model_zoo/deepfm_mtl_deep_heads.py` gives each task
its own small private MLP (`Linear(128→32)→ReLU→Linear(32→1)`) instead —
same shared trunk/embeddings, main task unchanged, only the auxiliary
heads' capacity is the variable being isolated. Registered directly in
`agent/model_zoo/registry.py` (reuses the existing `is_mtl` branch, no new
training-loop code).

Single seed first: valid primary 0.6048, near-identical training curve
shape to `deepfm_mtl_v1`'s own (best epoch 12/17 both times) — close
enough to the parent to need verification.

3-seed verified (`parent_id=deepfm_mtl_v1`):

| | Valid primary | vs. parent |
|---|---|---|
| `deepfm_mtl_deep_heads_v1` | 0.6038 ± 0.0008 | −0.0008, `noise_floor` (bar=0.0010) |
| `deepfm_mtl_v1` (parent, current best) | 0.6046 ± 0.0003 | — |

**Honest read:** giving the auxiliary tasks independent capacity doesn't
help — the shared-linear-layer bottleneck wasn't actually limiting
anything. Plausible reason: the 4 auxiliary signals
(`is_like`/`is_follow`/`is_comment`/`is_forward`) are all downstream of
the same latent "did this user genuinely enjoy this video" factor (the
same hypothesis `deepfm_mtl_v1`'s own design was built on) — if that's
right, a SHARED projection is actually the theoretically correct
inductive bias for these 4 tasks, not a compromise forced by insufficient
capacity; giving them independent heads just adds free parameters without
a genuine specialization need, explaining the clean noise_floor rather
than either a clear win or a clear loss. `deepfm_mtl_v1` remains the
project-best.

## 27. Graph-propagated embedding initialization (LightGCN-style) — a genuinely different mechanism, still a null result

Requested explicitly as a "non-standard" attempt after §20-26 exhausted
the standard toolbox (loss functions, architectures, ensembling,
checkpoint averaging, auxiliary-head capacity) without a win. Every prior
lever still treats embeddings as a pure lookup table learned from scratch
via gradient descent alone. This instead injects collaborative-filtering
STRUCTURE *before* training starts: LightGCN (He et al. 2020) -- normalized
bipartite-graph propagation over user-video interactions -- used to
initialize `deepfm_mtl_v1`'s user_id/video_id embedding rows instead of
random init. A genuinely different mechanism, not a variant of anything
already tried.

`tools/check_gnn_init.py`: graph built from TRAIN-split `is_click=1` rows
only (528,845 of 1,141,112) -- a distinct auxiliary signal from the
scored `long_view` label, same discipline as the 4 MTL aux heads, not a
circular use of the target. 2-layer symmetric-normalized-adjacency
propagation (`agent/features.py`'s `load_aux_labels` reused for `is_click`,
`data.encode()`'s own indices reused directly for graph node ids -- no
re-derived vocab, no drift risk), rescaled to match `deepfm_mtl.py`'s own
init std before overwriting 32,568 of 40,260 embedding-table rows (every
untouched row -- `author_id`/`tab`/`dur_bucket`, and any never-clicked
user/video -- keeps the model's own random init). The propagation math
itself was verified on hand-constructed synthetic data *before* touching
real data: two users sharing a graph neighbor end up measurably more
similar after propagation (cosine similarity 0.76) than two users with no
shared neighbor (−0.15) -- confirms the mechanism does what it's supposed
to before ever trusting a real-data number from it.

3-seed result (same architecture/loss/hyperparameters as `deepfm_mtl_v1`
throughout -- embedding initialization is the only variable):

| | Valid primary | vs. `deepfm_mtl_v1` |
|---|---|---|
| `deepfm_mtl_gnn_init_v1` | 0.6045 ± 0.0003 | −0.0001, `noise_floor` (bar=0.0005) |
| `deepfm_mtl_v1` (parent, current best) | 0.6046 ± 0.0003 | — |

**Honest read:** the graph-propagated initialization makes essentially no
difference -- an even cleaner null than §26's deeper auxiliary heads
(−0.0001 vs −0.0008). Plausible reason: with only 703,110 trainable
parameters and Adam's adaptive per-parameter updates, ~10-12 epochs of
full-batch-equivalent gradient descent is enough to move the embeddings
well past whatever information the initial condition encoded -- the model
converges to essentially the same solution regardless of starting point,
the same way two different random seeds already do (§4's `fm_seed_variance`
finding, extended here to structured vs. random init, not just two random
seeds). This is itself informative: it suggests the ~0.27 headroom to the
oracle ceiling (CLAUDE.md) isn't sitting in embedding *initialization*
either -- reinforcing, from yet another angle, that this benchmark's
learnable signal at this data volume has been substantially extracted by
`deepfm_mtl_v1`'s combination of embedding crossing + multi-task
regularization, not that any one training detail was quietly leaving
value on the table. A real regression test caught a subtle bug in the
verification process itself before it produced a misleading result:
`tools/verify_multiseed.py`'s standard re-training path rebuilds a model
via the plain registry (`deepfm_mtl.build()`), which would silently skip
the graph-init overlay entirely and 3-seed-verify plain random-init
`deepfm_mtl` instead -- caught before running it, so seeds 1/2 were run
directly through `check_gnn_init.py` (the actual treatment) instead.
`deepfm_mtl_v1` remains the project-best, now confirmed against **twelve**
structurally different levers, two of which (§26, §27) were built and
verified specifically in response to a direct request for a genuinely
different mechanism, not just another standard-toolbox variant.

## Net effect on the project-best

| | Valid primary (3-seed) | Test primary (3-seed mean) | vs. official baseline (test) |
|---|---|---|---|
| Official FM baseline | 0.6016 | 0.5946 | — |
| `deepfm_regularized` (Phase 4, prior best) | 0.6035 ± 0.0002 | 0.5977 | +0.0031 |
| **`deepfm_mtl_v1` (P2, current best)** | **0.6046 ± 0.0003** | 0.5974 | +0.0028 |

Engineered features, LightGBM, DIN sequence modeling, DeepFM_BPR, the
watch-time auxiliary head, combining DIN with MTL, uncertainty-weighted
MTL, listwise ranking loss, and PDAOM hard-pair mining were all genuinely
worth trying — public, well-motivated ideas backed by the problem
statement's own allowed toolset, several straight off this project's own
documented headroom list — and all came back negative, mixed, or (in
uncertainty-weighting's and listwise's cases) too close to call, each for
a different, specific, structural reason rather than "we didn't get to
it." Multi-task learning
(the original, fixed-weight version) was the one that paid off, and did
so on the first attempt with no diagnosis-driven iteration needed (unlike
BPR's 3 P1 rounds, or DeepFM_BPR's own regularization round here). The
randomized-exposure check
(§9) is the closest thing to independent confirmation this project has:
`deepfm_mtl_v1`'s edge over the baseline doesn't just survive an unbiased
data distribution, it grows there.
