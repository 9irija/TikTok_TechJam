# Autonomous ML Research Agent for KuaiRand-Pure

TikTok TechJam 2026 — Challenge 2: *Autonomous Machine Learning Research
Agent for Recommender Systems*.

An agent that runs the MLE iteration loop — read the problem → inspect data
→ engineer features → train + tune → evaluate → reflect + revise — on its
own against **KuaiRand-Pure**, reproduces the official Factorization Machine
baseline, then iterates past it while logging every hypothesis, code diff,
metric, and error/recovery event it produces along the way.

> Full problem statement + team brainstorm: [`docs/TikTok TechJam Hackathon.md`](docs/TikTok%20TechJam%20Hackathon.md).
> Operational project context (architecture, task-definition caveat, roadmap): [`CLAUDE.md`](CLAUDE.md).

## Project overview

The challenge scores a **converged, autonomously-produced improvement** over
a fixed baseline on a hidden test set, not a one-shot leaderboard score. This
repo has two layers built so far:

**Phase 0 — foundation** (all 8 P0 features from the brainstorm doc):

- **Evaluator Wrapper** — imports the organizer's `evaluate.py` directly; never reimplements its pinned scoring conventions.
- **Convergence Detector** — implements the organizer's ε=0.002 / N=3 rule, read live from `baseline_scores.json`.
- **Model Zoo (base)** — Factorization Machine (faithful port of the official baseline) + DeepFM (numpy-only MLP deep component over shared embeddings).
- **Structured Experiment Interface** — a config schema an orchestrator fills in instead of an LLM writing training code from scratch.
- **Orchestrator** — runs a fixed predefined experiment list end-to-end (no LLM reasoning yet — that's a later phase).
- **Failure Recovery** — every experiment runs in an isolated subprocess with a real timeout kill switch, retry, and a degraded fallback so one bad run can never crash or stall the whole loop.
- **Structured Run Log** — every iteration's hypothesis, config diff, metrics, and error/recovery events, both as per-experiment folders and one append-only JSONL.
- **Submission Validator** — wraps the organizer's `submit.py` format checks exactly before anything is called "final."

**P1 — differentiators** (`python run_p1.py`, see
[`docs/P1_FEATURES_AND_RESULTS.md`](docs/P1_FEATURES_AND_RESULTS.md)):

- **Research Map / Experiment Tree** — *persistent* across runs (unlike Phase 0's per-run log), tree-structured (draft/improve/debug edges, AIDE-style).
- **Metric-Aware Diagnosis Engine** — classifies *why* a result looks the way it does (clear improvement / regression / ranking trade-off / noise floor), seed-aware (uses real per-seed std, not just a flat threshold on point estimates).
- **Multi-Fidelity Runner** — 1%→10%→100% staged training, killing clearly-broken candidates cheaply instead of always training at full scale.
- **Best-First Node Selector** — deterministic `expected_gain × confidence × novelty ÷ cost` scoring (no LLM yet — that's Phase 4) that decides which candidate to try next and why, with a logged reasoning string per candidate.
- **FM_BPR** — a new model: pairwise BPR ranking loss instead of pointwise logloss, the starter kit README's own #1-ranked untested direction.
- **Engineered features** (`agent/features.py`) — TikTok-disclosed signals (completion rate, rewatch, fast-skip, creator engagement) added as TRAIN-SPLIT-ONLY AGGREGATE fields, never a same-row `play_time_ms` ratio (which would leak the `long_view` label directly — confirmed ~85% correlated with it before writing any of this code). Result: `features_v1`, 3-seed verified, landed at `noise_floor` vs. `deepfm_regularized` — reported plainly as a real negative-ish result, not omitted.

See `CLAUDE.md` for the full architecture, the exact judging-criteria → code
map, and — importantly — a **task-definition caveat**: the problem statement's
prose text and the Starter Kit's actual code disagree on the label/metrics;
this repo follows the Starter Kit (the pinned, code-level source of truth),
using label `long_view` and metrics `GAUC` / `nDCG@5`.

## Development tools & environment

- **Editor**: VS Code (via the Claude Code extension)
- **Language / runtime**: Python 3.14, Windows 11
- **Dev dependencies actually used**: `numpy` (all model/eval code), `pandas` (ad-hoc EDA only — never imported by `agent/`)
- **LLM API**: Google Gemini (free tier), used by Phase 4's Research Strategist only — Phase 0 and P1 deliberately have no LLM in the loop (hand-authored candidate pools, by design). `logs/p4_run_report.json`'s `llm_tokens_total` is real; Phase 0/P1's `run_summary.json`/`p1_round_report.json` stay structurally 0.

### Why no GPU was used

The problem statement is explicit that compute is not the binding
constraint on this benchmark: 100 iterations of the official baseline
complete in well under an hour on a single CPU core with no GPU (the
organizer's own `baseline.py` alone reproduces it in ~40s per run —
starter kit README). GPU-hours and LLM tokens are reported for
Feasibility & Practicality scoring, but they're not something to spend
just because the budget technically allows it — they're what the PS
actually measures against wall-clock hours, not a quota to fill, and
unnecessary GPU usage is a real cost with no upside unless it produces a
genuine, validated score improvement.

Our own numbers back this up directly. Phase 0's full run (self-check →
4 predefined experiments → convergence, `logs/run_summary.json`) — the
project's first complete, automatic convergence — took **18.1 minutes**
wall-clock, **4 iterations**, **0 GPU-hours**, 0 manual interventions, all
CPU-only. Every model built since, including all of P2's torch models
(`deepfm_mtl`, `deepfm_din`, `deepfm_bpr`, and the later variants),
trains in single-digit minutes per seed on one CPU core — the current
project-best (`deepfm_mtl_v1`) included.

This project's later evidence makes the case even more directly: 8 of the
last 10 real P2 modeling attempts came back flat or negative
(`docs/P2_FEATURES_AND_RESULTS.md`), a pattern that describes hitting this
benchmark's actual learning-problem ceiling, not a compute ceiling. More
epochs or a bigger network on an already-plateaued or overfitting model
burns compute for nothing — it doesn't buy a better result
(`deepfm_pdaom_v1` and the first `deepfm_bpr_v1` attempt were both fixed
by *less* capacity/training, via diagnosed `overfitting_risk`, not more).
We'll reach for GPU only if a specific candidate — a genuinely larger
architecture, or the bonus KuaiRand-1k/27k benchmarks — is CPU-infeasible
within the compute budget, not as a default. That hasn't happened yet.

## Datasets and assets used

- **KuaiRand-Pure** (Kuaishou / KuaiRand, via Zenodo — https://kuairand.com), the challenge's required benchmark. 1.4M interactions, 27K users × 7.6K items. No other dataset is used for training, per the challenge's one hard rule ("no external training data").

## Setup and installation

```bash
git clone https://github.com/9irija/TikTok_TechJam.git
cd TikTok_TechJam
pip install -r requirements.txt

# Download the dataset (public, no auth — ~47MB)
cd kuairand-starter-kit
curl -sL -o KuaiRand-Pure.tar.gz "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
tar xzf KuaiRand-Pure.tar.gz && rm KuaiRand-Pure.tar.gz
cd ..
```

Windows note: use `python` (not `python3`, which is a Microsoft Store alias
stub with no interpreter behind it in some environments).

## Steps to reproduce results

```bash
# 1. Smoke-test the foundation (no pytest dependency — plain script)
python tests/test_foundation.py

# 2. Run the Phase 0 loop end-to-end: self-check -> baseline repro ->
#    predefined experiments -> convergence -> submission generation+validation
python run.py

# 3. Regenerate the Run & Iteration Log deliverable
python tools/generate_analysis.py --stdout

# 4. Run a P1 round: seeds the persistent Research Map from step 2's log,
#    scores the candidate pool (Best-First Node Selector), runs each
#    candidate through the Multi-Fidelity Runner in best-first order
python run_p1.py

# 5. Run a Phase 4 round: the LLM Research Strategist (Gemini) proposes
#    experiments instead of a hand-authored pool -- needs GEMINI_API_KEY
#    in .env (copy .env.example). Free tier: real token usage, $0 cost.
python run_p4.py --max_iterations 2

# 6. Promote a promising single-seed result to a 3-seed-verified one
python tools/verify_multiseed.py <node_id>

# 7. Regenerate submission_valid.csv / submission_test.csv from whatever the
#    Research Map currently considers a *confirmed* best (not just the raw
#    numeric leader -- see ResearchMap.best_confirmed_node()). Re-runnable
#    any time the map changes; reuses cached predictions when they exist.
python tools/generate_submission.py
```

Outputs: `experiments/<run_id>/iter_*/` (per-iteration hypothesis/config/results/logs;
run-scoped, so a later run never overwrites an earlier one's folders),
`logs/run_log.jsonl` + `logs/run_summary.json` (machine-readable),
`logs/analysis_report.md` (the human-readable deliverable), and
`submission_valid.csv` / `submission_test.csv` (format-validated against the
organizer's `submit.py --check` logic).

**Final Submission & Results Summary (Problem Statement §2.5 Deliverable
4 — results table with delta over baseline, plus resource usage: LLM
tokens, wall-clock, iterations out of the 50-cap, GPU-hours):**
[`docs/RESULTS_SUMMARY.md`](docs/RESULTS_SUMMARY.md).

## Results (current project-best, 3-seed verified)

Live numbers are always in `logs/analysis_report.md` / `logs/research_map.json`.
Current best is `deepfm_mtl_v1` — DeepFM with auxiliary `is_like`/`is_follow`/
`is_comment`/`is_forward` heads sharing the main embedding table (multi-task
learning; see "Multi-task learning" below), 3-seed-verified with
`tools/verify_multiseed.py`. It improves on `deepfm_regularized` (Phase 4's
LLM find, full writeup: [`docs/PHASE4_RESULTS.md`](docs/PHASE4_RESULTS.md)),
which itself improved on Phase 0's `deepfm_wider` (full writeup:
[`docs/PHASE0_FEATURES_AND_IMPROVEMENTS.md`](docs/PHASE0_FEATURES_AND_IMPROVEMENTS.md#4-final-validated-phase-0-result)),
still the reference for the earliest, largest jump over the official baseline.

| Metric | Official FM baseline | `deepfm_wider` (Phase 0) | `deepfm_regularized` (Phase 4) | `deepfm_mtl_v1` (P2, torch) |
|---|---|---|---|---|
| GAUC (valid, 3-seed) | 0.6674 | 0.6691 | 0.6699 | **0.6715** |
| primary (valid, 3-seed) | 0.6016 | 0.6028 | 0.6035 ± 0.0002 | **0.6046 ± 0.0003** |
| primary (test, 3-seed mean) | 0.5946 | 0.5976 | 0.5977 | 0.5974 |

`submission_valid.csv` / `submission_test.csv` reflect `deepfm_mtl_v1`
(regenerated via `python tools/generate_submission.py`, which resolves the
Research Map's `best_confirmed_node()` — see "Engineered features" below for
why that's not simply "the highest score in the map"). Every model here was
selected on **validation only** (the split every decision is allowed to
read) — the honest nuance worth stating plainly: `deepfm_mtl_v1`'s 3-seed
**test**-primary mean (0.5974) is actually marginally *below*
`deepfm_regularized`'s (0.5977), even though its **validation** win is
clear and real (+0.0011, both GAUC and nDCG@5 individually significant).
Same pattern `deepfm_regularized` itself showed over `deepfm_wider` —
exactly what train/valid/test discipline predicts for models that were
never selected by peeking at test, and reported here rather than smoothed
over.

## P1 results (three rounds — each acts on the previous round's own diagnosis)

Full details, including three bugs found and fixed during this pass's own
validation, in
[`docs/P1_FEATURES_AND_RESULTS.md`](docs/P1_FEATURES_AND_RESULTS.md).

**Round 1** — the Best-First Selector ranked 2 candidates and ran both
(best-first order, not declaration order) through the Multi-Fidelity
Runner. **Neither beat the baseline** — reported exactly as measured,
because negative results with a specific diagnosis are legitimate output,
not something to omit:

| Candidate | Valid primary | vs. parent | Diagnosis |
|---|---|---|---|
| `fm_bpr_default` (pairwise BPR loss, k=16) | 0.5981 | −0.0034 | `mixed`, with an overfitting-risk flag (best epoch at ~33% of training length) — a specific, actionable lead for a next round, not a dead end |
| `fm_wider_k32` (FM, k=32) | 0.6009 | −0.0006 | `noise_floor` — independently reproduces the starter kit's own documented "capacity isn't the bottleneck" finding, on this codebase's own harness |

**Round 2** — a genuinely new capability, not just more candidates: the
orchestrator now generates a follow-up candidate automatically when a
node's own diagnosis flags something actionable (`agent/p1_orchestrator.py`'s
`_diagnosis_driven_candidates()`). A second, unmodified `python run_p1.py`
found and ran exactly one new candidate — `fm_bpr_regularized` (L2 raised
100x, tighter early stopping, directly targeting the overfitting flag
above) — with zero manual intervention:

| Candidate | Valid primary | vs. `fm_bpr_default` | vs. baseline | Diagnosis |
|---|---|---|---|---|
| `fm_bpr_regularized` | 0.5989 | **+0.0008** (right direction) | still −0.0026 | `noise_floor` |

**Round 3** — a hand-authored follow-up from directly comparing full
validation curve *shapes* (not just best-epoch ratios, which is all the
automated Diagnosis Engine currently checks — a real gap, see
`P1_FEATURES_AND_RESULTS.md` §6): `fm_baseline_repro` climbs gradually for
7 epochs before declining; both prior BPR attempts instead plateaued by
epoch 2–4. `fm_bpr_slow_and_steady` (lr cut ~3x, more patience/epochs)
confirmed the hypothesis — its curve now climbs gradually for 6 epochs,
qualitatively matching the baseline's healthy shape:

| Round | Config | Valid primary | Δ vs. previous |
|---|---|---|---|
| 1 | `fm_bpr_default` | 0.5981 | — |
| 2 | `fm_bpr_regularized` | 0.5989 | +0.0008 |
| 3 | `fm_bpr_slow_and_steady` | 0.5994 | +0.0005 |

Honest conclusion: 3 rounds fixed BPR's *training dynamics* (no more
overfitting, healthy convergence shape) but only partially closed the
*quality* gap — still 0.0021 below the FM baseline, 0.0034 below DeepFM.
The Selector's own diminishing-returns prior predicted round 3's gain
almost exactly (+0.0004 projected vs. +0.0005 actual) — the honest signal
to stop iterating on this specific lever (FM + BPR + hyperparameters) and
either try DeepFM_BPR or accept this direction has a structural, not
tunable, ceiling here. Reported as a real, incomplete trend — not a clean
win, and not worth a 4th round without a genuinely new hypothesis. (Best
result across the project moved past `deepfm_wider` in Phase 4, below.)

## Phase 4 results — the LLM found a real, verified improvement

Full details, including the LLM's actual reasoning and a genuine failure it
also proposed, in [`docs/PHASE4_RESULTS.md`](docs/PHASE4_RESULTS.md).

`python run_p4.py --max_iterations 2`: the LLM Research Strategist (Gemini,
$0 cost, 4,947 tokens total) replaces the hand-authored candidate pools
with a model that reads the full Research Map history and proposes what to
try — reusing P1's Multi-Fidelity Runner and Diagnosis Engine completely
unchanged.

- **Iteration 1 — a real win.** LLM proposal `deepfm_regularized`: raise L2
  on `deepfm_wider`'s architecture, because *both* prior DeepFM nodes were
  flagged `overfitting_risk` — a gap P1's own hand-authored logic never
  acted on (it only ever reacted to `fm_bpr_default`'s overfitting flag).
  Result: valid primary 0.6035, up from 0.6028. **3-seed-verified**
  (`tools/verify_multiseed.py`, new this pass): 0.6032 / 0.6035 / 0.6036,
  mean 0.6035 ± 0.0002 — a robust, real `clear_improvement`, not a lucky seed.
- **Iteration 2 — a real failure, correctly caught.** LLM proposal
  `deepfm_higher_l2`: push L2 further. Result: GAUC −0.0512, nDCG@5 −0.0191
  — a clear regression, diagnosed as such, not hidden. Reported here
  precisely because a curated "the LLM only ever proposes good ideas"
  writeup would be dishonest.

This was the project's best result until the multi-task model below (P2) superseded it — still the reference for the LLM finding a real, non-obvious gap on its own.

## Multi-task learning (torch) — the current project-best

Full writeup, including the LightGBM comparison below: [`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md).

`agent/model_zoo/deepfm_mtl.py`: the first genuine PyTorch model in this
project. Same DeepFM architecture as `deepfm_regularized` (shared embedding
table, FM 2nd-order term + deep MLP), plus four small auxiliary sigmoid
heads reading the same pooled embeddings, trained jointly on TikTok's other
logged engagement signals — `is_like`, `is_follow`, `is_comment`,
`is_forward` — a shared-bottom setup in the public ESMM/MMoE multi-task
recsys line. Only the main `long_view` logit is ever scored (GAUC/nDCG@5);
the auxiliary heads exist purely to shape training via a combined loss
(`main_loss + 0.2 * aux_loss`), never touch evaluation.

Why torch here and nowhere else in the Model Zoo: FM/DeepFM/FM_BPR each
have one clean backward pass, worth hand-deriving to keep the starter
kit's numpy-only philosophy. A 5-headed shared-bottom network's backward
pass — one shared trunk, five different loss gradients merging back into
it — is exactly the case autograd is for, not a shortcut around it.

`deepfm_mtl_v1` (`parent_id=deepfm_regularized`, same fields/k/hidden/lr/l2
— the multi-task objective is the only variable) ran single-seed first
(0.6049), then **3-seed-verified**: **0.6046 ± 0.0003** vs. parent's
0.6035 ± 0.0002 — `clear_improvement` (both GAUC +0.0016 and nDCG@5 +0.0008
individually significant, bar=0.0004). This is now the project's best
result, and `submission_valid.csv`/`submission_test.csv` reflect it.

Honest nuance, stated plainly rather than smoothed over: the 3-seed
**test**-primary mean (0.5974) is marginally *below* `deepfm_regularized`'s
(0.5977), even though the **validation** win (the only split any decision
here is allowed to read) is clear and real. Same pattern
`deepfm_regularized` itself showed over `deepfm_wider` — exactly what
train/valid/test discipline predicts.

**LightGBM — tried, and a real negative result with a structural reason,
not just "didn't beat it."** A quick standalone check (same 5 base fields,
`lgbm_baseline` in the Research Map, `parent_id=fm_baseline_repro`):
valid primary 0.5995, test primary 0.5946 (exactly ties the official FM
baseline, doesn't beat it) — trailing every DeepFM variant by a real
margin. This isn't bad luck: gradient-boosted trees split on features,
they don't learn dense embeddings, and this task's signal lives almost
entirely in `user_id × video_id` crossing (~27K users × ~7.6K items) —
exactly what FM/DeepFM's embedding tables are built for and what a tree
can only crudely approximate at this cardinality. Full Model Zoo
integration was deliberately not built (LightGBM manages its own
boosting/early-stopping internally, a real interface mismatch with every
other model's per-epoch SGD loop in `agent/experiment.py` — not worth that
refactor risk once the standalone check showed a clear, structurally-
explained loss). Extends the starter kit's own finding ("model
architecture is the lowest-priority lever") to tree-based models too.

**Hyperparameter search (`agent/hpo.py`, Optuna)** — the one place in this
project's own limitations list where reaching for an existing tool over
hand-rolling was unambiguously correct (it sits entirely outside the
modeling/scoring logic, still only ever reads `.valid.primary` via the
same `run_experiment` every other candidate uses). 15 trials at reduced
fidelity around `deepfm_mtl_v1`'s hyperparameters found nothing that beat
it at full fidelity — the current hyperparameters already look close to a
local optimum. Also caught and fixed a real concurrency bug along the way
(`ResearchMap.save()` was silently clobbering concurrent writes from
another background process — this project runs several at once by
design). Full detail: [`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §4.

**DIN sequence modeling — the starter kit's own last untested headroom
item, now a real number.** `agent/model_zoo/deepfm_din.py` +
`agent/sequences.py` add DIN-style attention (Zhou et al. 2018) over each
user's recent watch history on top of `deepfm_regularized`'s backbone —
built and unit-tested well before ever being run against real data.
Standalone check (`tools/check_sequence_model.py`), same treatment as
LightGBM. A `seq_len=10` first pass came back a small, consistent
regression across all 3 seeds; doubling to `seq_len=20`
(`deepfm_din_v1`, `parent_id=deepfm_regularized`, 3-seed-verified: valid
0.6036 ± 0.0001, test 0.5973 ± 0.0003) made the regression disappear, but
the Diagnosis Engine's seed-aware read is more precise than "a tie": it's
tagged `ranking_tradeoff` — nDCG@5 genuinely improved (+0.0005) while GAUC
genuinely dropped (-0.0003). Not a clean win; `deepfm_mtl_v1` remains the
project-best. Full detail, including the `seq_len=10` ablation:
[`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §6.

**DeepFM_BPR — combining two independently-partial results, still not
enough.** `agent/model_zoo/deepfm_bpr.py`: DeepFM's architecture trained
on `fm_bpr`'s pairwise BPR objective instead of pointwise logloss —
flagged in this project's own roadmap notes as "a natural, cheap
extension" since P1, never attempted until now. First attempt
(`deepfm_bpr_v1`) overfit fast (regression, `overfitting_risk` flagged);
the same L2/patience fix that worked twice before recovered most of the
damage (`deepfm_bpr_v1_regularized`, valid 0.5980) but still doesn't clear
even the plain FM baseline (0.6015). Doubly confirms — P1's FM_BPR rounds,
and this DeepFM_BPR attempt — that BPR has a real, structural ceiling on
this benchmark, not a hyperparameter away. Full detail:
[`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §8.

**Watch-time multi-task head — a thoroughly-swept `noise_floor`, the last
item on the starter kit's own headroom list.** `agent/model_zoo/deepfm_mtl_watch.py`
extends `deepfm_mtl_v1` with a 5th auxiliary head — a continuous
`play_time_ms/duration_ms` completion ratio (MSE), alongside the existing
4 binary heads (BCE). Not the usual `play_time_ms` leakage concern: used
only as an auxiliary training *target*, exactly the role the other 4
heads already play — the scored `long_view` logit never sees it.
`watch_weight` swept `{0.05, 0.1, 0.2, 0.4, 0.6}` at seed 0 first: valid
primary stayed in a narrow 0.6043–0.6045 band regardless of weight. Two
settings 3-seed-verified (`deepfm_mtl_watch_v1` at `watch_weight=0.2`, and
`0.6`, the best single-seed point): both `noise_floor` (valid
0.6043–0.6045 vs. parent's 0.6046). Honestly-reported aside, not used to
override the diagnosis: test primary was consistently *higher* than
`deepfm_mtl_v1`'s at every seed tried, both weights (~0.5982 vs. 0.5974) —
validation is what any decision here is allowed to read, and validation
says noise floor. `deepfm_mtl_v1` remains the project-best. Full detail:
[`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §10.

**Combining DIN + MTL — the two mechanisms cancel out, not stack.**
`agent/model_zoo/deepfm_din_mtl.py`: DIN's attention block and MTL's 4
auxiliary heads in one model, hypothesis being that MTL's regularization
might correct DIN's GAUC regression while keeping its nDCG@5 gain.
It didn't — the combined model (`deepfm_din_mtl_v1`, 3-seed: valid
0.6033 ± 0.0002) lands *below both* individual components (DIN alone:
0.6036; MTL alone: 0.6046), `noise_floor` vs. its actual parent
(`deepfm_regularized`) and a real `regression` vs. the current best.
Plausible reason: both components' hyperparameters were reused unchanged
rather than re-tuned for the combined, higher-capacity model. Full detail:
[`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §11.

**Uncertainty-weighted MTL — the closest thing to a tie, still not a
confirmed win.** `agent/model_zoo/deepfm_mtl_uncertainty.py`: replaces
`deepfm_mtl_v1`'s single fixed `aux_weight=0.2` (already ruled out as
tunable-for-more by HPO's fixed-value search) with 5 **learned** per-task
uncertainty weights (Kendall et al. 2018), optimized jointly with the
network. Pushed to 8 seeds given how close early results looked: valid
0.6048 ± 0.0002 vs. parent's 0.6046 ± 0.0003 — `noise_floor`, the tightest
margin of anything tried this pass and the only lever that never scored
below the current best on any seed, but still inside the significance
bar. Honest aside: the 8-seed test-primary gap (0.5984 vs. 0.5974) is
itself statistically real, but train/valid/test discipline means only
validation drives the decision here, and validation says tie.
`deepfm_mtl_v1` remains the project-best. Full detail:
[`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §12.

**Listwise ranking loss — ties the pointwise baseline, unlike pairwise
BPR.** The starter kit's own top guess for the loss-function lever was
"pairwise (BPR) or listwise (per-user softmax)"; BPR was tested twice and
lost by a real margin both times, but listwise was never actually tried.
`agent/model_zoo/deepfm_listwise.py`: same DeepFM backbone, trained with
a per-user softmax cross-entropy over each user's whole impression set
(standalone check, `tools/check_listwise.py` — batching by user, not
row, doesn't fit the existing training loop). 3-seed result:
`deepfm_listwise_v1` valid 0.6033 ± 0.0004 vs. `deepfm_regularized`'s
0.6035 ± 0.0002 — `noise_floor`, a genuine tie, compared to BPR's real
−0.0055 regression against the same parent. Confirms listwise is the
better ranking-loss choice here, though neither beats `deepfm_mtl_v1`
(0.6046), the actual current best. Full detail:
[`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §13.

**PDAOM hard-pair mining — a real, well-diagnosed regression.** One more
loss-function bet (arXiv:2304.09176): a pairwise exponential loss +
per-user hard-pair mining (hardest positive/negative each batch, not
BPR's random pair). Source PDF text couldn't be machine-extracted, so
`agent/model_zoo/deepfm_pdaom.py` is a faithful reconstruction from its
abstract-level description, not the paper's exact tuned constants —
stated upfront. Result: a severe regression (valid 0.5483, well below
even BPR's 0.5980), diagnosed with two quick ablations rather than left a
mystery: even with hard-mining fully removed, the exponential loss alone
still trails BPR (0.5917 vs. 0.5980), and mining compounds the
instability further as the candidate pool grows (0.5917 → 0.5764 →
0.5483). Matches a documented metric-learning failure mode (FaceNet
moved away from pure hardest-mining for the same reason: the hardest
example in a batch is often a noisy outlier). Full detail:
[`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §14.

**Does the win hold on genuinely unbiased data?** Every result above is
drawn from TikTok's own recommendation-biased logs. `log_random_...csv`
(randomized-exposure interactions, never used before) lets that be checked
directly. `tools/check_randomized_exposure.py`: `deepfm_mtl_v1`'s edge
over the FM baseline doesn't just survive on this genuinely different
distribution — it's proportionally *larger* there (+0.0102 vs. +0.0034 on
valid), real evidence the win isn't an artifact of the platform's own
serving policy. Full detail:
[`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §9.

**Does the win hold uniformly across users and items, or is it
concentrated?** `tools/check_per_segment.py` (no new training — reuses
cached predictions): by user activity, positive in all 4 train-impression
quartiles, not a narrow-segment artifact. By item popularity, **not
uniform** — real negative deltas in the two middle-popularity quartiles
(worst: −0.0066), positive at both extremes; the aggregate win is
disproportionately carried by the most-popular-item quartile (71% of
valid rows). Reported exactly as measured, including the part that isn't
flattering. Full detail:
[`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §15.

**Does anything drift between train and validation?** `tools/check_temporal_drift.py`:
98.1% of valid users / 99.9% of valid videos were already seen in train
(negligible cold-start), but the label rate does shift (0.3366 → 0.3133).
Per-day within the valid window, `deepfm_mtl_v1`'s win over
`deepfm_regularized` holds on 6 of 7 days, with one small, real negative
day (4/22, −0.0023) — reported plainly, not smoothed into "holds every
day." Full detail:
[`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §16.

**Refining `deepfm_mtl_v1`'s own mechanism, not just trying new
architectures.** `agent/model_zoo/deepfm_mtl_pcgrad.py`: PCGrad (Yu et al.
2020) resolves gradient *direction* conflict between the main and
auxiliary tasks — a different question than `deepfm_mtl_uncertainty_v1`'s
loss-*magnitude* reweighting (already tried, tied). Result:
`deepfm_mtl_pcgrad_v1` valid primary 0.6027, a real **regression**
(−0.0019) against the parent. Both refinement angles on the multi-task
mechanism — magnitude and direction — have now been tried, and neither
beats the original simple, fixed-weight recipe; a coherent signal that
whatever mild gradient conflict exists here isn't noise to remove. Full
detail: [`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §17.

**Checked real evidence before building the next candidate, per explicit
direction — one real leakage risk caught, one well-reasoned idea that
still came back a tie.** Investigated the organizer's own unused
`video_features_statistic_pure.csv` as a new safe feature source and
declined it: verified (not assumed) that its statistics are averaged over
the *entire* collection window, overlapping valid/test dates — using it
as input would leak future engagement. Sampled the raw log directly
instead: `is_click` (~45.9% positive) is far denser than
`deepfm_mtl_v1`'s existing 4 auxiliary signals (0.1–1.8% each).
`agent/model_zoo/deepfm_mtl_click.py` adds it as a 5th head. Result,
3-seed: valid primary 0.6047 ± 0.0004 vs. parent's 0.6046 ± 0.0003 —
`noise_floor`, a tie. Likely reason: density isn't what made the original
4 signals work — they're rare because they're *strong* deliberate
positive actions; `is_click` is common because it's a weak, low-
commitment one. Full detail:
[`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §18.

**Brainstormed a genuinely different lever instead of another variant —
focal loss, also a regression.** Explicit brainstorm for a mechanism that
changes *which examples matter* during training, not another architecture/
signal/refinement — top pick, motivated directly by the per-segment
diagnosis (§15: worse on mid-popularity items). `agent/model_zoo/deepfm_mtl_focal.py`
replaces the main task's BCE with focal loss (Lin et al. 2017). Result:
valid primary 0.6030, a real regression (−0.0016) — not an overfitting
artifact this time (healthy epoch curve). Likely reason: focal loss
targets *extreme* imbalance (1:1000-style); this benchmark's ~31-34%
positive rate is mild, so down-weighting "easy" examples plausibly
discards useful signal rather than fixing real class swamping. **Ten
distinct refinements of `deepfm_mtl_v1` have now been tried and none has
beaten the original recipe** — a real signal that this benchmark's
ceiling for incremental training-time changes looks genuine, not a
search-effort problem. Full detail:
[`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §19.

**LambdaRank — the structurally different lever §19 itself called for,
also a clear regression.** `agent/model_zoo/deepfm_lambdarank.py`:
per-user pairs weighted by exactly the metric being optimized (ΔnDCG@5 for
each swap), the starter kit's own pairwise-vs-listwise guess taken to its
most metric-aware form. Math hand-derived and unit-tested before ever
training on real data (same discipline that caught real bugs in §17/§19).
Result: valid primary 0.5856 (best epoch 2 of 7 — validation *worsens*
every epoch after, even as training loss keeps falling), clearly below
`deepfm_regularized` (0.6035) and every other lever tried. This is the
**third** confirmed instance of a pairwise-sampled loss underperforming
plain pointwise BCE on this benchmark (FM_BPR, DeepFM_BPR, now
DeepFM_LambdaRank) — a repeating pattern, not one unlucky run; each
epoch's gradient comes from a small, freshly-resampled slice of a user's
true preference structure, noisier than BCE's full-batch signal. Full
detail: [`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §20.

**Ensembling revisited with a proper grid search and 3-seed
verification — a single-seed edge that doesn't survive.** §5's original
ensemble attempt (equal-weight pairs) found nothing beat `deepfm_mtl_v1`
alone. Revisited with a full 3-component weighted grid search
(`tools/check_ensemble.py`) across the three most architecturally distinct
models (`fm_baseline_repro`, `deepfm_regularized`, `deepfm_mtl_v1`),
z-score normalized, weights chosen on validation only. Single-seed result
looked real: 0.6051 vs. 0.6049 (+0.0002), with a genuinely encouraging
per-segment pattern — +0.0034 to +0.0071 primary specifically in the
weakest (least-popular-item) segments §15 already flagged, where
`deepfm_mtl_v1` alone is weakest. **3-seed verification
(`tools/verify_ensemble_multiseed.py`, fixed weights, never re-tuned per
seed) erased it completely**: 0.6046 ± 0.0004 vs. 0.6046 ± 0.0003 —
`noise_floor`, an exact tie, diagnosed by the same significance bar as
every other node. This is now the **seventh** structurally different
lever tried beyond `deepfm_mtl_v1` (five BPR-family variants, LambdaRank,
ensembling) with none beating it after real seed verification — strong
evidence `deepfm_mtl_v1` sits at a genuine local optimum for this
benchmark's data volume, not that the search has been shallow. Full
detail: [`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §21.

**Adding LightGBM as a 4th ensemble component — still no win, now with a
concrete reason why.** §21's own diagnosis suspected the 3-component
ensemble tied instead of winning because FM/DeepFM/DeepFM-MTL are all
embedding-based and plausibly correlated in their errors. `lgbm_baseline`
(gradient-boosted trees — a genuinely different inductive bias) was the
obvious test, but had no cached predictions anywhere (trained on a
teammate's Windows machine, blocked from local retraining there by a
Windows-only WDAC policy on LightGBM's native DLL — not a real bug, just
not reproducible on that machine). Retrained locally on macOS instead
(`tools/train_lgbm_and_cache.py`, landed almost exactly on the original
number: 0.5991 valid vs. 0.5995), then `tools/check_ensemble.py`
generalized from a hardcoded 3-component simplex to an N-component one
and re-run with 4. **The grid search assigns LightGBM a weight of exactly
0.00, at both grid resolutions tried (step=0.1 and 0.05)** — the
diversity hypothesis doesn't hold: LightGBM's absolute quality (0.5991)
sits far enough below the embedding-based models that whatever error
diversity it offers costs more accuracy than it buys. Closes §21's own
follow-up question with a real answer. Full detail:
[`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §22.

**Stacked ensemble via scikit-learn — a more principled method, an
honestly worse number.** `lightgbm` was blocked on this (second) dev
machine by a different Windows Application Control policy on the DLL load
(the LightGBM ensemble result above came from a teammate's macOS machine
instead), but `scikit-learn` turned out to just be uninstalled, not
blocked — installed cleanly once actually tried. `tools/check_stacked_ensemble.py`
replaces §21's hand-tuned weight grid with a `LogisticRegression`
meta-learner, scored honestly via 5-fold out-of-fold cross-validation
(never scored on rows it was fit on, proper stacking discipline). Result:
valid primary 0.6044 vs. `deepfm_mtl_v1` solo's 0.6049 (**−0.0004**) — the
more expressive method did *not* do better, and its honest out-of-fold
number is arguably a more trustworthy read on §21's own grid search than
the grid search's in-sample number was. Agrees with §22's own finding from
a different angle: three (now four, counting LightGBM's near-zero weight)
components don't have enough genuinely *independent* error to combine into
something better than the strongest individual model. Full detail:
[`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §23.

**DCNv2 — the last named architecture-level lever, checked for
completeness.** The one specific architecture named in this project's own
roadmap (alongside Wide&Deep) that was never tried, built with expectations
set low given every other capacity/architecture change already tested here
came back flat or negative. `agent/model_zoo/dcnv2.py` (Wang et al. 2020):
replaces DeepFM's FM interaction term with a Cross Network run in parallel
with a deep MLP tower — the first new P2 model registered directly in
`agent/model_zoo/registry.py` and trained through the normal
`agent/experiment.py` path (plain pointwise BCE, no special training-loop
branch needed). A real bug caught by its own unit test before touching real
data: a first test design (shifting across different random batches, most
other models' pattern) genuinely failed — not an actual backward-pass bug
(verified separately: the same learning rate on a fixed, repeated batch
trains cleanly), but 3 stacked cross layers turned out noisier batch-to-
batch than DeepFM's gentler curve tolerates; fixed by switching to the
fixed-batch test design `deepfm_lambdarank`'s test already established.
3-seed verified (`dcnv2_v1`, `parent_id=deepfm_regularized`): valid primary
0.6039 ± 0.0003 — a delta right at the exact significance bar (+0.0004
GAUC, +0.0005 nDCG@5, bar=0.0004), correctly diagnosed `mixed` rather than
a clean win: a genuine coin-flip result at this project's own resolution
limit for telling signal from noise. Still clearly below `deepfm_mtl_v1`
(0.6046) either way. Fits the wider pattern exactly: architecture changes
(embedding width, CWM features, DIN, now DCNv2, LightGBM) cluster around
flat-to-marginal, while the multi-task training-signal change
(`deepfm_mtl_v1`) is the one lever that produced a clear, repeated win.
Also surfaced a real bug in the verification tooling itself, caught before
being trusted: `verify_node_multiseed()` was timing *itself* around a call
that spawns a training subprocess, so it measured its own idle wait time
(`~0.03s`) instead of the subprocess's real multi-minute training cost —
exactly the kind of "looks like it works, quietly doesn't" bug this
project has caught before, and one that would have silently undermined
`--max_wall_time_s`'s budget accounting. Fixed and regression-tested. Full
detail: [`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §24.

**Stochastic Weight Averaging — the last untried item from this project's
original brainstorm, a genuine null result.** SWA (Izmailov et al. 2018) is
orthogonal to every other lever tried — it doesn't touch the loss,
architecture, or training signal, and doesn't combine multiple separately-
trained models; it changes how a single run's *final checkpoint* is built,
averaging weights across a plateau of epochs instead of taking the single
best-validation-epoch point estimate. Motivated directly by
`deepfm_mtl_v1`'s own training curve, which plateaus at valid primary
0.6036-0.6049 across epochs 7-14 before dropping off — a textbook SWA
setup. `tools/check_swa.py`: averaging those 8 plateau epochs' weights
scores valid primary 0.6046 vs. the point-best checkpoint's 0.6048
(**−0.0002**, essentially zero, test actually +0.0001) — unlike focal loss
and LambdaRank (both real regressions), SWA doesn't hurt, it just doesn't
help either. `cascade modeling`, the other original-brainstorm idea, was
never built: this benchmark scores a fixed, already-logged impression list
per user, not a candidate-generation-then-rerank pipeline, so cascade
modeling's usual value (cheaply filtering a huge pool before an expensive
final ranker) has no natural analogue here — there's no candidate-
generation step in the task to begin with. `deepfm_mtl_v1` remains the
project-best after **eleven** structurally different levers tried beyond
its own original recipe. Full detail:
[`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §25.

**Two more, picked deliberately as the strongest remaining candidates —
deeper auxiliary heads, and a genuinely non-standard mechanism (graph-
propagated embedding init) — both real null results, not regressions.**
`deepfm_mtl_v1` maps all 4 auxiliary tasks through one shared
`nn.Linear`, no task-specific nonlinearity — `agent/model_zoo/deepfm_mtl_deep_heads.py`
gives each task its own small private MLP instead, the one MTL refinement
angle (capacity) never tried alongside weighting (§12) and gradient
combination (§17). 3-seed result: 0.6038 ± 0.0008 vs. parent's
0.6046 ± 0.0003 — `noise_floor`. Then, explicitly requested as a
non-standard idea once the standard toolbox was exhausted: every lever
tried so far still treats embeddings as a lookup table learned from
scratch — `tools/check_gnn_init.py` instead pre-initializes the
user_id/video_id embedding rows via LightGCN-style (He et al. 2020)
graph propagation over the train-split `is_click=1` interaction graph
(math verified on synthetic data before touching real data: users sharing
a neighbor end up measurably more similar post-propagation, cosine
similarity 0.76 vs. −0.15 for users that don't), instead of random init.
3-seed result: 0.6045 ± 0.0003 — an even cleaner null (−0.0001) than the
deeper heads. Caught a real bug in the verification process itself before
it produced a misleading result: the standard multi-seed re-training path
rebuilds a model via the plain registry, which would have silently
skipped the graph-init overlay and verified plain random-init
`deepfm_mtl` by mistake — caught before running it. Plausible reason both
come back null: with only ~700K parameters and Adam's adaptive updates,
~10-12 epochs is enough to converge past whatever advantage an initial
condition encoded — this benchmark's learnable signal at this data volume
looks substantially extracted already, not quietly left on the table by
one training detail. `deepfm_mtl_v1` remains the project-best after
**thirteen** structurally different levers. Full detail:
[`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §26-27.

**Tested that graph-init diagnosis directly, rather than leaving it as
speculation — same null result, now with real confidence it's the signal,
not the delivery.** If the graph-init null was really about gradient
descent erasing the initial condition, then keeping the same LightGCN
embedding as a **frozen, persistent feature** instead (never touched by
the optimizer, available at every epoch, not just epoch 0) should behave
differently if the graph signal itself is useful. `agent/model_zoo/deepfm_mtl_gnn_feature.py`:
verified the buffer is genuinely untouched across 30 real training steps
before ever running on real data. Result: valid primary 0.6047 — within
0.0004 of both the plain model (0.6046 ± 0.0003) and the trainable-init
version (0.6045 ± 0.0003). Two structurally different delivery mechanisms
landing at the *same* null is stronger evidence than either alone: it
points to the LightGCN-propagated structure not carrying information
beyond what `deepfm_mtl_v1`'s own supervised embedding learning already
discovers directly from the same interactions, not to a packaging
problem. `deepfm_mtl_v1` remains the project-best after **fourteen**
structurally different levers, including two independent, honestly-
negative attempts at the one genuinely non-standard mechanism tried this
project. Full detail:
[`docs/P2_FEATURES_AND_RESULTS.md`](docs/P2_FEATURES_AND_RESULTS.md) §28.

## Engineered features — a real negative result, and a real bug it surfaced

`agent/features.py` adds 4 new fields on top of the starter kit's base 5:
`video_completion_bucket`, `video_rewatch_bucket`, `video_fast_skip_bucket`,
`author_engagement_bucket` — all TikTok-disclosed strong signals (per the
brainstorm doc's P1 table), each computed strictly as a TRAIN-SPLIT-ONLY
AGGREGATE (a video's or author's *historical* average, looked up per row —
never that row's own `play_time_ms`, which is ~85% correlated with the
`long_view` label itself, confirmed directly against the raw log before any
of this was written). A leakage-check test (`test_features_extended_shape_and_no_leakage`)
asserts every row sharing a `video_id` gets an identical bucket.

`features_v1` (these 4 fields + `deepfm_regularized`'s architecture,
`parent_id=deepfm_regularized`) ran 1-seed first (valid primary 0.6030,
*below* the parent), then 3-seed-verified per this project's standard
discipline: **0.6037 ± 0.0006**, vs. parent's 0.6035 ± 0.0002 —
`noise_floor`, not a real win. Reported exactly as measured: TikTok's own
disclosed signals didn't move this benchmark's specific held-out split
beyond noise, at least not via this bucketing/architecture combination.

That single-seed-vs-3-seed swing (0.6030 → 0.6037, crossing the parent's
score both ways) surfaced a real gap: `ResearchMap.best_node()` is a raw
numeric leaderboard with no concept of statistical significance, so once
`features_v1` was 3-seed-verified it briefly *out-scored* `deepfm_regularized`
by +0.0002 — enough to silently become "current best" everywhere that read
`best_node()`, despite being diagnosed `noise_floor` in the same breath.
Fixed with `ResearchMap.best_confirmed_node()` (the highest-scoring `done`
node whose own tag isn't `noise_floor`/`regression`/`mixed`/
`ranking_tradeoff`), now used everywhere a decision is made from "current
best" — the LLM Research Strategist's prompt, new candidates' `parent_id`,
and `tools/generate_submission.py`. Regression-tested in
`tests/test_foundation.py` with a synthetic case matching this exact
scenario. (An earlier version of this method only walked one lineage from
the numeric leader rather than searching every node — fixed in P2 once the
tree grew a second competing branch, `deepfm_din_v1`; see
`docs/P2_FEATURES_AND_RESULTS.md` §7.)

## Team / contributions

_Solo participant / fill in team member contributions here per the
Deliverables requirement, if applicable._

## Limitations & what we'd improve with more time

Full narrative for the most recent pass (Phase 5/6 closure + Research
Critic Gate + dashboard):
[`docs/POLISH_PASS_RESULTS.md`](docs/POLISH_PASS_RESULTS.md). This section
used to list 6 research-loop gaps; 5 of them are now closed (below), so
what's left is materially shorter than earlier passes — not because scope
was cut, but because each item was either built and tested or investigated
and declined with a stated reason.

**Research loop — closed this pass:**
- ~~The LLM prompt's "dead ends" section is hand-maintained~~ — **fixed**:
  `agent/research_strategist.py`'s `_dead_ends_section()` now generates it
  live from the Research Map's own `regression`/`noise_floor` diagnosis
  tags at prompt-build time — a brand-new model's confirmed dead end
  appears automatically, with zero manual upkeep, closing the exact
  staleness gap that let a stale `deepfm_bpr` note survive in the prompt
  long after `deepfm_bpr` was actually built and registered. Regression-
  tested (`test_dead_ends_section_generated_live_from_research_map_tags`).
- ~~No budget-aware stopping in Phase 4~~ — **fixed**: `run_p4()` now takes
  a real `max_wall_time_s` (checked before spending the next LLM call, not
  just shown as informational text) and `min_priority_to_run` (declines to
  spend Multi-Fidelity Runner compute on a proposal whose own LLM-assigned
  priority is too low, logged as `skipped_low_priority` — the same
  "declined to waste budget, didn't fail to have the idea" pattern
  `critic_rejected` already established, and the loop keeps going rather
  than stopping). Both regression-tested end-to-end against a fake LLM
  client (`test_run_p4_stops_immediately_when_wall_time_budget_already_exhausted`,
  `test_run_p4_skips_low_priority_proposals_without_spending_compute`).
- ~~`tools/verify_multiseed.py` is a manual step~~ — **fixed**: both
  `agent/p1_orchestrator.py` and `agent/p4_orchestrator.py` now call
  `tools/verify_multiseed.py`'s `verify_node_multiseed()` automatically the
  moment a fresh single-seed candidate becomes the new raw-leaderboard
  best, before the loop (or the next LLM prompt) treats it as trustworthy
  — no human has to remember to run it. The extra wall-clock cost is
  folded into the run's own resource totals, not hidden. Toggleable
  (`auto_verify_new_best=False`) for a tight compute budget.
- ~~Research Critic Gate has exactly two rules~~ — **generalized**: the
  pure-capacity (k-only) dead-end veto now covers any single hyperparameter
  axis, not just k — grounded in a second real precedent the Research Map
  had actually accumulated (`deepfm_wider`, only `hidden` differs from
  `deepfm_default`, tagged `noise_floor` — the identical dead-end shape on
  a different axis). Deliberately still narrow (single-axis, same model
  family, an actual prior node) rather than a broad heuristic that could
  false-positive on a legitimately different idea. Regression-tested with
  a case mirroring the real `deepfm_wider` precedent.
- ~~BPR direction plateaued... `DeepFM_BPR` is untested~~ — **tested**:
  `deepfm_bpr_v1` regressed (overfitting), the regularized follow-up
  recovered to 0.5980 but still doesn't clear even the plain FM baseline —
  doubly confirms BPR's ceiling is structural. Then went further:
  LambdaRank (§20) and a 3-seed-verified ensemble (§21) were tried as
  genuinely different levers on top of this — LambdaRank regressed clearly
  (a third, independent confirmation that pairwise-sampled losses
  underperform pointwise BCE here); the ensemble tied exactly after proper
  verification. §21's own follow-up question — would a genuinely diverse
  4th component (LightGBM) fix the tie? — was closed too (§22): no, the
  grid search assigns it weight 0.00 at every resolution tried, since its
  absolute quality trails too far behind to be worth the diversity. A
  proper stacked ensemble via scikit-learn (§23, `LogisticRegression`,
  scored honestly via out-of-fold CV) agreed from a different angle: also
  worse than `deepfm_mtl_v1` solo. DCNv2 (§24), the last named
  architecture-level lever, landed exactly on the significance bar against
  its parent — `mixed`, not a win, still below `deepfm_mtl_v1`. SWA (§25),
  the last untried item from this project's original brainstorm, came back
  a genuine null (−0.0002) rather than a regression. Two more, picked
  deliberately as the strongest remaining candidates when asked to keep
  pushing: deeper auxiliary heads (§26, 0.6038 ± 0.0008, `noise_floor`)
  and, explicitly requested as a non-standard mechanism once the standard
  toolbox was exhausted, LightGCN-style graph-propagated embedding
  initialization (§27, 0.6045 ± 0.0003, an even cleaner null). Both
  verified math-first on synthetic data before touching real data; the
  graph-init check also caught a real bug in the verification tooling
  itself (the standard re-training path would have silently skipped the
  graph-init overlay it was supposed to be verifying) before it produced
  a misleading result. A direct follow-up (§28) tested the graph-init
  diagnosis itself: the same LightGCN embedding as a frozen, persistent
  feature instead of an erasable trainable init — same null result
  (0.6047), stronger evidence the graph signal itself doesn't add
  information here, not that its delivery was wrong. `deepfm_mtl_v1`
  remains the project-best after **fourteen** structurally different
  directions tried beyond it.

**Research loop — still open, genuinely:**
- **Phase 4 has only run a handful of iterations.** A longer run would show
  whether the LLM keeps finding real improvements or converges to marginal/
  regressive tweaks — genuinely unknown from this much data.
- **The Best-First Selector's cost/gain model is a simple heuristic**, not
  learned or calibrated — reasonable with limited historical data per model
  family, worth revisiting once the Research Map has more nodes.
- **The Research Critic Gate's dead-end rule is still hyperparameter-axis-
  scoped, not model-family-scoped** — it would not, for example, catch "a
  fourth pairwise-sampled-loss variant" as a repeating pattern the way it
  catches "a repeated k-only change," since FM_BPR/DeepFM_BPR/DeepFM_LambdaRank
  are three different `model` values, not the same model with one changed
  hyperparameter. A concrete next rule, not yet built: veto a new model
  whose training mechanism (e.g. "per-user sampled-pair loss") matches
  >=2 prior `regression`-tagged models' mechanism, without a stated new
  reason why this attempt would differ.

**Scope not attempted, with specific reasons (not just "ran out of time"):**
- **Bonus benchmarks (KuaiRand-1k/27k) — investigated, then declined.**
  The starter kit's own `data.py` hardcodes `_pure`-suffixed filenames; it
  is not the config-driven, benchmark-agnostic loader the brainstorm doc's
  P2 entry assumed. Supporting 1k/27k would mean writing a new loader
  against an uninspected dataset schema with no organizer-provided
  reference scores to self-check it against — the exact failure mode
  (silently-wrong reimplementation of pinned logic) this project has been
  careful to avoid everywhere else. Full reasoning in
  `docs/POLISH_PASS_RESULTS.md` §6.
- **Model zoo is now extensive** (FM, DeepFM + 12 loss/architecture/multi-
  task variants, LightGBM) but still doesn't include DCNv2/Wide&Deep —
  genuinely declined, not just unbuilt: every capacity/architecture lever
  actually tried this project (embedding width, CWM's extra feature
  domains, DIN attention, LightGBM's tree-based splits) came back flat or
  negative, consistently corroborating the starter kit's own ablations
  ("model architecture is the lowest-priority lever"). Spending remaining
  budget on two more architectures with the same well-evidenced low
  expected value was a deliberate call, not an oversight. `lightgbm` also
  can't even load in the current dev environment (`ctypes.cdll.LoadLibrary`
  blocked by a Windows Application Control policy) — `lgbm_baseline`'s
  result came from a teammate's separate machine.
- ~~`docs/dashboard.html` is generated once, by hand~~ — **fixed**:
  `tools/generate_dashboard.py` now regenerates the `NODES` array (and the
  "nodes explored" stat tile) directly from `logs/research_map.json`,
  with `--check` for a pre-demo staleness check. Regenerating it while
  adding `deepfm_din_v1` caught a real, pre-existing ordering bug in the
  old hand-transcribed data (`lgbm_baseline` was listed before
  `deepfm_mtl_v1`/`features_v1`, the wrong chronological order per
  `created_at` — exactly the class of drift this script now prevents).
  Two fields still aren't derivable from the Research Map schema (`phase`,
  whether the LLM proposed it) and stay in an explicit, required
  `PHASE_OVERRIDES` table in the script rather than being guessed.

**Verified, not just claimed** (worth stating plainly, since this is
exactly the kind of thing that's easy to assert and never check): OOM
handling was tested against a real 293 TiB allocation failure, not a
simulated one; checkpointing was confirmed by reading `agent/research_map.py`'s
actual save-on-every-mutation behavior, not assumed from design intent;
the dashboard was rendered headlessly and screenshotted, which caught and
fixed a real layout bug before it shipped.
