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

## Results (current project-best, 3-seed verified)

Live numbers are always in `logs/analysis_report.md` / `logs/research_map.json`
— this table is a snapshot of `deepfm_regularized`, found by Phase 4's LLM
Research Strategist and 3-seed-verified with `tools/verify_multiseed.py`
(full writeup: [`docs/PHASE4_RESULTS.md`](docs/PHASE4_RESULTS.md)). It
improves on Phase 0's `deepfm_wider` (full writeup:
[`docs/PHASE0_FEATURES_AND_IMPROVEMENTS.md`](docs/PHASE0_FEATURES_AND_IMPROVEMENTS.md#4-final-validated-phase-0-result)),
which is itself still the reference for the earlier, larger jump over the
official baseline.

Validation-best: `deepfm_regularized` — `deepfm_wider`'s architecture (hidden=[128,64]) with L2 raised 1e-5→1e-4, proposed by the LLM Research Strategist because both prior DeepFM nodes were flagged `overfitting_risk`.

| Metric | Official FM baseline (test) | `deepfm_wider` (Phase 0, 3-seed) | `deepfm_regularized` (Phase 4, 3-seed) |
|---|---|---|---|
| GAUC | 0.6610 | 0.6643 | **0.6646** (Δ **+0.0036**) |
| nDCG@5 | 0.5282 | 0.5306 | **0.5308** (Δ **+0.0026**) |
| primary | 0.5946 | 0.5976 | **0.5977** (Δ **+0.0031**) |

`submission_valid.csv` / `submission_test.csv` reflect `deepfm_regularized`
(regenerated via `python tools/generate_submission.py`, which resolves the
Research Map's `best_confirmed_node()` — see "Engineered features" below for
why that's not simply "the highest score in the map"). Both models converged
on the validation split only — see
`docs/PHASE4_RESULTS.md` §4 for the honest nuance that this improvement is
clear and statistically real on **validation** (0.6035 vs. 0.6028, the
split every decision here is allowed to read) while the **test**-split gap
over `deepfm_wider` is much smaller, exactly what train/valid/test
discipline predicts for a model that wasn't selected by peeking at test.

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

This is now the project's best result — see the Results table above.

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
Fixed with `ResearchMap.best_confirmed_node()` (walks the parent chain past
any run of `noise_floor`/`regression`/`mixed`/`ranking_tradeoff` nodes to
the nearest actual `clear_improvement`/`baseline_beat`), now used everywhere
a decision is made from "current best" — the LLM Research Strategist's
prompt, new candidates' `parent_id`, and `tools/generate_submission.py`.
Regression-tested in `tests/test_foundation.py` with a synthetic case
matching this exact scenario.

## Team / contributions

_Solo participant / fill in team member contributions here per the
Deliverables requirement, if applicable._

## Limitations & what we'd improve with more time

Full narrative for the most recent pass (Phase 5/6 closure + Research
Critic Gate + dashboard):
[`docs/POLISH_PASS_RESULTS.md`](docs/POLISH_PASS_RESULTS.md).

**Research loop:**
- **Phase 4 has only run 2 iterations.** A longer run would show whether
  the LLM keeps finding real improvements or converges to marginal/
  regressive tweaks — genuinely unknown from this much data.
- **The LLM prompt's "dead ends" section is hand-maintained**, not derived
  automatically from the Research Map's own diagnosis tags — see
  `docs/PHASE4_RESULTS.md` §6 for the concrete next step (generate it from
  `explored_summary()` instead of a hardcoded string).
- **No budget-aware stopping in Phase 4.** `--max_iterations` is a fixed
  count; the `budget` dict shown to the LLM in its prompt is informational
  only — nothing makes it change behavior as budget depletes.
- **`tools/verify_multiseed.py` is a manual step**, not auto-triggered when
  a new best node appears. (`tools/generate_submission.py`, added this pass,
  closes the *next* step in that chain — regenerating the submission CSVs
  from the Research Map's confirmed best is now a real, re-runnable script,
  not a manual/undocumented one — but promoting a fresh single-seed result
  to 3-seed-verified is still something a human has to remember to run.)
- **Research Critic Gate has exactly two rules** (duplicate; confirmed
  pure-capacity dead end) — both grounded in real project data, but a
  general "veto any candidate matching any prior `regression`-tagged
  pattern" rule would generalize this well beyond the one case it
  currently catches.
- **BPR direction plateaued after 3 diagnosis-driven rounds** (see the P1
  results above) — training dynamics fixed, but a ~0.002 quality gap to
  baseline looks structural, not a hyperparameter away. `DeepFM_BPR`
  (pairwise loss + the deep component) is untested and the natural next angle.

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
- **Model zoo**: FM, DeepFM, FM_BPR. DCNv2/Wide&Deep/LightGBM need
  torch/scikit-learn, not installed in the current dev environment.
- **Best-First Selector's cost/gain model is a simple heuristic**, not
  learned or calibrated — reasonable with almost no historical data per
  model family yet, worth revisiting once the Research Map has more nodes.
- **`docs/dashboard.html` is generated once, by hand**, from a snapshot of
  the Research Map at the time it was written — not regenerated
  automatically from `logs/research_map.json` on every run. A real next
  step: a small script that re-emits the `NODES` array from the live JSON
  instead of the current hand-transcribed data block.

**Verified, not just claimed** (worth stating plainly, since this is
exactly the kind of thing that's easy to assert and never check): OOM
handling was tested against a real 293 TiB allocation failure, not a
simulated one; checkpointing was confirmed by reading `agent/research_map.py`'s
actual save-on-every-mutation behavior, not assumed from design intent;
the dashboard was rendered headlessly and screenshotted, which caught and
fixed a real layout bug before it shipped.
