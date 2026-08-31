# Written Project Description

_Ready to paste into the Devpost submission form (PS Deliverables item 1).
Covers all 5 required points: problem fit, dev tools, APIs, libraries/frameworks,
datasets/assets._

## How this solution addresses the problem statement

The challenge asks for an autonomous ML research agent that runs the MLE
iteration loop (read → inspect → engineer features → train/tune → evaluate
→ reflect/revise) against KuaiRand-Pure, reproduces the official baseline,
then autonomously drives the validation score above it — with a full,
auditable log of what it tried and why, bounded compute, and minimal human
intervention.

**In one line: the agent reproduces the official baseline, diagnoses its
weaknesses, generates and ranks candidate experiments, rejects the weak
ones on cost/gain grounds before spending real compute on them, tests the
promising one cheaply before scaling it up, records the insight, and
repeats until the organizer's own convergence rule fires — and that exact
sequence is what beat the baseline, not a single lucky configuration.**
Every step in it is a real, logged transition (`logs/run_log.jsonl`,
`docs/dashboard.html`), not a narrative summary written after the fact.

This project builds that loop as five layered, increasingly autonomous
systems, each one reusing the last rather than replacing it:

- **Foundation (Phase 0).** An Evaluator Wrapper that imports the
  organizer's own `evaluate.py` directly (never reimplements pinned
  scoring logic — the single most common way teams silently drift from
  the real benchmark), a Convergence Detector that reads the organizer's
  ε=0.002/N=3 rule live from `baseline_scores.json`, a faithful numpy
  port of the official FM baseline plus a DeepFM extension, Failure
  Recovery (real subprocess isolation with a timeout kill switch, not
  just try/except), and a Structured Run Log. This alone reproduces the
  baseline and beats it automatically, with 0 manual interventions.
- **Differentiators (P1).** A persistent, tree-structured Research Map
  that survives across runs (unlike a flat per-run log), a Metric-Aware
  Diagnosis Engine that classifies *why* a result looks the way it does
  (clear improvement / regression / ranking trade-off / noise floor,
  seed-aware — not a flat threshold), a Multi-Fidelity Runner that kills
  clearly-broken candidates at 1% scale instead of always training at
  full scale, and a Best-First Node Selector that decides what to try
  next by `gain × confidence × novelty ÷ cost`, with a logged reasoning
  string for every candidate.
- **An LLM Research Strategist (Phase 4).** Google Gemini (free tier —
  genuinely $0, not an estimate) reads the live Research Map and proposes
  the next experiment on its own; a Validation Gate rejects anything
  structurally invalid before it ever runs. Across two real rounds this
  found the project's second-best result on its own reasoning
  (`deepfm_regularized`, raised L2 after noticing an `overfitting_risk`
  flag human-authored logic had missed) and correctly diagnosed its own
  misses (regressions caught and reported, never hidden).
- **Extended exploration (P2).** The one mechanism that actually worked —
  multi-task learning, four auxiliary engagement signals (`is_like`,
  `is_follow`, `is_comment`, `is_forward`) sharing DeepFM's embedding
  table — landed a real, 3-seed-verified improvement on its first
  attempt. We don't read that as generic multi-task learning: TikTok has
  publicly described its own recommender as optimizing a *blend* of
  engagement signals, not click/long-view alone (Appendix A.3's own
  framing). Only `long_view` is scored here, but training the shared
  embedding table against the other signals jointly is this project's
  approximation of that same disclosed multi-objective value function —
  the auxiliary heads shape what the embeddings learn even though they're
  never themselves evaluated.

  Everything since has been an honest, thorough search for
  more: 18+ further levers across loss functions (BPR, listwise,
  LambdaRank, focal, PDAOM), architecture (DCNv2, LightGBM), ensembling
  (grid search, stacking, a 4th tree-based component), checkpoint
  averaging (SWA), and two genuinely non-standard graph-based embedding
  mechanisms — each isolate-one-variable, 3-seed-verified where the
  result was close, and reported exactly as measured, negative or not.
  That negative-result density is itself the finding: this benchmark's
  learnable signal at this data volume looks substantially extracted by
  the multi-task objective, not that the search was shallow.
- **Robustness and honesty, throughout.** Every experiment's hypothesis,
  code diff, metrics, and error/recovery events are logged
  (`logs/run_log.jsonl` + `docs/dashboard.html`). Manual interventions are
  counted explicitly, not inferred — including the one genuine case this
  pass (extending a timed-out LLM-proposed candidate's budget by hand,
  logged plainly rather than quietly excluded). Failure Recovery is
  tested against a **real** OOM (a genuine 293 TiB allocation failure,
  not simulated) and a genuinely broken config, not just asserted to
  work.

**Result:** `deepfm_mtl_v1`, valid primary 0.6046 ± 0.0003 (3-seed
verified), **+0.0028 primary-metric delta over the official baseline on
the hidden test set** (0.5974 vs. 0.5946), independently confirmed to
generalize on TikTok's own unbiased randomized-exposure log (the edge
*grows* there, not shrinks). Full breakdown:
[`docs/RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md).

## Development tools used

VS Code (via the Claude Code extension), used as the primary development
environment throughout — an AI coding assistant (Claude, Anthropic) drove
implementation, working from a persistent project-context file
(`CLAUDE.md`) across sessions. Windows 11, Python 3.14.

## APIs used

Google Gemini (free tier) — the only external API this project calls,
used exclusively by the Phase 4 LLM Research Strategist
(`agent/llm_client.py`, `agent/research_strategist.py`) to propose new
experiments from the live Research Map. Chosen deliberately over a paid
API: the problem statement doesn't mandate a provider, and a genuinely
$0 real token cost is a cleaner, more defensible Feasibility &
Practicality number than an estimated dollar figure. Real usage this
project: 16,608 tokens across 2 rounds — real, not structurally zero.

## Libraries and frameworks used

- **NumPy** — the entire Phase 0/P1 foundation (evaluator, convergence
  detector, FM, DeepFM, FM_BPR, orchestrator, recovery, Research Map,
  diagnosis, multi-fidelity runner, selector) is numpy-only by design,
  matching the Starter Kit's own zero-dependency philosophy. Every
  backward pass in that layer is hand-derived, not autograd — a
  deliberate choice, not a limitation.
- **PyTorch** (CPU wheel) — added deliberately and scoped, only where
  autograd is a genuine win over hand-rolled backprop: the multi-task
  DeepFM family (5-headed shared-bottom networks, gradient-surgery
  variants), DIN-style attention, and the two graph-embedding mechanisms.
- **LightGBM** — one standalone comparison point (gradient-boosted trees
  vs. this task's embedding-based models); genuinely blocked from
  installing on the primary dev machine by a Windows Application Control
  policy, worked around by a teammate retraining it on macOS.
- **scikit-learn** — one standalone check (`LogisticRegression` as an
  ensemble meta-learner, scored via `cross_val_predict`).
- **Optuna** — hyperparameter search around the project-best's own
  hyperparameters, sitting entirely outside the modeling/scoring logic.
- **SciPy** (sparse) — the LightGCN-style graph propagation math for the
  two non-standard embedding-mechanism checks.
- **pandas** — ad-hoc EDA only, never imported by the actual agent code.
- **google-genai / python-dotenv** — the Gemini client and `.env` loading
  for Phase 4.

Every dependency is explained inline in
[`requirements.txt`](../requirements.txt) with the specific reason it's
there — an unused heavy dependency is treated as a Feasibility &
Practicality red flag in this project, not a convenience.

## Datasets and assets used

**KuaiRand-Pure** (Kuaishou / KuaiRand, via Zenodo, the challenge's
required benchmark) — 1.4M interactions, 27K users × 7.6K items, with
12 logged feedback signals and a randomized-exposure intervention subset
used for an unbiased generalization check. No other dataset is used for
training, per the challenge's one hard rule ("no external training
data") — verified directly before using any new field: the organizer's
own `video_features_statistic_pure.csv` (a candidate feature source) was
checked against the KuaiRand paper and found to average over the full
collection period including valid/test dates, so it was declined as a
leakage risk rather than assumed safe.
