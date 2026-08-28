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

See `CLAUDE.md` for the full architecture, the exact judging-criteria → code
map, and — importantly — a **task-definition caveat**: the problem statement's
prose text and the Starter Kit's actual code disagree on the label/metrics;
this repo follows the Starter Kit (the pinned, code-level source of truth),
using label `long_view` and metrics `GAUC` / `nDCG@5`.

## Development tools & environment

- **Editor**: VS Code (via the Claude Code extension)
- **Language / runtime**: Python 3.14, Windows 11
- **Dev dependencies actually used**: `numpy` (all model/eval code), `pandas` (ad-hoc EDA only — never imported by `agent/`)
- **No LLM API calls yet** — neither Phase 0 nor P1 has an LLM in the loop by design (see roadmap); `run_summary.json`'s `llm_tokens_total` is structurally 0 until the Phase 4 Research Strategist is wired in.

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
```

Outputs: `experiments/<run_id>/iter_*/` (per-iteration hypothesis/config/results/logs;
run-scoped, so a later run never overwrites an earlier one's folders),
`logs/run_log.jsonl` + `logs/run_summary.json` (machine-readable),
`logs/analysis_report.md` (the human-readable deliverable), and
`submission_valid.csv` / `submission_test.csv` (format-validated against the
organizer's `submit.py --check` logic).

## Results (validation-best config, latest run)

Live numbers are always in `logs/analysis_report.md` / `logs/run_summary.json`
(regenerate with `python tools/generate_analysis.py`) — this table is a
snapshot from the run described in
[`docs/PHASE0_FEATURES_AND_IMPROVEMENTS.md`](docs/PHASE0_FEATURES_AND_IMPROVEMENTS.md#4-final-validated-phase-0-result),
which also explains the efficiency/robustness fixes made to reach it, and
why this number is seed-robustness-checked (3 seeds), not a single lucky draw.

Validation-best: `deepfm_wider` (FM + numpy MLP deep component, shared embeddings, hidden=[128,64]).

| Metric | Official FM baseline (test) | This run (test, submitted seed) | Δ | 3-seed mean Δ |
|---|---|---|---|---|
| GAUC | 0.6610 | 0.6643 | **+0.0033** | — |
| nDCG@5 | 0.5282 | 0.5306 | **+0.0024** | — |
| primary | 0.5946 | 0.5975 | **+0.0029** | **+0.0030** (~8.9σ above noise floor) |

Converged automatically (ε=0.002, N=3) after 4 iterations, 0 manual
interventions, 1,083.3s wall-clock (10 individual training runs — 3 seeds
each for both DeepFM configs, to confirm the win isn't a single-seed
fluke), 0 LLM tokens, 0 GPU-hours (CPU-only numpy). Config selection used
only the validation split throughout — see
`docs/PHASE0_FEATURES_AND_IMPROVEMENTS.md` §4 for the full per-seed
breakdown and significance calculation.

## P1 results (one round, both real outcomes reported)

Full details, including two bugs found and fixed during this pass's own
validation, in
[`docs/P1_FEATURES_AND_RESULTS.md`](docs/P1_FEATURES_AND_RESULTS.md).

The Best-First Selector ranked 2 candidates and ran both (best-first order,
not declaration order) through the Multi-Fidelity Runner. **Neither beat the
baseline** — reported here exactly as measured, because negative results
with a specific diagnosis are legitimate output, not something to omit:

| Candidate | Valid primary | vs. parent | Diagnosis |
|---|---|---|---|
| `fm_bpr_default` (pairwise BPR loss, k=16) | 0.5981 | −0.0034 | `mixed`, with an overfitting-risk flag (best epoch at ~33% of training length) — a specific, actionable lead for a next round, not a dead end |
| `fm_wider_k32` (FM, k=32) | 0.6009 | −0.0006 | `noise_floor` — independently reproduces the starter kit's own documented "capacity isn't the bottleneck" finding, on this codebase's own harness |

This is the "reflect → revise" loop working as intended: a real hypothesis
(the starter kit's own #1-ranked untested direction) was tested, it didn't
pay off, and the Diagnosis Engine produced a specific, actionable reason
instead of just a lower number.

## Team / contributions

_Solo participant / fill in team member contributions here per the
Deliverables requirement, if applicable._

## Limitations & what we'd improve with more time

- **No LLM reasoning yet.** Both Phase 0's and P1's candidate pools are
  fixed and human-written (by design — see `CLAUDE.md` roadmap Phase 4).
  The Innovation & Autonomy scores both depend heavily on the Research
  Strategist layer that isn't built yet.
- **P1 is one round, not a loop.** The candidate pool (`p1_candidate_pool()`)
  is a fixed 2-item list; a second `python run_p1.py` finds nothing new to
  run. A real iterative loop needs either more hand-authored "improve"
  templates reacting to each round's diagnosis, or Phase 4's LLM.
- **BPR direction not beaten yet.** `fm_bpr_default` underperformed the
  baseline with pointwise-FM hyperparameters reused unchanged; the
  Diagnosis Engine's overfitting flag suggests fewer epochs / different
  learning rate as the natural next candidate, not that BPR itself is a
  dead end. `DeepFM_BPR` (pairwise loss + the deep component) is also
  untested.
- **Model zoo**: FM, DeepFM, FM_BPR. DCNv2/Wide&Deep/LightGBM need
  torch/scikit-learn, not installed in the current dev environment.
- **Bonus benchmarks (KuaiRand-1k/27k) not attempted** — deliberately
  deprioritized until the required KuaiRand-Pure path is fully hardened.
- **Best-First Selector's cost/gain model is a simple heuristic**, not
  learned or calibrated — reasonable with almost no historical data per
  model family yet, worth revisiting once the Research Map has more nodes.
