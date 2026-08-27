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
repo's current state is **Phase 0**: the foundation layer that has to be
correct before any research reasoning gets layered on top of it —

- **Evaluator Wrapper** — imports the organizer's `evaluate.py` directly; never reimplements its pinned scoring conventions.
- **Convergence Detector** — implements the organizer's ε=0.002 / N=3 rule, read live from `baseline_scores.json`.
- **Model Zoo** — Factorization Machine (faithful port of the official baseline) + DeepFM (numpy-only MLP deep component over shared embeddings).
- **Structured Experiment Interface** — a config schema an orchestrator fills in instead of an LLM writing training code from scratch.
- **Orchestrator** — runs a fixed predefined experiment list end-to-end (no LLM reasoning yet — that's a later phase).
- **Failure Recovery** — every experiment runs in an isolated subprocess with a real timeout kill switch, retry, and a degraded fallback so one bad run can never crash or stall the whole loop.
- **Structured Run Log** — every iteration's hypothesis, config diff, metrics, and error/recovery events, both as per-experiment folders and one append-only JSONL.
- **Submission Validator** — wraps the organizer's `submit.py` format checks exactly before anything is called "final."

See `CLAUDE.md` for the full architecture, the exact judging-criteria → code
map, and — importantly — a **task-definition caveat**: the problem statement's
prose text and the Starter Kit's actual code disagree on the label/metrics;
this repo follows the Starter Kit (the pinned, code-level source of truth),
using label `long_view` and metrics `GAUC` / `nDCG@5`.

## Development tools & environment

- **Editor**: VS Code (via the Claude Code extension)
- **Language / runtime**: Python 3.14, Windows 11
- **Dev dependencies actually used**: `numpy` (all model/eval code), `pandas` (ad-hoc EDA only — never imported by `agent/`)
- **No LLM API calls yet** — Phase 0 has no LLM in the loop by design (see roadmap); `run_summary.json`'s `llm_tokens_total` is structurally 0 until the Phase 4 Research Strategist is wired in.

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

## Team / contributions

_Solo participant / fill in team member contributions here per the
Deliverables requirement, if applicable._

## Limitations & what we'd improve with more time

- **No LLM reasoning yet.** Phase 0's experiment list is fixed and human-written
  (by design — see `CLAUDE.md` roadmap Phase 4). The Innovation & Autonomy
  scores both depend heavily on the Research Strategist layer that isn't built yet.
- **No Research Map / Experiment Tree.** Experiments are a flat predefined
  list today; `ExperimentConfig.parent_id` is wired but unused for branching.
  AIDE reports ~4x more MLE-Bench medals with tree search vs. linear iteration
  — this is the highest-priority next build per the brainstorm doc's own analysis.
- **No multi-fidelity runner.** Every experiment currently trains at full
  scale; a 1%→10%→100% staged runner would meaningfully cut GPU-hours (the
  P1 item the doc calls the "biggest single lever" on the Feasibility score).
- **Model zoo is FM + DeepFM only.** DCNv2/Wide&Deep/LightGBM need
  torch/scikit-learn, not installed in the current dev environment.
- **Bonus benchmarks (KuaiRand-1k/27k) not attempted** — deliberately
  deprioritized until the required KuaiRand-Pure path is fully hardened.
- **Loss function still pointwise logloss** despite the starter kit's own
  README ranking a pairwise/listwise loss change as the most promising
  unexplored direction — a natural first Phase-3/4 experiment.
