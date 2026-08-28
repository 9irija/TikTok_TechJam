# CLAUDE.md -- Autonomous ML Research Agent for KuaiRand-Pure

Context file for whichever Claude session picks this project up next. Read
this before touching code. The full problem statement + team brainstorm live
in [`docs/TikTok TechJam Hackathon.md`](docs/TikTok%20TechJam%20Hackathon.md)
(also as PDF, same content, with the two diagrams under `docs/images/`) --
this file is the *operational* summary: what's true, what's built, what's
next, and the traps that will quietly cost points if missed.

## What this project is

TikTok TechJam 2026, Challenge 2: build an **autonomous ML research agent**
that runs the MLE loop (read → inspect → engineer features → train/tune →
evaluate → reflect/revise) on its own against **KuaiRand-Pure**, a
short-video recommendation dataset, and drives the **validation** score
above a fixed official baseline with minimal human intervention, bounded
compute, and a fully auditable log of what it tried and why.

Required repo connection: **https://github.com/9irija/TikTok_TechJam.git**
(empty at time of writing -- see "Git" section below for push status).

## ⚠️ The one thing that will bite you if you skip it: which task definition is real

The problem-statement *prose* (top of `docs/TikTok TechJam Hackathon.md`)
says: label = `click`, metrics = **NDCG@10 / Recall@50**.

The **Starter Kit** — code, `evaluate.py`, `baseline_scores.json`, README —
says: label = `long_view`, metrics = **GAUC / nDCG@5**, primary =
`mean(GAUC, nDCG@5)`.

These directly conflict. **The Starter Kit wins.** The PS itself says so
explicitly: *"The exact label definition and K values are pinned in the
Starter Kit so every team solves the same task"* and *"evaluate.py... Pinned
conventions... never reimplement it."* `evaluate.py` is the literal scoring
code the judges will run. Everything in this repo (`agent/evaluator.py`,
`agent/convergence.py`, the whole loop) is built against the Starter Kit's
definition: **label `long_view`, metrics GAUC + nDCG@5, primary =
mean(GAUC, nDCG@5)**. If you ever see "NDCG@10/Recall@50" mentioned in a
deliverable template, treat it as the PS prose being stale, not an
instruction to build a second, different evaluator. Worth a one-line caveat
in the Devpost write-up so judges see we noticed, not that we got it wrong.

## Ground truth numbers (kuairand-starter-kit/baseline_scores.json)

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (sanity floor) | valid 0.4993 / test 0.4996 | valid 0.4675 / test 0.4511 | valid **0.4834** / test **0.4753** |
| item popularity (trivial) | test 0.6308 | test 0.5121 | test 0.5715 |
| **FM official baseline** (the one to beat) | valid 0.6674 / test 0.6610 | valid 0.5357 / test 0.5282 | valid **0.6016** / test **0.5946** |
| oracle ceiling | test 1.0000 | test 0.7289 | test **0.8645** |

- FM baseline config: `k=16, lr=0.001, batch=8192, max_epochs=40, patience=4`, 5 fields
  (`user_id, video_id, author_id, tab, dur_bucket`). 5-seed test std = **0.0008**.
- **Convergence rule** (read live from `baseline_scores.json`, not hardcoded anywhere):
  `epsilon=0.002, N=3` -- converged when validation primary hasn't improved by
  more than epsilon over the last 3 consecutive iterations.
- **nDCG@5 ceiling is 0.7289, not 1.0** — 27.1% of test users are all-negative
  (nDCG forced to 0, unfixable by any model) and 9.2% are all-positive (nDCG
  forced to 1). Judge progress against the oracle (headroom = 0.8645 − 0.5946
  = **0.27**), not against 1.0 (which would wrongly read as 0.41 of headroom).
- Splits are **date-based**, fixed: train `20220408–20220421` (1,141,112 rows),
  valid `20220422–20220428` (124,909), test `20220429–20220508` (170,588).
  This local "test" split has real labels (it's the full public dataset) --
  see the "train/valid/test discipline" section below for how we still
  behave as if blind to it.

### Dead ends the organizers already tested (don't repeat these)

From the Starter Kit README, tested and **no gain**:
- Adding CWM's 13 feature domains (vs. the 5 base fields): 0.5940 vs 0.5950 — noise, slightly worse.
- Larger embedding dim k = 8/16/32: 0.5895/0.5902/0.5887 — flat.
- **Why**: `user_id × video_id` crossing already captures most learnable signal;
  114K rows can't support more capacity; pure user-side first-order terms are
  provably zero-signal for within-user ranking (a constant per user doesn't
  change in-group order) — they only help through **item-side interactions**.

### Unexplored headroom, ranked by the organizers' own guess (their words, not scored/tested by them)

1. **Loss function**: pointwise logloss now; GAUC/nDCG are *ranking* metrics —
   pairwise (BPR) or listwise (per-user softmax) is their top guess.
2. **User history / sequences**: zero sequential modeling currently exists
   (DIN/SIM-style interest modeling is untouched).
3. **Multi-task**: `is_click, is_like, is_follow, is_comment, is_forward,
   play_time_ms` all exist in the logs, unused as auxiliary signals.
4. **Watch-time modeling**: censored regression on `play_time` (see CWM,
   github.com/hyz20/CWM — reference only, don't adopt its `torch==1.6.0` dep
   or its self-redefined `long_view2` label; it evaluates against something
   other than this task's pinned label).
5. **Model architecture** (DeepFM/DCN/xDeepFM) — explicitly ranked *last*
   since capacity is empirically not the bottleneck (see dead ends above).
6. Temporal features / train↔test drift.
7. `log_random_4_22_to_5_08_pure.csv` — randomized-exposure subset, usable
   for an unbiased secondary validation set / IPW / counterfactual eval.

## Train/valid/test discipline

Task Requirement 2: *"develops using only the training split and the public
validation feedback -- it never has access to the hidden test set."* This
local Starter Kit copy's "test" split **does** contain real labels (it's the
public dataset), and `agent/experiment.py` **does** compute test metrics
every run (so does the organizer's own `baseline.py` -- printing test scores
locally is normal Starter Kit practice, not a violation). The rule we
actually enforce in code: **nothing that picks a winner** -- Convergence
Detector, config selection, early stopping, the Orchestrator's "best
iteration" -- **ever reads `.test`, only `.valid.primary`.** Test scores are
recorded purely for our own tracking parity with `baseline.py`. Keep it that
way in any change you make; it's the single easiest rule to accidentally
break while iterating fast.

## Architecture (Phase 0 + P1 -- what's built)

```
run.py                     Phase 0 entry point: python run.py
run_p1.py                   P1 entry point: python run_p1.py (run Phase 0 first at least once)
agent/
  paths.py                  sys.path wiring into kuairand-starter-kit/
  config.py                  Structured Experiment Interface (ExperimentConfig, diff_configs)
  evaluator.py                 wraps evaluate.py exactly + harness self-check (random -> ~0.4834 valid)
  convergence.py                epsilon/N read live from baseline_scores.json
  cache.py                       disk cache for encoded training data (~66x faster on a cache hit)
  model_zoo/
    base.py                      RankingModel Protocol (step/predict/get_state/set_state/num_params)
    fm.py                         numerically identical to baseline.py's FM
    deepfm.py                      FM component + numpy MLP deep component, shared embeddings
    fm_bpr.py                       [P1] FM + pairwise BPR loss instead of pointwise logloss
    registry.py                     config.model string -> builder
  experiment.py               trains one config+seed, valid AND test metrics (test never drives decisions);
                                supports train_fraction (P1 multi-fidelity) and BPR's pairwise training loop
  recovery.py                   subprocess isolation (real timeout kill) + retry + degraded fallback
  run_logger.py                   experiments/<run_id>/iter_NNN/{config.json,hypothesis.md,results.json,logs.txt}
                                    + append-only logs/run_log.jsonl + manual-intervention hook
  orchestrator.py                   Phase 0: PREDEFINED_EXPERIMENTS (no LLM yet) + the loop tying it together
  submission.py                      wraps submit.py's write/read exactly, never reimplements format rules
  research_map.py              [P1] persistent, tree-structured Research Map (logs/research_map.json)
  diagnosis.py                  [P1] Metric-Aware Diagnosis Engine, seed-aware significance (not flat epsilon)
  multi_fidelity.py              [P1] 1%->10%->100% staged runner, kills clearly-broken candidates cheaply
  selector.py                     [P1] Best-First Node Selector: gain x confidence x novelty / cost
  p1_orchestrator.py               [P1] ties the above together; seeds the map from Phase 0's run_log.jsonl
tools/
  generate_analysis.py         logs/run_log.jsonl -> logs/analysis_report.md (the Run & Iteration Log deliverable)
tests/
  test_foundation.py            plain-assert smoke tests (no pytest in this env), run: python tests/test_foundation.py
experiments/                  generated per run, run-scoped (<run_id>/iter_001/, <run_id>/iter_002/, ...)
                                 -- a second run gets its own subdirectory, never overwrites a prior one
logs/                          run_log.jsonl (shared, append-only, run_id per entry), run_summary.json,
                                 analysis_report.md (Phase 0), research_map.json + p1_round_report.json (P1)
kuairand-starter-kit/          organizer's code, UNMODIFIED (evaluate.py/data.py/submit.py/baseline.py)
  KuaiRand-Pure/data/            downloaded dataset (gitignored -- see Quick start)
docs/                         the full PS + brainstorm doc, PDF, its 2 extracted diagrams, and both
                                Phase 0 / P1 features+results writeups
```

### Design decisions worth knowing before you change anything

- **Numpy-only, on purpose.** This dev environment has numpy + pandas, no
  torch/sklearn/matplotlib. `agent/` never imports beyond numpy so `python
  run.py` keeps working in the Starter Kit's own zero-dependency spirit.
  DeepFM's MLP is hand-rolled (manual forward + backprop + Adam) rather than
  reaching for torch -- see `agent/model_zoo/deepfm.py` docstring. Don't add
  a torch/sklearn import to anything under `agent/` without updating
  `requirements.txt`'s honesty comment and confirming it's actually needed.
- **FM is a faithful port, not a rewrite.** `agent/model_zoo/fm.py` must stay
  numerically identical to `kuairand-starter-kit/baseline.py`'s FM class. If
  a baseline-reproduction run doesn't hit ~0.6016 valid primary, that's a bug
  in this port, not a modeling result.
- **evaluate.py / submit.py are imported, never reimplemented.** `agent/evaluator.py`
  and `agent/submission.py` are thin wrappers around the organizer's actual
  functions (`ensure_starter_kit_on_path()` puts `kuairand-starter-kit/` on
  `sys.path`). This is deliberate against the "Tips -- How We Win" warning
  in the brainstorm doc: subtly-wrong reimplementations of pinned scoring
  logic silently change your score without anyone noticing.
- **Recovery uses real subprocess isolation** (`multiprocessing`, spawn --
  required on Windows), not just try/except, because numpy code inside a C
  loop won't yield to a Python-level timeout check, and Windows has no
  SIGALRM. Any script that imports `agent.recovery` (directly or
  transitively via `agent.orchestrator`) must guard its entry point with
  `if __name__ == "__main__":` or spawn will recursively re-execute it.
- **Config diffs stand in for code diffs**, per the brainstorm doc's own
  reasoning: *"most experiments go through a structured config... only
  genuinely novel ideas cost real tokens."* `diff_configs()` in
  `agent/config.py` produces the "code diff applied" field the Run &
  Iteration Log deliverable asks for, for every config-driven experiment.
  Once Phase 4 lets the LLM write raw code for a genuinely novel idea, THAT
  diff should be a real unified diff of the generated file, logged
  alongside (not instead of) the config diff.
- **Convergence Detector reads epsilon/N from `baseline_scores.json` at
  runtime**, never hardcoded, so a republished Starter Kit value is picked
  up automatically.
- **Lineage field pre-wired, tree not built yet.** `ExperimentConfig.parent_id`
  exists in Phase 0's flat predefined list purely so P1's Research Map /
  Experiment Tree (AIDE-style: node = full config, edges = modification
  type) doesn't require a schema migration later -- see Roadmap.

## Quick start

```bash
# 1. Dataset (already done in this repo's dev copy -- 47MB, Zenodo, no auth)
cd kuairand-starter-kit
curl -sL -o KuaiRand-Pure.tar.gz "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
tar xzf KuaiRand-Pure.tar.gz && rm KuaiRand-Pure.tar.gz
cd ..

# 2. Smoke test the foundation (no pytest in this env -- plain script, ~1 min)
python tests/test_foundation.py

# 3. Run the Phase 0 loop end-to-end
python run.py                      # full run: self-check -> 4 predefined experiments -> submission
python run.py --skip_submission    # faster: skip the final retrain+CSV step
python run.py --timeout_s 120      # tighter per-experiment recovery budget

# 4. Regenerate the Run & Iteration Log deliverable from logs/run_log.jsonl
python tools/generate_analysis.py --stdout
```

Windows notes: use `python`, not `python3` (the latter is a Microsoft Store
alias stub in this environment that fails with no interpreter installed).
`py` also works. PowerShell is the primary shell; Bash tool available too.

## Judging-criteria -> code map

| Criterion | What in this repo earns it |
|---|---|
| Technical Execution (35%) — primary metric | `agent/evaluator.py` (exact scoring), `agent/model_zoo/` |
| Technical Execution — robustness | `agent/recovery.py` (subprocess isolation, retry, degraded fallback) |
| Innovation & Problem Insight (20%) | Hypotheses in `PREDEFINED_EXPERIMENTS` (orchestrator.py) + `p1_candidate_pool()` citing starter-kit headroom + prior work; `agent/research_map.py` (persistent tree) + `agent/diagnosis.py` (why, not just what) |
| Impact & Relevance (20%) — autonomy | `RunLogger.manual_intervene()` -- every human nudge counted, logged, never silent; `agent/selector.py`'s scored best-first order (not source order) is a real, if not yet LLM-driven, autonomy step |
| Feasibility & Practicality (15%) | `Orchestrator.wall_time_total_s` / `llm_tokens_total` / `gpu_hours_total` in `run_summary.json` |
| Presentation (10%, final only) | `tools/generate_analysis.py` report; dashboard artifact (P2, pulled forward per doc's own advice -- not built yet, see Roadmap) |
| Deliverable: Run & Iteration Logs | `logs/run_log.jsonl` + `experiments/<run_id>/iter_*/` + `tools/generate_analysis.py` -> `logs/analysis_report.md` |
| Deliverable: Final submission | `agent/submission.py` writes+validates `submission_valid.csv` / `submission_test.csv` |

## Roadmap (from the brainstorm doc -- Phase 0 + the two highest-ROI P1 items are DONE; this is what's next)

- **Phase 0 (done, validated end-to-end against real data, 3-seed
  robustness-checked):** all 8 P0 "Foundation" features -- Orchestrator,
  Model Zoo (FM+DeepFM), Evaluator Wrapper, Convergence Detector, Structured
  Experiment Interface, Structured Run Log, Failure Recovery, Submission
  Validator. A real run converges automatically (ε/N rule) and **beats the
  official baseline** (test primary +0.0029 on the submitted seed, +0.0030
  as a 3-seed mean, ~8.9σ above the measured noise floor -- GAUC +0.0033,
  nDCG@5 +0.0024), 0 manual interventions. Full writeup:
  [`docs/PHASE0_FEATURES_AND_IMPROVEMENTS.md`](docs/PHASE0_FEATURES_AND_IMPROVEMENTS.md).
- **P1 (done, one round, validated end-to-end):** the brainstorm doc's own
  "highest-ROI pair" -- Research Map / Experiment Tree (persistent,
  AIDE-style edges) + Multi-Fidelity Runner (1%→10%→100% staged) -- plus the
  Best-First Node Selector and Metric-Aware Diagnosis Engine needed to make
  them functional, plus one new algorithmic direction (FM_BPR, pairwise
  loss). Real round result: **two honest negative results** (BPR
  underperformed with unchanged pointwise hyperparameters, `k=32`
  independently reproduces a documented dead end), each with a specific,
  actionable diagnosis rather than just a lower number. Full writeup,
  including two more real bugs found + fixed during this pass's own
  validation (a 12h host-sleep wall-clock measurement bug, a
  PYTHONHASHSEED reproducibility gap, and a Diagnosis Engine fix that
  corrected a false "noise_floor" label on a real ~8.9σ effect):
  [`docs/P1_FEATURES_AND_RESULTS.md`](docs/P1_FEATURES_AND_RESULTS.md).
  Read both docs before touching `agent/` performance-sensitive or
  statistics code again -- they document *why* several non-obvious things
  are the way they are (`time.process_time()` not `time.time()`; `sorted()`
  over a set of string keys; a seed-aware significance bar instead of a flat
  epsilon) so a "cleanup" pass doesn't quietly reintroduce a fixed bug.
- **Not yet built from the P1 tier:** Multi-Task Feature Exploitation,
  TikTok-disclosed features (completion rate, rewatch flag, fast-skip,
  creator aggregates), Per-Segment Metric Diagnosis. `DeepFM_BPR` (pairwise
  loss + the deep component) is a natural, cheap extension of what's
  already built. P1 is currently **one round**, not a loop -- the candidate
  pool is a fixed 2-item list; extending it (more hand-authored "improve"
  templates reacting to each round's diagnosis) is the natural next step
  before Phase 4's LLM replaces hand-authoring entirely.
- **Phase 3 (Model Zoo + tuning):** fill out DCNv2/Wide&Deep/LightGBM (needs
  torch/sklearn — currently absent from this env, install when this phase starts).
- **Phase 4 (LLM research reasoning) -- the actual differentiator:** replace
  the hand-authored candidate pools (`PREDEFINED_EXPERIMENTS`,
  `p1_candidate_pool()`) with an LLM-driven Research Strategist that reads
  dataset summary + metric history + the Research Map + budget remaining and
  outputs `{hypothesis, experiment, reasoning, expected_effect, cost, priority}`
  -- `agent/selector.py`'s scoring formula and `agent/diagnosis.py`'s
  insights are exactly the *input context* this strategist would consume;
  neither needs to change shape when Phase 4 starts. This is also where
  `llm_tokens_total` in `run_summary.json` starts being real instead of
  structurally zero.
- **P2 stretch, but the doc explicitly says pull the dashboard earlier:**
  *"Consider building the dashboard/log-replay earlier than its P2 slot
  suggests -- it's cheap and it's your closing argument."* Generate an HTML
  dashboard from `logs/run_log.jsonl` + `logs/research_map.json` (Claude's
  `dataviz` skill covers chart/palette conventions; `artifact-design` before
  publishing as an Artifact) — a good next session's first task now that
  both logs have real, multi-node content.

## Open questions from the brainstorm doc, still open

- Orchestration framework for Phase 4 (custom loop vs. LangGraph vs. CrewAI) — undecided.
- Actual GPU budget/access for later phases — TBD per organizer; Phase 0 is CPU-only.
- Attempt bonus benchmarks (KuaiRand-1k/27k)? Config-driven scale-up is designed
  for later (P2) — don't attempt before the required KuaiRand-Pure path is solid.
- Who owns the Phase 4 "next hypothesis" prompt design — likely the highest-leverage
  piece for the Innovation score.

## Git

Remote: `https://github.com/9irija/TikTok_TechJam.git` (empty repo, confirmed
via `git ls-remote` before any push). Repo root (`c:\Users\mgiri\Desktop\tiktok_techjam`)
is the git root — `kuairand-starter-kit/` is a plain subdirectory, not a submodule.
