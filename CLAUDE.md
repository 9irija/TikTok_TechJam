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
   **Done (P1):** `fm_bpr` -- 3 diagnosis-driven rounds, real but plateaued
   ~0.002 below baseline, see P1_FEATURES_AND_RESULTS.md.
2. **User history / sequences**: zero sequential modeling currently exists
   (DIN/SIM-style interest modeling is untouched). **Still open** -- the
   next natural PyTorch target (attention backward pass is a genuinely
   good autograd use case, same reasoning as deepfm_mtl below).
3. **Multi-task**: `is_click, is_like, is_follow, is_comment, is_forward,
   play_time_ms` all exist in the logs, unused as auxiliary signals.
   **Done (P2):** `agent/model_zoo/deepfm_mtl.py` -- first torch model in
   the project, auxiliary is_like/is_follow/is_comment/is_forward heads
   sharing DeepFM's embedding table. `deepfm_mtl_v1`, 3-seed verified:
   valid primary 0.6046 +/- 0.0003 vs. deepfm_regularized's 0.6035 +/-
   0.0002 -- real `clear_improvement`, now the project-best. See README
   "Multi-task learning" for the honest test-split nuance.
4. **Watch-time modeling**: censored regression on `play_time` (see CWM,
   github.com/hyz20/CWM — reference only, don't adopt its `torch==1.6.0` dep
   or its self-redefined `long_view2` label; it evaluates against something
   other than this task's pinned label). Still open.
5. **Model architecture** (DeepFM/DCN/xDeepFM) — explicitly ranked *last*
   since capacity is empirically not the bottleneck (see dead ends above).
   **LightGBM tried too (P2, `lgbm_baseline`):** valid primary 0.5995,
   confirms the same finding extends to tree-based models -- this task's
   signal lives in high-cardinality `user_id x video_id` embedding
   crossing, which trees can't represent as efficiently as a learned
   embedding table. Not integrated into the Model Zoo proper (real
   training-loop interface mismatch, not worth the refactor once the
   standalone result was this clearly behind) -- see README "Multi-task
   learning" section for the full reasoning.
6. Temporal features / train↔test drift. Still open.
7. `log_random_4_22_to_5_08_pure.csv` — randomized-exposure subset, usable
   for an unbiased secondary validation set / IPW / counterfactual eval.
   Still open.

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

## Architecture (Phase 0 + P1 + Phase 4 -- what's built)

```
run.py                     Phase 0 entry point: python run.py
run_p1.py                   P1 entry point: python run_p1.py (run Phase 0 first at least once)
run_p4.py                    Phase 4 entry point: python run_p4.py (needs GEMINI_API_KEY in .env)
agent/
  paths.py                  sys.path wiring into kuairand-starter-kit/
  config.py                  Structured Experiment Interface (ExperimentConfig, diff_configs, edge_type)
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
  selector.py                     [P1] Best-First Node Selector: gain x confidence x novelty / cost (no LLM)
  p1_orchestrator.py               [P1] ties the above together; seeds the map from Phase 0's run_log.jsonl
  llm_client.py                     [Phase 4] Gemini API wrapper (free tier), retry logic
  research_strategist.py             [Phase 4] builds the prompt from the Research Map, validates the LLM's
                                       proposed ExperimentConfig against the model registry before executing it
  p4_orchestrator.py                  [Phase 4] swaps P1's candidate source for the Strategist; reuses P1's
                                        Multi-Fidelity Runner + Diagnosis Engine completely unchanged
  research_critic.py                   [P2] deterministic pre-flight veto (duplicate / confirmed dead end),
                                         runs before Multi-Fidelity Runner ever sees a candidate
tools/
  generate_analysis.py         logs/run_log.jsonl -> logs/analysis_report.md (the Run & Iteration Log deliverable)
  verify_multiseed.py            [Phase 4] promotes a single-seed Research Map node to a 3-seed-verified one
docs/dashboard.html            [P2] self-contained Research Map dashboard (tree + trajectory chart); generated
                                 once by hand from a map snapshot, not auto-regenerated from the live JSON
tests/
  test_foundation.py            plain-assert smoke tests (no pytest in this env), run: python tests/test_foundation.py
                                  -- Phase 4's LLM calls are tested via a mock client, never a real API call
experiments/                  generated per run, run-scoped (<run_id>/iter_001/, <run_id>/iter_002/, ...)
                                 -- a second run gets its own subdirectory, never overwrites a prior one
logs/                          run_log.jsonl (shared, append-only, run_id per entry), run_summary.json,
                                 analysis_report.md (Phase 0), research_map.json (shared, persistent, P1+Phase 4),
                                 p1_round_report.json (P1), p4_run_report.json (Phase 4, real llm_tokens_total)
kuairand-starter-kit/          organizer's code, UNMODIFIED (evaluate.py/data.py/submit.py/baseline.py)
  KuaiRand-Pure/data/            downloaded dataset (gitignored -- see Quick start)
docs/                         the full PS + brainstorm doc, PDF, its 2 extracted diagrams, and the
                                Phase 0 / P1 / Phase 4 / P2 features+results writeups
.env / .env.example            GEMINI_API_KEY -- .env is gitignored (real key), .env.example is the
                                 committed template (empty value) -- NEVER put a real key in .env.example
```

### Design decisions worth knowing before you change anything

- **Numpy-only by default, torch added deliberately and scoped (P2).**
  `python run.py`/`run_p1.py` (minus one new model) still work in the
  Starter Kit's zero-dependency spirit -- FM/DeepFM/FM_BPR stay hand-rolled
  numpy (manual forward + backprop + Adam), each with one clean,
  worth-hand-deriving backward pass -- see `agent/model_zoo/deepfm.py`
  docstring. `agent/model_zoo/deepfm_mtl.py` is the one deliberate
  exception: a 5-headed shared-bottom multi-task network (1 main + 4
  auxiliary losses merging into one shared embedding table) is exactly the
  case hand-rolled backprop stops paying for itself, so it's built in
  torch (CPU wheel, `requirements.txt` now lists it as a real, used
  dependency, not aspirational). Don't add a torch/sklearn import to
  anything else under `agent/` without the same justification -- "autograd
  is genuinely the right tool here," not "it would be more convenient."
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
  Phase 4's LLM still only ever proposes a structured config too (see
  below) -- there is no code-generation call anywhere in this codebase yet;
  if that's ever added for a genuinely novel idea, THAT diff should be a
  real unified diff of the generated file, logged alongside (not instead
  of) the config diff.
- **The LLM never executes anything it returns directly.** `research_strategist.py`'s
  `_validate_and_build()` is a real Validation Gate: unknown model name,
  non-existent `parent_id`, wrong hyperparameter types, missing fields are
  all rejected with a specific error fed back to the LLM as a re-prompt
  (`max_validation_retries`, default 2) before giving up and returning
  `None` -- the caller logs that and moves on, never crashes. Tested with a
  mock client (`_FakeGeminiClient` in `tests/test_foundation.py`) so the
  test suite makes zero real API calls.
- **Gemini (free tier), not Anthropic/OpenAI** -- the problem statement
  doesn't mandate a provider; a genuinely $0 token cost is a cleaner,
  more defensible number for Feasibility & Practicality than an estimated
  dollar figure from a paid API. `GEMINI_MODEL` env var overrides
  `agent/llm_client.py`'s `DEFAULT_MODEL` if Google retires the current one
  again (already happened once during this project -- `gemini-2.0-flash`
  was retired mid-build, caught by a live API call returning 404 with the
  replacement name in the error message, not by advance knowledge).
- **Convergence Detector reads epsilon/N from `baseline_scores.json` at
  runtime**, never hardcoded, so a republished Starter Kit value is picked
  up automatically.
- **Lineage was pre-wired in Phase 0, the tree is now built and in active
  use.** `ExperimentConfig.parent_id` (Phase 0) plus `edge_type` (added in
  P1) are what `agent/research_map.py`'s persistent tree runs on --
  draft/improve/debug edges, AIDE-style. Both P1's hand-authored candidates
  and Phase 4's LLM proposals set these on every node.

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

# 5. Run a P1 round (persistent Research Map, Best-First Selector, Multi-Fidelity Runner)
python run_p1.py

# 6. Run a Phase 4 round (LLM Research Strategist) -- needs GEMINI_API_KEY in .env
#    (copy .env.example; get a free key at https://aistudio.google.com/apikey)
python run_p4.py --max_iterations 2

# 7. Promote a promising single-seed result to 3-seed-verified
python tools/verify_multiseed.py <node_id>

# 8. Regenerate submission_valid.csv / submission_test.csv from the Research
#    Map's confirmed best (ResearchMap.best_confirmed_node(), not the raw
#    numeric leaderboard -- see the "Engineered features" note in README.md)
python tools/generate_submission.py

# 9. Optuna hyperparameter search around the confirmed best (P2) -- reduced
#    fidelity trials, full-fidelity confirmation before trusting a winner
python tools/run_hyperparameter_search.py --n_trials 15
```

Windows notes: use `python`, not `python3` (the latter is a Microsoft Store
alias stub in this environment that fails with no interpreter installed).
`py` also works. PowerShell is the primary shell; Bash tool available too.

## Judging-criteria -> code map

| Criterion | What in this repo earns it |
|---|---|
| Technical Execution (35%) — primary metric | `agent/evaluator.py` (exact scoring), `agent/model_zoo/` |
| Technical Execution — robustness | `agent/recovery.py` (subprocess isolation, retry, degraded fallback) |
| Innovation & Problem Insight (20%) | Hypotheses in `PREDEFINED_EXPERIMENTS`/`p1_candidate_pool()` citing starter-kit headroom + prior work; `agent/research_map.py` (persistent tree) + `agent/diagnosis.py` (why, not just what); Phase 4's LLM finding a gap (DeepFM overfitting) the hand-authored logic missed -- see `docs/PHASE4_RESULTS.md`; `agent/features.py` (TikTok-disclosed, leakage-safe train-only-aggregate features -- honest `noise_floor` result, see README "Engineered features") |
| Impact & Relevance (20%) — autonomy | `RunLogger.manual_intervene()` -- every human nudge counted, logged, never silent; Phase 4's full run (propose -> validate -> execute -> diagnose -> record) with 0 manual interventions is the real autonomy story, not `agent/selector.py`'s heuristic (still useful, but not LLM-driven) |
| Feasibility & Practicality (15%) | `wall_time_total_s` / `gpu_hours_total` throughout; `llm_tokens_total` is real (not structurally zero) only in `logs/p4_run_report.json` -- Gemini free tier means it's also genuinely $0, not an estimate |
| Presentation (10%, final only) | `tools/generate_analysis.py` report; dashboard artifact (P2, pulled forward per doc's own advice -- not built yet, see Roadmap) |
| Deliverable: Run & Iteration Logs | `logs/run_log.jsonl` + `experiments/<run_id>/iter_*/` + `tools/generate_analysis.py` -> `logs/analysis_report.md` |
| Deliverable: Final submission | `tools/generate_submission.py` resolves the Research Map's confirmed best (`ResearchMap.best_confirmed_node()`) and calls `agent/submission.py` to write+validate `submission_valid.csv` / `submission_test.csv` -- currently `deepfm_mtl_v1` (P2, torch multi-task), 3-seed verified |

## Roadmap (Phase 0 + P1 + Phase 4 + Phase 5 are DONE; Phase 6 + P2 substantially closed; what's left below)

- **Phase 0 (done):** all 8 P0 "Foundation" features. Converges
  automatically, **beat the official baseline** (test primary +0.0029 on
  the submitted seed at the time, +0.0030 3-seed mean, ~8.9σ above noise),
  0 manual interventions. Superseded as "project best" by Phase 4 (below),
  still the reference for the largest single jump over the official
  baseline. Full writeup: [`docs/PHASE0_FEATURES_AND_IMPROVEMENTS.md`](docs/PHASE0_FEATURES_AND_IMPROVEMENTS.md).
- **P1 (done, 3 rounds):** the brainstorm doc's own "highest-ROI pair" --
  Research Map / Experiment Tree + Multi-Fidelity Runner -- plus the
  Best-First Node Selector, Metric-Aware Diagnosis Engine, and one new
  algorithmic direction (FM_BPR, pairwise loss). 3 rounds of
  diagnosis-driven iteration on BPR (round 1 hand-authored, rounds 2-3
  automated + hand-authored reactions to its own diagnosis) recovered
  +0.0013 and fixed BPR's training dynamics, but plateaued ~0.002 below
  baseline -- a real, reported, incomplete trend, not a clean win. Also
  found + fixed 3 real bugs (a 12h host-sleep wall-clock measurement bug
  now using `time.process_time()`; a PYTHONHASHSEED reproducibility gap;
  a Diagnosis Engine fix for a false "noise_floor" label on a real ~8.9σ
  effect -- read *why* before touching this logic again). Full writeup:
  [`docs/P1_FEATURES_AND_RESULTS.md`](docs/P1_FEATURES_AND_RESULTS.md).
- **Phase 4 (done, LLM research reasoning) -- the actual differentiator,
  and it delivered:** `agent/research_strategist.py` + `agent/llm_client.py`
  (Gemini, free tier -- $0 real cost) replace the hand-authored candidate
  pools. On its first real run: proposed `deepfm_regularized` (raised L2 on
  `deepfm_wider`'s architecture, reasoning from *both* DeepFM nodes'
  `overfitting_risk` flags -- a gap P1's own hand-authored logic never
  acted on), **3-seed-verified as a real improvement** (0.6035±0.0002 vs.
  0.6028, `clear_improvement`) -- **project-best at the time**, since
  superseded by `deepfm_mtl_v1` (P2, see the "Unexplored headroom" #3
  entry above and README "Multi-task learning"). Iteration 2's
  proposal was a genuine regression, correctly caught and reported, not
  hidden. `tools/verify_multiseed.py` (new this pass) promotes a promising
  single-seed result to 3-seed-verified on demand. Full writeup:
  [`docs/PHASE4_RESULTS.md`](docs/PHASE4_RESULTS.md).
- **P2 (post-roadmap, done this pass): Multi-Task Feature Exploitation +
  TikTok-disclosed features -- both closed.** `agent/features.py`
  (completion rate, rewatch flag, fast-skip, creator aggregates as
  train-only aggregates; `features_v1` came back `noise_floor`, a real
  negative result) and `agent/model_zoo/deepfm_mtl.py` (multi-task,
  auxiliary is_like/is_follow/is_comment/is_forward heads; `deepfm_mtl_v1`
  came back a real, 3-seed-verified `clear_improvement` -- now the
  project-best). LightGBM also tried (`lgbm_baseline`, standalone, not
  integrated into the Model Zoo) -- see README "Multi-task learning" for
  the full reasoning on both.
- **Still not built from the P1 tier:** Per-Segment Metric Diagnosis,
  generalized (not per-node-id-hardcoded) diagnosis-driven candidate
  generation. `DeepFM_BPR` is a natural, cheap extension of what's already
  built. Sequence modeling (DIN/SIM-style, the starter kit's own #2-ranked
  untested item) is the next natural PyTorch target -- attention's
  backward pass is a genuinely good autograd use case, same reasoning as
  `deepfm_mtl`.
- **Not yet built from Phase 4:** budget-aware stopping (the `budget` dict
  in the LLM prompt is informational only -- nothing changes behavior as it
  depletes), auto-triggering `verify_multiseed.py` when a new best node
  appears, generating the prompt's "dead ends" section from the Research
  Map's own tags instead of a hand-maintained string. See
  `docs/PHASE4_RESULTS.md` §6 for the complete list.
- **Phase 3 (Model Zoo + tuning) -- mostly covered:** FM/DeepFM/FM_BPR
  (numpy) + DeepFM_MTL (torch, P2) done; LightGBM tried standalone (not
  integrated -- real interface mismatch with the per-epoch SGD loop, not
  worth the refactor given the result); DCNv2/Wide&Deep still not started.
  **Hyperparameter search (Optuna) done (P2):** `agent/hpo.py` +
  `tools/run_hyperparameter_search.py`, reduced-fidelity search with a
  full-fidelity confirmation gate. 15 trials around `deepfm_mtl_v1`'s
  hyperparameters found nothing that beat it (`deepfm_mtl_v1_hpo`, tagged
  `regression`) -- current hyperparameters already look close to a local
  optimum for this search space. Also caught and fixed a real concurrency
  bug in `ResearchMap.save()` along the way (see `docs/P2_FEATURES_AND_RESULTS.md` §4).
- **Phase 5 (multi-fidelity + early termination) -- done, including the
  "log GPU saved" requirement that was initially missed:** built as part
  of P1 (`agent/multi_fidelity.py`), but per-stage costs were computed for
  kill/escalate decisions and then discarded, never persisted. Closed:
  `MultiFidelityResult.stage_wall_times_s` + `.estimated_time_saved_s()`,
  threaded through `RunLogger.log_iteration()`'s new `fidelity_info`
  parameter. Full writeup: [`docs/POLISH_PASS_RESULTS.md`](docs/POLISH_PASS_RESULTS.md) §2.
- **Phase 6 (recovery, polish & bonus) -- robustness claims now verified,
  not just asserted; Research Critic Gate done; bonus benchmarks assessed
  and declined with a specific reason:**
  - OOM handling: tested against a **real** MemoryError (`k=10**9`, a
    genuine 293 TiB allocation request numpy fails fast on), not simulated
    -- `test_recovery_catches_a_genuine_oom`.
  - Checkpointing: verified already satisfied by `agent/research_map.py`'s
    existing `save()`-on-every-mutation design -- no new code needed.
  - Research Critic Gate: `agent/research_critic.py`, deterministic-only
    (duplicate check + confirmed-dead-end veto), wired into both
    orchestrators before any candidate reaches the Multi-Fidelity Runner.
    Verified against the real, committed `research_map.json`.
  - Bonus benchmarks (KuaiRand-1k/27k): investigated, not attempted --
    `kuairand-starter-kit/data.py` hardcodes `_pure` filenames, so this is
    NOT the "config change only" scale-up the brainstorm doc's P2 entry
    assumed. Supporting 1k/27k needs a new, unvalidated loader against an
    uninspected schema with no organizer reference scores to self-check --
    exactly the failure mode this project avoids everywhere else. Full
    reasoning: `docs/POLISH_PASS_RESULTS.md` §6.
  - Full writeup: [`docs/POLISH_PASS_RESULTS.md`](docs/POLISH_PASS_RESULTS.md).
- **P2 stretch -- Research Critic Gate, the dashboard, Extended Model Zoo
  (DeepFM_MTL, torch), and Hyperparameter Search (Optuna) all done; still
  open:** Config-Driven Scale-Up (blocked on the bonus-benchmark finding
  above). `docs/dashboard.html`: validated palette (dataviz skill),
  built per artifact-design principles, screenshot-verified with headless
  Edge (caught and fixed a real card-layout collision bug before shipping)
  -- generated once by hand from a Research Map snapshot, not auto-regenerated
  from the live JSON (a real next step, not done here).

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
