# Final Submission & Results Summary

This is the single, consolidated artifact for Problem Statement §2.5
Deliverable 4 ("Final Submission & Results Summary"). Every number below is
pulled directly from `logs/research_map.json` / `logs/run_summary.json` /
`logs/p4_run_report.json` — nothing here is hand-estimated.

## Results table (KuaiRand-Pure, required benchmark)

Current best: **`deepfm_mtl_v1`** (multi-task DeepFM, see README "Multi-task
learning"), resolved via `ResearchMap.best_confirmed_node()` — see
`tools/generate_submission.py`, which writes `submission_valid.csv` /
`submission_test.csv` from this exact node.

| Metric | Official baseline | `deepfm_mtl_v1` (valid, 3-seed) | `deepfm_mtl_v1` (test, 3-seed mean) | Δ vs. baseline (test) |
|---|---|---|---|---|
| GAUC | 0.6674 (valid) / 0.6610 (test) | 0.6715 ± 0.0005 | 0.6641 ± 0.0008 | **+0.0031** |
| nDCG@5 | 0.5357 (valid) / 0.5282 (test) | 0.5378 ± 0.0001 | 0.5308 ± 0.0004 | **+0.0026** |
| **Primary** (mean of the two) | 0.6016 (valid) / 0.5946 (test) | **0.6046 ± 0.0003** | **0.5974 ± 0.0006** | **+0.0028** |

Delta computed exactly per Problem Statement §2.6's formula
(`delta(m) = score_agent(m) − score_baseline(m)`, `score_dataset =
mean over m of delta(m)`) — mathematically identical here to
`delta(mean(GAUC, nDCG@5))` since the mean is linear, so both readings of
"the primary delta" agree.

Context for reading these numbers (per §2.6's own framing): the metrics
don't span [0, 1] — a perfect ranking only reaches primary 0.8645 (27.1%
of hidden-test users have no positive label at all), and random scoring
sits at 0.4753. The official baseline (0.5946) already captures ~31% of
the attainable range; this submission adds another ~+0.0028 of test
primary on top of that, verified 3-seed and independently confirmed to
generalize on the unbiased randomized-exposure split (see README
"Does the win hold on genuinely unbiased data?" — the edge *grows* there,
not shrinks).

KuaiRand-1k / KuaiRand-27k (bonus benchmarks): not attempted. Investigated
and explicitly declined — `kuairand-starter-kit/data.py` hardcodes
`_pure`-suffixed filenames, so supporting them isn't the config-only
scale-up it might look like; it would mean a new, unvalidated loader
against an uninspected schema with no organizer reference score to
self-check against. Full reasoning: `docs/POLISH_PASS_RESULTS.md` §6.

## Resource usage

Two honest figures, not one cherry-picked number — this project's actual
best model came from a longer research process than a single convergence
loop, and both numbers are worth reporting plainly.

**Phase 0's own single continuous run** (the literal case Problem
Statement §2.6's convergence rule describes — read the baseline,
iterate, converge automatically, no LLM involved yet):

| | Value | Cap | 
|---|---|---|
| Iterations used | 4 | out of 50 (hard cap) |
| Agent wall-clock | ~18.1 min (1083 s) | out of 6h (backstop) |
| LLM tokens | 0 | — (no LLM in this phase) |
| GPU-hours | 0.0 | CPU-only throughout |
| Manual interventions (this run specifically) | 0 | — |

**Full project total** (every experiment across every phase — P0 through
this pass's P2 work — that led to the actual submitted best model,
`deepfm_mtl_v1`, which came from 5 further phases of research after Phase
0's own convergence):

| | Value |
|---|---|
| Total experiment nodes in the Research Map | 33 |
| Total individual training runs (all seeds, all nodes) | 68 |
| Total training wall-clock, summed across every run | **at least** ~2.69 hours (9,677 s)† |
| Total LLM tokens (Gemini free tier, Phase 4, both rounds) | 16,608 — **$0 real cost** |
| Total GPU-hours | 0.0 — CPU-only throughout the entire project |
| Manual interventions, per-run (no single run required a human fix mid-execution) | 0 |
| Manual interventions, total (see below) | 1 |

† A lower bound, stated honestly rather than padded: `deepfm_mtl_gnn_init_v1`'s
3 runs and `deepfm_mtl_gnn_feature_v1`'s 1 run (`tools/check_gnn_init.py` /
`tools/check_gnn_feature.py`, standalone checks, not the standard
`agent/experiment.py` training loop) don't record a per-seed
`wall_time_s` the way every other node's runs do — a real, minor gap in
those two scripts, not a hidden number. The true total is somewhat higher.

**The one logged manual intervention, in full** (`logs/manual_interventions.jsonl`):
a real, bounded Phase 4 round's 3rd proposal (`deepfm_mtl_capacity_v1`)
genuinely timed out at its 240s per-stage budget — Failure Recovery itself
already tried a retry and a degraded-epoch fallback, both also timing out,
before correctly abandoning it (a real, honest demonstration of Task
Requirement 3, not a bug). Extending `timeout_s` to 1500s and re-running
the identical config by hand is "tuning something by hand" per the
workshop's own definition, even though it didn't change *what* was being
tried — logged plainly rather than quietly deciding it doesn't count. The
real result once it finished: valid primary 0.6036, a genuine regression
vs. `deepfm_mtl_v1` (0.6046) — confirms the timeout was a budget issue,
not a sign the idea might have won with a bug fix.

Both figures are far under the §2.3 "Limits" caps (50 iterations / 6h per
run) even summed together — compute was never close to the binding
constraint on this benchmark, consistent with the Problem Statement's own
framing ("compute is deliberately not the binding constraint... 100
iterations of the official baseline take about 28 min on a single CPU
core").

**How Feasibility & Practicality is actually scored** (§2.6): gated, not
continuous — "scored only among submissions whose hidden-test primary
score exceeds the official baseline" (we do, by +0.0028 — see the Results
table above), then graded in three coarse tiers (low/medium/high
consumption) rather than a fine-grained ranking. The scored compute
measure is **agent wall-clock**, not GPU-hours — GPU-hours are 0.0
throughout this project (CPU-only), reported anyway per the PS's own
"report GPU-hours if any were used" instruction, but wall-clock is what's
actually graded. Phase 0's own single convergence run (~18.1 min) is the
cleanest read on "wall-clock to reach a converged result" in the PS's own
singular framing; the full-project total (~2.7 hours, spanning 5 further
phases of research after that convergence) is the honest, complete number
for everything that produced the actual submitted best model.

## Autonomy breakdown (Impact & Relevance, 20%)

Per the Track 2 technical workshop's own clarification (28 Aug 2026):
manual intervention is judged by whether a human *changed the agent's
behavior* (picked what to try, tuned something by hand), not by whether a
crashed process was restarted (that's Failure Recovery's job, and
`agent/recovery.py` does it automatically, with zero human involvement,
verified against a genuine OOM). Reported plainly rather than left for a
judge to infer from raw logs — of the 33 total Research Map nodes:

| Source | Count | Autonomy level |
|---|---|---|
| Phase 0's predefined loop | 4 | Fully mechanical — reproduce baseline, run a fixed list, converge automatically. Zero human choice of *what* to try. |
| P1's Best-First Selector | 4 | Deterministic, non-LLM heuristic (`gain × confidence × novelty ÷ cost`) picks from a **hand-authored** candidate pool — the pool itself was human-designed, the pick from it was not. |
| Phase 4's LLM Research Strategist | 5 | **Genuinely autonomous** — Gemini reads the live Research Map and proposes the next experiment on its own, across 2 real rounds. Round 1: `deepfm_regularized` (the project-best's ancestor) and `deepfm_higher_l2` (a correctly-caught regression). Round 2 (run against the full 30-node history): `deepfm_mtl_aux_weight_tuning` and `deepfm_mtl_focal_soft_v1` (both real, correctly-diagnosed regressions), and `deepfm_mtl_capacity_v1` (the one candidate that needed the single logged manual intervention above to reach a real result — the LLM's *proposal* was fully autonomous; getting it to *completion* was not). |
| P2's manually-directed research | 20 | A human (or a human directing an AI coding assistant) chose which idea to test next (BPR variants, DIN, MTL extensions, loss functions, ensembling, DCNv2, deeper MTL heads, two graph-propagated embedding variants, etc.) — the training/evaluation of each idea then runs autonomously once launched, but the choice of *what* to try was not agent-driven. |

**Honest framing:** the fully-autonomous story is real and grew this pass
(Phase 0's loop, plus 5 genuine LLM-driven proposals across 2 real Phase 4
rounds) — the bulk of this project's actual research volume (P1's pool
design, all of P2) remains human-directed, with automation covering
*execution* (training, recovery,
logging, diagnosis) rather than *ideation*. This is reported directly
rather than let a "0 manual interventions" figure (accurate for any
single run, per the workshop's own definition) imply more end-to-end
autonomy across the whole project than actually happened.

## Convergence rule compliance

`agent/convergence.py`'s `ConvergenceDetector` implements the organizer's
exact rule (ε=0.002, N=3, read live from
`kuairand-starter-kit/baseline_scores.json`, not hardcoded) plus the two
hard backstops from §2.3 ("50 iterations... 6h wall-clock ceiling"),
added this pass as named constants (`MAX_ITERATIONS`, `MAX_WALL_CLOCK_S`
in `agent/convergence.py` — no organizer-published file carries these two
numbers the way `baseline_scores.json` does epsilon/N, so they can't be
read live the same way; update by hand if a republished PS ever changes
them). Regression-tested (`test_convergence_detector_hits_iteration_cap_before_epsilon_n_rule`,
`test_convergence_detector_hits_wall_clock_cap`) to confirm both backstops
actually fire, not just assumed to work. In every real run so far, the
epsilon/N rule fired first, exactly as the Problem Statement predicts —
the hard caps have never actually been the binding stop condition, but
they're now a real, checked guarantee rather than an unverified assumption.
