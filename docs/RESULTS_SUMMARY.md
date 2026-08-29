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
| Manual interventions | 0 | — |

**Full project total** (every experiment across every phase — P0 through
this pass's P2 work — that led to the actual submitted best model,
`deepfm_mtl_v1`, which came from 5 further phases of research after Phase
0's own convergence):

| | Value |
|---|---|
| Total experiment nodes in the Research Map | 22 |
| Total individual training runs (all seeds, all nodes) | 49 |
| Total training wall-clock, summed across every run | ~1.41 hours (5,077 s) |
| Total LLM tokens (Gemini free tier, Phase 4 only) | 4,947 — **$0 real cost** |
| Total GPU-hours | 0.0 — CPU-only throughout the entire project |
| Manual interventions (Phase 0's own automatic run) | 0 |

Both figures are far under the §2.3 "Limits" caps (50 iterations / 6h per
run) even summed together — compute was never close to the binding
constraint on this benchmark, consistent with the Problem Statement's own
framing ("compute is deliberately not the binding constraint... 100
iterations of the official baseline take about 28 min on a single CPU
core").

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
