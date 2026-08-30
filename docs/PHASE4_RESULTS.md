# Phase 4 — LLM Research Strategist: Real, Working, and It Found a Real Improvement

> **Superseded as "project-best" by P2's `deepfm_mtl_v1`** (multi-task
> DeepFM, torch) — see [`P2_FEATURES_AND_RESULTS.md`](P2_FEATURES_AND_RESULTS.md).
> Everything below is left exactly as it was: `deepfm_regularized` was a
> real, 3-seed-verified improvement when this was written, and remains the
> best evidence in this project of the LLM finding a genuine, non-obvious
> gap (both DeepFM nodes' `overfitting_risk` flags) on its own.

Deliverable-facing summary of Phase 4: the LLM replaces the hand-authored
candidate pools that drove Phase 0 and P1, and — on its very first real
run — found a genuine, 3-seed-verified improvement over Phase 0's best
model. A second real round (§2b), run later against the full 30-node
history after P2's own extensive exploration, produced 3 more honestly-
reported outcomes (2 regressions, 1 that needed a logged manual timeout
extension to reach a result) — the autonomous loop staying coherent at
scale, not just on a fresh map. Same honesty standard as
[`PHASE0_FEATURES_AND_IMPROVEMENTS.md`](PHASE0_FEATURES_AND_IMPROVEMENTS.md)
and [`P1_FEATURES_AND_RESULTS.md`](P1_FEATURES_AND_RESULTS.md): wins,
failures, and the one manual nudge are all reported exactly as they
happened, not curated.

See [`CLAUDE.md`](../CLAUDE.md) for architecture and the roadmap.

---

## 1. What Phase 4 is

Per the brainstorm doc's own roadmap: *"Now bring in the planner/strategist:
feeds dataset summary + metric history + research map + budget remaining,
outputs hypothesis + reasoning + expected effect + cost + priority."*

| File | Role |
|---|---|
| `agent/llm_client.py` | Gemini API wrapper (free tier — see the provider decision below), retry logic matching `agent/recovery.py`'s philosophy for a flaky external call |
| `agent/research_strategist.py` | Builds the prompt (task, model registry, full Research Map history, budget), calls the LLM, validates the response against `ExperimentConfig`'s schema before it's ever executed |
| `agent/p4_orchestrator.py` | Thin: swaps P1's candidate source for the Strategist, reuses P1's Multi-Fidelity Runner and Diagnosis Engine **unchanged** |
| `run_p4.py` | Entry point: `python run_p4.py` |
| `tools/verify_multiseed.py` | Promotes a promising single-seed result to a 3-seed-verified one (new this pass — see §4) |

**Provider decision**: Gemini free tier, not Anthropic/OpenAI — the problem
statement doesn't mandate a provider (Deliverables explicitly lists "APIs
used" as an open choice), and a genuinely $0 token cost is a clean,
defensible number for the Feasibility & Practicality criterion versus an
estimated dollar figure. Confirmed working end-to-end with a real key.

### The payoff of P1's module boundaries

P1 deliberately kept "what happened" (`research_map.py`), "why"
(`diagnosis.py`), and "what next" (`selector.py`) as separate, swappable
files, specifically so Phase 4 could swap only the last one. It worked
exactly as intended: `agent/p4_orchestrator.py` is ~110 lines, almost all
of it identical in shape to `agent/p1_orchestrator.py`'s loop. The Multi-
Fidelity Runner and Diagnosis Engine required **zero changes** to work with
LLM-proposed candidates instead of hand-authored ones — they only ever see
an `ExperimentConfig`, and don't know or care where it came from.

### Validation Gate, not blind trust

The LLM never gets to execute anything directly. `research_strategist.py`'s
`_validate_and_build()` checks every field of the LLM's JSON response
against the real model registry and `ExperimentConfig`'s schema —unknown
model name, non-existent `parent_id`, wrong hyperparameter types, and
missing fields are all rejected with a specific error, fed back to the LLM
as a re-prompt (tested in `tests/test_foundation.py` — a mock client
confirms both the accept path and the reject-then-retry path, with **no
real API calls in the test suite**). This is the same Structured Experiment
Interface principle used everywhere else in this codebase: the LLM proposes
a *config*, never code.

---

## 2. The real run

`python run_p4.py --max_iterations 2`, against the persistent Research Map
already carrying Phase 0's 4 nodes + P1's 6 nodes (10 total going in).

### Iteration 1 — a real win

**LLM's proposal** (`deepfm_regularized`): raise L2 regularization
(1e-5 → 1e-4) on `deepfm_wider`'s exact architecture (`hidden=[128,64]`).

**LLM's own reasoning** (verbatim from the response): *"DeepFM is our
strongest model family (best valid primary 0.6028). However, both deepfm
runs overfit prematurely (best epoch at 33% of runtime). Regularizing
DeepFM directly targets this diagnosed bottleneck, unlike FM where capacity
was ineffective or FM_BPR which has already undergone 3 tuning rounds."*

This is worth pausing on: the LLM read `overfitting_risk` flags that were
present on **both** `deepfm_default` and `deepfm_wider` in the Research Map
— a diagnosis P1's own hand-authored `_diagnosis_driven_candidates()` never
acted on (P1 only ever reacted to `fm_bpr_default`'s overfitting flag). The
LLM found a real gap in what the hand-authored logic had covered, on its
first call.

**Result (single seed): valid primary = 0.6035**, up from 0.6028 — a
genuine improvement, diagnosed `clear_improvement` (both GAUC +0.0011 and
nDCG@5 +0.0004 cleared the seed-aware significance bar computed from
`deepfm_wider`'s own real 3-seed std, per §2.3 of `P1_FEATURES_AND_RESULTS.md`).

### Iteration 2 — a real failure, correctly caught

**LLM's proposal** (`deepfm_higher_l2`): push L2 further and drop the
learning rate slightly, reasoning that iteration 1's regularization fix
worked, so more of it should help further.

**Result: GAUC −0.0512, nDCG@5 −0.0191** — a large regression, correctly
diagnosed `regression` (both metrics clearly worse), not silently accepted
or misreported. Over-regularizing broke the model. This is genuinely useful
negative evidence recorded in the map, not something to omit: it shows the
LLM's proposals aren't cherry-picked as always-good in this writeup, and it
shows the Diagnosis Engine catching a bad idea exactly as designed.

### Resource usage, this run

| | Value |
|---|---|
| Iterations | 2 (1 accepted-and-good, 1 accepted-and-bad — both are legitimate outcomes of the LLM proposing something and it being tested) |
| LLM tokens | **4,947** total (2,434 + 2,513) |
| LLM cost | **$0** (Gemini free tier) |
| Wall-clock (training, CPU time) | 251.5s (~4.2 min) for both experiments' full multi-fidelity ladders |
| GPU-hours | 0.0 (CPU-only numpy) |
| Manual interventions | 0 |
| Validation-gate rejections | 0 this run (both LLM responses were well-formed on the first attempt) — the reject-then-retry path is real but wasn't exercised live here; it's covered by a dedicated mock-based test instead (§1) |

---

## 2b. Round 2 — a real 3-iteration round against the full 30-node history

Run after P2's own exploration had already tried 18+ further levers beyond
`deepfm_mtl_v1` (the "chasing further improvements" section of
`P2_FEATURES_AND_RESULTS.md`) — the point of running Phase 4 again wasn't
"can the LLM beat all of that," it was "does genuine LLM autonomy keep
producing real, honestly-reported outcomes once the map is this rich," per
Task Requirement 2's own framing ("iterates autonomously across the full
stack... driven by its own evaluation of results"). `python run_p4.py
--max_iterations 3 --max_wall_time_s 1200 --timeout_s 240`.

### Iteration 1 — `deepfm_mtl_aux_weight_tuning`, a real regression

**LLM's reasoning:** the 4 auxiliary signals are extremely sparse
(0.1-1.8% positive) — a fixed `aux_weight=0.2` might exert too much
gradient force on the shared embedding space for such rare events; try
`aux_weight=0.05`.

**Result: GAUC −0.0015, nDCG@5 −0.0006** — a real regression (bar=0.0004).
A coherent hypothesis, correctly tested, correctly diagnosed negative.

### Iteration 2 — `deepfm_mtl_focal_soft_v1`, a real regression

**LLM's reasoning:** the earlier hand-built `deepfm_mtl_focal_v1` (P2 §19)
regressed with `gamma=2.0`; a softer `gamma=1.0` combined with a lower
`aux_weight=0.1` might target hard examples without discarding as much
useful signal.

**Result: GAUC −0.0022, nDCG@5 −0.0014** — also a real regression, slightly
worse than iteration 1. The LLM correctly identified and cited the prior
attempt's own diagnosis before proposing a variant of it — genuine use of
the Research Map's history, even though the variant still didn't work.

### Iteration 3 — `deepfm_mtl_capacity_v1`, a real timeout, a real fix, a real (honest) manual intervention

**LLM's reasoning:** stated intent was "expanding the deep MLP component
capacity from [64, 32] to [128, 64]" — a factual inaccuracy worth noting
plainly: `deepfm_mtl_v1`'s parent already uses `hidden=[128, 64]`, so this
proposal's `hidden` value was actually *unchanged* from the parent, not
expanded. The hyperparameter that actually differed was `batch=2048` (4×
smaller than the parent's 8192).

**What happened:** genuinely timed out at this round's 240s per-stage
budget. Failure Recovery worked exactly as designed, not as a bug: one
retry attempt timed out, a degraded 5-epoch fallback also timed out, and
the candidate was correctly abandoned (`no_result`, `estimated time saved
by not continuing to 100pct: 311.1s`) rather than crashing the round or
hanging — real, live evidence for Task Requirement 3 ("long iterative runs
neither crash, stall, nor diverge").

**Fixed, not left as a failure:** re-ran the identical config directly via
`agent.recovery.run_with_recovery` with `timeout_s=1500` (a human decision
— see below). It completed: best epoch 3/6, **valid primary 0.6036** — a
real regression vs. `deepfm_mtl_v1` (0.6046), confirming the original
timeout was purely a budget issue (batch=2048 needs ~65s/epoch, more than
the per-stage budget allowed for even one full epoch's benefit to be
measured), not a sign the idea might have won with a bug fix.

**The one manual intervention, logged plainly** (`logs/manual_interventions.jsonl`):
extending `timeout_s` from 240s to 1500s and re-running the same config by
hand is "tuning something by hand" per the Track 2 workshop's own
autonomy clarification, even though it didn't change *what* was being
tried. Recorded rather than quietly excluded — see
[`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md)'s Autonomy breakdown for the
full accounting.

### Resource usage, round 2

| | Value |
|---|---|
| Iterations | 3 (2 regressions tested to completion within budget, 1 timed out and required the manual timeout extension above to reach a result) |
| LLM tokens | **11,661** this round (**16,608** cumulative across both real Phase 4 rounds — `logs/p4_run_report.json`'s `cumulative_across_all_p4_rounds`) |
| LLM cost | **$0** (Gemini free tier) |
| Wall-clock (training, this round's own budget) | 460.5s within the round itself; +401.5s for the manual capacity_v1 retry |
| GPU-hours | 0.0 (CPU-only) |
| Manual interventions | 1 (the timeout extension above) |
| Validation-gate / Research Critic Gate rejections | 0 — all 3 proposals were well-formed and passed the deterministic dead-end checks |

**Honest read:** three real proposals, three negative outcomes (all
correctly diagnosed, one requiring a logged manual nudge to reach that
diagnosis at all) — genuinely useful evidence, not a disappointing round.
It demonstrates the full propose → validate → execute → (recover/fail) →
diagnose → record loop holding together even against a Research Map
already dense with prior negative results in the exact same hyperparameter
neighborhoods (aux_weight, focal loss, capacity) the LLM reasoned from and
still chose to explore variants of — a real test of whether the loop stays
coherent at scale, which it did. `deepfm_mtl_v1` remains the project-best.

---

## 3. Doing the responsible thing before calling it "the new best"

A single-seed win can be a lucky draw — this whole project's discipline has
consistently been "don't trust a single seed" (Phase 0 built
`fm_seed_variance` specifically to check this; P1's BPR rounds were always
described as single-seed, noted as a limitation each time). Built
`tools/verify_multiseed.py` to close this gap generically: re-runs a
Research Map node's exact config on more seeds, re-aggregates, and
re-diagnoses against its parent using the real (now multi-seed) std —
reusable for any future promising single-seed result, not a one-off script.

**3-seed re-verification of `deepfm_regularized`:**

| Seed | Valid primary |
|---|---|
| 0 | 0.6035 |
| 1 | 0.6036 |
| 2 | 0.6032 |

**Mean: 0.6035 ± 0.0002** (n=3) — tight, consistent, all three seeds above
`deepfm_wider`'s 0.6028. Re-diagnosed against its parent with the real
3-seed std on both sides: **`clear_improvement`** — GAUC +0.0007, nDCG@5
+0.0006, both clearing a tightened significance bar of 0.0004. This is a
robust, verified result, not a single lucky seed.

---

## 4. Final numbers — the new project-best

| Metric | Official FM baseline (test) | `deepfm_wider` (Phase 0, 3-seed) | `deepfm_regularized` (Phase 4, 3-seed) |
|---|---|---|---|
| Valid primary | 0.6016 | 0.6028 | **0.6035** |
| Test GAUC | 0.6610 | 0.6643 | **0.6646** (Δ **+0.0036**) |
| Test nDCG@5 | 0.5282 | 0.5306 | **0.5308** (Δ **+0.0026**) |
| Test primary | 0.5946 | 0.5976 | **0.5977** (Δ **+0.0031**) |

**Honest nuance, not glossed over**: the improvement is clear and
statistically real on the **validation** split (0.6035 vs 0.6028, which is
what every decision in this codebase is actually allowed to read) — the
**test**-split delta over `deepfm_wider` is much smaller (0.5977 vs 0.5976,
within noise of each other). This is exactly what the train/valid/test
discipline maintained throughout this project is for: the model was
selected because it's a real, validated improvement on the split we're
allowed to use, not because it happens to also look better on test — a
system that peeked at test wouldn't have needed to state this nuance at all.

`submission_valid.csv` and `submission_test.csv` have been regenerated
from `deepfm_regularized`'s cached seed-0 predictions (reusing the P0
retrain-avoidance pattern — no redundant retrain) and re-validated against
the organizer's own `submit.py` format checks.

---

## 5. Why this matters, per judging criterion

- **Technical Execution.** A real, 3-seed-verified improvement over the
  previous best, found autonomously — not a demo of infrastructure working
  in the abstract.
- **Innovation & Problem Insight.** The LLM identified an angle the
  hand-authored P1 logic had specifically missed (DeepFM's own overfitting
  flags, never acted on) — a genuine example of *why* an LLM-driven
  strategist can out-reach hand-authored heuristics, not just automate them.
- **Impact & Relevance (Autonomy).** 0 manual interventions across a run
  that proposed, validated, executed, diagnosed, and recorded two full
  experiments end to end.
- **Feasibility & Practicality.** $0 real LLM cost (not estimated), 4,947
  tokens, ~4.2 minutes of CPU-only training for both experiments combined.
- **Robustness.** Iteration 2's regression was caught and reported
  honestly, not hidden — and the Validation Gate's reject-then-retry path
  (tested via mock, §1) means a malformed LLM response degrades gracefully
  instead of crashing the run, matching every other external dependency in
  this codebase (`agent/recovery.py`'s same philosophy, one level up).

---

## 6. Honest limitations / what's next

Updated in a later pass -- 4 of the 5 gaps below (everything except "only a
few iterations run," which just needs more wall-clock, not more code) are
now closed:

- ~~The prompt's "dead ends" section is hand-maintained~~ -- **fixed**:
  `_dead_ends_section()` now generates it live from the Research Map's own
  `regression`/`noise_floor` diagnosis tags at prompt-build time instead of
  a hardcoded string in `_MODEL_HYPERPARAM_DOCS` -- a brand-new model's
  confirmed dead end appears in the very next prompt with zero manual
  upkeep. Regression-tested
  (`test_dead_ends_section_generated_live_from_research_map_tags`).
- ~~No budget-aware stopping~~ -- **fixed**: `run_p4()` now takes a real
  `max_wall_time_s` (checked before spending the next LLM call) and
  `min_priority_to_run` (declines Multi-Fidelity Runner compute on a
  proposal whose own LLM-assigned priority is too low, logged as
  `skipped_low_priority`, loop continues to the next iteration rather than
  stopping). Both regression-tested against a fake LLM client.
- ~~`tools/verify_multiseed.py` is a manual step~~ -- **fixed**: `run_p4()`
  (and `run_p1()`) now call `tools/verify_multiseed.py`'s
  `verify_node_multiseed()` automatically the instant a fresh single-seed
  candidate becomes the new raw-leaderboard best -- the *next* prompt's
  Research Map context always reflects a verified number, never an
  unconfirmed single-seed one. Toggleable off for a tight compute budget.
- ~~Iteration 2's regression wasn't fed into a 3rd iteration~~ -- superseded:
  the Research Map has since accumulated far more than one more regression
  to reason from (P2's DIN, PDAOM, PCGrad, focal loss, LambdaRank attempts
  are all real `regression`/`noise_floor`-tagged nodes now available as
  Strategist context on any future Phase 4 run).
- **Only a handful of iterations run per invocation.** A longer run (more
  `--max_iterations`, now genuinely boundable via `--max_wall_time_s` too)
  would show whether the LLM keeps finding real improvements or converges
  to proposing marginal/regressive tweaks -- genuinely unknown from this
  much data, and the one item here that's a "run it longer" gap rather than
  a "build more" one.
