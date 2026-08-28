# P1 — Research Map, Best-First Selection, Multi-Fidelity Runner & a New Direction

Deliverable-facing summary of P1: what was built, the bugs found and fixed
during its own validation (same honesty standard as
[`PHASE0_FEATURES_AND_IMPROVEMENTS.md`](PHASE0_FEATURES_AND_IMPROVEMENTS.md)),
and the real, unedited results — including two genuine negative results,
because that is what actually happened and negative results are legitimate
scientific output, not a failure to hide.

See [`CLAUDE.md`](../CLAUDE.md) for architecture and the roadmap this
continues from.

---

## 1. What P1 is

Per the brainstorm doc's own "Tips — How We Win": *"Highest-ROI pair if
forced to choose: Research Map + Multi-Fidelity Runner."* P1 builds those
two, plus the two pieces needed to make them actually functional rather
than decorative, plus one genuinely new algorithmic direction to give the
new machinery something real to explore:

| # | Feature | File(s) | What changed vs. Phase 0 |
|---|---|---|---|
| 1 | Research Map / Experiment Tree | `agent/research_map.py` | **Persistent** across runs (`logs/research_map.json`) — Phase 0's log resets every invocation; this doesn't. Tree-structured (draft/improve/debug edges, AIDE-style), not flat. |
| 2 | Metric-Aware Diagnosis Engine | `agent/diagnosis.py` | New — classifies *why* a result looks the way it does (clear_improvement / regression / ranking_tradeoff / noise_floor / mixed + overfitting risk), not just the score. |
| 3 | Multi-Fidelity Runner | `agent/multi_fidelity.py` | New — 1%→10%→100% staged training with kill criteria, instead of always training at full scale. |
| 4 | Best-First Node Selector | `agent/selector.py` | New — deterministic `expected_gain × confidence × novelty ÷ cost` scoring (the brainstorm doc's exact formula), replacing "run the fixed list in declaration order." |
| 5 | FM_BPR (new model) | `agent/model_zoo/fm_bpr.py` | New algorithmic direction — pairwise BPR ranking loss instead of pointwise logloss, the starter kit README's own #1-ranked untested idea. |

Wired together in `agent/p1_orchestrator.py` / `run_p1.py`.

Still **no LLM in the loop** — per the numbered roadmap in `CLAUDE.md`,
that's Phase 4. The candidate pool here is hand-authored (2 configs); what's
new relative to Phase 0 is the scored, best-first *order* they run in, the
persistent cross-run memory, and the staged compute spend — not autonomous
hypothesis generation yet.

---

## 2. Bugs found and fixed during P1's own validation

Same pattern as Phase 0: building the pieces was necessary but not
sufficient — running the full system against real data surfaced three real
issues, two of them substantive enough to have quietly corrupted future
decisions if left in place.

### 2.1 A ~12-hour host sleep inflated a wall-clock measurement by ~480x

**Symptom.** A background validation run reported `wall_time_s: 43371.1`
(~12 hours) for a single FM training run that should take ~90-100s.

**Root cause.** `agent/experiment.py` measured training duration with
`time.time()` (wall-clock). The host machine suspended (sleep) for roughly
12 hours mid-run; `time.time()` continues advancing during a suspend once
the system resumes, so the elapsed-time delta silently absorbed the entire
suspended duration. The training itself was unaffected (it simply paused
and resumed correctly) — only the *measurement* was wrong.

**Fix.** Switched to `time.process_time()` (actual CPU time consumed by the
process). A suspended process accrues zero CPU time, so this measurement
cannot be inflated by a host sleep — and it's also the more semantically
correct choice for what Feasibility & Practicality resource reporting
actually wants (real compute spent), not wall-clock time that might include
the process sitting idle.

**Why this mattered beyond one bad number.** The corrupted value would have
been averaged into the Best-First Selector's `_historical_cost()` for the
`fm` model family (`agent/selector.py`), badly skewing every future cost
estimate for that family. Caught and cleaned before it persisted — the
Research Map was rebuilt from scratch after the fix rather than patched.

### 2.2 Python's per-process hash randomization broke exact reproducibility

**Symptom.** Re-running the identical `fm_bpr_default` config (same seed,
same data) produced slightly different results across separate process
invocations (~0.0002 spread on valid primary) — small, but a real
reproducibility gap in a system whose whole design principle is "seeded and
deterministic."

**Root cause.** `agent/model_zoo/fm_bpr.py`'s `build_user_pos_neg_index`
built its output dict by iterating `pos.keys() & neg.keys()` — a Python
`set` of string user IDs. Python randomizes string hash seeds per process
by default (`PYTHONHASHSEED`), so this set's iteration order (and therefore
which user a given `rng`-sampled *index* lands on in `sample_bpr_batches`)
silently differed between runs, even with an identical `numpy` seed.

**Fix.** Iterate `sorted(pos.keys() & neg.keys())` instead of the raw set.
Verified directly: the exact same synthetic input now produces byte-
identical sampled batches across two independent process invocations (see
the test added for this, `test_fm_bpr_forward_backward_reduces_loss` plus a
standalone before/after check during debugging).

A very small residual variation (~0.0001, roughly an order of magnitude
below the organizer's own 0.0008 baseline std) remained after this fix in
full end-to-end runs — traced as far as confirming the sampling mechanism
itself is now fully deterministic, but not fully root-caused further given
its size is immaterial to every conclusion in this document. Documented
honestly rather than either overclaiming perfect reproducibility or hiding
the loose end.

### 2.3 The Diagnosis Engine mislabeled a real, statistically significant effect as noise

**Symptom.** While seeding P1's Research Map from Phase 0's own validated
log, the Diagnosis Engine tagged `deepfm_default` and `deepfm_wider` as
`noise_floor` versus the FM baseline — despite
[`PHASE0_FEATURES_AND_IMPROVEMENTS.md §4`](PHASE0_FEATURES_AND_IMPROVEMENTS.md#4-final-validated-phase-0-result)
separately establishing that exact comparison as a real, ~8.9σ effect using
proper 3-seed statistics.

**Root cause.** `diagnose()` compared point-estimate mean deltas against a
single flat threshold (the organizer's convergence epsilon, 0.002) with no
regard for how many seeds — and therefore how much statistical confidence —
backed each side of the comparison. Epsilon answers "is this practically
meaningful for convergence"; it does not answer "is this distinguishable
from zero," which is a different question once real per-seed variance data
exists (as it does for every 3-seed Phase 0 node).

**Fix.** `agent/diagnosis.py`'s new `_significance_bar()`: when either side
of a comparison has ≥2 seeds, the bar for calling a delta real becomes
`2 × combined standard error of the mean` (computed from the `_std` and
`n_seeds` fields already sitting in every aggregated metrics dict) instead
of the flat epsilon — which can be *tighter or looser* than epsilon
depending on how noisy the actual comparison is, not clamped to never go
below it (an earlier version of this fix used `max(eps, 2×SEM)`, which is
mathematically incapable of ever flagging a below-epsilon delta as
significant — caught before it shipped). Falls back to flat epsilon only
when there isn't enough seed data on either side to compute a std at all
(the honest, only-available option for a single-seed comparison).

**Result after the fix**, re-diagnosing the same Phase 0 nodes:
`deepfm_default` now reads `mixed` (GAUC's delta, +0.0017, clears its
seed-aware bar of 0.0006; nDCG@5's delta, +0.0005, falls just short of that
same bar) rather than a blanket `noise_floor` — a more honest and more
granular characterization than either the original bug (falsely "noise") or
a hoped-for "clear_improvement" would have been. `deepfm_wider` still reads
`noise_floor`, correctly — its comparison is against `deepfm_default` (its
actual parent in the tree), and the two are genuinely close to each other
(the point already made in Phase 0's own write-up: their 0.0002 gap is
smaller than either model's own seed-to-seed noise).

---

## 3. The real round: results, unedited

Ranking (Best-First Node Selector, both candidates being fresh directions
with no prior family history, so scores are close):

```
#1 score=0.000005  fm_bpr_default   (novelty=1.00 -- new family; confidence=0.30 -- no prior data)
#2 score=0.000003  fm_wider_k32     (novelty=0.33 -- fm family already has 2 nodes; confidence=0.60)
```

Both ran (in that best-first order, through the full 1%→10%→100%
Multi-Fidelity ladder — neither was killed early, both survived to 100%):

| Config | Valid primary | vs. parent | Diagnosis | Wall-clock (CPU time) |
|---|---|---|---|---|
| `fm_bpr_default` (BPR loss, k=16) | 0.5981 | **−0.0034** (parent: FM baseline, 0.6015) | `mixed`, overfitting_risk flagged (best epoch at ~33% of training length) | 136.8s |
| `fm_wider_k32` (FM, k=32) | 0.6009 | −0.0006 | `noise_floor` (within seed-aware significance bar) | 122.9s |

**Neither candidate beat the baseline.** Both are genuine negative results,
reported exactly as measured, not adjusted or omitted:

- **BPR pairwise loss underperformed pointwise FM here**, and the Diagnosis
  Engine's overfitting flag points at a specific, actionable cause: the
  best validation epoch landed early (~33% of the training run) across
  seeds, meaning later epochs kept improving pairwise ranking loss on the
  *sampled training pairs* without any corresponding validation benefit.
  The natural next-round candidate this insight generates: the same BPR
  loss with fewer max epochs / stronger L2, or a learning rate tuned for
  BPR's different gradient landscape rather than reusing pointwise FM's
  `lr=0.001` unchanged.
- **k=32 independently reproduces the starter kit's own documented dead
  end** — their ablation (`ablation_features.py`) already found k=8/16/32
  barely move the score; this run confirms it again, on this codebase's own
  harness rather than trusting the starter kit's separate script. That's a
  genuinely useful (if unglamorous) outcome: the Research Map now has *our
  own* confirmation of a known dead end, which the Best-First Selector's
  regression-penalty logic (`agent/selector.py`'s `_expected_gain`) can use
  to deprioritize future pure-capacity candidates in the `fm` family.

This is exactly the "reflect → revise" loop working as designed — a real
hypothesis was tested, it didn't pay off, and the system produced a
specific, actionable diagnosis instead of just a lower number. It's also
the precise beat the brainstorm doc's own demo narrative calls out:
*"Tested temporal features. Performance dropped. Agent recorded insight and
avoided similar experiments."* — same shape, applied to a real result
instead of a hypothetical one.

### Resource usage, this round

| | Value |
|---|---|
| Candidates run | 2 (both survived to 100% fidelity — neither was cheap to kill, since neither was *broken*, just not better; see §4 on multi-fidelity's actual value here) |
| Manual interventions | 0 |
| Wall-clock (CPU time, both candidates) | 259.7s (~4.3 min) |
| LLM tokens | 0 (no LLM in the loop yet) |
| GPU-hours | 0.0 (CPU-only numpy) |

---

## 4. Honest note: multi-fidelity didn't get to show off here, and that's fine

Both candidates survived every stage of the 1%→10%→100% ladder — neither
was killed early. That's not the Multi-Fidelity Runner failing to do its
job; the kill criteria (`agent/multi_fidelity.py`'s `_floor_for_stage`) are
deliberately lenient — "must beat random scoring," not "must beat the
baseline" — because the mechanism's real purpose is catching *broken*
ideas (a bug, a NaN, a config that can't learn anything) cheaply, not
prejudging legitimately-uncertain-but-sound ones before they've seen
enough data. `fm_bpr_default` and `fm_wider_k32` are both sound, working
ideas that simply didn't beat the baseline — multi-fidelity correctly let
both run to completion rather than second-guessing a legitimate experiment.

The mechanism's actual safety-net value is demonstrated separately and
concretely: `test_multi_fidelity_kills_a_broken_config`
(`tests/test_foundation.py`) verifies a genuinely broken config (an
unresolvable model name) gets killed at the cheapest (1%) stage rather than
burning a full-scale run to find out. That test passing is the real
evidence the mechanism works; this round simply didn't need it.

---

## 5. Why this matters, per judging criterion

- **Technical Execution (Robustness).** §2.1 and §2.2 are genuine bugs
  this pass's own validation found and fixed before they could quietly
  corrupt downstream decisions (the Selector's cost model; every future
  diagnosis) — the same "found it, fixed it, verified it" pattern as
  Phase 0's BLAS bug, not a hypothetical robustness story.
- **Innovation & Problem Insight.** The BPR direction is cited directly
  from the starter kit's own stated priority ranking, not a generic model
  tweak — and the honest negative result plus its specific diagnosis
  (overfitting, not "just didn't work") is exactly what this criterion asks
  for: *"Judged on what the agent identified as worth trying and why — not
  on implementation."*
- **Impact & Relevance (Autonomy).** The Best-First Selector picked which
  candidate to run first from a computed score with a logged reasoning
  string, not source-order — a small but real step from "run a fixed list"
  toward "decide what to try next," still short of Phase 4's LLM-driven
  version but a real, working intermediate stage.
- **Feasibility & Practicality.** §2.1's fix protects every future
  resource-usage number this system reports from an entire class of
  environmental measurement error (host sleep/suspend), not just the one
  instance that surfaced it.

---

## 6. Honest limitations / what's next

- **One round, not a loop.** `p1_candidate_pool()` is a fixed, hand-authored
  list of 2 candidates; a second `python run_p1.py` invocation today finds
  nothing new to run (both are already in the map). A real iterative loop
  needs either more hand-authored "improve" templates reacting to each
  round's diagnosis (e.g., automatically proposing a lower-epoch BPR variant
  after §3's overfitting finding) or Phase 4's LLM to generate genuinely new
  candidates from the accumulated insights — neither is built yet.
- **DeepFM_BPR not built.** Pairwise loss combined with the DeepFM
  architecture (rather than plain FM) is untested — a natural next
  candidate, and cheap to add given `fm_bpr.py`'s pattern.
- **BPR hyperparameters were reused from pointwise FM unchanged**, per
  §3's own diagnosis, likely not appropriate for BPR's different gradient
  landscape — a natural, diagnosis-informed next candidate, not a dead end
  for the BPR direction overall.
- **Residual ~0.0001 nondeterminism** in full BPR runs, noted in §2.2, not
  fully root-caused given its immaterial size.
- **The Best-First Selector's cost/gain model is a simple heuristic**, not
  learned or calibrated — reasonable for a first pass with almost no
  historical data per family, but worth revisiting once the Research Map
  has enough nodes per family for the "extrapolate from parent's own gain"
  logic to have real signal to work with.
