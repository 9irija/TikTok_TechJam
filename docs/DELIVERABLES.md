# Deliverables Index

Maps this repo directly onto the Problem Statement's own numbered
Deliverables list (item 5 in the PS's own top-level numbering — "1.
Background, 2. Problem Statement, 3. Constraints & Scope, 4. Available
Resources & Data, 5. Deliverables, 6. Judging Criteria") — one entry per
item, exact file paths, nothing paraphrased. Use this as the checklist
when filling in the Devpost form.

## 1. Written Project Description (via Devpost)

**[`docs/WRITTEN_PROJECT_DESCRIPTION.md`](WRITTEN_PROJECT_DESCRIPTION.md)**
— a complete, self-contained write-up covering all 5 of the PS's required
points (problem fit, dev tools, APIs, libraries/frameworks, datasets/
assets), written to be pasted directly into the Devpost form's text box
(Devpost itself is an external site, not a repo file — this is the actual
content that goes there, not just pointers to where the pieces live).

## 2. Public Code/GitHub Repository

**Repo:** [`github.com/9irija/TikTok_TechJam`](https://github.com/9irija/TikTok_TechJam)

| PS requires | Lives at |
|---|---|
| Well-structured, commented code covering all components | `agent/`, `tools/`, `tests/` — every non-trivial function has a docstring explaining *why*, not just what |
| README: Project overview | [`README.md`](../README.md) §"Project overview" |
| README: Setup and installation | [`README.md`](../README.md) §"Setup and installation" |
| README: Steps to reproduce results | [`README.md`](../README.md) §"Steps to reproduce results" |
| README: Limitations + what we'd improve | [`README.md`](../README.md) §"Limitations & what we'd improve with more time" |
| README: Team member contributions | [`README.md`](../README.md) §"Team / contributions" |

## 3. Run & Iteration Logs

| PS requires | Lives at |
|---|---|
| Per-iteration hypothesis, code diff, metrics, error/recovery events | [`logs/run_log.jsonl`](../logs/run_log.jsonl) — raw, machine-readable, append-only, one line per iteration across every phase (30 entries; `"backfilled": true` on the 13 whose own standalone check script logged elsewhere first — see `tools/backfill_run_log.py`) |
| ...same, human-readable | [`logs/analysis_report.md`](../logs/analysis_report.md), regenerated on demand by `python tools/generate_analysis.py` — headline "Project-best" section, the full trajectory, and a per-iteration table |
| ...same, visual/interactive | [`docs/dashboard.html`](dashboard.html) — the persistent Research Map as a clickable tree + trajectory chart, regenerated on demand by `python tools/generate_dashboard.py` |
| Summary of manual intervention count (Autonomy) | [`docs/RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) §"Resource usage" (the flat count) + §"Autonomy breakdown" (the honest per-node source split — mechanical / heuristic-selector / LLM-proposed / human-directed, not just a single number) |

## 4. Final Submission & Results Summary

| PS requires | Lives at |
|---|---|
| Final model output/checkpoint (KuaiRand-Pure schema) | [`submission_valid.csv`](../submission_valid.csv) / [`submission_test.csv`](../submission_test.csv), regenerated on demand by `python tools/generate_submission.py` (resolves `ResearchMap.best_confirmed_node()`, currently `deepfm_mtl_v1`) |
| Results table: validation-best score + delta over baseline | [`docs/RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) §"Results table" |
| Resource usage: LLM tokens + GPU-hours to reach the converged result | [`docs/RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) §"Resource usage" |

## The one-line score, if you only need one number

**0.6046 valid primary** (`deepfm_mtl_v1`, 3-seed verified) — **+0.0028**
over the official baseline on hidden test (0.5974 vs. 0.5946), GAUC +0.0031,
nDCG@5 +0.0026. Full breakdown: [`docs/RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md).

## Task Requirements (PS §2, separate from the Deliverables list above)

| Requirement | Evidence |
|---|---|
| 1. Runs end-to-end, aims to beat the baseline | `python run.py` — self-check → baseline repro → predefined experiments → automatic convergence → submission, all in one command. Beats baseline on hidden test (above). |
| 2. Iterates autonomously across the full stack | `agent/research_map.py` (persistent cross-run memory) + `agent/diagnosis.py` (why, not just what) + `agent/research_strategist.py`/`agent/llm_client.py` (Gemini proposes real experiments on its own — `docs/PHASE4_RESULTS.md`). Reported honestly, not oversold: `docs/RESULTS_SUMMARY.md`'s Autonomy breakdown shows exactly how much of the 30-node history was LLM-proposed vs. human-directed, rather than letting a "0 manual interventions per run" figure imply more end-to-end autonomy than actually happened. |
| 3. Robust operation | `agent/recovery.py` — real subprocess isolation (not just try/except), retry, degraded fallback. Tested against a **genuine** OOM (a real 293 TiB allocation failure, not simulated) and a genuinely broken config — see `tests/test_foundation.py`'s `test_recovery_catches_a_genuine_oom` / `test_recovery_catches_broken_config`. |

## Starter Kit compliance (evaluation script + convergence rule)

- **`evaluate.py` is imported, never reimplemented** — `agent/evaluator.py` calls the organizer's own function directly (`ensure_starter_kit_on_path()`), and self-checks against the published random-scoring reference (`primary ≈ 0.4834` on valid) before trusting any other result. Verified live: `python -c "from agent.evaluator import self_check; print(self_check())"` → `ok: True`.
- **Convergence rule (ε=0.002, N=3) read live from `baseline_scores.json`**, never hardcoded — `agent/convergence.py`. Both the epsilon/N rule and the PS §2.3 hard backstops (50-iteration cap, 6h wall-clock ceiling) are regression-tested to confirm they actually fire (`test_convergence_detector_hits_iteration_cap_before_epsilon_n_rule`, `test_convergence_detector_hits_wall_clock_cap`), not just assumed to work.
- **Label/metric caveat** (worth restating since it's easy to miss): the PS's own prose text says NDCG@10/Recall@50; the Starter Kit's actual pinned code (`evaluate.py`, `baseline_scores.json`) uses `long_view` as the label and GAUC/nDCG@5 as the metrics. This repo follows the Starter Kit — the code-level, pinned source of truth the PS itself says to follow ("the exact label definition and K values are pinned in the Starter Kit so every team solves the same task"). See `CLAUDE.md`'s own flagged caveat section for the full reasoning.
