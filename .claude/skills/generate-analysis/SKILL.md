---
name: generate-analysis
description: Regenerate the Run & Iteration Log analysis report (logs/analysis_report.md) from logs/run_log.jsonl and summarize it for the user. Use when the user asks for the analysis report, the run log summary, or to see how the last run went.
---

# Generate Analysis

`logs/analysis_report.md` is the human-readable form of the Run & Iteration
Log deliverable (PS Deliverables item 3: hypothesis, code diff, metrics,
error/recovery events per iteration + a manual-intervention count).

## Steps

1. Run `python tools/generate_analysis.py --stdout` from the repo root.
2. If `logs/run_log.jsonl` doesn't exist yet, tell the user no run has
   happened yet and offer to run one (see the `run-phase0` skill) rather
   than fabricating numbers.
3. This script only *formats* what `agent/run_logger.py` already wrote —
   it must never recompute a metric itself. If a number looks wrong, the
   bug is in `agent/evaluator.py` or `agent/run_logger.py`, not here.
4. Summarize for the user: convergence status + validation-best config,
   the per-iteration table highlights, any error/recovery events, and the
   manual intervention count (this is what judges use to score Autonomy —
   flag if it's non-zero and why).
5. If the user wants a visual/shareable version (not just the markdown
   report) and a run with real multi-iteration history exists, that's the
   log-replay dashboard idea from `CLAUDE.md`'s roadmap (P2, pulled
   forward) — load the `dataviz` skill before writing any chart code, and
   `artifact-design` before publishing it as an Artifact.
