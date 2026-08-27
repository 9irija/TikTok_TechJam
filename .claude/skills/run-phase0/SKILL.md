---
name: run-phase0
description: Run the Phase 0 autonomous ML research agent loop end-to-end against KuaiRand-Pure (evaluator self-check, predefined experiments, convergence check, submission generation) and report results. Use when the user asks to run the agent, run the pipeline, run an experiment, or "run phase 0".
---

# Run Phase 0

Runs `python run.py` from the repo root (`c:\Users\mgiri\Desktop\tiktok_techjam`)
and reports what happened. See `CLAUDE.md` for full architecture context.

## Steps

1. Confirm the dataset is present: `kuairand-starter-kit/KuaiRand-Pure/data/`
   should contain the 6 CSVs. If missing, download it first (see CLAUDE.md
   "Quick start" for the exact `curl`/`tar` commands — ~47MB, public, no auth).
2. Run `python run.py` (add `--skip_submission` for a faster smoke run, or
   `--timeout_s N` to change the per-experiment recovery budget). This takes
   several minutes: each predefined experiment trains in an isolated
   subprocess that reloads and re-encodes the ~1.14M-row training split.
3. If it fails, read the traceback: the most common categories are (a) a
   JSON-serialization bug from a numpy scalar leaking somewhere it shouldn't
   (fix at the boundary where numpy meets `agent/evaluator.py`'s `score()`
   or `agent/run_logger.py`'s JSON writes — don't scatter `float()` casts
   everywhere), or (b) a genuinely broken experiment config, in which case
   Failure Recovery should have caught it — check `logs/run_log.jsonl` for
   a `"status": "failed"` entry with `recovery_events` before assuming the
   whole run needs a fix.
4. After a successful run, regenerate the analysis report:
   `python tools/generate_analysis.py --stdout` and summarize the key
   numbers for the user: convergence status, validation-best config +
   score, delta vs. the official FM baseline, manual intervention count,
   wall-clock/token/GPU-hour totals.
5. Never edit `kuairand-starter-kit/evaluate.py`, `data.py`, or `submit.py`
   — those are the organizer's pinned files. If a run's numbers look wrong,
   the bug is in `agent/`, not there.
