---
name: validate-submission
description: Validate a submission CSV against the organizer's exact format rules (header, row_id contiguity, user_id/video_id alignment, no NaN/Inf) before it's treated as final. Use when the user asks to check, validate, or finalize a submission file.
---

# Validate Submission

Never eyeball a submission CSV by hand — the organizer's `submit.py --check`
rejects a wrong header, a row-count mismatch, `row_id` gaps, misalignment
against the evaluation split, and non-numeric/NaN/Inf scores, and this repo
must match that exactly (`agent/submission.py` imports `submit.py`'s own
`read_submission`/`write_submission`, never reimplements the format rules).

## Steps

1. From the repo root, run:
   ```
   cd kuairand-starter-kit
   python submit.py --check --split test  ../submission_test.csv
   python submit.py --check --split valid ../submission_valid.csv
   ```
   (adjust the path/split to whatever file the user means).
2. Or, from Python: `agent.submission.check_only(path, split, data_dir)` —
   returns `{"ok": bool, ...}` without needing to shell out.
3. `run.py`'s own final step already calls `agent.submission.write_and_validate()`
   after every full run, so a submission produced by `python run.py` should
   already be format-valid — this skill is for re-checking an existing file
   (e.g. before the actual Devpost/GitHub submission) or a file a human
   edited by hand.
4. If validation fails, report the exact error message from `submit.py` —
   it's already precise about what's wrong (which row, what mismatch).
   Do not "fix" a bad submission by re-deriving row_id/user_id/video_id
   yourself; regenerate it from `agent.submission.write_and_validate()`
   against the correct split instead, since `(user_id, video_id)` is not a
   unique key (3.06% of test rows repeat, up to 12 times) and only
   `data.load()`'s own row order is authoritative.
