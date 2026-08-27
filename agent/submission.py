"""Submission Validator (P0).

Wraps kuairand-starter-kit/submit.py exactly -- writes with its own
`write_submission()` and validates with its own `read_submission()` /
`--check` semantics, so we can never drift from the pinned format (header,
0-based contiguous row_id, (user_id,video_id) alignment, no NaN/Inf) that
the starter kit README says `--check` rejects. Avoids the "disqualified over
a formatting error" failure mode called out in the brainstorm doc's P0 table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .paths import DEFAULT_DATA_DIR, ensure_starter_kit_on_path

ensure_starter_kit_on_path()
from data import load  # noqa: E402
from submit import read_submission, write_submission  # noqa: E402


def write_and_validate(scores: np.ndarray, out_path: Path, split: str = "test",
                        data_dir: Path | None = None) -> dict:
    """Writes `scores` (one per row of `split`, in data.load() row order) as
    a submission CSV, then immediately re-reads it through the organizer's
    own alignment/format checks -- so a bad write is caught before anyone
    downstream (a human, Devpost, the judges' harness) ever sees the file.
    """
    d = data_dir or DEFAULT_DATA_DIR
    splits = load(str(d))
    rows = splits[split]
    if len(scores) != len(rows):
        raise ValueError(f"scores has {len(scores)} rows, split '{split}' has {len(rows)} rows")

    write_submission(str(out_path), rows, scores)
    checked_scores = read_submission(str(out_path), rows)  # raises ValueError on any format issue
    return {
        "ok": True,
        "path": str(out_path),
        "split": split,
        "n_rows": len(checked_scores),
    }


def check_only(path: Path, split: str = "test", data_dir: Path | None = None) -> dict:
    d = data_dir or DEFAULT_DATA_DIR
    splits = load(str(d))
    rows = splits[split]
    try:
        scores = read_submission(str(path), rows)
        return {"ok": True, "path": str(path), "split": split, "n_rows": len(scores)}
    except ValueError as e:
        return {"ok": False, "path": str(path), "split": split, "error": str(e)}
