"""Evaluator Wrapper (P0).

Wraps the organizer's kuairand-starter-kit/evaluate.py exactly -- we import
its `evaluate()` function directly and never reproduce its logic. Getting
GAUC weighting or the zero-positive-user nDCG=0 convention subtly wrong would
silently change every score downstream without anyone noticing, which is
exactly the failure mode the "Tips -- How We Win" section of the brainstorm
doc calls out as how most teams actually lose points. So: zero reimplementation,
by construction -- this file cannot drift from the pinned conventions because
it has no scoring logic of its own to drift.

Also implements the starter kit's own self-check: `--model random` must score
primary ~= 0.4753 (+/- ~0.001) on the fixed test split (0.4834 on valid, per
baseline_scores.json). If that fails, the harness itself is broken and no
downstream result can be trusted -- so `self_check()` is the first thing
run.py calls before any real experiment.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .paths import STARTER_KIT_DIR, ensure_starter_kit_on_path

ensure_starter_kit_on_path()
from evaluate import evaluate as _official_evaluate  # noqa: E402  (organizer's file, unmodified)


@dataclass
class EvalResult:
    gauc: float
    ndcg5: float
    primary: float

    @classmethod
    def from_dict(cls, d: dict) -> "EvalResult":
        return cls(gauc=d["GAUC"], ndcg5=d["nDCG@5"], primary=d["primary"])

    def to_dict(self) -> dict:
        return {"GAUC": self.gauc, "nDCG@5": self.ndcg5, "primary": self.primary}


def score(user_ids: Iterable, labels: Iterable, scores: Iterable) -> EvalResult:
    """Thin pass-through to the organizer's evaluate().

    Coerces inputs to plain Python types before calling it: evaluate.py does
    pure-Python arithmetic with no numpy import, so feeding it numpy scalars
    (e.g. `model.predict()`'s float32 array, iterated) silently turns its
    output into numpy float32 too -- which then fails to JSON-serialize deep
    inside the Structured Run Log. Converting at this one boundary keeps
    evaluate.py itself untouched (still zero reimplementation) while
    guaranteeing everything downstream of this wrapper is JSON-safe.
    """
    u = list(user_ids)
    y = [int(v) for v in labels]
    s = [float(v) for v in scores]
    return EvalResult.from_dict(_official_evaluate(u, y, s))


def load_baseline_scores() -> dict:
    with open(STARTER_KIT_DIR / "baseline_scores.json", encoding="utf-8") as fh:
        return json.load(fh)


def baseline_deltas(agent_test: EvalResult) -> dict[str, float]:
    """Primary-metric scoring formula from the PS (Technical Execution section):
    delta(m) = score_agent(m) - score_baseline(m); score_dataset = mean over m.
    Computed against the official FM baseline's published hidden-test scores.
    """
    b = load_baseline_scores()["scores"]["fm_official"]["test"]
    d_gauc = agent_test.gauc - b["GAUC"]
    d_ndcg = agent_test.ndcg5 - b["nDCG@5"]
    return {
        "delta_GAUC": d_gauc,
        "delta_nDCG@5": d_ndcg,
        "delta_primary_mean": (d_gauc + d_ndcg) / 2.0,  # equal-weighted avg per PS formula
    }


def self_check(data_dir: Path | None = None, seed: int = 0, tol: float = 0.0015) -> dict:
    """Reproduce the starter kit's own sanity check: random scoring on the
    *valid* split should land at primary ~= 0.4834 (baseline_scores.json).
    Raises AssertionError with a diagnostic message if the harness disagrees --
    that means evaluate.py, data.py, or this wrapper drifted, and nothing else
    in the pipeline should be trusted until it's fixed.
    """
    from .paths import DEFAULT_DATA_DIR, data_dir_available
    d = data_dir or DEFAULT_DATA_DIR
    if not data_dir_available(d):
        return {"skipped": True, "reason": f"data_dir not found: {d}"}

    from data import load  # organizer's file
    splits = load(str(d))
    rng = np.random.default_rng(seed)
    rows = splits["valid"]
    users = [x[1] for x in rows]
    labels = [x[6] for x in rows]
    rand_scores = rng.random(len(rows))
    got = score(users, labels, rand_scores)

    expected = load_baseline_scores()["scores"]["random"]["valid"]["primary"]
    ok = abs(got.primary - expected) <= tol
    result = {
        "skipped": False,
        "ok": ok,
        "got_primary": got.primary,
        "expected_primary": expected,
        "tolerance": tol,
    }
    assert ok, (
        f"Evaluator self-check FAILED: random scoring on valid gave primary="
        f"{got.primary:.4f}, expected ~{expected:.4f} (+/-{tol}). The harness "
        f"(evaluate.py wiring, data.py split logic, or this wrapper) is broken "
        f"-- fix before trusting any other result."
    )
    return result
