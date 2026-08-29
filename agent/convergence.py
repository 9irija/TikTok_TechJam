"""Convergence Detector (P0).

Implements the organizer's exact rule, read live from
kuairand-starter-kit/baseline_scores.json["convergence_rule"] rather than
hardcoded -- if the organizers ever republish that file with a different
epsilon/N, this detector picks it up automatically instead of silently
scoring against a stale threshold:

    "A run is converged when the validation primary score has not improved
    by more than epsilon (default 0.002, ~2.5 sigma of the baseline's 5-seed
    std of 0.0008) over the last N (default 3) consecutive iterations."

We track the *best-so-far* validation primary score after every iteration.
An iteration counts as "improved" only if it raises best-so-far by more than
epsilon. Once N consecutive iterations in a row fail to do that, the run is
converged, and the flagged checkpoint is the validation-best one seen at any
point in the whole history (not necessarily the most recent iteration).

Problem Statement Sec 2.3 "Limits" pins two more stop conditions alongside
this rule: "Compute budget: 50 iterations per benchmark run (hard cap; the
convergence rule normally triggers first), plus a 6h wall-clock ceiling per
run as a backstop." Unlike epsilon/N, no organizer-published file carries
these two numbers (baseline_scores.json's "convergence_rule" key only ever
had epsilon/N -- checked directly, not assumed), so MAX_ITERATIONS/
MAX_WALL_CLOCK_S below are named constants to update by hand if a
republished PS ever changes them, not something read live. This only
matters for agent/orchestrator.py's Phase 0 loop, the one place in this
project actually built as "keep iterating the same lineage until epsilon/N
plateau" -- P1's and Phase 4's rounds are a different paradigm (a finite
candidate budget via --max_iterations, defaulting to 3) that structurally
can't approach either cap, so they don't consult this detector at all.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .paths import STARTER_KIT_DIR

MAX_ITERATIONS = 50
MAX_WALL_CLOCK_S = 6 * 3600.0


@dataclass
class ConvergencePoint:
    iteration_id: str
    valid_primary: float
    is_new_best: bool
    stagnant_streak: int  # consecutive iterations (including this one) with improvement <= epsilon


@dataclass
class ConvergenceDetector:
    epsilon: float
    n: int
    history: list[ConvergencePoint] = field(default_factory=list)
    _best_id: str | None = field(default=None, repr=False)
    _best_score: float = field(default=float("-inf"), repr=False)
    _run_start_s: float = field(default_factory=time.time, repr=False)

    @classmethod
    def from_starter_kit(cls, path: Path | None = None) -> "ConvergenceDetector":
        p = path or (STARTER_KIT_DIR / "baseline_scores.json")
        with open(p, encoding="utf-8") as fh:
            rule = json.load(fh)["convergence_rule"]
        return cls(epsilon=rule["epsilon"], n=rule["N"])

    def update(self, iteration_id: str, valid_primary: float) -> ConvergencePoint:
        improved = valid_primary > self._best_score + self.epsilon
        if valid_primary > self._best_score:
            self._best_score = valid_primary
            self._best_id = iteration_id

        prev_streak = self.history[-1].stagnant_streak if self.history else 0
        streak = 0 if improved else prev_streak + 1

        pt = ConvergencePoint(
            iteration_id=iteration_id,
            valid_primary=valid_primary,
            is_new_best=(iteration_id == self._best_id and valid_primary == self._best_score),
            stagnant_streak=streak,
        )
        self.history.append(pt)
        return pt

    def stop_reason(self) -> str | None:
        """None if the run hasn't hit any stop condition yet. Otherwise one
        of "iteration_cap" / "wall_clock_cap" (Problem Statement Sec 2.3's
        hard backstops -- should essentially never fire given how tight
        epsilon/N normally is on this benchmark, but real, checked
        conditions rather than an assumption) or "epsilon_n_rule" (the
        organizer's own rule, the expected/normal case). Checked in that
        order since a run that both plateaued AND ran long should be
        reported as hitting whichever bound is more informative to know
        about -- the hard caps are the more surprising outcome."""
        if len(self.history) >= MAX_ITERATIONS:
            return "iteration_cap"
        if time.time() - self._run_start_s >= MAX_WALL_CLOCK_S:
            return "wall_clock_cap"
        if len(self.history) >= self.n and self.history[-1].stagnant_streak >= self.n:
            return "epsilon_n_rule"
        return None

    def is_converged(self) -> bool:
        return self.stop_reason() is not None

    def best(self) -> tuple[str | None, float]:
        return self._best_id, self._best_score

    def to_dict(self) -> dict:
        bid, bscore = self.best()
        return {
            "epsilon": self.epsilon,
            "N": self.n,
            "max_iterations": MAX_ITERATIONS,
            "max_wall_clock_s": MAX_WALL_CLOCK_S,
            "converged": self.is_converged(),
            "stop_reason": self.stop_reason(),
            "best_iteration_id": bid,
            "best_valid_primary": bscore,
            "num_iterations": len(self.history),
            "trajectory": [
                {"iteration_id": p.iteration_id, "valid_primary": p.valid_primary,
                 "is_new_best": p.is_new_best, "stagnant_streak": p.stagnant_streak}
                for p in self.history
            ],
        }
