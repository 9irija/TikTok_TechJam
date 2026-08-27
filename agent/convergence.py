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
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .paths import STARTER_KIT_DIR


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

    def is_converged(self) -> bool:
        if len(self.history) < self.n:
            return False
        return self.history[-1].stagnant_streak >= self.n

    def best(self) -> tuple[str | None, float]:
        return self._best_id, self._best_score

    def to_dict(self) -> dict:
        bid, bscore = self.best()
        return {
            "epsilon": self.epsilon,
            "N": self.n,
            "converged": self.is_converged(),
            "best_iteration_id": bid,
            "best_valid_primary": bscore,
            "num_iterations": len(self.history),
            "trajectory": [
                {"iteration_id": p.iteration_id, "valid_primary": p.valid_primary,
                 "is_new_best": p.is_new_best, "stagnant_streak": p.stagnant_streak}
                for p in self.history
            ],
        }
