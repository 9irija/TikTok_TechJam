"""Multi-Fidelity Runner (P1).

"Smoke-tests a new idea on ~1% of the data for one epoch before committing
to a full run, and only escalates to 10%, then 100%, if it looks promising,
so weak ideas get killed cheaply instead of burning GPU-hours finding out
at full scale." The brainstorm doc calls this "the biggest single lever on
GPU-hours -- could be worth more than any model-architecture choice."

Each stage trains on a random, seeded, deterministic subsample of the
*training* split only (agent/experiment.py's `train_fraction`) -- validation
and test always stay full-size, so every stage's score is a real (if
noisier, smaller-N) read on generalization, not a synthetic proxy metric.

Kill/escalate decisions use fixed, cheap floors (the organizer's own
random-baseline reference rung from baseline_scores.json) rather than
comparing against other candidates in the same batch -- that keeps each
stage decision self-contained and stateless: one candidate's fate never
has to wait on the whole cohort finishing. The floors are deliberately
lenient (see `_floor_for_stage`) -- this stage's job is to catch completely
broken ideas cheaply (a bug, a NaN, a config that can't learn anything at
all), not to reject legitimately weak-but-sound ideas before they've seen
enough data to show their strength.

Smoke stages are single-seed by design -- cheap, not noise-robust. A
100%-stage survivor that a run wants to trust as a real result should be
promoted to multi-seed separately (the way fm_seed_variance /
deepfm_default's 3 seeds already work in Phase 0), not inferred from one
multi-fidelity run alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .config import ExperimentConfig
from .evaluator import load_baseline_scores
from .recovery import RecoveryEvent, run_with_recovery

# (fidelity name, train_fraction, max_epochs_override -- None means "use the
# config's own `epochs` hyperparam", i.e. a genuine full-scale run)
STAGES: list[tuple[str, float, int | None]] = [
    ("1pct", 0.01, 5),
    ("10pct", 0.10, 15),
    ("100pct", 1.0, None),
]


@dataclass
class MultiFidelityResult:
    node_id: str
    final_stage: str = "none"          # highest stage actually reached
    killed_at: str | None = None       # stage name if killed, else None (survived to 100pct)
    kill_reason: str | None = None
    stage_results: dict[str, dict[str, Any]] = field(default_factory=dict)  # stage -> aggregated metrics dict
    all_events: list[RecoveryEvent] = field(default_factory=list)
    total_wall_time_s: float = 0.0

    def survived(self) -> bool:
        return self.killed_at is None and "100pct" in self.stage_results

    def final_metrics(self) -> dict[str, Any] | None:
        return self.stage_results.get(self.final_stage)


def _floor_for_stage(stage: str) -> float:
    r = load_baseline_scores()["scores"]["random"]["valid"]["primary"]
    if stage == "1pct":
        return r - 0.01   # only kills a run that's doing *worse* than random -- catches real breakage
    return r + 0.01        # 10pct+: must clearly be learning something above the random floor


def run_multi_fidelity(config: ExperimentConfig, data_dir: str, timeout_s: float = 240.0,
                        max_retries: int = 1, save_predictions_on_survive: str | None = None
                        ) -> MultiFidelityResult:
    """Runs `config` through the staged 1%->10%->100% ladder, killing it at
    the first stage where it looks broken or clearly not-learning. Returns
    a MultiFidelityResult whose `final_metrics()` is the highest-fidelity
    result reached -- callers (agent/orchestrator.py, agent/selector.py)
    should only trust `final_metrics()` when `survived()` is True.
    """
    from .orchestrator import _aggregate_seed_results  # local import: avoids a circular import at module load

    result = MultiFidelityResult(node_id=config.id)
    seed = config.seeds[0]

    for stage_name, frac, epochs_override in STAGES:
        save_to = save_predictions_on_survive if stage_name == "100pct" else None
        r, events = run_with_recovery(
            config, data_dir, seed=seed, timeout_s=timeout_s, max_retries=max_retries,
            train_fraction=frac, max_epochs_override=epochs_override, save_predictions_to=save_to,
        )
        result.all_events.extend(events)

        if r is None:
            result.killed_at = stage_name
            result.kill_reason = "training failed / timed out at this fidelity (see recovery events)"
            return result

        result.total_wall_time_s += r["wall_time_s"]
        agg = _aggregate_seed_results([r])
        result.stage_results[stage_name] = agg
        result.final_stage = stage_name

        primary = agg["valid"]["primary_mean"]
        if not math.isfinite(primary):
            result.killed_at, result.kill_reason = stage_name, "non-finite validation score (NaN/Inf)"
            return result
        if primary < _floor_for_stage(stage_name):
            result.killed_at = stage_name
            result.kill_reason = (f"valid primary {primary:.4f} below the {stage_name} floor "
                                   f"({_floor_for_stage(stage_name):.4f}) -- not learning anything useful")
            return result

    return result
