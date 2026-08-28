"""Structured Run Log (P0).

Directly satisfies Deliverables item 3 (Run & Iteration Logs): "each
iteration should record its hypothesis, the code diff, the resulting
metrics, and any error/recovery events." Writes both a per-experiment folder
(human-browsable, the experiments/experiment_NNN/ layout from the brainstorm
doc's Phase 2) and one append-only JSONL (machine-readable -- what
tools/generate_analysis.py and the log-replay dashboard both parse).

Also tracks manual interventions via an explicit, always-logged hook --
judges use that count to score Autonomy (Impact & Relevance), so it must
never be possible for a human nudge to go unrecorded.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, diff_configs
from .convergence import ConvergencePoint
from .recovery import RecoveryEvent


@dataclass
class ManualIntervention:
    reason: str
    timestamp: float = field(default_factory=time.time)
    iteration_id: str | None = None


class RunLogger:
    def __init__(self, logs_dir: Path, experiments_dir_root: Path, run_id: str | None = None):
        """`experiments_dir_root` is the shared parent (e.g. `experiments/`);
        each run gets its own `experiments_dir_root/<run_id>/` subdirectory
        (`self.experiments_dir`, below) so a second `python run.py` invocation
        can never silently overwrite a previous run's `iter_NNN/` folders --
        only `logs/run_log.jsonl` is deliberately append-only/shared across
        runs (each entry already carries its own `run_id` to disambiguate).
        """
        self.logs_dir = Path(logs_dir)
        self.run_id = run_id or time.strftime("run_%Y%m%d_%H%M%S")
        self.experiments_dir = Path(experiments_dir_root) / self.run_id
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.logs_dir / "run_log.jsonl"
        self._prev_config: ExperimentConfig | None = None
        self.manual_interventions: list[ManualIntervention] = []
        self._iteration_counter = 0

    def next_iteration_id(self) -> str:
        self._iteration_counter += 1
        return f"iter_{self._iteration_counter:03d}"

    def log_iteration(self, iteration_id: str, config: ExperimentConfig,
                       metrics: dict[str, Any] | None, wall_time_s: float,
                       recovery_events: list[RecoveryEvent],
                       convergence_point: ConvergencePoint | None, status: str,
                       fidelity_info: dict[str, Any] | None = None) -> dict:
        """`fidelity_info` (P1's Multi-Fidelity Runner -- see
        agent/multi_fidelity.py's MultiFidelityResult): per-stage wall
        times and, if the candidate was killed early, the estimated
        wall-clock saved by not running the remaining stages. Without this,
        every stage below the one that survives (the 1%/10% smoke tests)
        would compute a real kill/escalate decision but leave zero
        persisted record of what that decision cost or saved -- Phase 5's
        "log GPU saved" requirement was silently unmet until this was added.
        """
        code_diff = diff_configs(self._prev_config, config)
        self._prev_config = config

        entry = {
            "run_id": self.run_id,
            "iteration_id": iteration_id,
            "timestamp": time.time(),
            "status": status,  # "ok" | "failed"
            "config_id": config.id,
            "model": config.model,
            "hypothesis": config.hypothesis,
            "parent_id": config.parent_id,
            "code_diff": code_diff,
            "hyperparams": config.hyperparams,
            "metrics": metrics,
            "wall_time_s": wall_time_s,
            "recovery_events": [e.to_dict() for e in recovery_events],
            "convergence": (asdict(convergence_point) if convergence_point else None),
            "fidelity_info": fidelity_info,
        }

        exp_dir = self.experiments_dir / iteration_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "config.json").write_text(config.to_json(), encoding="utf-8")
        (exp_dir / "hypothesis.md").write_text(
            f"# {iteration_id} -- {config.id}\n\n## Hypothesis\n{config.hypothesis}\n\n"
            f"## Notes\n{config.notes or '(none)'}\n\n## Diff vs previous iteration\n"
            f"```json\n{json.dumps(code_diff, indent=2)}\n```\n", encoding="utf-8")
        (exp_dir / "results.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
        log_lines = [f"[{time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))}] "
                     f"status={status} wall_time={wall_time_s:.1f}s"]
        for e in recovery_events:
            log_lines.append(f"  [{e.kind}] attempt {e.attempt}: {e.message}")
        (exp_dir / "logs.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        with open(self.jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

        return entry

    def manual_intervene(self, reason: str, iteration_id: str | None = None) -> None:
        """Explicit hook for a human to nudge the run. Every call is
        automatically counted and logged -- it costs the Autonomy score, so
        it must never be possible to intervene silently."""
        mi = ManualIntervention(reason=reason, iteration_id=iteration_id)
        self.manual_interventions.append(mi)
        with open(self.logs_dir / "manual_interventions.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(mi)) + "\n")

    def finalize_summary(self, convergence_dict: dict, resource_totals: dict,
                          best_submission_metrics: dict | None = None) -> dict:
        summary = {
            "run_id": self.run_id,
            "finished_at": time.time(),
            "num_iterations": self._iteration_counter,
            "manual_intervention_count": len(self.manual_interventions),
            "manual_interventions": [asdict(m) for m in self.manual_interventions],
            "convergence": convergence_dict,
            "resource_totals": resource_totals,
            "best_submission_metrics": best_submission_metrics,
        }
        (self.logs_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
