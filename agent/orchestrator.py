"""Orchestrator / Planner (P0).

"Drives read -> inspect -> engineer -> train/tune -> evaluate -> reflect; a
fixed short list of predefined experiments is fine at this stage -- no LLM
reasoning yet." Phase 4 (see /CLAUDE.md roadmap) is what replaces
PREDEFINED_EXPERIMENTS with an LLM-driven Research Strategist; this file's
job is to make the *rest of the loop* (recovery, logging, convergence,
submission) solid enough that swapping the planner later is a one-file change.

Also owns:
  * resource totals (wall-clock as the Phase-0 compute proxy; LLM tokens are
    0 here by construction -- there is no LLM call anywhere in this loop)
  * the manual-intervention hook, delegated to RunLogger
  * picking the validation-best config once Convergence Detector fires, and
    handing it to the Submission Validator
"""
from __future__ import annotations

import time
from pathlib import Path
from statistics import mean, pstdev

from .config import BASE_FIELDS, ExperimentConfig
from .convergence import ConvergenceDetector
from .evaluator import baseline_deltas
from .paths import DEFAULT_DATA_DIR, EXPERIMENTS_DIR, LOGS_DIR
from .recovery import run_with_recovery
from .run_logger import RunLogger

# ---------------------------------------------------------------------------
# Predefined experiment list (Phase 0 -- no LLM reasoning). Each hypothesis
# is written the way the Deliverables section asks for: what we intended to
# try and why. iter_001 MUST be the baseline reproduction (Task Requirement 1:
# "Stand up a working end-to-end pipeline and confirm it reaches the official
# baseline's reported validation score" before anything else is trusted).
# ---------------------------------------------------------------------------
PREDEFINED_EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig(
        id="fm_baseline_repro",
        model="fm",
        hypothesis=(
            "Reproduce the official FM baseline exactly (k=16, lr=0.001, batch=8192, "
            "5 fields, seed 0) before writing anything original. If our harness can't "
            "hit the published validation primary (~0.6016), nothing downstream -- "
            "including every later experiment's delta-over-baseline -- can be trusted."
        ),
        hyperparams={"k": 16, "lr": 0.001, "batch": 8192, "epochs": 40, "patience": 4},
        fields=list(BASE_FIELDS),
        parent_id=None,
        seeds=[0],
        notes="Config matches kuairand-starter-kit/baseline_scores.json exactly.",
    ),
    ExperimentConfig(
        id="fm_seed_variance",
        model="fm",
        hypothesis=(
            "Run the identical baseline config on 3 seeds to measure our own harness's "
            "noise floor before accepting any future result as a real improvement. The "
            "organizer's own 5-seed std is 0.0008 against a convergence epsilon of 0.002 "
            "(only ~2.5 sigma of margin) -- a single noisy run can look like a false "
            "improvement or a false convergence, so this is cheap insurance, not busywork."
        ),
        hyperparams={"k": 16, "lr": 0.001, "batch": 8192, "epochs": 40, "patience": 4},
        fields=list(BASE_FIELDS),
        parent_id="fm_baseline_repro",
        seeds=[0, 1, 2],
        notes="Noise-Aware Convergence Check, P1 table -- pulled forward because it's cheap "
              "and protects every later decision from being poisoned by seed noise.",
    ),
    ExperimentConfig(
        id="deepfm_default",
        model="deepfm",
        hypothesis=(
            "Test whether a deep MLP component over the same shared embeddings captures "
            "higher-order, nonlinear feature interactions the FM's pairwise term structurally "
            "can't. Starter kit README headroom item #5 ranks 'change model' behind loss "
            "function / sequence / multi-task changes, but it is the cheapest of the P0 Model "
            "Zoo entries to stand up first and gives later experiments a second model family "
            "to branch from, not just tuning the same FM. Run on 3 seeds (matching "
            "fm_baseline_repro's own noise-floor rigor from fm_seed_variance) rather than 1 -- "
            "a single-seed 'win' over a baseline whose own 5-seed std is 0.0008 is exactly the "
            "false-improvement risk fm_seed_variance exists to guard against, so the model this "
            "run may end up selecting as validation-best deserves the same scrutiny."
        ),
        hyperparams={"k": 16, "hidden": [64, 32], "lr": 0.001, "batch": 8192, "epochs": 40, "patience": 4},
        fields=list(BASE_FIELDS),
        parent_id="fm_baseline_repro",
        seeds=[0, 1, 2],
        notes="Guo et al. 2017 DeepFM architecture; shared field embeddings between the FM and "
              "deep components (see agent/model_zoo/deepfm.py docstring).",
    ),
    ExperimentConfig(
        id="deepfm_wider",
        model="deepfm",
        hypothesis=(
            "If deepfm_default improves on the FM baseline, test whether more MLP capacity "
            "(64,32 -> 128,64) buys further gain. If deepfm_default does NOT improve, this "
            "experiment is a diagnostic: capacity is unlikely to be the bottleneck (the "
            "starter kit's own ablation already showed embedding-dim 8/16/32 barely moves "
            "the score for FM), so a wider net failing too would point at the deep pathway "
            "itself, not under-parameterization -- a concrete negative result for the log."
        ),
        hyperparams={"k": 16, "hidden": [128, 64], "lr": 0.001, "batch": 8192, "epochs": 40, "patience": 4},
        fields=list(BASE_FIELDS),
        parent_id="deepfm_default",
        seeds=[0, 1, 2],
        notes="Same 3-seed rigor as deepfm_default -- this is the wider variant of whichever "
              "model may end up being validation-best, so it gets the same scrutiny.",
    ),
]


class Orchestrator:
    def __init__(self, data_dir: Path | None = None, logs_dir: Path | None = None,
                 experiments_dir: Path | None = None, timeout_s: float = 300.0,
                 max_retries: int = 2):
        self.data_dir = str(data_dir or DEFAULT_DATA_DIR)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.convergence = ConvergenceDetector.from_starter_kit()
        self.logger = RunLogger(logs_dir or LOGS_DIR, experiments_dir or EXPERIMENTS_DIR)
        # Run-scoped (experiments/<run_id>/) -- resolved by RunLogger, not
        # recomputed here, so this can never drift from where the logger
        # actually writes iter_NNN/ folders.
        self.experiments_dir = self.logger.experiments_dir
        self.wall_time_total_s = 0.0
        self.llm_tokens_total = 0  # no LLM in the loop yet -- Phase 4 wires this up for real

    def manual_intervene(self, reason: str) -> None:
        self.logger.manual_intervene(reason)

    def run(self, experiments: list[ExperimentConfig] | None = None) -> dict:
        experiments = experiments if experiments is not None else PREDEFINED_EXPERIMENTS
        run_t0 = time.time()

        for config in experiments:
            iteration_id = self.logger.next_iteration_id()
            per_seed_results = []
            all_events = []
            # Real wall-clock elapsed for the whole iteration (all seeds, all
            # recovery attempts) -- NOT a sum of only-successful in-subprocess
            # training times. A timeout/retry saga can burn minutes of real
            # compute before Failure Recovery gives up on a config; that time
            # is genuinely spent (and would be genuine GPU-hours on a GPU
            # run), so it must count toward the resource totals the
            # Feasibility score is judged on -- silently excluding failed
            # attempts would understate real cost.
            iter_t0 = time.time()

            for seed in config.seeds:
                predictions_path = None
                if seed == config.seeds[0]:
                    # Only the "primary" seed's predictions are ever reused
                    # downstream (run.py retrains seeds[0] for the final
                    # submission) -- caching every seed's predictions would
                    # waste disk for no benefit.
                    predictions_path = str(self.experiments_dir / iteration_id / "predictions.npz")
                result, events = run_with_recovery(
                    config, self.data_dir, seed=seed,
                    timeout_s=self.timeout_s, max_retries=self.max_retries,
                    save_predictions_to=predictions_path,
                )
                all_events.extend(events)
                if result is not None:
                    per_seed_results.append(result)

            iter_wall = time.time() - iter_t0
            self.wall_time_total_s += iter_wall

            if not per_seed_results:
                self.logger.log_iteration(iteration_id, config, metrics=None, wall_time_s=iter_wall,
                                           recovery_events=all_events, convergence_point=None, status="failed")
                continue  # route around: skip to next predefined config, run is not stalled

            metrics = _aggregate_seed_results(per_seed_results)
            cp = self.convergence.update(iteration_id, metrics["valid"]["primary_mean"])
            self.logger.log_iteration(iteration_id, config, metrics=metrics, wall_time_s=iter_wall,
                                       recovery_events=all_events, convergence_point=cp, status="ok")

            if self.convergence.is_converged():
                break

        return self._finalize(run_t0)

    def _finalize(self, run_t0: float) -> dict:
        convergence_dict = self.convergence.to_dict()
        resource_totals = {
            "wall_time_total_s": self.wall_time_total_s,
            "run_wall_time_s": time.time() - run_t0,
            "llm_tokens_total": self.llm_tokens_total,
            "gpu_hours_total": 0.0,  # CPU-only numpy in Phase 0 -- no GPU time to report yet
        }
        summary = self.logger.finalize_summary(convergence_dict, resource_totals)
        return summary


def _aggregate_seed_results(per_seed_results: list[dict]) -> dict:
    def agg(split: str) -> dict:
        gaucs = [r[split]["GAUC"] for r in per_seed_results]
        ndcgs = [r[split]["nDCG@5"] for r in per_seed_results]
        prims = [r[split]["primary"] for r in per_seed_results]
        return {
            "GAUC_mean": mean(gaucs), "GAUC_std": pstdev(gaucs) if len(gaucs) > 1 else 0.0,
            "nDCG@5_mean": mean(ndcgs), "nDCG@5_std": pstdev(ndcgs) if len(ndcgs) > 1 else 0.0,
            "primary_mean": mean(prims), "primary_std": pstdev(prims) if len(prims) > 1 else 0.0,
        }

    valid_agg, test_agg = agg("valid"), agg("test")
    from .evaluator import EvalResult
    test_point = EvalResult(gauc=test_agg["GAUC_mean"], ndcg5=test_agg["nDCG@5_mean"],
                             primary=test_agg["primary_mean"])
    return {
        "n_seeds": len(per_seed_results),
        "valid": valid_agg,
        "test": test_agg,
        "delta_vs_baseline_test": baseline_deltas(test_point),
        "per_seed": per_seed_results,
    }
