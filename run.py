"""Entry point: `python run.py` -- the Phase 0 "dumb autonomous loop".

    evaluator self-check -> baseline reproduction -> predefined experiments
    -> convergence detection -> reuse validation-best's cached predictions
    (retrain only as a fallback) -> submission generation + validation
    -> final report

No LLM reasoning yet (PREDEFINED_EXPERIMENTS is a fixed list -- that's
Phase 4, see /CLAUDE.md roadmap). The point of Phase 0 is proving every
*other* piece -- recovery, structured logging, convergence detection,
submission validation -- is solid before any research reasoning gets
layered on top of it.

Multiprocessing note: Windows only supports the 'spawn' start method, which
re-imports this module in each worker process -- hence the `if __name__ ==
"__main__"` guard below is required, not just style.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agent.config import ExperimentConfig
from agent.evaluator import EvalResult, self_check
from agent.experiment import run_experiment
from agent.orchestrator import PREDEFINED_EXPERIMENTS, Orchestrator
from agent.paths import DEFAULT_DATA_DIR, LOGS_DIR, REPO_ROOT
from agent.submission import write_and_validate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR),
                     help="KuaiRand-Pure/data directory (default: kuairand-starter-kit/KuaiRand-Pure/data)")
    ap.add_argument("--timeout_s", type=float, default=300.0,
                     help="Per-experiment wall-clock budget before Failure Recovery kills and retries it")
    ap.add_argument("--skip_submission", action="store_true",
                     help="Skip retraining + writing the final submission CSVs (faster smoke tests)")
    args = ap.parse_args()

    print("=" * 78)
    print("Phase 0 -- Autonomous ML Research Agent for KuaiRand-Pure")
    print("=" * 78)

    print("\n[1/4] Evaluator self-check (random scoring on valid should be ~0.4834)...")
    check = self_check(Path(args.data_dir))
    if check.get("skipped"):
        print(f"  SKIPPED: {check['reason']}")
        print("  Cannot proceed without data. See docs/TikTok TechJam Hackathon.md 'Starter Kit'")
        print("  section, or CLAUDE.md 'Quick start', for the download command.")
        return 1
    print(f"  OK: got primary={check['got_primary']:.4f} (expected ~{check['expected_primary']:.4f}, "
          f"tol {check['tolerance']})")

    print(f"\n[2/4] Running {len(PREDEFINED_EXPERIMENTS)} predefined experiments "
          f"(no LLM reasoning yet -- Phase 0)...")
    orch = Orchestrator(data_dir=Path(args.data_dir), timeout_s=args.timeout_s)
    summary = orch.run()

    conv = summary["convergence"]
    print(f"\n[3/4] Convergence: {conv['converged']} after {conv['num_iterations']} iteration(s).")
    if conv["best_iteration_id"]:
        print(f"  Validation-best: {conv['best_iteration_id']} "
              f"(valid primary={conv['best_valid_primary']:.4f})")
    print(f"  Manual interventions: {summary['manual_intervention_count']}")
    rt = summary["resource_totals"]
    print(f"  Resource usage -- wall-clock: {rt['wall_time_total_s']:.1f}s | "
          f"LLM tokens: {rt['llm_tokens_total']} | GPU-hours: {rt['gpu_hours_total']}")

    if args.skip_submission or conv["best_iteration_id"] is None:
        print("\n[4/4] Skipped submission generation (--skip_submission or no successful iteration).")
        _print_footer(summary, orch.experiments_dir)
        return 0

    best_iter_dir = orch.experiments_dir / conv["best_iteration_id"]  # run-scoped: experiments/<run_id>/iter_NNN/
    best_config = ExperimentConfig.from_dict(json.loads((best_iter_dir / "config.json").read_text(encoding="utf-8")))
    predictions_path = best_iter_dir / "predictions.npz"

    if predictions_path.exists():
        # Efficiency: '{best_iteration_id}' already trained this exact
        # config+seed during the loop above (agent/orchestrator.py caches
        # the primary seed's predictions precisely so this step never has
        # to retrain from scratch just to materialize a submission CSV).
        print(f"\n[4/4] Reusing cached predictions from '{conv['best_iteration_id']}' "
              f"(no retrain needed) -- writing + validating submission CSVs...")
        npz = np.load(predictions_path)
        valid_scores, test_scores = npz["valid_scores"], npz["test_scores"]
        results_json = json.loads((best_iter_dir / "results.json").read_text(encoding="utf-8"))
        primary_seed_result = next(r for r in results_json["metrics"]["per_seed"]
                                    if r["seed"] == best_config.seeds[0])
        valid_eval = EvalResult.from_dict(primary_seed_result["valid"])
        test_eval = EvalResult.from_dict(primary_seed_result["test"])
    else:
        # Fallback: predictions weren't cached (e.g. the winning iteration's
        # primary-seed attempt only succeeded via the degraded fallback,
        # which deliberately never caches -- see recovery.run_with_recovery).
        # Retraining here is the correctness-preserving fallback, not the
        # common path.
        print(f"\n[4/4] No cached predictions for '{conv['best_iteration_id']}' -- retraining to "
              f"materialize predictions, then writing + validating submission CSVs...")
        result, model, enc = run_experiment(best_config, args.data_dir, seed=best_config.seeds[0],
                                             return_model=True)
        Xva, _, _ = enc["valid"]
        Xte, _, _ = enc["test"]
        valid_scores, test_scores = model.predict(Xva), model.predict(Xte)
        valid_eval, test_eval = result.valid, result.test

    print(f"  '{best_config.id}': valid primary={valid_eval.primary:.4f} "
          f"(test primary={test_eval.primary:.4f}, logged for tracking only -- "
          f"never used to pick this config)")

    valid_out = REPO_ROOT / "submission_valid.csv"
    test_out = REPO_ROOT / "submission_test.csv"
    valid_report = write_and_validate(valid_scores, valid_out, split="valid", data_dir=Path(args.data_dir))
    test_report = write_and_validate(test_scores, test_out, split="test", data_dir=Path(args.data_dir))
    print(f"  {valid_out.name}: {'OK' if valid_report['ok'] else 'FAILED'} "
          f"({valid_report['n_rows']:,} rows)")
    print(f"  {test_out.name}: {'OK' if test_report['ok'] else 'FAILED'} "
          f"({test_report['n_rows']:,} rows)")

    from agent.evaluator import baseline_deltas
    final_summary = orch.logger.finalize_summary(
        summary["convergence"], summary["resource_totals"],
        best_submission_metrics={
            "config_id": best_config.id,
            "valid": valid_eval.to_dict(),
            "test": test_eval.to_dict(),
            "delta_vs_baseline_test": baseline_deltas(test_eval),
        },
    )
    _print_footer(final_summary, orch.experiments_dir)
    return 0


def _print_footer(summary: dict, experiments_dir) -> None:
    print("\n" + "=" * 78)
    print(f"Run complete. Structured logs: {LOGS_DIR / 'run_log.jsonl'}")
    print(f"Per-experiment folders:        {experiments_dir}  (run-scoped -- never overwritten by a later run)")
    print(f"Run summary:                   {LOGS_DIR / 'run_summary.json'}")
    print("Next: python tools/generate_analysis.py")
    print("=" * 78)


if __name__ == "__main__":
    raise SystemExit(main())
