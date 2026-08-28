"""Generates `submission_valid.csv` / `submission_test.csv` from whatever the
persistent Research Map (P1/P4's cross-run memory, `logs/research_map.json`)
currently considers a *confirmed* best -- closing a real gap: `run.py`'s own
submission step (Phase 0) only ever looks at its own single run's flat
`PREDEFINED_EXPERIMENTS` loop, so it has no way to know P1's `deepfm_regularized`
or a future Phase 4 find even exists. Before this script, promoting a
Research-Map discovery into `submission_*.csv` was a manual, undocumented
step (see the note in README.md's Results section) -- this makes it a real,
re-runnable, auditable one.

Deliberately uses `ResearchMap.best_confirmed_node()`, not `best_node()`:
the raw leaderboard can be led by a node that's numerically on top only by
noise (this is exactly what happened live -- `features_v1` out-scored
`deepfm_regularized` by +0.0002 valid primary on 3 seeds, well inside the
significance bar, and was correctly tagged `noise_floor` by
agent/diagnosis.py; shipping it as "the improvement" would have been a false
claim). `best_confirmed_node()` walks the lineage back to the nearest node
that's an actual, statistically real win.

Usage:
    python tools/generate_submission.py [--data_dir ...] [--node_id ID]

`--node_id` overrides auto-selection (useful to force-submit a specific,
already-validated node instead of whatever's currently on top).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.evaluator import EvalResult, baseline_deltas  # noqa: E402
from agent.experiment import run_experiment  # noqa: E402
from agent.paths import DEFAULT_DATA_DIR, EXPERIMENTS_DIR, LOGS_DIR  # noqa: E402
from agent.research_map import ResearchMap  # noqa: E402
from agent.submission import write_and_validate  # noqa: E402


def _find_cached_predictions(config_id: str, seed: int) -> tuple[Path, dict] | tuple[None, None]:
    """Scans every run's `experiments/<run_id>/iter_*/` folder for one that
    already trained this exact `config_id` at this exact `seed` and cached
    its predictions -- avoids a pointless retrain if any earlier P0/P1/P4
    round already did the work. Returns (predictions.npz path, that
    iteration's results.json dict) or (None, None) if nothing matches."""
    if not EXPERIMENTS_DIR.exists():
        return None, None
    for run_dir in sorted(EXPERIMENTS_DIR.iterdir(), reverse=True):  # newest run first
        if not run_dir.is_dir():
            continue
        for iter_dir in run_dir.iterdir():
            results_path = iter_dir / "results.json"
            npz_path = iter_dir / "predictions.npz"
            if not (results_path.exists() and npz_path.exists()):
                continue
            try:
                results = json.loads(results_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if results.get("config_id") != config_id:
                continue
            per_seed = (results.get("metrics") or {}).get("per_seed") or []
            if any(r.get("seed") == seed for r in per_seed):
                return npz_path, results
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--node_id", default=None,
                     help="Force a specific Research Map node instead of best_confirmed_node()")
    args = ap.parse_args()

    map = ResearchMap(LOGS_DIR / "research_map.json")
    if args.node_id:
        if args.node_id not in map.nodes:
            print(f"'{args.node_id}' not found in the Research Map.")
            return 1
        node = map.get(args.node_id)
    else:
        node = map.best_confirmed_node()
        if node is None:
            print("Research Map has no confirmed-best `done` node yet -- run run.py/run_p1.py/run_p4.py first.")
            return 1

    raw_best = map.best_node()
    if raw_best is not None and raw_best.node_id != node.node_id:
        print(f"Note: raw leaderboard leader is '{raw_best.node_id}' "
              f"(valid primary={(raw_best.metrics or {}).get('valid', {}).get('primary_mean')}), "
              f"but it isn't a confirmed win (diagnosis_tag='{raw_best.diagnosis_tag}') -- "
              f"submitting the nearest confirmed node instead: '{node.node_id}'.")

    config = node.config
    seed = config.seeds[0]
    print(f"Submitting '{node.node_id}' ({config.model}, seed={seed}) -- "
          f"valid primary={(node.metrics or {}).get('valid', {}).get('primary_mean')}, "
          f"diagnosis_tag='{node.diagnosis_tag}'")

    npz_path, cached_results = _find_cached_predictions(config.id, seed)
    if npz_path is not None:
        print(f"  Reusing cached predictions from {npz_path.parent.name} (no retrain needed).")
        npz = np.load(npz_path)
        valid_scores, test_scores = npz["valid_scores"], npz["test_scores"]
        primary_seed_result = next(r for r in cached_results["metrics"]["per_seed"] if r["seed"] == seed)
        valid_eval = EvalResult.from_dict(primary_seed_result["valid"])
        test_eval = EvalResult.from_dict(primary_seed_result["test"])
    else:
        print(f"  No cached predictions found for '{config.id}' seed={seed} -- retraining...")
        result, model, enc = run_experiment(config, args.data_dir, seed=seed, return_model=True)
        Xva, _, _ = enc["valid"]
        Xte, _, _ = enc["test"]
        valid_scores, test_scores = model.predict(Xva), model.predict(Xte)
        valid_eval, test_eval = result.valid, result.test

    print(f"  valid primary={valid_eval.primary:.4f} (test primary={test_eval.primary:.4f}, "
          f"logged for tracking only -- never used to pick this config)")

    valid_out = REPO_ROOT / "submission_valid.csv"
    test_out = REPO_ROOT / "submission_test.csv"
    valid_report = write_and_validate(valid_scores, valid_out, split="valid", data_dir=Path(args.data_dir))
    test_report = write_and_validate(test_scores, test_out, split="test", data_dir=Path(args.data_dir))
    print(f"  {valid_out.name}: {'OK' if valid_report['ok'] else 'FAILED'} ({valid_report['n_rows']:,} rows)")
    print(f"  {test_out.name}: {'OK' if test_report['ok'] else 'FAILED'} ({test_report['n_rows']:,} rows)")

    deltas = baseline_deltas(test_eval)
    print(f"\n  vs. official FM baseline (test): GAUC {deltas['delta_GAUC']:+.4f}, "
          f"nDCG@5 {deltas['delta_nDCG@5']:+.4f}, primary {deltas['delta_primary_mean']:+.4f}")
    return 0 if (valid_report["ok"] and test_report["ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
