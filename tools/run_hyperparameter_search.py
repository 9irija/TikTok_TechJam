"""Hyperparameter search around the Research Map's current confirmed best,
using Optuna (agent/hpo.py) at reduced fidelity, then -- only if the winner
actually beats the base config at that SAME reduced fidelity -- runs the
winning hyperparameters for real (full data, full epoch budget, single
seed first) and records it as a proper Research Map node, same discipline
as every other candidate in this project (diagnosed against its parent,
never trusted off a single seed without `tools/verify_multiseed.py`).

Usage:
    python tools/run_hyperparameter_search.py [--node_id ID] [--n_trials 15]
                                               [--train_fraction 0.15] [--data_dir ...]

`--node_id` overrides which Research Map node to search around (default:
`best_confirmed_node()`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.config import ExperimentConfig  # noqa: E402
from agent.diagnosis import diagnose  # noqa: E402
from agent.experiment import run_experiment  # noqa: E402
from agent.hpo import SEARCH_SPACES, run_search  # noqa: E402
from agent.paths import DEFAULT_DATA_DIR, LOGS_DIR  # noqa: E402
from agent.research_map import ResearchMap  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--node_id", default=None)
    ap.add_argument("--n_trials", type=int, default=15)
    ap.add_argument("--train_fraction", type=float, default=0.15)
    ap.add_argument("--max_epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    args = ap.parse_args()

    map = ResearchMap(LOGS_DIR / "research_map.json")
    base = map.get(args.node_id) if args.node_id else map.best_confirmed_node()
    if base is None:
        print("No confirmed-best node in the Research Map to search around.")
        return 1
    if base.config.model not in SEARCH_SPACES:
        print(f"No search space registered for model '{base.config.model}'. "
              f"Available: {sorted(SEARCH_SPACES)}")
        return 1

    print(f"Searching around '{base.node_id}' ({base.config.model}), "
          f"base hyperparams: {base.config.hyperparams}")
    print(f"{args.n_trials} trials @ train_fraction={args.train_fraction}, "
          f"max_epochs={args.max_epochs}, patience={args.patience} "
          f"(reduced fidelity -- agent/hpo.py's whole point is not spending full compute per trial)\n")

    result = run_search(base.config, args.data_dir, n_trials=args.n_trials,
                         train_fraction=args.train_fraction, max_epochs=args.max_epochs,
                         patience=args.patience)

    print(f"Base config @ reduced fidelity: valid primary = {result.base_score_reduced:.4f}")
    print(f"Best trial @ reduced fidelity:  valid primary = {result.study.best_value:.4f} "
          f"(trial #{result.study.best_trial.number}, params={result.study.best_params})")
    print(f"Search wall-clock: {result.wall_time_s:.1f}s ({args.n_trials} trials)\n")

    if result.best_trial_hyperparams is None:
        print("No trial beat the base config at this fidelity -- current hyperparameters already "
              "look locally optimal for this search space. Nothing new to register.")
        return 0

    node_id = f"{base.node_id}_hpo"
    if node_id in map.nodes:
        print(f"'{node_id}' already in the Research Map -- run again with a fresh --node_id/id scheme "
              f"or delete it first if you want to re-search.")
        return 0

    print(f"Best trial beat the base config at reduced fidelity "
          f"({result.study.best_value:.4f} vs {result.base_score_reduced:.4f}) -- "
          f"running it for real (full data, full epoch budget, seed=0)...")
    full_hp = dict(result.best_trial_hyperparams)
    full_hp["epochs"] = base.config.hyperparams.get("epochs", 40)
    full_hp["patience"] = base.config.hyperparams.get("patience", 4)
    full_hp["batch"] = base.config.hyperparams.get("batch", 8192)

    config = ExperimentConfig(
        id=node_id, model=base.config.model,
        hypothesis=(
            f"Optuna search (agent/hpo.py) around '{base.node_id}''s hyperparameters, "
            f"{args.n_trials} trials at reduced fidelity (train_fraction={args.train_fraction}, "
            f"{args.max_epochs} epochs) -- winner ({result.best_trial_hyperparams}) beat the base "
            f"config's own reduced-fidelity score ({result.study.best_value:.4f} vs "
            f"{result.base_score_reduced:.4f}), now run at full fidelity to see if the gain holds."
        ),
        hyperparams=full_hp, fields=list(base.config.fields), parent_id=base.node_id, seeds=[0],
        edge_type="improve",
        notes=f"agent/hpo.py Optuna search, {args.n_trials} trials @ train_fraction={args.train_fraction}.",
    )

    exp_result = run_experiment(config, args.data_dir, seed=0)
    metrics = {
        "n_seeds": 1,
        "valid": {"GAUC_mean": exp_result.valid.gauc, "GAUC_std": 0.0,
                  "nDCG@5_mean": exp_result.valid.ndcg5, "nDCG@5_std": 0.0,
                  "primary_mean": exp_result.valid.primary, "primary_std": 0.0},
        "test": {"GAUC_mean": exp_result.test.gauc, "GAUC_std": 0.0,
                 "nDCG@5_mean": exp_result.test.ndcg5, "nDCG@5_std": 0.0,
                 "primary_mean": exp_result.test.primary, "primary_std": 0.0},
        "per_seed": [{"experiment_id": node_id, "seed": 0, "best_epoch": exp_result.best_epoch,
                      "epochs_run": exp_result.epochs_run, "valid": exp_result.valid.to_dict(),
                      "test": exp_result.test.to_dict(), "wall_time_s": exp_result.wall_time_s,
                      "num_params": exp_result.num_params, "train_fraction": 1.0}],
    }
    d = diagnose(metrics, base.metrics)
    map.add_node(config, edge_type="improve", parent_id=base.node_id)
    map.update_node(node_id, status="done", metrics=metrics, fidelity="100pct",
                     insight=d.insight, diagnosis_tag=d.tag)

    print(f"\nFull-fidelity result: valid primary = {exp_result.valid.primary:.4f} "
          f"(base: {(base.metrics or {}).get('valid', {}).get('primary_mean')})")
    print(f"Diagnosis vs '{base.node_id}': [{d.tag}] {d.insight}")
    if d.tag == "clear_improvement":
        print(f"\nReal win at single-seed -- verify before trusting it: "
              f"python tools/verify_multiseed.py {node_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
