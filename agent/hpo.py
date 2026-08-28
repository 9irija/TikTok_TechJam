"""Hyperparameter Search (P2 stretch) -- Optuna, not hand-rolled.

The one place in this project's own limitations list where reaching for an
existing tool is unambiguously the right call instead of building from
scratch: unlike modeling (where a heavier framework's own train/eval loop
risked silently drifting from `evaluate.py`'s pinned scoring -- see
CLAUDE.md's RecBole/TorchRec reasoning), a search *orchestrator* sits
entirely outside the modeling/scoring logic. Optuna picks which
hyperparameters to try next; `agent/experiment.py`'s `run_experiment` (the
same function every other candidate in this project runs through) still
does the actual training and, critically, still only ever reads
`.valid.primary` -- Optuna never sees `.test`, so this can't become a
backdoor around train/valid/test discipline.

Runs at REDUCED fidelity (a `train_fraction` < 1.0 subsample, a capped
epoch budget) -- the same "don't spend full compute on an unproven
candidate" principle as `agent/multi_fidelity.py`'s 1%/10%/100% staging,
just applied across many candidates instead of one. A trial's score is
only ever compared to the *base config's own* score at the identical
reduced fidelity (never to a different config's full-fidelity number --
that would be comparing across two different noise regimes) to decide
whether the winning hyperparameters are worth a full, 3-seed-verified run.

Deliberately in-process (no subprocess-per-trial isolation via
agent/recovery.py, unlike every other experiment path in this codebase):
a crash inside one trial's objective just loses that trial's score
(caught, reported to Optuna as a large negative value so the sampler
learns to avoid that region), not the whole search -- correct behavior for
a many-cheap-trials search loop, and a real, disclosed departure from this
project's usual subprocess-isolation-everywhere norm, not an oversight.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable

import optuna

from .config import ExperimentConfig
from .experiment import run_experiment

optuna.logging.set_verbosity(optuna.logging.WARNING)  # Optuna's own per-trial spam -- we print our own summary


# One search space per model family that's actually been through this --
# adding a new model here is one dict entry, not a new script.
SEARCH_SPACES: dict[str, Callable[[optuna.Trial, dict[str, Any]], dict[str, Any]]] = {}


def _deepfm_family_space(trial: optuna.Trial, base_hp: dict[str, Any]) -> dict[str, Any]:
    hp = dict(base_hp)
    hp["k"] = trial.suggest_categorical("k", [8, 16, 32, 64])
    hidden_choice = trial.suggest_categorical("hidden", ["64_32", "128_64", "256_128"])
    hp["hidden"] = [int(x) for x in hidden_choice.split("_")]
    hp["lr"] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hp["l2"] = trial.suggest_float("l2", 1e-6, 1e-3, log=True)
    return hp


def _deepfm_mtl_space(trial: optuna.Trial, base_hp: dict[str, Any]) -> dict[str, Any]:
    hp = _deepfm_family_space(trial, base_hp)
    hp["aux_weight"] = trial.suggest_float("aux_weight", 0.05, 0.5)
    return hp


SEARCH_SPACES["deepfm"] = _deepfm_family_space
SEARCH_SPACES["deepfm_mtl"] = _deepfm_mtl_space


@dataclass
class SearchResult:
    study: optuna.Study
    base_score_reduced: float  # the base config's own score at the SAME reduced fidelity -- the fair comparison
    best_trial_hyperparams: dict[str, Any] | None
    wall_time_s: float


def run_search(base_config: ExperimentConfig, data_dir: str, n_trials: int = 15,
                train_fraction: float = 0.15, max_epochs: int = 8, patience: int = 3,
                seed: int = 0) -> SearchResult:
    """Searches around `base_config`'s hyperparameters (its own values are
    the starting point Optuna's sampler explores from, not discarded) using
    `base_config.model`'s registered search space. Every trial and the base
    config itself are scored identically (same seed, same train_fraction,
    same epoch/patience budget) so the comparison is fair.
    """
    if base_config.model not in SEARCH_SPACES:
        raise ValueError(f"No search space registered for model '{base_config.model}'. "
                          f"Available: {sorted(SEARCH_SPACES)}")
    space_fn = SEARCH_SPACES[base_config.model]
    t0 = time.process_time()

    def _score(hp: dict[str, Any]) -> float:
        cfg = ExperimentConfig(id="__hpo_trial__", model=base_config.model, hypothesis="hpo trial",
                                hyperparams=hp, fields=list(base_config.fields))
        result = run_experiment(cfg, data_dir, seed=seed, train_fraction=train_fraction,
                                 max_epochs_override=max_epochs)
        return result.valid.primary

    base_score_reduced = _score(dict(base_config.hyperparams) | {"patience": patience})

    def objective(trial: optuna.Trial) -> float:
        hp = space_fn(trial, base_config.hyperparams)
        hp["patience"] = patience
        try:
            return _score(hp)
        except Exception as e:  # noqa: BLE001 -- a broken trial (e.g. a bad hidden/k combination)
            # must not kill the whole study; Optuna's sampler learns to avoid this region from a
            # clearly-bad score instead.
            trial.set_user_attr("error", str(e))
            return -1.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    best_hp = None
    if study.best_value > base_score_reduced:
        best_hp = dict(base_config.hyperparams)
        best_hp.update({k: v for k, v in study.best_params.items() if k != "hidden"})
        if "hidden" in study.best_params:
            best_hp["hidden"] = [int(x) for x in study.best_params["hidden"].split("_")]

    return SearchResult(study=study, base_score_reduced=base_score_reduced,
                         best_trial_hyperparams=best_hp, wall_time_s=time.process_time() - t0)
