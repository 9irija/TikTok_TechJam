"""Turns an ExperimentConfig into a trained model + metrics (P0).

This is the function that actually runs inside the Failure Recovery
subprocess (agent/recovery.py) -- it must be a plain, picklable, top-level
function (no closures) so `multiprocessing` can ship it to a worker process
and really kill it on timeout, not just hope a flag gets checked in time.

Training loop mirrors the starter kit's own FM early-stopping pattern
(track best validation primary, stop after `patience` epochs without
improvement) so results are comparable apples-to-apples with the FM
baseline's own reported numbers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import cache as _cache
from .config import ExperimentConfig
from .evaluator import EvalResult, score
from .model_zoo import build as build_model
from .paths import ensure_starter_kit_on_path

ensure_starter_kit_on_path()
from data import encode, load  # organizer's file, unmodified  # noqa: E402


@dataclass
class ExperimentResult:
    experiment_id: str
    seed: int
    best_epoch: int
    epochs_run: int
    valid: EvalResult
    test: EvalResult
    wall_time_s: float
    num_params: int
    train_loss_curve: list[float] = field(default_factory=list)
    valid_primary_curve: list[float] = field(default_factory=list)
    train_fraction: float = 1.0  # 1.0 = full data; <1.0 = a Multi-Fidelity Runner smoke stage

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "best_epoch": self.best_epoch,
            "epochs_run": self.epochs_run,
            "valid": self.valid.to_dict(),
            "test": self.test.to_dict(),
            "wall_time_s": self.wall_time_s,
            "num_params": self.num_params,
            "train_loss_curve": self.train_loss_curve,
            "valid_primary_curve": self.valid_primary_curve,
            "train_fraction": self.train_fraction,
        }


_ENCODED_CACHE: dict[tuple, Any] = {}  # (data_dir, tuple(fields)) -> (enc, dim), in-process memo


def _load_encoded(data_dir: str, fields: list[str]):
    """Two-level cache for the encoded arrays: in-process (helps if multiple
    experiments ever run sequentially in one process) over a disk cache
    (agent/cache.py -- helps every subprocess spawn after the first, which
    is the common case under Failure Recovery's one-subprocess-per-experiment
    isolation). Without the disk layer, every experiment attempt re-parses
    the 1.14M-row CSVs from scratch (~10.6s) purely as spawn overhead.
    """
    key = (data_dir, tuple(fields))
    if key not in _ENCODED_CACHE:
        def _build():
            splits = load(data_dir)
            # data.encode() currently always builds the 5 base FIELDS; custom
            # field subsets are a P1 feature (agent/features.py, not built yet).
            return encode(splits)
        _ENCODED_CACHE[key] = _cache.load_or_build(data_dir, fields, _build)
    return _ENCODED_CACHE[key]


def run_experiment(config: ExperimentConfig, data_dir: str, seed: int,
                    max_epochs_override: int | None = None, return_model: bool = False,
                    save_predictions_to: str | None = None, train_fraction: float = 1.0):
    """Trains `config.model` and returns metrics.

    Test-split metrics are computed and logged here purely for parity with
    the organizer's own kuairand-starter-kit/baseline.py (which prints both
    valid and test every run) -- NOTHING in the Orchestrator, Convergence
    Detector, or model/config selection logic ever reads `.test`; only
    `.valid.primary` drives any decision, per Task Requirement 2 ("develops
    using only the training split and the public validation feedback -- it
    never has access to the hidden test set" during iteration).

    If `return_model=True`, also returns `(model, enc)` so a caller can call
    `model.predict(...)` directly.

    If `save_predictions_to` is given, the valid+test prediction arrays used
    to compute this result are also written there (.npz) -- this is what
    lets run.py's final submission step reuse the validation-best config's
    *already-trained* predictions instead of retraining it a second time
    from scratch purely to materialize a submission CSV (see Orchestrator:
    only `config.seeds[0]` gets this set, since that's the one seed run.py
    ever needs again).

    `train_fraction` (P1's Multi-Fidelity Runner, agent/multi_fidelity.py):
    trains on a random, seeded, deterministic subsample of the *training*
    split only -- validation and test always stay full-size, so a low-
    fidelity stage's score is a real (if noisier, smaller-N) read on
    generalization, not a synthetic proxy metric.

    Models exposing `bpr_step` instead of `step` (P1's FM_BPR -- see
    agent/model_zoo/fm_bpr.py) are trained with the pairwise BPR sampling
    loop instead of the pointwise loop below -- evaluation is identical
    either way, only the training signal differs.
    """
    hp = dict(config.hyperparams)
    epochs = max_epochs_override or hp.get("epochs", 40)
    bs = hp.get("batch", 8192)
    patience = hp.get("patience", 4)
    verbose = hp.get("verbose", False)

    enc, dim = _load_encoded(data_dir, config.fields)
    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]
    Xte, yte, ute = enc["test"]

    if train_fraction < 1.0:
        n = max(1, int(len(ytr) * train_fraction))
        sub_idx = np.random.default_rng(seed).choice(len(ytr), size=n, replace=False)
        Xtr, ytr = Xtr[sub_idx], ytr[sub_idx]
        utr = [utr[i] for i in sub_idx]

    model = build_model(config.model, dim, n_fields=Xtr.shape[1],
                         **{k: v for k, v in hp.items() if k not in ("epochs", "batch", "patience", "verbose")},
                         seed=seed)

    rng = np.random.default_rng(seed)
    best_primary, best_state, best_epoch, bad = -1.0, None, 0, 0
    loss_curve: list[float] = []
    valid_curve: list[float] = []
    # time.process_time(), not time.time(): this measures CPU time actually
    # consumed by this process, not wall-clock elapsed. A host that suspends
    # (sleep) mid-training would inflate a time.time() delta by the full
    # suspended duration once resumed (observed once during P1 validation --
    # a background run survived an ~12h host sleep and reported ~43,371s for
    # what was really ~90s of actual compute); process_time() cannot be
    # inflated this way since a suspended process accrues zero CPU time.
    # This is also the more semantically correct choice for what Feasibility
    # & Practicality resource reporting actually wants: real compute spent,
    # not wall-clock time that might include the process sitting idle.
    t0 = time.process_time()

    is_bpr = hasattr(model, "bpr_step")
    if is_bpr:
        from .model_zoo.fm_bpr import build_user_pos_neg_index, sample_bpr_batches
        user_index = build_user_pos_neg_index(utr, ytr)
        n_batches_per_epoch = max(1, len(ytr) // bs)

    for ep in range(1, epochs + 1):
        if is_bpr:
            losses = [model.bpr_step(xp, xn) for xp, xn in
                      sample_bpr_batches(rng, Xtr, user_index, n_batches_per_epoch, bs)]
        else:
            idx = rng.permutation(len(ytr))
            losses = [model.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        mean_loss = float(np.mean(losses))
        loss_curve.append(mean_loss)

        va = score(uva, yva, model.predict(Xva))
        valid_curve.append(va.primary)
        if verbose:
            print(f"  epoch {ep:2d} | loss {mean_loss:.4f} | valid primary {va.primary:.4f}")

        if va.primary > best_primary + 1e-5:
            best_primary, best_epoch, bad = va.primary, ep, 0
            best_state = model.get_state()
        else:
            bad += 1
            if bad >= patience:
                break

    model.set_state(best_state)
    valid_scores = model.predict(Xva)
    test_scores = model.predict(Xte)
    valid_final = score(uva, yva, valid_scores)
    test_final = score(ute, yte, test_scores)
    wall = time.process_time() - t0

    if save_predictions_to is not None:
        from pathlib import Path
        p = Path(save_predictions_to)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, valid_scores=valid_scores, test_scores=test_scores)

    result = ExperimentResult(
        experiment_id=config.id, seed=seed, best_epoch=best_epoch, epochs_run=len(loss_curve),
        valid=valid_final, test=test_final, wall_time_s=wall, num_params=model.num_params(),
        train_loss_curve=loss_curve, valid_primary_curve=valid_curve, train_fraction=train_fraction,
    )
    if return_model:
        return result, model, enc
    return result
