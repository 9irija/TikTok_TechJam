"""Stochastic Weight Averaging (Izmailov et al. 2018) on `deepfm_mtl_v1` --
one of the two items from this project's own original brainstorm (focal
loss, LambdaRank, SWA, cascade modeling) never actually tried; focal loss
and LambdaRank both came back regressions (docs/P2_FEATURES_AND_RESULTS.md
Sec.19-20). Genuinely orthogonal to everything else tried this project:
every prior lever changed the loss function, the architecture, the
training signal, or combined multiple already-trained MODELS (ensembling).
SWA changes none of that -- it changes how a single training run's final
checkpoint is CONSTRUCTED, averaging weights across a window of epochs
instead of taking the single best-validation-epoch point estimate every
other model in this project uses.

Motivated directly by `deepfm_mtl_v1`'s own real training curve (seed 0,
already on record): valid primary sits in a flat plateau roughly epochs
7-14 (0.6036-0.6049), with epoch 12 the single best point, before genuinely
dropping off by epoch 15+. A flat plateau across several epochs is exactly
the textbook case for SWA -- the individual epoch checkpoints in a
plateau are different, roughly-equally-good local solutions; averaging
them in weight space, not prediction space (unlike this project's
ensembling checks, which average PREDICTIONS of separately-trained
models), can land in a flatter, better-generalizing region than any single
epoch's point estimate. No BatchNorm layers in `DeepFM_MTL`'s architecture
(agent/model_zoo/deepfm_mtl.py's `_Net` -- embeddings, Linear, ReLU only),
so plain elementwise parameter averaging needs no extra recalibration step
(the usual SWA caveat about BN running-stats doesn't apply here).

Selection window: every epoch whose valid primary is within
`--tolerance` of the best epoch's (default 0.0015, roughly matching the
project's own ~0.0004-0.002 significance-bar range) -- not a fixed "last K
epochs" rule, which would be sensitive to exactly when early stopping
happened to trigger.

Usage: python tools/check_swa.py [--data_dir ...] [--seed 0] [--tolerance 0.0015]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from agent.evaluator import score  # noqa: E402
from agent.features import AUX_LABEL_FIELDS, load_aux_labels  # noqa: E402
from agent.model_zoo.deepfm_mtl import build  # noqa: E402
from agent.paths import DEFAULT_DATA_DIR, ensure_starter_kit_on_path  # noqa: E402

ensure_starter_kit_on_path()
from data import encode, load  # organizer's file, unmodified  # noqa: E402

# Exact deepfm_mtl_v1 hyperparameters -- SWA is the only variable.
HP = {"k": 16, "hidden": [128, 64], "lr": 0.001, "l2": 1e-4, "aux_weight": 0.2}
EPOCHS, PATIENCE, BATCH = 20, 5, 8192


def _average_states(states: list[dict]) -> dict:
    keys = states[0].keys()
    return {k: torch.stack([s[k].float() for s in states], dim=0).mean(dim=0) for k in keys}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=0.0015)
    args = ap.parse_args()

    splits = load(args.data_dir)
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]
    Xte, yte, ute = enc["test"]
    aux = load_aux_labels(args.data_dir, fields=list(AUX_LABEL_FIELDS))
    aux_tr = aux["train"]

    model = build(dim=dim, n_fields=Xtr.shape[1], seed=args.seed, **HP)
    rng = np.random.default_rng(args.seed)

    best_primary, best_epoch, best_state, bad = -1.0, 0, None, 0
    epoch_states: list[dict] = []
    epoch_primaries: list[float] = []
    for ep in range(1, EPOCHS + 1):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            model.mtl_step(Xtr[b], ytr[b], aux_tr[b])
        va = score(uva, yva, model.predict(Xva))
        epoch_states.append(model.get_state())
        epoch_primaries.append(va.primary)
        print(f"  epoch {ep:2d} | valid primary {va.primary:.4f}")

        if va.primary > best_primary + 1e-5:
            best_primary, best_epoch, best_state, bad = va.primary, ep, model.get_state(), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    print(f"\nBest single epoch: {best_epoch}, valid primary {best_primary:.4f}")

    window = [i for i, p in enumerate(epoch_primaries) if p >= best_primary - args.tolerance]
    print(f"SWA window (tolerance={args.tolerance}): epochs {[i + 1 for i in window]} "
          f"({len(window)} of {len(epoch_primaries)} trained epochs)")

    swa_state = _average_states([epoch_states[i] for i in window])
    model.set_state(swa_state)
    swa_valid = score(uva, yva, model.predict(Xva))
    swa_test = score(ute, yte, model.predict(Xte))

    model.set_state(best_state)
    point_valid = score(uva, yva, model.predict(Xva))
    point_test = score(ute, yte, model.predict(Xte))

    print(f"\nPoint-best checkpoint (epoch {best_epoch}): valid primary={point_valid.primary:.4f}, "
          f"test primary={point_test.primary:.4f}")
    print(f"SWA-averaged checkpoint ({len(window)} epochs): valid primary={swa_valid.primary:.4f}, "
          f"test primary={swa_test.primary:.4f}")
    print(f"Delta (SWA - point-best): valid {swa_valid.primary - point_valid.primary:+.4f}, "
          f"test {swa_test.primary - point_test.primary:+.4f}")
    print(f"\ndeepfm_mtl_v1 (3-seed verified, for comparison): valid primary=0.6046 +/- 0.0003")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
