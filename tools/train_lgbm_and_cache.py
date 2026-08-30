"""Retrains `lgbm_baseline` locally and caches its predictions in the same
`predictions.npz` format every other cached model uses (see
`tools/check_ensemble.py`'s `_find_cached`), so LightGBM can finally be a
4th ensemble component. It previously existed only as a metrics-only
Research Map node (trained on a teammate's machine originally, no cached
predictions locally) -- blocked on Windows by a WDAC application-control
policy refusing to load LightGBM's native DLL (`OSError: [WinError 4551]`),
not a real bug; that policy doesn't exist on macOS.

Same organizer's own encode() pipeline every other model in this project
uses (`data.load()` + `data.encode()`, the 5 BASE_FIELDS) -- guarantees
identical row order/alignment with the cached FM/DeepFM/DeepFM-MTL
predictions this gets ensembled with, which matters more here than trying
to byte-for-byte reproduce whatever standalone encoding the original
teammate's run used. Same hyperparameters as the existing `lgbm_baseline`
Research Map node (native API, categorical_feature on all 5 columns):
num_leaves=63, lr=0.05, min_data_in_leaf=50, feature_fraction=0.9,
bagging_fraction=0.9, num_boost_round=500, early_stopping_rounds=30.

Usage: python tools/train_lgbm_and_cache.py [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

from agent.evaluator import score  # noqa: E402
from agent.paths import DEFAULT_DATA_DIR, EXPERIMENTS_DIR, ensure_starter_kit_on_path  # noqa: E402

ensure_starter_kit_on_path()
from data import encode, load  # organizer's file, unmodified  # noqa: E402

HYPERPARAMS = {
    "objective": "binary",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "verbose": -1,
}
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_id", default="run_lgbm_local")
    args = ap.parse_args()

    splits = load(args.data_dir)
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]
    Xte, yte, ute = enc["test"]
    n_fields = Xtr.shape[1]

    dtrain = lgb.Dataset(Xtr, label=ytr, categorical_feature=list(range(n_fields)), free_raw_data=False)
    dvalid = lgb.Dataset(Xva, label=yva, categorical_feature=list(range(n_fields)), reference=dtrain, free_raw_data=False)

    hp = dict(HYPERPARAMS)
    hp["seed"] = args.seed
    hp["bagging_seed"] = args.seed
    hp["feature_fraction_seed"] = args.seed

    t0 = time.process_time()
    model = lgb.train(
        hp, dtrain, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dvalid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(period=0)],
    )
    wall = time.process_time() - t0

    valid_scores = model.predict(Xva, num_iteration=model.best_iteration)
    test_scores = model.predict(Xte, num_iteration=model.best_iteration)

    valid_result = score(uva, yva, valid_scores)
    test_result = score(ute, yte, test_scores)

    print(f"best_iteration={model.best_iteration}, wall_time={wall:.1f}s")
    print(f"valid: GAUC={valid_result.gauc:.4f} nDCG@5={valid_result.ndcg5:.4f} primary={valid_result.primary:.4f}")
    print(f"test:  GAUC={test_result.gauc:.4f} nDCG@5={test_result.ndcg5:.4f} primary={test_result.primary:.4f}")
    print("\nlgbm_baseline (existing Research Map node, single seed 0, original teammate run,"
          " for comparison): valid primary=0.5995, test primary=0.5946")

    out_dir = EXPERIMENTS_DIR / args.run_id / "iter_001"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "predictions.npz", valid_scores=valid_scores, test_scores=test_scores)
    (out_dir / "results.json").write_text(json.dumps({
        "run_id": args.run_id,
        "iteration_id": "iter_001",
        "timestamp": time.time(),
        "status": "ok",
        "config_id": "lgbm_baseline",
        "model": "lgbm",
        "hyperparams": {**HYPERPARAMS, "num_boost_round": NUM_BOOST_ROUND,
                        "early_stopping_rounds": EARLY_STOPPING_ROUNDS, "seed": args.seed,
                        "best_iteration": model.best_iteration},
        "metrics": {
            "n_seeds": 1,
            "valid": {"GAUC_mean": valid_result.gauc, "GAUC_std": 0.0,
                      "nDCG@5_mean": valid_result.ndcg5, "nDCG@5_std": 0.0,
                      "primary_mean": valid_result.primary, "primary_std": 0.0},
            "test": {"GAUC_mean": test_result.gauc, "GAUC_std": 0.0,
                     "nDCG@5_mean": test_result.ndcg5, "nDCG@5_std": 0.0,
                     "primary_mean": test_result.primary, "primary_std": 0.0},
        },
        "wall_time_s": wall,
        "note": "Retrained locally on macOS (tools/train_lgbm_and_cache.py) specifically to cache "
                "predictions.npz for tools/check_ensemble.py's 4-component run -- the existing "
                "lgbm_baseline Research Map node was trained on a teammate's Windows machine with "
                "no predictions cached there. Uses the organizer's own data.encode() pipeline "
                "(same as every other cached model here) rather than a from-scratch categorical "
                "encoding, so results may not be bit-identical to that original single-seed run, "
                "though they should land in the same ballpark (same task, same fields, same "
                "hyperparameters).",
    }, indent=2))
    print(f"\nCached predictions.npz + results.json under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
