"""Stacked ensemble via scikit-learn -- a more principled combination method
than tools/check_ensemble.py's hand-tuned grid search over a weight
simplex. Same 3 already-trained, architecturally distinct components
(`fm_baseline_repro`, `deepfm_regularized`, `deepfm_mtl_v1`), but the blend
weights are now genuinely LEARNED by a meta-learner (`LogisticRegression`)
instead of brute-forced on a 0.1-resolution grid -- and scored honestly via
out-of-fold cross-validation on the valid split, not by fitting and scoring
the meta-learner on the exact same rows (which is the standard stacking
discipline: even a tiny, 3-coefficient linear meta-learner should never be
scored on rows it was fit on).

Why this might do better than the grid search's exact tie (§21,
docs/P2_FEATURES_AND_RESULTS.md): a weight simplex search is still
restricted to combinations that sum to 1 with no bias term; logistic
regression on the same 3 features can express a genuinely different
function (its own intercept, coefficients that need not sum to anything in
particular, and it optimizes log-loss directly rather than this project's
scoring metric via grid search). Whether that extra flexibility helps or
is just extra variance to overfit is exactly what the honest out-of-fold
check below answers -- not assumed either way.

Train/valid/test discipline unchanged: the meta-learner's out-of-fold
score on valid is what any decision here would read; test is scored purely
for tracking, exactly like every other node in this project.

Usage: python tools/check_stacked_ensemble.py [--data_dir ...] [--n_splits 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from agent.evaluator import score  # noqa: E402
from agent.paths import DEFAULT_DATA_DIR, EXPERIMENTS_DIR, ensure_starter_kit_on_path  # noqa: E402

ensure_starter_kit_on_path()
from data import encode, load  # organizer's file, unmodified  # noqa: E402

COMPONENTS = ["fm_baseline_repro", "deepfm_regularized", "deepfm_mtl_v1"]


def _find_cached(config_id: str) -> tuple[np.ndarray, np.ndarray]:
    for run_dir in sorted(EXPERIMENTS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        for iter_dir in run_dir.iterdir():
            results_path, npz_path = iter_dir / "results.json", iter_dir / "predictions.npz"
            if not (results_path.exists() and npz_path.exists()):
                continue
            results = json.loads(results_path.read_text(encoding="utf-8"))
            if results.get("config_id") == config_id:
                npz = np.load(npz_path)
                return npz["valid_scores"], npz["test_scores"]
    raise FileNotFoundError(f"No cached predictions found for '{config_id}'")


def _zscore(fit_on: np.ndarray, *arrs: np.ndarray) -> list[np.ndarray]:
    mu, sd = float(fit_on.mean()), float(fit_on.std())
    sd = sd if sd > 1e-9 else 1.0
    return [(a - mu) / sd for a in arrs]


def main() -> int:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
    except ImportError:
        print("scikit-learn is not installed -- `pip install scikit-learn` first. "
              "(Not a hard blocker like lightgbm's DLL issue on this machine -- just not installed yet.)")
        return 1

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("Loading raw splits + organizer's own encode() for users/labels...")
    splits = load(args.data_dir)
    enc, _ = encode(splits)
    _, yva, uva = enc["valid"]
    _, yte, ute = enc["test"]

    print(f"Loading cached predictions for {COMPONENTS}...")
    raw_valid, raw_test = {}, {}
    for cid in COMPONENTS:
        v, t = _find_cached(cid)
        assert len(v) == len(yva) and len(t) == len(yte), f"{cid}: cached predictions don't align with the splits"
        raw_valid[cid], raw_test[cid] = v, t

    z_valid = {cid: _zscore(raw_valid[cid], raw_valid[cid])[0] for cid in COMPONENTS}
    z_test = {cid: _zscore(raw_valid[cid], raw_test[cid])[0] for cid in COMPONENTS}

    X_valid = np.column_stack([z_valid[cid] for cid in COMPONENTS])
    X_test = np.column_stack([z_test[cid] for cid in COMPONENTS])

    solo = score(uva, yva, raw_valid["deepfm_mtl_v1"]).primary
    print(f"\ndeepfm_mtl_v1 solo valid primary (for reference): {solo:.4f}")

    print(f"\nFitting LogisticRegression meta-learner, {args.n_splits}-fold out-of-fold on valid...")
    meta = LogisticRegression(max_iter=1000, random_state=args.seed)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof_scores = cross_val_predict(meta, X_valid, yva, method="predict_proba", cv=skf)[:, 1]
    r_oof = score(uva, yva, oof_scores)
    print(f"  out-of-fold valid primary = {r_oof.primary:.4f} "
          f"(GAUC={r_oof.gauc:.4f}, nDCG@5={r_oof.ndcg5:.4f})")
    print(f"  vs. deepfm_mtl_v1 solo = {solo:.4f} (delta {r_oof.primary - solo:+.4f})")

    print("\nRefitting on the FULL valid split for the deployable meta-learner (test scoring only)...")
    meta.fit(X_valid, yva)
    coefs = dict(zip(COMPONENTS, meta.coef_[0]))
    print(f"  learned coefficients: {coefs}")
    print(f"  intercept: {meta.intercept_[0]:.4f}")

    test_scores = meta.predict_proba(X_test)[:, 1]
    r_test = score(ute, yte, test_scores)
    r_test_mtl = score(ute, yte, raw_test["deepfm_mtl_v1"])
    print(f"\n  test primary (tracking only, never used to pick anything) = {r_test.primary:.4f}")
    print(f"  vs. deepfm_mtl_v1 solo test primary = {r_test_mtl.primary:.4f} "
          f"(delta {r_test.primary - r_test_mtl.primary:+.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
