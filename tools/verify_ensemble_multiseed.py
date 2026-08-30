"""3-seed verification of tools/check_ensemble.py's blend -- same discipline
this project applies to every other single-seed result close to its parent
(features_v1, deepfm_mtl_click_v1): a seed-0-only aggregate delta of +0.0002
valid primary is well inside the noise floor on its own, so before calling
this an improvement we check whether it survives real seed variance.

Blend WEIGHTS are fixed from check_ensemble.py's seed-0 grid search
(fm=0.20, deepfm_regularized=0.10, deepfm_mtl_v1=0.70) and never re-tuned
per seed here -- re-optimizing weights per seed would leak seed-specific
noise into the "decision" and make this check meaningless. Each seed's
z-score normalization is still fit fresh on that seed's own valid scores
(an unsupervised transform, not a decision), matching check_ensemble.py.

Trains each of the 3 component models at seeds [0, 1, 2] (reusing seed 0's
already-cached predictions.npz, training only 1 and 2), computes the fixed
blend at each seed, and runs it through the exact same seed-aware
significance bar (agent/diagnosis.py) already used everywhere else in this
project, so the verdict here is the same kind of "clear_improvement" /
"noise_floor" tag used throughout, not a bespoke one.

Usage: python tools/verify_ensemble_multiseed.py [--data_dir ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from agent.diagnosis import diagnose  # noqa: E402
from agent.evaluator import score  # noqa: E402
from agent.experiment import run_experiment  # noqa: E402
from agent.paths import DEFAULT_DATA_DIR, EXPERIMENTS_DIR, ensure_starter_kit_on_path  # noqa: E402
from agent.research_map import ResearchMap  # noqa: E402

ensure_starter_kit_on_path()
from data import encode, load  # organizer's file, unmodified  # noqa: E402

COMPONENTS = ["fm_baseline_repro", "deepfm_regularized", "deepfm_mtl_v1"]
FIXED_WEIGHTS = {"fm_baseline_repro": 0.20, "deepfm_regularized": 0.10, "deepfm_mtl_v1": 0.70}
SEEDS = [0, 1, 2]


def _find_cached(config_id: str, seed: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if not EXPERIMENTS_DIR.exists():
        return None, None
    for run_dir in sorted(EXPERIMENTS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        for iter_dir in run_dir.iterdir():
            results_path, npz_path = iter_dir / "results.json", iter_dir / "predictions.npz"
            if not (results_path.exists() and npz_path.exists()):
                continue
            results = json.loads(results_path.read_text(encoding="utf-8"))
            if results.get("config_id") != config_id:
                continue
            per_seed = (results.get("metrics") or {}).get("per_seed") or []
            if any(r.get("seed") == seed for r in per_seed):
                npz = np.load(npz_path)
                return npz["valid_scores"], npz["test_scores"]
    return None, None


def _zscore(fit_on: np.ndarray, *arrs: np.ndarray) -> list[np.ndarray]:
    mu, sd = float(fit_on.mean()), float(fit_on.std())
    sd = sd if sd > 1e-9 else 1.0
    return [(a - mu) / sd for a in arrs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    args = ap.parse_args()

    map = ResearchMap(Path("logs") / "research_map.json")
    configs = {}
    for cid in COMPONENTS:
        if cid not in map.nodes:
            print(f"'{cid}' not found in the Research Map -- run it first.")
            return 1
        configs[cid] = map.nodes[cid].config

    print("Loading raw splits + organizer's own encode() for users/labels...")
    splits = load(args.data_dir)
    enc, _ = encode(splits)
    _, yva, uva = enc["valid"]
    _, yte, ute = enc["test"]

    per_seed_mtl, per_seed_blend = [], []
    for seed in SEEDS:
        print(f"\n{'=' * 60}\nseed {seed}\n{'=' * 60}")
        raw_valid, raw_test = {}, {}
        for cid in COMPONENTS:
            v, t = _find_cached(cid, seed)
            if v is None:
                print(f"  training {cid} @ seed={seed} (not cached)...")
                iter_dir = EXPERIMENTS_DIR / "ensemble_verify" / f"{cid}_seed{seed}"
                iter_dir.mkdir(parents=True, exist_ok=True)
                result, model, this_enc = run_experiment(
                    configs[cid], args.data_dir, seed=seed, return_model=True,
                    save_predictions_to=str(iter_dir / "predictions.npz"))
                (iter_dir / "results.json").write_text(json.dumps({
                    "config_id": cid,
                    "metrics": {"per_seed": [{"seed": seed,
                                               "valid": result.valid.to_dict(),
                                               "test": result.test.to_dict()}]},
                }, indent=2), encoding="utf-8")
                npz = np.load(iter_dir / "predictions.npz")
                v, t = npz["valid_scores"], npz["test_scores"]
                print(f"    valid primary={result.valid.primary:.4f}")
            else:
                print(f"  {cid} @ seed={seed}: reusing cached predictions.")
            raw_valid[cid], raw_test[cid] = v, t

        z_valid = {cid: _zscore(raw_valid[cid], raw_valid[cid])[0] for cid in COMPONENTS}
        z_test = {cid: _zscore(raw_valid[cid], raw_test[cid])[0] for cid in COMPONENTS}

        blend_valid = sum(FIXED_WEIGHTS[cid] * z_valid[cid] for cid in COMPONENTS)
        blend_test = sum(FIXED_WEIGHTS[cid] * z_test[cid] for cid in COMPONENTS)

        r_mtl_valid = score(uva, yva, raw_valid["deepfm_mtl_v1"])
        r_mtl_test = score(ute, yte, raw_test["deepfm_mtl_v1"])
        r_blend_valid = score(uva, yva, blend_valid)
        r_blend_test = score(ute, yte, blend_test)

        print(f"  deepfm_mtl_v1 solo : valid primary={r_mtl_valid.primary:.4f}  test primary={r_mtl_test.primary:.4f}")
        print(f"  fixed blend        : valid primary={r_blend_valid.primary:.4f}  test primary={r_blend_test.primary:.4f}")

        per_seed_mtl.append(r_mtl_valid)
        per_seed_blend.append(r_blend_valid)

    def _agg(results):
        gauc = np.array([r.gauc for r in results])
        ndcg = np.array([r.ndcg5 for r in results])
        prim = np.array([r.primary for r in results])
        return {"n_seeds": len(results), "GAUC_mean": float(gauc.mean()), "GAUC_std": float(gauc.std()),
                "nDCG@5_mean": float(ndcg.mean()), "nDCG@5_std": float(ndcg.std()),
                "primary_mean": float(prim.mean()), "primary_std": float(prim.std())}

    agg_mtl = _agg(per_seed_mtl)
    agg_blend = _agg(per_seed_blend)

    print(f"\n{'=' * 60}\n3-seed summary (valid)\n{'=' * 60}")
    print(f"  deepfm_mtl_v1 solo : {agg_mtl['primary_mean']:.4f} +/- {agg_mtl['primary_std']:.4f}")
    print(f"  fixed blend        : {agg_blend['primary_mean']:.4f} +/- {agg_blend['primary_std']:.4f}")
    print(f"  delta              : {agg_blend['primary_mean'] - agg_mtl['primary_mean']:+.4f}")

    d = diagnose({"valid": agg_blend, "n_seeds": 3}, {"valid": agg_mtl, "n_seeds": 3})
    print(f"\nDiagnosis (agent/diagnosis.py, same significance bar as every other Research Map node):")
    print(f"  [{d.tag}] {d.insight}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
