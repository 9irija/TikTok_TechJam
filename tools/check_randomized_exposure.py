"""Does deepfm_mtl_v1's improvement over the FM baseline hold under an
UNBIASED data distribution, not just TikTok's own recommendation-biased
logs? Every result so far -- train, valid, test -- is drawn from the same
underlying policy (whatever the platform already chose to show users).
`log_random_4_22_to_5_08_pure.csv` is a separate, real file in the dataset:
interactions logged under *randomized* exposure (`is_rand=1`), covering the
same user population and date range as valid+test. The brainstorm doc flags
it explicitly as usable for "an unbiased secondary validation set / IPW /
counterfactual eval" -- never used anywhere in this project until now.

Used STRICTLY as a held-out evaluation set, exactly like valid/test --
never trained on, never used to pick a config. Encoded via the organizer's
own pinned `encode()` (kuairand-starter-kit/data.py), not reimplemented: a
`splits` dict with an extra "random" key, built from a vocab that's still
derived from `train` only, so there is zero risk of an independently-built
vocab silently drifting from what these already-trained models actually
learned.

Retrains both models fresh (single seed=0, their own committed
hyperparameters) since only their predictions on the ORIGINAL valid/test
splits were ever cached, not the trained weights themselves or predictions
on this new split.

Usage: python tools/check_randomized_exposure.py [--data_dir ...]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent.config import ExperimentConfig  # noqa: E402
from agent.evaluator import score  # noqa: E402
from agent.experiment import run_experiment  # noqa: E402
from agent.paths import DEFAULT_DATA_DIR, ensure_starter_kit_on_path  # noqa: E402

ensure_starter_kit_on_path()
from data import encode, load  # organizer's file, unmodified  # noqa: E402

RANDOM_LOG_FILE = "log_random_4_22_to_5_08_pure.csv"


def _load_random_split(data_dir: str) -> list[tuple]:
    """Same row tuple shape as data.py's own `load()` -- (date, user_id,
    video_id, author_id, tab, duration_ms, label) -- built independently
    since this file isn't one `load()` reads, but through the identical
    author-lookup + label logic so the row content is directly comparable."""
    vid2author: dict[str, str] = {}
    with open(os.path.join(data_dir, "video_features_basic_pure.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]

    rows = []
    with open(os.path.join(data_dir, RANDOM_LOG_FILE), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r["date"]), r["user_id"], r["video_id"],
                         vid2author.get(r["video_id"], "UNK"), r["tab"],
                         float(r["duration_ms"]), 1 if r["long_view"] != "0" else 0))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("Loading + encoding the randomized-exposure split via the organizer's own encode()...")
    splits = load(args.data_dir)
    splits["random"] = _load_random_split(args.data_dir)
    enc, dim = encode(splits)
    Xrand, yrand, urand = enc["random"]
    print(f"  {len(yrand):,} randomized-exposure rows, {yrand.sum():,.0f} positive ({yrand.mean():.1%})")
    print(f"  (for reference, valid/test are {enc['valid'][1].mean():.1%} / {enc['test'][1].mean():.1%} positive "
          f"-- TikTok's own recommendation policy already skews toward videos users tend to watch)")

    configs = {
        "fm_baseline_repro": ExperimentConfig(
            id="fm_baseline_repro", model="fm", hypothesis="",
            hyperparams={"k": 16, "lr": 0.001, "batch": 8192, "epochs": 40, "patience": 4}),
        "deepfm_mtl_v1": ExperimentConfig(
            id="deepfm_mtl_v1", model="deepfm_mtl", hypothesis="",
            hyperparams={"k": 16, "lr": 0.001, "l2": 0.0001, "batch": 8192, "epochs": 20,
                         "patience": 5, "hidden": [128, 64], "aux_weight": 0.2}),
    }

    results = {}
    for name, config in configs.items():
        print(f"\nTraining {name} (seed={args.seed})...")
        result, model, _ = run_experiment(config, args.data_dir, seed=args.seed, return_model=True)
        rand_scores = model.predict(Xrand)
        rand_result = score(urand, yrand, rand_scores)
        results[name] = (result, rand_result)
        print(f"  valid:  GAUC={result.valid.gauc:.4f} nDCG@5={result.valid.ndcg5:.4f} primary={result.valid.primary:.4f}")
        print(f"  test:   GAUC={result.test.gauc:.4f} nDCG@5={result.test.ndcg5:.4f} primary={result.test.primary:.4f}")
        print(f"  random: GAUC={rand_result.gauc:.4f} nDCG@5={rand_result.ndcg5:.4f} primary={rand_result.primary:.4f}")

    print("\n" + "=" * 78)
    print("Does the improvement survive an unbiased distribution?")
    print("=" * 78)
    fm_v, fm_t, fm_r = results["fm_baseline_repro"][0].valid.primary, results["fm_baseline_repro"][0].test.primary, results["fm_baseline_repro"][1].primary
    mtl_v, mtl_t, mtl_r = results["deepfm_mtl_v1"][0].valid.primary, results["deepfm_mtl_v1"][0].test.primary, results["deepfm_mtl_v1"][1].primary
    print(f"  {'split':10s} {'FM baseline':>12s} {'deepfm_mtl_v1':>14s} {'delta':>10s}")
    print(f"  {'valid':10s} {fm_v:12.4f} {mtl_v:14.4f} {mtl_v - fm_v:+10.4f}")
    print(f"  {'test':10s} {fm_t:12.4f} {mtl_t:14.4f} {mtl_t - fm_t:+10.4f}")
    print(f"  {'random':10s} {fm_r:12.4f} {mtl_r:14.4f} {mtl_r - fm_r:+10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
