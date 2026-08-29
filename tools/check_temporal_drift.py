"""Temporal drift check (priority #3 from the user's own next-steps list) --
does anything that "worked" on the train window quietly stop holding up
across the train->valid boundary, or within the valid window itself? Two
checks, both using only already-available data (no new training):

1. Distribution drift: label positive rate, and cold-start ratio (how many
   valid users/videos were never seen in train at all) across the
   train/valid boundary -- the basic sanity check for whether train and
   valid are actually comparable populations, not just adjacent dates.
2. Per-day model performance: deepfm_regularized vs. deepfm_mtl_v1,
   scored separately for each of the 7 days in the valid window (reusing
   already-cached predictions, same reuse pattern as
   tools/check_per_segment.py) -- does the multi-task win hold steady
   day-by-day, or decay/grow across the window?

Usage: python tools/check_temporal_drift.py [--data_dir ...]
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
from data import load  # organizer's file, unmodified  # noqa: E402


def _find_cached_valid_predictions(config_id: str) -> np.ndarray:
    for run_dir in sorted(EXPERIMENTS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        for iter_dir in run_dir.iterdir():
            results_path, npz_path = iter_dir / "results.json", iter_dir / "predictions.npz"
            if not (results_path.exists() and npz_path.exists()):
                continue
            results = json.loads(results_path.read_text(encoding="utf-8"))
            if results.get("config_id") == config_id:
                return np.load(npz_path)["valid_scores"]
    raise FileNotFoundError(f"No cached valid predictions found for '{config_id}'")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    args = ap.parse_args()

    splits = load(args.data_dir)
    tr, va = splits["train"], splits["valid"]
    # tuple layout: (date, user_id, video_id, author_id, tab, duration_ms, label)

    print("=" * 78)
    print("1. Distribution drift across the train -> valid boundary")
    print("=" * 78)
    tr_users, va_users = {r[1] for r in tr}, {r[1] for r in va}
    tr_videos, va_videos = {r[2] for r in tr}, {r[2] for r in va}
    tr_labels = np.array([r[6] for r in tr], dtype=np.float32)
    va_labels = np.array([r[6] for r in va], dtype=np.float32)
    print(f"  Valid users also seen in train:  {len(va_users & tr_users):,} / {len(va_users):,} "
          f"({len(va_users & tr_users) / len(va_users):.1%}) -- {100 - len(va_users & tr_users) / len(va_users) * 100:.1f}% cold-start")
    print(f"  Valid videos also seen in train: {len(va_videos & tr_videos):,} / {len(va_videos):,} "
          f"({len(va_videos & tr_videos) / len(va_videos):.1%}) -- {100 - len(va_videos & tr_videos) / len(va_videos) * 100:.1f}% cold-start")
    print(f"  Train positive rate (long_view): {tr_labels.mean():.4f}")
    print(f"  Valid positive rate (long_view): {va_labels.mean():.4f}  (delta {va_labels.mean() - tr_labels.mean():+.4f})")

    print("\n" + "=" * 78)
    print("2. Per-day model performance within the valid window")
    print("=" * 78)
    scores_base = _find_cached_valid_predictions("deepfm_regularized")
    scores_best = _find_cached_valid_predictions("deepfm_mtl_v1")
    uva = [r[1] for r in va]
    yva = va_labels
    dates = np.array([r[0] for r in va])

    print(f"  {'date':>10s} {'n_rows':>9s} {'pos_rate':>9s} {'base(reg)':>11s} {'best(mtl)':>11s} {'delta':>9s}")
    for d in sorted(set(dates)):
        idx = np.where(dates == d)[0]
        u_sub = [uva[i] for i in idx]
        y_sub = yva[idx]
        r_base = score(u_sub, y_sub, scores_base[idx])
        r_best = score(u_sub, y_sub, scores_best[idx])
        print(f"  {int(d):>10d} {len(idx):>9,d} {y_sub.mean():>9.4f} {r_base.primary:>11.4f} "
              f"{r_best.primary:>11.4f} {r_best.primary - r_base.primary:>+9.4f}")

    r_all_base = score(uva, yva, scores_base)
    r_all_best = score(uva, yva, scores_best)
    print(f"\n  {'ALL 7 DAYS':>10s} {len(yva):>9,d} {yva.mean():>9.4f} {r_all_base.primary:>11.4f} "
          f"{r_all_best.primary:>11.4f} {r_all_best.primary - r_all_base.primary:>+9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
