"""Per-Segment Metric Diagnosis (the "cheapest, do first" item from the
user's own prioritized next-steps list) -- does deepfm_mtl_v1's aggregate
win hold *uniformly* across user/item segments, or is it concentrated in
one and roughly flat/negative elsewhere? An aggregate GAUC/nDCG@5 number
can hide that; this is the direct check. Genuinely no new training run
needed -- reuses each model's already-cached valid-split predictions
(`experiments/*/iter_*/predictions.npz`), exactly the reuse pattern
`tools/generate_submission.py` already established.

Segments, both computed from TRAIN-split counts only (never valid/test --
the same train-only-aggregate discipline `agent/features.py` already
uses, so a segment boundary is never informed by the very labels being
scored):
  - User activity: each user's number of TRAIN-split impressions, bucketed
    into quartiles (Q1 = least active ... Q4 = most active).
  - Item popularity: each video's number of TRAIN-split impressions,
    bucketed into quartiles the same way. A row's segment is its OWN
    video's popularity bucket (not the user's).

For each segment, scores `deepfm_mtl_v1` against `deepfm_regularized`
(same architecture, only the multi-task objective differs -- the cleanest
apples-to-apples comparison already established in this project) on the
valid split, restricted to just that segment's rows.

Usage: python tools/check_per_segment.py [--data_dir ...]
"""
from __future__ import annotations

import argparse
import collections
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


def _quartile_labels(counts: dict[str, int], keys: list[str]) -> dict[str, int]:
    """Maps each key to a quartile index 0-3 (0=least, 3=most) by its count,
    using train-split counts only -- computed once, applied to every row
    referencing that key regardless of split, same convention as
    agent/features.py's train-only aggregates."""
    values = np.array([counts.get(k, 0) for k in keys])
    edges = np.quantile(values, [0.25, 0.5, 0.75])
    return {k: int(np.searchsorted(edges, counts.get(k, 0))) for k in keys}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    args = ap.parse_args()

    print("Loading raw splits (organizer's own load()) to build train-only segment boundaries...")
    splits = load(args.data_dir)
    train_rows, valid_rows = splits["train"], splits["valid"]

    user_counts = collections.Counter(r[1] for r in train_rows)   # tuple layout: (date, uid, vid, aid, tab, dur, y)
    video_counts = collections.Counter(r[2] for r in train_rows)
    all_valid_users = sorted({r[1] for r in valid_rows})
    all_valid_videos = sorted({r[2] for r in valid_rows})
    user_bucket = _quartile_labels(user_counts, all_valid_users)
    video_bucket = _quartile_labels(video_counts, all_valid_videos)

    uva = [r[1] for r in valid_rows]
    vva = [r[2] for r in valid_rows]
    yva = np.array([r[6] for r in valid_rows], dtype=np.float32)

    print("Loading cached valid-split predictions for deepfm_regularized and deepfm_mtl_v1...")
    scores_base = _find_cached_valid_predictions("deepfm_regularized")
    scores_best = _find_cached_valid_predictions("deepfm_mtl_v1")
    assert len(scores_base) == len(yva) == len(scores_best), "cached predictions must align with the valid split"

    def _segment_report(label: str, bucket_of: dict[str, int], row_keys: list[str]) -> None:
        print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
        print(f"  {'segment':10s} {'n_rows':>9s} {'base(reg)':>11s} {'best(mtl)':>11s} {'delta':>9s}")
        for q in range(4):
            idx = np.array([i for i, k in enumerate(row_keys) if bucket_of.get(k, -1) == q])
            if len(idx) == 0:
                continue
            u_sub = [uva[i] for i in idx]
            y_sub = yva[idx]
            r_base = score(u_sub, y_sub, scores_base[idx])
            r_best = score(u_sub, y_sub, scores_best[idx])
            qlabel = f"Q{q + 1}" + (" (least)" if q == 0 else " (most)" if q == 3 else "")
            print(f"  {qlabel:10s} {len(idx):9,d} {r_base.primary:11.4f} {r_best.primary:11.4f} "
                  f"{r_best.primary - r_base.primary:+9.4f}")

    _segment_report("By USER activity (train-split impression count quartiles)", user_bucket, uva)
    _segment_report("By ITEM (video) popularity (train-split impression count quartiles)", video_bucket, vva)

    r_all_base = score(uva, yva, scores_base)
    r_all_best = score(uva, yva, scores_best)
    print(f"\n{'=' * 78}\nOverall (for reference)\n{'=' * 78}")
    print(f"  deepfm_regularized: {r_all_base.primary:.4f}")
    print(f"  deepfm_mtl_v1:      {r_all_best.primary:.4f}  (delta {r_all_best.primary - r_all_base.primary:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
