"""Follow-up to `tools/check_per_segment.py` §15 (docs/P2_FEATURES_AND_RESULTS.md):
that check found deepfm_mtl_v1's win over deepfm_regularized is NOT uniform by
item (video) popularity -- positive at both popularity extremes, a real
negative delta in the two middle quartiles (worst Q2: -0.0066). One specific,
untested hypothesis was written down but never checked: mid-popularity videos
have enough interaction volume for the POINTWISE (long_view) signal to already
be well-estimated, but not enough for the sparser AUXILIARY signals
(is_like/is_follow/is_comment/is_forward) to add anything beyond noise at that
density -- while low-popularity videos might behave differently for a reason
not yet identified.

This script tests the DENSITY half of that hypothesis directly: no new
training, no new model -- purely descriptive statistics over each item-
popularity quartile's TRAIN-split auxiliary-signal rates, using the exact
same quartile boundaries §15 already used (reuses check_per_segment.py's own
bucketing function and video-count source so the buckets are guaranteed
identical, not recomputed and hoped to match).

Two loaders are cross-checked, not assumed to agree: the organizer's own
data.load() (tuple rows, used for the quartile boundaries, matching §15
exactly) and agent/features.py's load_splits() (dict rows, carries the aux
label fields data.load() doesn't). Joined by video_id (both are raw,
uncast strings from csv.DictReader against the same two log files and the
same train date range -- verified via an assertion below, not just claimed
in this docstring).

Usage: python tools/check_popularity_dip_hypothesis.py [--data_dir ...]
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

import numpy as np  # noqa: E402

from agent.features import AUX_LABEL_FIELDS, load_splits  # noqa: E402
from agent.paths import DEFAULT_DATA_DIR, ensure_starter_kit_on_path  # noqa: E402
from check_per_segment import _quartile_labels  # noqa: E402

ensure_starter_kit_on_path()
from data import load  # organizer's file, unmodified  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    args = ap.parse_args()

    print("Loading raw splits both ways (organizer's data.load() for the exact")
    print("Section 15 quartile boundaries; agent/features.py's load_splits() for the")
    print("auxiliary label fields data.load() doesn't carry)...")
    splits_official = load(args.data_dir)          # tuple rows: (date, uid, vid, aid, tab, dur, y)
    splits_features = load_splits(args.data_dir)    # dict rows: video_id, is_like, is_follow, ...

    train_official = splits_official["train"]
    train_features = splits_features["train"]
    valid_official = splits_official["valid"]

    # Cross-check the two loaders actually agree before trusting a join by
    # video_id across them -- exactly the kind of silent-mismatch risk this
    # project treats as a real bug class, not a formality.
    assert len(train_official) == len(train_features), (
        f"train row-count mismatch: data.load()={len(train_official)} vs "
        f"load_splits()={len(train_features)} -- the two loaders have diverged, "
        f"do not trust a video_id join below."
    )
    official_vids = collections.Counter(r[2] for r in train_official)
    features_vids = collections.Counter(r["video_id"] for r in train_features)
    assert official_vids == features_vids, (
        "per-video train impression counts disagree between data.load() and "
        "load_splits() -- do not trust a video_id join below."
    )
    print(f"  Cross-check OK: both loaders agree on {len(train_official):,} train rows "
          f"across {len(official_vids):,} distinct videos.")

    # Exactly §15's own bucketing: train-split impression-count quartiles,
    # applied to the videos that actually appear in valid (same call, same
    # function, reused rather than reimplemented).
    video_counts = official_vids
    all_valid_videos = sorted({r[2] for r in valid_official})
    video_bucket = _quartile_labels(dict(video_counts), all_valid_videos)

    # Per-video aux-signal totals, computed once over the full train split.
    aux_pos_by_video: dict[str, np.ndarray] = collections.defaultdict(lambda: np.zeros(len(AUX_LABEL_FIELDS)))
    for row in train_features:
        v = np.array([row[f] for f in AUX_LABEL_FIELDS], dtype=np.float64)
        aux_pos_by_video[row["video_id"]] += v

    print(f"\n{'=' * 100}")
    print("Item-popularity quartiles (train-split impression counts) vs. auxiliary-signal density")
    print(f"{'=' * 100}")
    header = f"  {'segment':10s} {'videos':>8s} {'rows/video':>11s}"
    for f in AUX_LABEL_FIELDS:
        header += f" {f + '_rate':>13s} {f + '/vid':>9s}"
    print(header)

    quartile_summary = []
    for q in range(4):
        vids_in_q = [v for v in all_valid_videos if video_bucket.get(v, -1) == q]
        if not vids_in_q:
            continue
        rows_in_q = sum(video_counts[v] for v in vids_in_q)
        rows_per_video = rows_in_q / len(vids_in_q)

        line_parts = []
        rates, per_video_counts = [], []
        for i, f in enumerate(AUX_LABEL_FIELDS):
            total_pos = sum(aux_pos_by_video[v][i] for v in vids_in_q)
            rate = total_pos / rows_in_q if rows_in_q else 0.0
            per_video = total_pos / len(vids_in_q)
            rates.append(rate)
            per_video_counts.append(per_video)
            line_parts.append(f" {rate:13.4%} {per_video:9.2f}")

        qlabel = f"Q{q + 1}" + (" (least)" if q == 0 else " (most)" if q == 3 else "")
        print(f"  {qlabel:10s} {len(vids_in_q):8,d} {rows_per_video:11.1f}" + "".join(line_parts))
        quartile_summary.append({
            "q": q, "n_videos": len(vids_in_q), "rows_per_video": rows_per_video,
            "rates": rates, "per_video_counts": per_video_counts,
        })

    # §15's own already-measured deltas, printed alongside for direct visual
    # comparison -- not recomputed here (would need cached predictions this
    # script deliberately doesn't load, staying pure read-only over raw data).
    known_deltas = {0: +0.0035, 1: -0.0066, 2: -0.0012, 3: +0.0019}
    print(f"\n{'=' * 100}")
    print("For reference -- Section 15's already-measured deepfm_mtl_v1 vs. deepfm_regularized delta per quartile:")
    print(f"{'=' * 100}")
    for row in quartile_summary:
        q = row["q"]
        qlabel = f"Q{q + 1}" + (" (least)" if q == 0 else " (most)" if q == 3 else "")
        mean_rate = float(np.mean(row["rates"]))
        mean_per_video = float(np.mean(row["per_video_counts"]))
        print(f"  {qlabel:10s} known delta {known_deltas[q]:+.4f}   "
              f"mean aux rate {mean_rate:.4%}   mean aux events/video {mean_per_video:.2f}")

    print(f"\n{'=' * 100}")
    print("Hypothesis check")
    print(f"{'=' * 100}")
    print("""The hypothesis (docs/P2_FEATURES_AND_RESULTS.md Section 15) was that Q2/Q3's
negative delta comes from auxiliary-signal SPARSITY: enough volume for the
main pointwise signal to already be well-estimated, not enough for the
sparser auxiliary signals to add real gradient beyond noise. If that were
the driver, mean aux rate and/or mean aux-events-per-video should be
lower in Q2/Q3 than in Q1/Q4, tracking the delta's own Q1(+) Q2(--) Q3(-) Q4(+)
shape. Read the two tables above directly against that prediction -- this
script reports the numbers, it does not pre-declare a verdict.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
