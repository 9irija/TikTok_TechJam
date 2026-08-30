"""Ensemble check -- a structurally different lever from every single-model
refinement tried so far (5 BPR variants, LambdaRank, focal loss, PCGrad,
is_click aux head all failed or tied; see docs/P2_FEATURES_AND_RESULTS.md
Sec.17-20). Combines three ALREADY-TRAINED models with genuinely different
inductive biases -- fm_baseline_repro (pure bilinear), deepfm_regularized
(+ MLP deep component), deepfm_mtl_v1 (+ 4 auxiliary engagement heads,
current best) -- via a validation-only-tuned weighted blend of their
prediction scores. No retraining: reuses each model's cached
`predictions.npz` exactly the way tools/generate_submission.py and
tools/check_per_segment.py already do.

Why this might help where more single-model tweaks haven't: GAUC/nDCG@5
reward correctly RANKING each user's impressions, and three models that
make different mistakes on different rows can rank-correct each other on
average (classic ensemble variance reduction) -- especially relevant given
tools/check_per_segment.py's finding that deepfm_mtl_v1's win over
deepfm_regularized is NOT uniform (negative in item-popularity Q2/Q3,
positive only at the extremes). This is exactly the kind of segment-level
disagreement an ensemble can average away, which is also why this script
re-runs the per-segment check on the blended result -- directly answering
"does this generalize across other types of data" rather than trusting one
aggregate number.

Row-alignment note: all three cached runs used the same 5 BASE_FIELDS, so
they all went through the *same* deterministic `agent/experiment.py:_load_encoded`
-> organizer's own `data.encode()` path (verified: no shuffling in encode(),
rows processed in split order) -- valid_scores[i] / test_scores[i] refer to
the identical underlying row across all three files, checked defensively
below (matching users lists) before combining anything.

Blend weights are grid-searched using VALID primary ONLY (never test) --
same discipline as every other "config selection" in this project. Test
primary is reported purely for tracking, per the project's train/valid/test
rule (see CLAUDE.md).

Usage: python tools/check_ensemble.py [--data_dir ...]
"""
from __future__ import annotations

import argparse
import itertools
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

COMPONENTS = ["fm_baseline_repro", "deepfm_regularized", "deepfm_mtl_v1", "lgbm_baseline"]
# lgbm_baseline added this pass: a genuinely different inductive bias (gradient-
# boosted trees, not learned embeddings) than the other three, which are all
# embedding-based and plausibly correlated in their errors -- the leading
# candidate explanation for why the original 3-component (all-embedding) blend
# only tied deepfm_mtl_v1 instead of beating it. Retrained locally
# (tools/train_lgbm_and_cache.py) specifically to get a real predictions.npz
# cached for it -- previously metrics-only, no cached predictions anywhere.


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
    raise FileNotFoundError(f"No cached predictions found for '{config_id}' -- run it via run.py/run_p1.py first")


def _zscore(fit_on: np.ndarray, *arrs: np.ndarray) -> list[np.ndarray]:
    """Standardizes using stats fit on `fit_on` (valid), applied to every
    array passed -- keeps the transform itself picked without looking at
    test, while still letting test go through the identical affine map."""
    mu, sd = float(fit_on.mean()), float(fit_on.std())
    sd = sd if sd > 1e-9 else 1.0
    return [(a - mu) / sd for a in arrs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--grid_step", type=float, default=0.1, help="Weight grid resolution (simplex over 3 models)")
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
        assert len(v) == len(yva), f"{cid}: valid_scores length {len(v)} != valid split {len(yva)}"
        assert len(t) == len(yte), f"{cid}: test_scores length {len(t)} != test split {len(yte)}"
        raw_valid[cid], raw_test[cid] = v, t

    print("Z-scoring each model's scores (stats fit on valid, same affine map applied to test)...")
    z_valid, z_test = {}, {}
    for cid in COMPONENTS:
        zv, zt = _zscore(raw_valid[cid], raw_valid[cid], raw_test[cid])
        z_valid[cid], z_test[cid] = zv, zt

    solo = {cid: score(uva, yva, raw_valid[cid]).primary for cid in COMPONENTS}
    print("\nSolo valid primary (for reference):")
    for cid in COMPONENTS:
        print(f"  {cid:24s} {solo[cid]:.4f}")

    print(f"\nGrid-searching blend weights (step={args.grid_step}) on VALID primary only "
          f"({len(COMPONENTS)} components -> {len(COMPONENTS) - 1}-dimensional simplex search)...")
    steps = np.arange(0.0, 1.0 + 1e-9, args.grid_step)
    best = None
    for free_weights in itertools.product(steps, repeat=len(COMPONENTS) - 1):
        if sum(free_weights) > 1.0 + 1e-9:
            continue
        w_last = max(1.0 - sum(free_weights), 0.0)
        weights = list(free_weights) + [w_last]
        blend = sum(w * z_valid[cid] for w, cid in zip(weights, COMPONENTS))
        p = score(uva, yva, blend).primary
        if best is None or p > best[0]:
            best = (p, weights)

    best_valid_primary, weights = best
    weight_str = "  ".join(f"{cid}={w:.2f}" for cid, w in zip(COMPONENTS, weights))
    print(f"\nBest blend: {weight_str}")
    print(f"  blend valid primary = {best_valid_primary:.4f}")
    print(f"  vs. deepfm_mtl_v1 solo (current best) = {solo['deepfm_mtl_v1']:.4f} "
          f"(delta {best_valid_primary - solo['deepfm_mtl_v1']:+.4f})")

    blend_test = sum(w * z_test[cid] for w, cid in zip(weights, COMPONENTS))
    r_test = score(ute, yte, blend_test)
    r_test_mtl = score(ute, yte, raw_test["deepfm_mtl_v1"])
    print(f"\n  blend test primary (tracking only, never used to pick the blend) = {r_test.primary:.4f}")
    print(f"  vs. deepfm_mtl_v1 solo test primary = {r_test_mtl.primary:.4f} "
          f"(delta {r_test.primary - r_test_mtl.primary:+.4f})")

    # Per-segment breakdown -- does the blend actually smooth out
    # deepfm_mtl_v1's known item-popularity Q2/Q3 weakness, or just win on
    # average while still failing the same segments?
    import collections
    train_rows, valid_rows = splits["train"], splits["valid"]
    video_counts = collections.Counter(r[2] for r in train_rows)
    all_valid_videos = sorted({r[2] for r in valid_rows})
    values = np.array([video_counts.get(k, 0) for k in all_valid_videos])
    edges = np.quantile(values, [0.25, 0.5, 0.75])
    video_bucket = {k: int(np.searchsorted(edges, video_counts.get(k, 0))) for k in all_valid_videos}
    vva = [r[2] for r in valid_rows]

    blend_valid = sum(w * z_valid[cid] for w, cid in zip(weights, COMPONENTS))

    print(f"\n{'=' * 78}\nPer-segment: blend vs. deepfm_mtl_v1 solo, by ITEM popularity quartile (valid)\n{'=' * 78}")
    print(f"  {'segment':18s} {'n_rows':>9s} {'mtl_solo':>10s} {'blend':>10s} {'delta':>9s}")
    for q in range(4):
        idx = np.array([i for i, k in enumerate(vva) if video_bucket.get(k, -1) == q])
        if len(idx) == 0:
            continue
        u_sub, y_sub = [uva[i] for i in idx], yva[idx]
        r_mtl = score(u_sub, y_sub, raw_valid["deepfm_mtl_v1"][idx])
        r_blend = score(u_sub, y_sub, blend_valid[idx])
        qlabel = f"Q{q + 1}" + (" (least)" if q == 0 else " (most)" if q == 3 else "")
        print(f"  {qlabel:18s} {len(idx):9,d} {r_mtl.primary:10.4f} {r_blend.primary:10.4f} "
              f"{r_blend.primary - r_mtl.primary:+9.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
