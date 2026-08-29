"""Standalone check for agent/model_zoo/deepfm_pdaom.py (DeepFM trained
with a pairwise exponential AUC loss + per-user hard-pair mining, aka
"PDAOM", arXiv:2304.09176) -- a genuinely different bet than fm_bpr/
deepfm_bpr's uniform-sampling pairwise BPR loss (tested twice already,
both plateaued below the FM baseline).

Reuses agent.model_zoo.fm_bpr.build_user_pos_neg_index verbatim for
per-user grouping (identical "at least one positive AND one negative"
filter fm_bpr.py already established) -- only the batch construction and
loss differ: instead of one random pos/neg pair per user
(sample_bpr_batches), this gathers up to --max_candidates positives and
negatives per user, pads/masks them, and lets the model select its own
hardest pair each step.

Same base hyperparameters as deepfm_regularized (k=16, hidden=[128,64],
lr=0.001, l2=1e-4, epochs=20, patience=5) so the loss function is the
only variable versus that parent.

Usage: python tools/check_pdaom.py [--max_candidates 8] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from agent.evaluator import score  # noqa: E402
from agent.model_zoo.deepfm_pdaom import build  # noqa: E402
from agent.model_zoo.fm_bpr import build_user_pos_neg_index  # noqa: E402
from agent.paths import DEFAULT_DATA_DIR, ensure_starter_kit_on_path  # noqa: E402

ensure_starter_kit_on_path()
from data import encode, load  # organizer's file, unmodified  # noqa: E402


def make_pdaom_batch(user_ids: list, user_index: dict, X: np.ndarray,
                      max_candidates: int, rng: np.random.Generator):
    B, Fd = len(user_ids), X.shape[1]
    Xp = np.zeros((B, max_candidates, Fd), dtype=np.int32)
    Xn = np.zeros((B, max_candidates, Fd), dtype=np.int32)
    mp = np.zeros((B, max_candidates), dtype=bool)
    mn = np.zeros((B, max_candidates), dtype=bool)
    for b, u in enumerate(user_ids):
        p, n = user_index[u]
        if len(p) > max_candidates:
            p = rng.choice(p, size=max_candidates, replace=False)
        if len(n) > max_candidates:
            n = rng.choice(n, size=max_candidates, replace=False)
        Xp[b, :len(p)] = X[p]
        mp[b, :len(p)] = True
        Xn[b, :len(n)] = X[n]
        mn[b, :len(n)] = True
    return Xp, mp, Xn, mn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--max_candidates", type=int, default=8)
    ap.add_argument("--batch_users", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    splits = load(args.data_dir)
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]
    Xte, yte, ute = enc["test"]

    t0 = time.process_time()
    user_index = build_user_pos_neg_index(utr, ytr)
    eligible_users = list(user_index.keys())
    print(f"{len(eligible_users)}/{len(set(utr))} train users have both a positive and negative "
          f"impression (PDAOM-eligible), built in {time.process_time() - t0:.1f}s")

    model = build(dim, n_fields=Xtr.shape[1], k=16, hidden=[128, 64], lr=args.lr, l2=args.l2, seed=args.seed)

    rng = np.random.default_rng(args.seed)
    n_batches_per_epoch = max(1, len(eligible_users) // args.batch_users)
    best_primary, best_state, best_epoch, bad = -1.0, None, 0, 0
    t1 = time.process_time()
    for ep in range(1, args.epochs + 1):
        losses = []
        for _ in range(n_batches_per_epoch):
            sampled = rng.choice(eligible_users, size=args.batch_users, replace=True)
            Xp, mp, Xn, mn = make_pdaom_batch(list(sampled), user_index, Xtr, args.max_candidates, rng)
            losses.append(model.pdaom_step(Xp, mp, Xn, mn))
        mean_loss = float(np.mean(losses))
        va = score(uva, yva, model.predict(Xva))
        print(f"  epoch {ep:2d} | loss {mean_loss:.4f} | valid primary {va.primary:.4f}")
        if va.primary > best_primary + 1e-5:
            best_primary, best_epoch, bad = va.primary, ep, 0
            best_state = model.get_state()
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.set_state(best_state)
    valid_final = score(uva, yva, model.predict(Xva))
    test_final = score(ute, yte, model.predict(Xte))
    wall = time.process_time() - t1

    print(f"\nBest epoch: {best_epoch}, wall_time={wall:.1f}s, num_params={model.num_params()}")
    print(f"valid: GAUC={valid_final.gauc:.4f} nDCG@5={valid_final.ndcg5:.4f} primary={valid_final.primary:.4f}")
    print(f"test:  GAUC={test_final.gauc:.4f} nDCG@5={test_final.ndcg5:.4f} primary={test_final.primary:.4f}")
    print(f"\ndeepfm_regularized (parent, pointwise BCE, for comparison): valid primary=0.6035, test primary=0.5977")
    print(f"deepfm_bpr_v1_regularized (uniform-pair BPR, for comparison): valid primary=0.5980")
    print(f"deepfm_listwise_v1 (per-user softmax, for comparison): valid primary=0.6033")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
